#!/usr/bin/env python3
# ingredient_recognition.py - pubblica solo elementi "food" e solo se cambiano
# ora invia anche messaggi di feedback sintetico "vedo <label>" su /robot_feedback
# e saluta una persona ("Ciao umano!") su /robot_speech quando rilevata via YOLO.
# Modifiche: pubblica "vedo <label>" SOLO LA PRIMA VOLTA che quella label viene vista
# durante l'esecuzione (evita ripetizioni consecutive o multiple).

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from loader import Loader
import threading
import time

CONF_THRESH_PARAM = "~conf_threshold"
PUBLISH_RATE_PARAM = "~publish_rate"

# minimal food whitelist (italian + english). Estendila secondo necessità.
FOOD_SET = {
    'uova','uovo','pasta','spaghetti','salsiccia','formaggio','pepe','sale','pomodoro','latte','pane','farina',
    'burro','olio','pollo','manzo','carne','pesce','riso','patate','cipolla','aglio','yogurt','miele','zucchero',
    'pane','formaggio','mozzarella','parmigiano','pomodori','carote','insalata','lattuga', 'wurstel'
}

PERSON_KEYWORDS = ('person','persona','human','uomo','donna','people','walking_person','person_detected','personne')

# cooldowns / rate limits (seconds)
PERSON_GREET_COOLDOWN = 10.0    # greet a detected person at most once every N seconds
INGREDIENT_PUBLISH_MIN_INTERVAL = 0.5  # will also be honored by publish_rate param

def is_food_label(lbl: str) -> bool:
    if not lbl:
        return False
    t = lbl.lower().strip()
    if t in FOOD_SET:
        return True
    # substring heuristic
    for f in FOOD_SET:
        if f in t:
            return True
    return False

class IngredientRecognitionNode:
    def __init__(self):
        rospy.init_node("ingredient_recognition", anonymous=True)
        self.bridge = CvBridge()
        self.loader = Loader()
        # topic publishers
        self.pub = rospy.Publisher("/detected_ingredients", String, queue_size=1)
        # new publishers for feedback and speech
        self.pub_feedback = rospy.Publisher("/robot_feedback", String, queue_size=4)
        self.pub_speech = rospy.Publisher("/robot_speech", String, queue_size=2)

        self.lock = threading.Lock()
        self.latest_ingredients = []
        self.last_pub = 0.0

        # seen labels: publish "vedo <label>" only once per label during this run
        self._seen_feedback_labels = set()

        # last person greet time (to avoid repeated greetings)
        self._last_person_greet_time = 0.0

        self.conf_threshold = rospy.get_param(CONF_THRESH_PARAM, 0.3)
        self.publish_rate = rospy.get_param(PUBLISH_RATE_PARAM, 1.0)  # Hz

        rospy.Subscriber("/xtion/rgb/image_rect_color", Image, self.image_cb, queue_size=1)
        rospy.loginfo("ingredient_recognition started, conf_thresh=%s publish_rate=%s", self.conf_threshold, self.publish_rate)

    def _maybe_publish_feedback_and_greet(self, label: str):
        """
        Publish a short feedback "vedo <label>" on /robot_feedback only if label
        hasn't been published before during this run. If label suggests a person,
        greet (respecting a cooldown).
        """
        if not label:
            return
        label_norm = label.lower().strip()
        now = time.time()

        # Only publish feedback if this label hasn't been seen before
        if label_norm in self._seen_feedback_labels:
            return

        # mark seen and publish feedback
        self._seen_feedback_labels.add(label_norm)
        try:
            fb = f"vedo {label_norm}"
            self.pub_feedback.publish(String(data=fb))
            rospy.loginfo("Published feedback: %s", fb)
        except Exception as e:
            rospy.logwarn("Failed to publish robot_feedback: %s", e)

        # greet if person and cooldown elapsed
        if any(pk in label_norm for pk in PERSON_KEYWORDS):
            if (now - self._last_person_greet_time) > PERSON_GREET_COOLDOWN:
                try:
                    # also publish on robot_speech for TTS / terminal display
                    self.pub_speech.publish(String(data="Ciao umano!"))
                    rospy.loginfo("Detected person -> greeted with 'Ciao umano!'")
                except Exception as e:
                    rospy.logwarn("Failed to publish robot_speech greeting: %s", e)
                self._last_person_greet_time = now

    def image_cb(self, msg: Image):
        """
        Main image callback: runs YOLO via loader.yolow_model.predict(img).
        - builds list of (label, conf)
        - publishes food labels as CSV on /detected_ingredients when changed (existing behavior)
        - publishes a single succinct feedback "vedo <label>" (highest-confidence label)
          but ONLY the first time that label appears (per run).
        """
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", e)
            return

        try:
            bboxes, classes, confidences = self.loader.yolow_model.predict(img)
        except Exception as e:
            rospy.logerr("YOLO predict error: %s", e)
            return

        # collect detections with their confidences
        detections = []
        for cls, conf in zip(classes, confidences):
            try:
                conf_val = float(conf)
            except Exception:
                conf_val = 0.0
            detections.append((str(cls).strip(), conf_val))

        # filter by confidence threshold
        detections = [(c, p) for (c, p) in detections if p >= self.conf_threshold]

        if not detections:
            rospy.logdebug("No detections above threshold.")
            return

        # choose top detection by confidence for feedback
        detections_sorted = sorted(detections, key=lambda x: -x[1])
        top_label, top_conf = detections_sorted[0]

        # Publish concise feedback & possible greeting based on top label (only if first time seen)
        try:
            with self.lock:
                self._maybe_publish_feedback_and_greet(top_label)
        except Exception as e:
            rospy.logwarn("Error while publishing feedback/greeting: %s", e)

        # existing: build ingredients list (food-only)
        ingredients = []
        for cls, conf in detections:
            cname = str(cls).lower().strip()
            if is_food_label(cname):
                ingredients.append(cname)
            else:
                rospy.logdebug("Non-food detected (filtered out for detected_ingredients): %s (conf=%.2f)", cname, conf)

        # deduplicate & publish only on change and rate-limit (same logic as before)
        ingredients = list(dict.fromkeys(ingredients))
        now = time.time()
        with self.lock:
            # respect publish rate param and minimum safe interval
            min_interval = max(1.0 / max(1.0, float(self.publish_rate)), INGREDIENT_PUBLISH_MIN_INTERVAL)
            if ingredients == self.latest_ingredients:
                # nothing new for ingredients
                return
            if (now - self.last_pub) < min_interval:
                rospy.logdebug("Ingredient publish rate-limited (now-last=%.3f min_interval=%.3f)", now - self.last_pub, min_interval)
                return
            self.latest_ingredients = ingredients
            self.last_pub = now

        if ingredients:
            msg = ",".join(ingredients)
            try:
                self.pub.publish(String(data=msg))
                rospy.loginfo("Published detected ingredients: %s", msg)
            except Exception as e:
                rospy.logwarn("Failed to publish /detected_ingredients: %s", e)
        else:
            rospy.loginfo("No food detected in image (filtered).")

    def get_latest_ingredients(self):
        with self.lock:
            return list(self.latest_ingredients)

if __name__ == "__main__":
    node = IngredientRecognitionNode()
    rospy.spin()