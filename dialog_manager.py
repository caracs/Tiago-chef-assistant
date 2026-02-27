#!/usr/bin/env python3
# dialog_manager.py
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from agents import llm_call  # usa l'LLM per NLU se disponibile
import re
import json
import os
import time

# Named places dictionary: map name->(x,y,yaw)
NAMED_PLACES = {
    "dispensa": (-2.419, 0.692, 1.571),
    "tavolo1": (0.223, -6.820, 3.1415),
    # aggiungi altri
}

# Topic used by DispensaInspector to suppress local NL handling while it asks questions
SUPPRESS_TOPIC = "/suppress_local_nl"
# Default suppression timeout (s): after this time a stale "on" is auto-cleared
DEFAULT_SUPPRESS_TIMEOUT = float(os.getenv("SUPPRESS_TIMEOUT", rospy.get_param("~suppress_timeout", 30.0)))

class DialogManager:
    def __init__(self):
        rospy.init_node('dialog_manager', anonymous=True)

        self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.pub_actions = rospy.Publisher('/arrival_actions_str', String, queue_size=1)
        self.pub_robot_speech = rospy.Publisher('/robot_speech', String, queue_size=1)

        # suppression: ignore local handling when another node (e.g. dispensa_inspector) asks questions
        self._suppress_local = False
        self._suppress_ts = 0.0
        self._suppress_timeout = DEFAULT_SUPPRESS_TIMEOUT

        rospy.Subscriber(SUPPRESS_TOPIC, String, self._suppress_cb, queue_size=1)
        rospy.Subscriber('/voice_commands', String, self.cb_voice, queue_size=1)

        rospy.loginfo("dialog_manager ready (listening /voice_commands). Suppress topic: %s timeout=%.1f",
                      SUPPRESS_TOPIC, self._suppress_timeout)

    def _suppress_cb(self, msg: String):
        try:
            val = msg.data.strip().lower() if msg and msg.data else ""
            if val == "on":
                self._suppress_local = True
                self._suppress_ts = time.time()
                rospy.loginfo("dialog_manager: suppression ON (ts=%.3f)", self._suppress_ts)
            elif val == "off":
                self._suppress_local = False
                self._suppress_ts = 0.0
                rospy.loginfo("dialog_manager: suppression OFF")
            else:
                rospy.loginfo("dialog_manager: suppression topic got unknown value '%s' (ignoring)", val)
        except Exception as e:
            rospy.logwarn("dialog_manager: error in _suppress_cb: %s", str(e))

    def _check_and_reset_suppression_if_stale(self):
        """Se la soppressione è attiva ma è passata troppo tempo, resettala automaticamente."""
        if self._suppress_local:
            elapsed = time.time() - (self._suppress_ts or 0.0)
            if elapsed > self._suppress_timeout:
                rospy.logwarn("dialog_manager: suppression stale (%.1f s > %.1f s). Auto-resetting suppression.",
                              elapsed, self._suppress_timeout)
                self._suppress_local = False
                self._suppress_ts = 0.0

    def cb_voice(self, msg: String):
        # safety: reset suppression if it's stale
        self._check_and_reset_suppression_if_stale()

        if self._suppress_local:
            # while suppression active, ignore voice commands (they belong to a guided Q/A)
            rospy.loginfo("dialog_manager: suppression active -> ignoring voice command.")
            return

        if not msg or not msg.data:
            return
        text = msg.data.strip().lower()
        rospy.loginfo("DialogManager received: %s", text)

        # RULES (fast path) -- ampliato per cogliere anche "va" oltre a "vai"
        # riconosci frasi tipo: "vai in X", "vai a X", "va al X", "va al tavolo uno", "vai"
        nav_verb_match = re.search(r"\b(vai|va)\b", text)
        if nav_verb_match or text.startswith("vai") or text.startswith("va"):
            # find place token
            for place in NAMED_PLACES.keys():
                # check multiple patterns: place name directly, "al/alla <place>", "a <place>", "in <place>"
                if (place in text) or re.search(r"\b(al|alla|a|in)\s+" + re.escape(place) + r"\b", text):
                    self._goto_place(place)
                    return

        if "prendi" in text or "afferra" in text:
            # instruct mission controller to pick at arrival
            self.pub_actions.publish(String(data="pick"))
            self.pub_robot_speech.publish(String(data="Ok, cerco e prendo l'oggetto."))
            return

        # migliorata la condizione per "ispeziona dispensa"
        if ("ispeziona" in text and "dispensa" in text) or (("dispensa" in text) and ("ispeziona" in text)):
            if "dispensa" in NAMED_PLACES:
                self._goto_place("dispensa")
                # upon arrival mission_controller can call /dispensa_inspect via arrival_actions
                self.pub_actions.publish(String(data="custom:/dispensa_inspect"))
                self.pub_robot_speech.publish(String(data="Vado in dispensa e controllo gli ingredienti."))
                return

        # FALLBACK: use LLM to parse intent (requires agents.llm_call)
        try:
            system_prompt = "You are an intent parser: given a user's command, output JSON {intent:...,place:...,actions:[...]}."
            user_prompt = f"Command: {text}\nReturn only valid JSON."
            resp = llm_call(system_prompt, user_prompt)
            # attempt to parse
            data = json.loads(resp)
            self._handle_intent(data)
        except Exception as e:
            rospy.logwarn("LLM fallback failed or returned non-json: %s", e)
            # rispondi cortesemente all'utente
            try:
                self.pub_robot_speech.publish(String(data="Scusa, non ho capito. Puoi ripetere?"))
            except Exception:
                pass

    def _goto_place(self, place_name):
        try:
            x,y,yaw = NAMED_PLACES[place_name]
        except Exception:
            rospy.logerr("dialog_manager: place '%s' not found in NAMED_PLACES", str(place_name))
            try:
                self.pub_robot_speech.publish(String(data=f"Non riconosco il luogo {place_name}."))
            except Exception:
                pass
            return

        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        # convert yaw to quaternion (simple)
        try:
            import tf
            q = tf.transformations.quaternion_from_euler(0,0,float(yaw))
            ps.pose.orientation.x = q[0]
            ps.pose.orientation.y = q[1]
            ps.pose.orientation.z = q[2]
            ps.pose.orientation.w = q[3]
        except Exception:
            # se tf non disponibile, usa quaternion identità
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = 0.0
            ps.pose.orientation.w = 1.0

        self.pub_goal.publish(ps)
        try:
            self.pub_robot_speech.publish(String(data=f"Sto andando a {place_name}."))
        except Exception:
            pass

    def _handle_intent(self, data):
        if not isinstance(data, dict):
            try:
                self.pub_robot_speech.publish(String(data="Non posso eseguire questa azione."))
            except Exception:
                pass
            return

        intent = data.get("intent")
        if intent == "navigate" and data.get("place"):
            self._goto_place(data["place"])
        elif intent == "pick":
            self.pub_actions.publish(String(data="pick"))
            try:
                self.pub_robot_speech.publish(String(data="Ok, eseguo pick."))
            except Exception:
                pass
        else:
            try:
                self.pub_robot_speech.publish(String(data="Non posso eseguire questa azione."))
            except Exception:
                pass

if __name__ == '__main__':
    try:
        DialogManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass