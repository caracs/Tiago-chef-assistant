#!/usr/bin/env python3
# voice_terminal.py - unified terminal that shows robot speech + chef suggestions but avoids duplicates

import rospy
from std_msgs.msg import String
import threading
import sys
import os
import time

VOICE_TOPIC = "/voice_commands"
ROBOT_SPEECH = "/robot_speech"
ROBOT_FEEDBACK = "/robot_feedback"
CHEF_SUGGESTIONS = "/chef_suggestions"
CHEF_STATUS = "/chef_status"

class VoiceTerminal:
    def __init__(self):
        rospy.init_node("voice_terminal", anonymous=True)
        self.pub = rospy.Publisher(VOICE_TOPIC, String, queue_size=4)
        rospy.Subscriber(ROBOT_SPEECH, String, self.cb_robot_speech, queue_size=4)
        rospy.Subscriber(CHEF_SUGGESTIONS, String, self.cb_chef_suggestions, queue_size=2)
        rospy.Subscriber(CHEF_STATUS, String, self.cb_chef_status, queue_size=2)
        # NEW: subscribe robot_feedback so detection messages and other feedback appear here
        rospy.Subscriber(ROBOT_FEEDBACK, String, self.cb_robot_feedback, queue_size=4)

        # dedupe: keep last printed payload + timestamp to ignore close duplicates
        self._last_printed_text = None
        self._last_printed_time = 0.0
        self._dedup_seconds = 2.0  # if identical text within 2s skip

        # store last feedback to suppress immediate unhelpful robot_speech errors
        self._last_feedback_text = None
        self._last_feedback_time = 0.0

        print("Voice terminal ready. Type commands (or 'exit').")
        t = threading.Thread(target=self._input_loop, daemon=True)
        t.start()
        rospy.sleep(0.1)

    def _input_loop(self):
        while not rospy.is_shutdown():
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                print("Exiting.")
                os._exit(0)
            try:
                self.pub.publish(String(data=line))
            except Exception as e:
                print("[ERROR] failed to publish voice command:", e)

    def _should_print(self, text: str) -> bool:
        if not text:
            return False
        normalized = text.strip()
        now = time.time()
        if self._last_printed_text == normalized and (now - self._last_printed_time) <= self._dedup_seconds:
            return False
        self._last_printed_text = normalized
        self._last_printed_time = now
        return True

    def cb_robot_speech(self, msg: String):
        text = msg.data if msg and msg.data else ""
        if not text:
            return

        # Suppress generic error-like robot messages if a recent positive feedback exists.
        low = text.strip().lower()
        now = time.time()
        recent_feedback = (self._last_feedback_text or "").lower()
        feedback_recent = (now - self._last_feedback_time) <= 3.0

        # phrases considered "unhelpful" / noisy in context
        unhelpful_phrases = [
            "non posso eseguire questa azione",
            "scusa, non ho capito",
            "non ho riconosciuto il comando",
            "non ho ricevuto"
        ]
        if any(p in low for p in unhelpful_phrases) and feedback_recent:
            # if last feedback indicated action started, suppress the robot_speech message
            if ("sto andando" in recent_feedback) or ("ricevuto" in recent_feedback) or ("ho capito" in recent_feedback) or ("inizio movimento" in recent_feedback):
                # do not log to rospy to avoid polluting stdout (use DEBUG)
                rospy.logdebug("voice_terminal: suppressed robot_speech '%s' (recent feedback: '%s')", text, self._last_feedback_text)
                return

        # otherwise print if not duplicate
        if self._should_print(text):
            print(f"[ROBOT] {text}\n")

    def cb_robot_feedback(self, msg: String):
        text = msg.data if msg and msg.data else ""
        if not text:
            return
        # update last feedback (used to suppress spurious robot_speech)
        self._last_feedback_text = text
        self._last_feedback_time = time.time()

        # feedback messages (from nodes) -> print succinctly and not duplicate
        if self._should_print(text):
            print(f"[FEEDBACK] {text}\n")

    def cb_chef_suggestions(self, msg: String):
        text = msg.data if msg and msg.data else ""
        # chef_suggestions is detailed text -> print if not duplicate of last printed text
        if self._should_print(text):
            print(f"[CHEF SUGGESTIONS]\n{text}\n")

    def cb_chef_status(self, msg: String):
        text = msg.data if msg and msg.data else ""
        if text:
            # always print status (short)
            print(f"[CHEF STATUS] {text}\n")

def main():
    vt = VoiceTerminal()
    rospy.spin()

if __name__ == "__main__":
    main()