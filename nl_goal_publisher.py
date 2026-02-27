#!/usr/bin/env python3
# nl_goal_publisher.py
"""
Natural language -> navigation helper.

Usage:
  # direct: try to match place locally, publish PoseStamped
  python3 nl_goal_publisher.py "tiago vai al tavolo"

  # force LLM: publish to /voice_commands instead (semantic_voice_node will handle)
  python3 nl_goal_publisher.py "prendi la bottiglia sul tavolo" --use-llm

The script will:
 - try to match a place name from rosparam /semantic_places
 - if matched, publish a PoseStamped on /move_base_simple/goal
 - otherwise, publish the command on /voice_commands (for LLM processing)
"""
import rospy
import sys
import argparse
import json
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import tf
import re

PLACES_PARAM = "/semantic_places"
VOICE_TOPIC = "/voice_commands"
GOAL_TOPIC = "/move_base_simple/goal"

# simple synonyms map (extendable)
SYNONYMS = {
    "tavolo": ["table", "tavolo", "table1", "table_1", "tavolo1"],
    "cucina": ["kitchen", "cucina"],
    "ingresso": ["entrance", "ingresso", "door"],
    "ufficio": ["office", "ufficio"],
    "tavolo1": ["table1", "first table", "primo tavolo", "tavolo uno"],
    # aggiungi altre mapping come preferisci
}

def load_places():
    # reads rosparam /semantic_places; expected dict name -> [x,y,z,yaw]
    if not rospy.has_param(PLACES_PARAM):
        return {}
    try:
        p = rospy.get_param(PLACES_PARAM)
        if isinstance(p, dict):
            return p
        # if given as a string containing JSON
        if isinstance(p, str):
            return json.loads(p)
    except Exception as e:
        rospy.logerr("Failed to read /semantic_places: %s", e)
    return {}

def match_place_from_text(text, places):
    """
    Try to find a place key that matches the natural language text.
    Strategy:
     - lowercase text
     - substring match place name or synonyms
     - return place_name (key in places)
    """
    text_l = text.lower()
    # direct key match
    for key in places.keys():
        if key.lower() in text_l:
            return key
    # synonyms
    for canonical, variants in SYNONYMS.items():
        for v in variants:
            if v.lower() in text_l:
                # try to find canonical in places keys
                for k in places.keys():
                    if canonical.lower() == k.lower() or canonical.lower() in k.lower():
                        return k
                # otherwise return canonical if present
                if canonical in places:
                    return canonical
    # try fuzzy numeric matching (e.g. "primo tavolo" -> table1)
    # a simple heuristic: look for "primo", "secondo", "1", "2"
    if re.search(r'\b(primo|1|uno)\b', text_l):
        # try common names with 1
        for candidate in ("table1", "tavolo1", "table_1"):
            if candidate in places:
                return candidate
    return None

def yaw_to_quaternion(yaw):
    q = tf.transformations.quaternion_from_euler(0, 0, yaw)
    return q  # (x,y,z,w)

def publish_goal_from_place(place_key, coords):
    """
    coords: [x, y, z, yaw]
    """
    pub = rospy.Publisher(GOAL_TOPIC, PoseStamped, queue_size=1)
    rospy.sleep(0.2)  # allow publisher to register
    x, y, z, yaw = coords[0], coords[1], coords[2] if len(coords) > 2 else 0.0, coords[3] if len(coords) > 3 else 0.0
    ps = PoseStamped()
    ps.header.stamp = rospy.Time.now()
    ps.header.frame_id = "map"
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)
    q = yaw_to_quaternion(float(yaw))
    ps.pose.orientation.x = q[0]
    ps.pose.orientation.y = q[1]
    ps.pose.orientation.z = q[2]
    ps.pose.orientation.w = q[3]
    pub.publish(ps)
    rospy.loginfo("Published PoseStamped to %s for place '%s': (%.3f, %.3f, yaw=%.3f)", GOAL_TOPIC, place_key, x, y, yaw)

def publish_to_voice_commands(text):
    pub = rospy.Publisher(VOICE_TOPIC, String, queue_size=1)
    rospy.sleep(0.2)
    pub.publish(String(data=text))
    rospy.loginfo("Published raw command to %s for LLM processing: \"%s\"", VOICE_TOPIC, text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="Natural language command (quoted).")
    parser.add_argument("--use-llm", action="store_true", help="Force sending to LLM (/voice_commands) instead of local mapping.")
    args = parser.parse_args()

    rospy.init_node("nl_goal_publisher_cli", anonymous=True)

    text = args.text.strip()
    rospy.loginfo("NL command: %s", text)

    places = load_places()
    if args.use_llm:
        publish_to_voice_commands(text)
        return 0

    match = match_place_from_text(text, places)
    if match:
        coords = places.get(match)
        if coords is None:
            rospy.logerr("Found place key '%s' but no coords available.", match)
            publish_to_voice_commands(text)
            return 1
        # coords must be list-like: [x,y,z,yaw]
        try:
            publish_goal_from_place(match, coords)
        except Exception as e:
            rospy.logerr("Failed to publish PoseStamped: %s", e)
            publish_to_voice_commands(text)
            return 1
    else:
        # fallback: send to voice_commands / LLM
        rospy.logwarn("No local match for command; forwarding to /voice_commands for LLM interpretation.")
        publish_to_voice_commands(text)

    return 0

if __name__ == "__main__":
    sys.exit(main())
