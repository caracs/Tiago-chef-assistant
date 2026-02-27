#!/usr/bin/env python3
# coding: utf-8
"""
mission_controller_3d.py (orchestrator) - versione robusta per arrival_actions
Gestisce l'esecuzione di azioni all'arrivo (pick/place/custom services).
Supporta arrival_actions passati come:
 - lista Python (es. ['pick'])
 - stringa che rappresenta una lista (es. "['pick','place']")
 - stringa separata da virgole (es. "pick,place")
In aggiunta supporta aggiornamento dinamico delle arrival_actions pubblicando
una stringa su /arrival_actions_str (es: "pick" o "pick,custom:/chef_suggestion_service").
Pubblica inoltre il risultato dell'esecuzione su /arrival_actions_result (std_msgs/String).
"""
from __future__ import annotations
import rospy
import math
import subprocess
import shlex
import threading
import tf2_ros
import ast
import time
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
from std_msgs.msg import String  # per ricevere arrival_actions dinamiche e pubblicare risultati

def strip_leading_slash(s: str) -> str:
    return s[1:] if s and s.startswith("/") else s

def _parse_arrival_actions(raw) -> list:
    """
    Accepts raw param and returns a list of action strings.
    Supports:
      - actual list -> returns cleaned list
      - string representation of list (literal) -> parses and returns list
      - csv string -> splits on comma and returns list
      - single string action -> returns [action]
    Normalizes items with strip() and keeps original case (but caller may use lower()).
    """
    if raw is None:
        return []

    # Already a list-like
    if isinstance(raw, (list, tuple)):
        out = []
        for it in raw:
            if it is None:
                continue
            s = str(it).strip()
            if s:
                out.append(s)
        return out

    # If it's a string, try to interpret it
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        # Try to parse Python literal like "['pick','place']"
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                out = []
                for it in parsed:
                    if it is None:
                        continue
                    out.append(str(it).strip())
                return out
        except Exception:
            pass
        # If not a literal list, treat as CSV: "pick,place" or single "pick"
        if ',' in s:
            parts = [p.strip() for p in s.split(',') if p.strip()]
            return parts
        # fallback single value
        return [s]

    # Fallback: convert to string then parse
    try:
        return _parse_arrival_actions(str(raw))
    except Exception:
        return []

def _normalize_action(act: str) -> str:
    """
    Normalizza una azione libera in:
      - "pick"  -> pick standard
      - "place" -> place standard
      - "custom:/service_name" -> custom service (se presente)
    Altre frasi verranno restituite come sono (caller deciderà).
    """
    if not act:
        return act
    s = str(act).strip().lower()

    # Se già custom: pass through
    if s.startswith("custom:") or "/" in s and s.count("/")>=1:
        # se sembra un servizio (es: /dispensa_inspect), convertilo in custom:/servicename
        if s.startswith("/"):
            return f"custom:{s}"
        return act  # mantieni originale, caller invierà il servizio come dato

    # keyword english
    pick_keywords = ["pick", "grab", "take", "pickup", "pick-up", "pick_up"]
    # italian keywords
    pick_keywords_it = ["prendi", "prendere", "prendi l'oggetto", "afferra", "afferrare", "prendilo", "prendimi", "prendi l'oggetto sul", "vai a prendere"]
    for kw in pick_keywords + pick_keywords_it:
        if kw in s:
            return "pick"

    place_keywords = ["place", "drop", "put", "posa", "posare", "lascia", "riporta"]
    for kw in place_keywords:
        if kw in s:
            return "place"

    # also handle snake/camel case variants like pick_object, pickObject
    if "pick" in s and ("object" in s or "oggetto" in s or "item" in s):
        return "pick"

    # if exactly 'pick_object' or 'pickobject'
    if s.replace("_","").replace("-","") == "pickobject" or s == "pick_object":
        return "pick"

    # fallback: return original (could be unknown)
    return act

class MissionController3DMinimal:
    def __init__(self):
        rospy.init_node("mission_controller_3d", anonymous=False)

        # frames and TF
        self.robot_base_frame = rospy.get_param("~robot_base_frame", "base_footprint")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.tf_timeout = float(rospy.get_param("~tf_lookup_timeout", 1.0))

        # approach threshold
        self.approach_reach_tol = float(rospy.get_param("~approach_reach_tol", 0.35))

        # pick/place services and arrival actions (robust parsing)
        self.pick_service_name = rospy.get_param("~pick_service_name", "/pick_gui")
        self.place_service_name = rospy.get_param("~place_service_name", "/place_gui")
        raw_actions = rospy.get_param("~arrival_actions", ["pick"])
        self.arrival_actions = _parse_arrival_actions(raw_actions)
        # normalize to lower-case for comparisons
        self.arrival_actions = [a.lower() for a in self.arrival_actions]

        # optional prepare service to call before the actions (e.g. a motion prep)
        self.prepare_service_name = rospy.get_param("~prepare_service_name", "")

        # service call timeout
        self.pick_service_timeout = float(rospy.get_param("~pick_service_timeout", 5.0))

        # optional: automatically start pick nodes using rosrun/roslaunch commands if the service does not exist
        self.start_pick_nodes = rospy.get_param("~start_pick_nodes", False)
        # list of shell commands to start pick-related nodes (roslaunch/rosrun)
        self.start_pick_nodes_cmds = rospy.get_param("~start_pick_nodes_cmds", [])

        # process handles for spawned processes (if used)
        self._spawned_processes = []

        # TF buffer/listener
        self.tf_buf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # concurrency protection
        self._goal_lock = threading.Lock()
        self._busy = False

        # publisher for arrival actions result
        self.arrival_result_pub = rospy.Publisher('/arrival_actions_result', String, queue_size=1)

        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._goal_cb, queue_size=1)

        # NEW: subscribe to dynamic arrival actions string topic
        rospy.Subscriber("/arrival_actions_str", String, self._arrival_actions_cb, queue_size=1)
        rospy.loginfo("MissionController3DMinimal ready. Waiting for /move_base_simple/goal. On arrival will execute: %s", str(self.arrival_actions))
        rospy.loginfo("Also subscribed to /arrival_actions_str for dynamic arrival action updates")

    def _goal_cb(self, goal: PoseStamped):
        # ensure single goal handled at a time
        if not self._goal_lock.acquire(blocking=False):
            rospy.logwarn("Nuovo goal ricevuto ma un altro è in gestione -> ignoro")
            return
        try:
            if self._busy:
                rospy.logwarn("Busy handling previous goal -> ignoring")
                return
            self._busy = True

            rospy.loginfo("New goal received at (%.3f, %.3f) frame=%s", goal.pose.position.x, goal.pose.position.y, goal.header.frame_id)
            reached = self._wait_robot_reaches(goal, timeout=120.0)
            if not reached:
                rospy.logwarn("Robot did not reach goal within timeout -> aborting trigger")
                # publish failure
                try:
                    self.arrival_result_pub.publish(String(data="failure"))
                except Exception:
                    pass
                self._busy = False
                return

            rospy.loginfo("Robot reached goal -> executing arrival actions: %s", str(self.arrival_actions))

            # Normalize the actions list before executing
            normalized = []
            for act in self.arrival_actions:
                normalized_act = _normalize_action(act)
                normalized.append(normalized_act)
            self.arrival_actions = normalized
            rospy.loginfo("Normalized arrival actions: %s", str(self.arrival_actions))

            # execute configured sequence of actions (pick/place/custom)
            success = self._execute_arrival_actions()

            if success:
                rospy.loginfo("Arrival actions executed successfully.")
                try:
                    self.arrival_result_pub.publish(String(data="success"))
                except Exception:
                    pass
            else:
                rospy.logerr("Arrival actions failed or partially executed.")
                try:
                    self.arrival_result_pub.publish(String(data="failure"))
                except Exception:
                    pass

            self._busy = False

        finally:
            try:
                self._goal_lock.release()
            except Exception:
                pass

    def _arrival_actions_cb(self, msg: String):
        """
        Update the arrival_actions dynamically from a CSV or list-like string.
        The message payload can be:
         - "pick"
         - "pick,place"
         - "['pick','place']"
         - "custom:/chef_suggestion_service"
         - free-form language (will be normalized)
        """
        try:
            new_actions = _parse_arrival_actions(msg.data)
            if not new_actions:
                rospy.logwarn("Received empty arrival_actions on /arrival_actions_str -> ignoring")
                return
            # Keep original casing until normalization on goal arrive
            self.arrival_actions = [a for a in new_actions]
            rospy.loginfo("Updated arrival_actions dynamically: %s", str(self.arrival_actions))
        except Exception as e:
            rospy.logerr("Failed to parse arrival_actions_str: %s", str(e))

    def _wait_robot_reaches(self, goal_pose: PoseStamped, timeout=120.0) -> bool:
        start = rospy.Time.now()
        goal_frame = strip_leading_slash(goal_pose.header.frame_id) if goal_pose.header.frame_id else self.map_frame
        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > timeout:
                return False
            try:
                t = self.tf_buf.lookup_transform(goal_frame, self.robot_base_frame, rospy.Time(0), rospy.Duration(self.tf_timeout))
                # transform holds translation of robot base expressed in goal_frame
                tx = t.transform.translation.x
                ty = t.transform.translation.y
                # compute distance between robot base and goal coordinates (both in goal_frame)
                dx = tx - goal_pose.pose.position.x
                dy = ty - goal_pose.pose.position.y
                dist = math.hypot(dx, dy)
                if dist <= self.approach_reach_tol:
                    rospy.loginfo("Robot within %.3f m of goal (dist=%.3f).", self.approach_reach_tol, dist)
                    return True
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
                rospy.logwarn_throttle(5.0, "TF lookup while waiting for arrival failed: %s", str(e))
            except Exception as e:
                rospy.logwarn("Unexpected error while checking arrival: %s", str(e))
            rate.sleep()
        return False

    def _execute_arrival_actions(self) -> bool:
        """
        Execute the list of self.arrival_actions in order.
        Supported normalized actions: "pick", "place", "custom:<ros_service_name>".
        Returns True if all configured actions returned successfully.
        """
        overall_ok = True
        # optionally call prepare service before the actions
        if self.prepare_service_name:
            rospy.loginfo("Calling prepare service '%s'...", self.prepare_service_name)
            ok = self._call_service_with_optional_spawn(self.prepare_service_name)
            if not ok:
                rospy.logwarn("Prepare service '%s' failed.", self.prepare_service_name)
                overall_ok = False

        for act in self.arrival_actions:
            if not act:
                rospy.logwarn("Skipping empty action entry.")
                overall_ok = False
                continue

            # allow both "pick" and "place" normalized names
            if act == "pick":
                srv_to_call = self.pick_service_name
                rospy.loginfo("Executing arrival action: pick -> calling service '%s'", srv_to_call)
                ok = self._call_service_with_optional_spawn(srv_to_call)
            elif act == "place":
                srv_to_call = self.place_service_name
                rospy.loginfo("Executing arrival action: place -> calling service '%s'", srv_to_call)
                ok = self._call_service_with_optional_spawn(srv_to_call)
            elif isinstance(act, str) and act.startswith("custom:"):
                svc_name = act.split(":", 1)[1]
                # if custom: received as "custom:/service" or "custom:service"
                if not svc_name.startswith("/"):
                    svc_name = "/" + svc_name
                rospy.loginfo("Executing arrival action: custom:%s", svc_name)
                ok = self._call_service_with_optional_spawn(svc_name)
            else:
                rospy.logwarn("Unknown arrival action '%s' -> attempting normalization fallback", act)
                # fallback: if string contains '/': treat as service
                if isinstance(act, str) and "/" in act:
                    # try to call it as service
                    try:
                        ok = self._call_service_with_optional_spawn(act)
                    except Exception:
                        ok = False
                else:
                    ok = False

            if not ok:
                rospy.logerr("Action '%s' failed.", act)
                overall_ok = False
            else:
                rospy.loginfo("Action '%s' succeeded.", act)
        return overall_ok

    def _call_service_with_optional_spawn(self, service_name: str) -> bool:
        """Generic: try to wait+call a ROS service (Empty type) or spawn processes if configured."""
        if not service_name:
            rospy.logwarn("Empty service_name provided -> skipping call.")
            return False
        # ensure service name starts with '/'
        if not service_name.startswith("/"):
            service_name = "/" + service_name
        try:
            rospy.loginfo("Waiting for service '%s' (timeout=%.1fs)...", service_name, self.pick_service_timeout)
            rospy.wait_for_service(service_name, timeout=self.pick_service_timeout)
            rospy.loginfo("Service '%s' available. Calling...", service_name)
            srv = rospy.ServiceProxy(service_name, Empty)
            srv()
            rospy.loginfo("Service '%s' call succeeded.", service_name)
            return True
        except rospy.ROSException:
            rospy.logwarn("Service '%s' not available within %.1fs.", service_name, self.pick_service_timeout)
        except Exception as e:
            rospy.logwarn("Call to '%s' failed: %s", service_name, str(e))

        # service not available: optionally spawn external nodes if configured
        if self.start_pick_nodes and isinstance(self.start_pick_nodes_cmds, list) and len(self.start_pick_nodes_cmds) > 0:
            rospy.loginfo("start_pick_nodes=True and commands provided. Spawning pick nodes via rosrun/roslaunch commands...")
            success = self._spawn_commands(self.start_pick_nodes_cmds)
            if not success:
                rospy.logerr("Failed to spawn pick nodes.")
                return False
            # after spawning, wait and try to call service again
            try:
                rospy.loginfo("Waiting for service '%s' after spawning (timeout=10s)...", service_name)
                rospy.wait_for_service(service_name, timeout=10.0)
                rospy.loginfo("Service now available. Calling...")
                srv = rospy.ServiceProxy(service_name, Empty)
                srv()
                rospy.loginfo("Service call succeeded after spawn.")
                return True
            except Exception as e:
                rospy.logerr("Service still unavailable after spawn: %s", str(e))
                return False
        else:
            rospy.logwarn("start_pick_nodes disabled or no start commands configured. Not spawning pick nodes.")
            return False

    def _spawn_commands(self, cmds: list) -> bool:
        """
        Spawn shell commands (strings) with subprocess.Popen. Commands should be full roslaunch/rosrun commands.
        Returns True if at least one process started.
        """
        ok = False
        for c in cmds:
            try:
                rospy.loginfo("Spawning command: %s", c)
                args = shlex.split(c)
                p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self._spawned_processes.append(p)
                ok = True
            except Exception as e:
                rospy.logerr("Failed to spawn '%s': %s", c, str(e))
        return ok

if __name__ == "__main__":
    try:
        mc = MissionController3DMinimal()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass