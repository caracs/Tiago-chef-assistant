#!/usr/bin/env python3
# coding: utf-8
"""
pick_client.py (adattato)
- Parametrizzato su topic ArUco, frame base e timeout.
- Pubblica /detected_aruco_pose (latch) per permettere al spherical_grasps_server di ascoltare.
- Mantiene la logica originale della demo ma:
  * pick_aruco("pick") -> esegue solo il pick e SALVA last_object_pose
  * pick_aruco("place") -> esegue la fase di place usando last_object_pose
  * i service /pick_gui e /place_gui ritornano EmptyResponse()
"""
import rospy
import time
from tiago_pick_demo.msg import PickUpPoseAction, PickUpPoseGoal
from geometry_msgs.msg import PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from actionlib import SimpleActionClient

import tf2_ros
from tf2_geometry_msgs import do_transform_pose

import numpy as np
from std_srvs.srv import Empty, EmptyResponse

import cv2
from cv_bridge import CvBridge

from moveit_msgs.msg import MoveItErrorCodes
moveit_error_dict = {}
for name in MoveItErrorCodes.__dict__.keys():
    if not name[:1] == '_':
        code = MoveItErrorCodes.__dict__[name]
        moveit_error_dict[code] = name

class SphericalService(object):
    def __init__(self):
        rospy.loginfo("Starting Spherical Grab Service")
        self.pick_type = PickAruco()
        rospy.loginfo("Finished SphericalService constructor")
        # expose services: /pick_gui and /place_gui (std_srvs/Empty)
        self.place_gui = rospy.Service("/place_gui", Empty, self.start_aruco_place)
        self.pick_gui = rospy.Service("/pick_gui", Empty, self.start_aruco_pick)
        rospy.loginfo("Services exposed: /pick_gui, /place_gui")

    def start_aruco_pick(self, req):
        # run pick in separate thread if you want non blocking; here we run inline but return EmptyResponse()
        try:
            self.pick_type.pick_aruco("pick")
        except Exception as e:
            rospy.logerr("Exception in start_aruco_pick: %s", str(e))
        return EmptyResponse()

    def start_aruco_place(self, req):
        try:
            self.pick_type.pick_aruco("place")
        except Exception as e:
            rospy.logerr("Exception in start_aruco_place: %s", str(e))
        return EmptyResponse()

class PickAruco(object):
    def __init__(self):
        rospy.loginfo("Initalizing PickAruco...")
        self.bridge = CvBridge()

        # params (customizzabili)
        self.aruco_topic = rospy.get_param("~aruco_topic", "/aruco_single/pose")
        self.aruco_wait_timeout = float(rospy.get_param("~aruco_wait_timeout", 15.0))
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")

        # TF buffer/listener
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_l = tf2_ros.TransformListener(self.tfBuffer)

        rospy.loginfo("Waiting for /pickup_pose AS...")
        self.pick_as = SimpleActionClient('/pickup_pose', PickUpPoseAction)
        time.sleep(1.0)
        if not self.pick_as.wait_for_server(rospy.Duration(20)):
            rospy.logerr("Could not connect to /pickup_pose AS")
            exit()
        rospy.loginfo("Waiting for /place_pose AS...")
        self.place_as = SimpleActionClient('/place_pose', PickUpPoseAction)
        if not self.place_as.wait_for_server(rospy.Duration(20)):
            rospy.logerr("Could not connect to /place_pose AS")
            exit()

        rospy.loginfo("Setting publishers to torso and head controller...")
        self.torso_cmd = rospy.Publisher(
            '/torso_controller/command', JointTrajectory, queue_size=1)
        self.head_cmd = rospy.Publisher(
            '/head_controller/command', JointTrajectory, queue_size=1)

        # Publisher that other nodes (e.g. spherical_grasps_server) can listen to
        self.detected_pose_pub = rospy.Publisher('/detected_aruco_pose',
                                PoseStamped,
                                queue_size=1,
                                latch=True)

        rospy.loginfo("Waiting for '/play_motion' AS...")
        self.play_m_as = SimpleActionClient('/play_motion', PlayMotionAction)
        if not self.play_m_as.wait_for_server(rospy.Duration(20)):
            rospy.logerr("Could not connect to /play_motion AS")
            exit()
        rospy.loginfo("Connected!")
        rospy.sleep(1.0)
        rospy.loginfo("Done initializing PickAruco.")

        # keep last picked object pose for deferred place
        self.last_object_pose = None

    def strip_leading_slash(self, s):
        return s[1:] if s and s.startswith("/") else s

    def pick_aruco(self, string_operation):
        """
        Modes:
         - "pick": perform pick (attach object), store self.last_object_pose, DO NOT place
         - "place": perform place using self.last_object_pose (if available)
         - other: log and return
        """
        if string_operation not in ["pick", "place"]:
            rospy.logwarn("Unsupported pick_aruco operation: %s", string_operation)
            return

        if string_operation == "pick":
            # Prepare robot for pick
            self.prepare_robot()

            rospy.sleep(0.5)
            rospy.loginfo("spherical_grasp_gui: Waiting for an aruco detection on '%s' (timeout=%.1fs)", self.aruco_topic, self.aruco_wait_timeout)
            try:
                aruco_pose = rospy.wait_for_message(self.aruco_topic, PoseStamped, timeout=self.aruco_wait_timeout)
            except rospy.ROSException:
                rospy.logerr("Timeout waiting for aruco on %s", self.aruco_topic)
                return

            aruco_pose.header.frame_id = self.strip_leading_slash(aruco_pose.header.frame_id)
            rospy.loginfo("Got: %s", str(aruco_pose))

            rospy.loginfo("spherical_grasp_gui: Transforming from frame: %s to '%s'", aruco_pose.header.frame_id, self.base_frame)
            ps = PoseStamped()
            ps.pose.position = aruco_pose.pose.position
            # set stamp to latest common time for robust transform
            try:
                latest_common = self.tfBuffer.get_latest_common_time(self.base_frame, aruco_pose.header.frame_id)
                ps.header.stamp = latest_common
            except Exception:
                ps.header.stamp = rospy.Time(0)
            ps.header.frame_id = aruco_pose.header.frame_id

            transform_ok = False
            while not transform_ok and not rospy.is_shutdown():
                try:
                    transform = self.tfBuffer.lookup_transform(self.base_frame,
                                           ps.header.frame_id,
                                           rospy.Time(0),
                                           rospy.Duration(1.0))
                    aruco_ps = do_transform_pose(ps, transform)
                    transform_ok = True
                except Exception as e:
                    rospy.logwarn("Exception on transforming point... trying again (%s)", str(e))
                    rospy.sleep(0.01)
                    try:
                        ps.header.stamp = self.tfBuffer.get_latest_common_time(self.base_frame, aruco_pose.header.frame_id)
                    except Exception:
                        ps.header.stamp = rospy.Time(0)

            pick_g = PickUpPoseGoal()

            rospy.loginfo("Setting object pose based on detection (in %s)", self.base_frame)
            pick_g.object_pose.pose.position = aruco_ps.pose.position
            # small adjustment to z to place the box center correctly (demo did -0.1*(1/2))
            pick_g.object_pose.pose.position.z -= 0.1 * 0.5
            pick_g.object_pose.header.frame_id = self.base_frame
            pick_g.object_pose.pose.orientation.w = 1.0

            # publish for other nodes (spherical_grasps_server) the centroid pose
            self.detected_pose_pub.publish(pick_g.object_pose)

            rospy.loginfo("Gonna pick: %s", str(pick_g))
            self.pick_as.send_goal_and_wait(pick_g)
            rospy.loginfo("Pick action finished. Checking result")

            result = self.pick_as.get_result()
            if result is None:
                rospy.logerr("Pick action returned no result")
                return
            if str(moveit_error_dict[result.error_code]) != "SUCCESS":
                rospy.logerr("Failed to pick, not trying further")
                return

            # Save pose for later placing
            self.last_object_pose = deepcopy(pick_g.object_pose)
            rospy.loginfo("Pick successful. Saved last_object_pose for future place.")

            # Raise arm (optional), but DO NOT place automatically
            self.prepare_placing_robot()
            rospy.loginfo("Robot prepared and holding object (pick-only mode).")

            return

        # --- place mode ---
        if string_operation == "place":
            if self.last_object_pose is None:
                rospy.logerr("No last_object_pose available for place operation. Aborting place.")
                return

            rospy.loginfo("Starting place operation using stored last_object_pose.")
            self.prepare_placing_robot()
            pick_g = PickUpPoseGoal()
            pick_g.object_pose = deepcopy(self.last_object_pose)

            # attempt placing
            rospy.loginfo("Gonna place: %s", str(pick_g))
            self.place_as.send_goal_and_wait(pick_g)
            rospy.loginfo("Place action finished. Checking result")
            result = self.place_as.get_result()
            if result is None:
                rospy.logerr("Place action returned no result")
            else:
                rospy.loginfo("Place result: %s", str(moveit_error_dict[result.error_code.val]))
            # Clear stored pose after place attempt
            self.last_object_pose = None
            return

    def lower_head(self):
        rospy.loginfo("Moving head down")
        jt = JointTrajectory()
        jt.joint_names = ['head_1_joint', 'head_2_joint']
        jtp = JointTrajectoryPoint()
        jtp.positions = [0.0, -0.75]
        jtp.time_from_start = rospy.Duration(2.0)
        jt.points.append(jtp)
        self.head_cmd.publish(jt)
        rospy.loginfo("Done.")

    def prepare_robot(self):
        rospy.loginfo("Unfold arm safely")
        pmg = PlayMotionGoal()
        pmg.motion_name = 'pregrasp'
        pmg.skip_planning = False
        self.play_m_as.send_goal_and_wait(pmg)
        rospy.loginfo("Done.")

        self.lower_head()

        rospy.loginfo("Robot prepared.")

    def prepare_placing_robot(self):
        rospy.loginfo("Grasp Success")
        pmg = PlayMotionGoal()
        pmg.motion_name = 'pick_final_pose'
        pmg.skip_planning = False
        self.play_m_as.send_goal_and_wait(pmg)
        rospy.loginfo("Done.")

        self.lower_head()

        rospy.loginfo("Robot prepared to place")


if __name__ == '__main__':
    rospy.init_node('pick_aruco_demo')
    sphere = SphericalService()
    rospy.spin()