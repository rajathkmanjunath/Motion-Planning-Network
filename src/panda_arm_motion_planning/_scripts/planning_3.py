#!/usr/bin/env python
import copy
import sys
from math import pi

import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
import numpy as np
import rospy
from moveit_commander.conversions import pose_to_list
from std_msgs.msg import String


def all_close(goal, actual, tolerance):
    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False
    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)
    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)
    return True


class Movearmtopose(object):
    def __init__(self):
        super(Movearmtopose, self).__init__()  # initialize the object

        # initialize the Node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_panda_arm_to_pose', anonymous=True, disable_signals=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)
        self.box_name = ''
        self.planning_frame = self.group.get_planning_frame()
        self.eef_link = self.group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        self.count = 0
        self.group.set_planner_id("RRTstarConnectkConfigDefault")

    def go_to_joint_state(self):
        joint_goal = self.group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -pi / 4
        joint_goal[2] = 0
        joint_goal[3] = -pi / 2
        joint_goal[4] = 0
        joint_goal[5] = pi / 3
        joint_goal[6] = 0
        self.group.go(joint_goal, wait=True)
        self.group.stop()

        current_joints = self.group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def go_to_desired_pose(self):
        dic = {0: "x negetive y negetive", 1: "x positive y negetic", 2: "x positive y negetive",
               3: "x negetive y positive"}
        pose_goal = geometry_msgs.msg.Pose()
        print(dic[self.count])
        if (self.count == 0):
            # pose_goal.orientation.w = 1.0
            pose_goal.position.x = -0.5
            pose_goal.position.y = -0.5
            pose_goal.position.z = 0.1
            # self.count += 1

        elif (self.count == 1):
            # pose_goal.orientation.w = 1.0
            pose_goal.position.x = 0.5
            pose_goal.position.y = -0.5
            pose_goal.position.z = 0.1
            # self.count += 1

        elif (self.count == 2):
            # pose_goal.orientation.w = 1.0
            pose_goal.position.x = 0.5
            pose_goal.position.y = 0.5
            pose_goal.position.z = 0.1
            # self.count += 1

        elif (self.count == 3):
            # pose_goal.orientation.w = 1.0
            pose_goal.position.x = -0.5
            pose_goal.position.y = 0.5
            pose_goal.position.z = 0.1
            # self.count = 0

        self.count = (self.count + 1) % 4
        self.group.set_pose_target(pose_goal)
        current_pose = self.group.get_current_pose().pose
        # (plan, fraction) = self.group.compute_cartesian_path([pose_goal], 0.01, 0.0)
        plan = self.group.plan()
        for i in range(len(plan.joint_trajectory.points)):
            print(plan.joint_trajectory.points[i].positions)
        self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def add_box1(self, timeout=4):
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "world"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = 0.6
        # box_pose.pose.position.y = 1.0
        box_pose.pose.position.z = 0.5
        self.box_name = "box1"
        self.scene.add_box(self.box_name, box_pose, size=(0.1, 0.1, 1))

        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def add_box2(self, timeout=4):
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "world"
        box_pose.pose.orientation.w = 1.0
        # box_pose.pose.position.x = 1.0
        box_pose.pose.position.y = 0.6
        box_pose.pose.position.z = 0.5
        self.box_name = "box2"
        self.scene.add_box(self.box_name, box_pose, size=(0.1, 0.1, 1))

        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def remove_box1(self, timeout=4):
        box_name = "box1"
        scene = self.scene
        scene.remove_world_object(box_name)
        return self.wait_for_state_update(box_is_attached=False, box_is_known=False, timeout=timeout)

    def remove_box2(self, timeout=4):
        box_name = "box2"
        scene = self.scene
        scene.remove_world_object(box_name)
        return self.wait_for_state_update(box_is_attached=False, box_is_known=False, timeout=timeout)

    def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            attached_objects = self.scene.get_attached_objects([self.box_name])
            is_attached = len(attached_objects.keys()) > 0

            is_known = self.box_name in self.scene.get_known_object_names()

            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            rospy.sleep(0.1)
            seconds = rospy.get_time()

        return False


def main():
    try:
        arm = Movearmtopose()
        while (not arm.add_box1()): continue
        while (not arm.add_box2()): continue
        arm.go_to_joint_state()

        while (1):
            arm.go_to_desired_pose()

    except rospy.ROSInterruptException:
        return

    except KeyboardInterrupt:
        arm.remove_box1()
        arm.remove_box2()


if __name__ == '__main__':
    main()
