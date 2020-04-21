#!/usr/bin/env python
import numpy as np
import os
import sys

import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
import rospy
import torch
from moveit_commander.conversions import pose_to_list

from utils.pnet import PNet


def all_close(goal, actual, tolerance):
    """
    used to check if the current eff position is close to the goal position.
    :param goal:
    :param actual:
    :param tolerance:
    :return type bool:
    """
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


class NeuralPandaPlanner(object):
    """
    Neural Motion Planner.
    """

    def __init__(self):
        """
        Constructor for NeuralPandaPlanner
        """
        super(NeuralPandaPlanner, self).__init__()  # initialize the object

        # initialize the Node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('Neural_Planner', anonymous=True, disable_signals=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("panda_arm")
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)
        self.group.set_planner_id("RRTstarConnectkConfigDefault")
        self.planning_frame = self.group.get_planning_frame()
        self.eef_link = self.group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        self.joint_goal = self.group.get_current_joint_values()
        self.pose_goal = self.group.get_current_pose()
        self.pnet = PNet(14, 7)
        self.pnet.load_state_dict(torch.load(os.path.join(os.curdir, 'pnet_weights.pt')))
        self.box_name = {}
        self.box_position = {}
        self.box_size = {}

    def plan_path(self, goal_states):
        current_states = self.group.get_current_joint_values()
        states = torch.from_numpy(np.array([current_states, goal_states]).flatten()).float()
        joint_values = self.group.get_random_joint_values()
        plan = self.pnet(states)
        for i in range(7):
            joint_values[i] = plan[i].item()
        # self.group.set_joint_value_target()
        try:
            self.group.go(joint_values, wait=True)
        except:
            pass
        self.group.stop()
        return all_close(goal_states, current_states, 0.01)

    def add_box(self, i, pose, dimension, timeout=4):
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "world"
        box_pose.pose.orientation.w = pose[3]
        box_pose.pose.position.x = pose[0]
        box_pose.pose.position.y = pose[1]
        box_pose.pose.position.z = pose[2]
        self.box_name[i] = ("box" + str(i))
        self.box_position[i] = pose
        self.box_size[i] = dimension
        self.scene.add_box(self.box_name[i], box_pose, size=tuple(dimension))
        return self.wait_for_state_update(i, box_is_known=True, timeout=timeout)

    def remove_box(self, i, timeout=4):
        self.scene.remove_world_object(self.box_name[i])
        return self.wait_for_state_update(i, box_is_attached=False, box_is_known=False, timeout=timeout)

    def wait_for_state_update(self, i, box_is_known=False, box_is_attached=False, timeout=4):
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            attached_objects = self.scene.get_attached_objects([self.box_name[i]])
            is_attached = len(attached_objects.keys()) > 0
            is_known = self.box_name[i] in self.scene.get_known_object_names()
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True
            rospy.sleep(0.1)
            seconds = rospy.get_time()
        return False


def main():
    arm = NeuralPandaPlanner()
    while (not arm.add_box(1, [0.7, 0., 0.1, 0.], [0.4, 0.4, 0.1])): continue
    while (not arm.add_box(2, [0.9, 0., 0.4, 0.], [0.4, 0.4, 0.1])): continue
    while (not arm.add_box(3, [1.1, 0., 0.7, 0.], [0.4, 0.4, 0.1])): continue
    while (not arm.add_box(4, [0., 0.7, 0.1, 0.], [0.4, 0.4, 0.1])): continue
    while (not arm.add_box(5, [0., 0.9, 0.4, 0.], [0.4, 0.4, 0.1])): continue
    while (not arm.add_box(6, [0., 1.1, 0.7, 0.], [0.4, 0.4, 0.1])): continue

    # for i in range(100):
    joint_goal = arm.group.get_current_joint_values()
    joint_goal[0] = 0
    joint_goal[1] = 0
    joint_goal[2] = 0
    joint_goal[3] = 0
    joint_goal[4] = 0
    joint_goal[5] = 0
    joint_goal[6] = 0
    while (not arm.plan_path(joint_goal)):
        print("False")
        continue
    print("True")


if __name__ == '__main__':
    main()
