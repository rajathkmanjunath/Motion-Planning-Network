#!/usr/bin/env python
import numpy as np
import os
import sys

import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
import rospy
from moveit_commander.conversions import pose_to_list


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


class PandaArmPlanner(object):
    """
    Panda arm class
    """

    def __init__(self):
        """
        Contructor for the PandaArmPlanner
        initialized the following parameters:
        move_it commander Robot Commander
        move_it commander Planning Scene Interface
        Move_it Move Group Commander
        obstacle data(box_name, box position box_size)
        step size
        workspace
        """
        super(PandaArmPlanner, self).__init__()  # initialize the object

        # initialize the Node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_panda_arm_to_pose', anonymous=True, disable_signals=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("panda_arm")
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)
        self.group.set_planner_id("RRTstarConnectkConfigDefault")
        self.box_name = {}
        self.box_position = {}
        self.box_size = {}
        self.planning_frame = self.group.get_planning_frame()
        self.eef_link = self.group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        self.index = 1
        self.joint_values = []
        self.joint_goal = self.group.get_current_joint_values()
        self.pose_goal = self.group.get_current_pose()

    def go_to_joint_state(self):
        """
        class function to move the manipulator from present state to the desired joint state
        :return: bool, if possible to reach goal
        """
        self.joint_goal = self.group.get_random_joint_values()
        self.group.set_joint_value_target(self.joint_goal)
        plan = self.group.plan()
        self.joint_values = [list(plan.joint_trajectory.points[i].positions) for i in
                             range(len(plan.joint_trajectory.points))]
        self.group.execute(plan, wait=True)
        self.group.stop()
        current_joints = self.group.get_current_joint_values()
        return all_close(self.joint_goal, current_joints, 0.01)

    def go_to_desired_pose(self):
        """
        class function to plan the path from the current state to the desired cartesian position using an RRT* planner
        and move the manipulator the desired cartesian pose.
        :return: if it is possible to reach the desired state.
        """
        self.pose_goal = geometry_msgs.msg.Pose()
        self.pose_goal.position = self.group.get_random_pose()
        self.group.set_pose_target(self.pose_goal)
        plan = self.group.plan()
        self.joint_values = np.array(
            [list(plan.joint_trajectory.points[i].positions) for i in range(len(plan.joint_trajectory.points))])
        self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        return all_close(self.pose_goal, current_pose, 0.01)

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

    def get_plan(self, path):
        if (len(self.joint_values)):
            for i in range(len(self.joint_values) - 1):
                temp = np.array([self.joint_values[i], self.joint_goal]).reshape(-1)
                np.save(os.path.join(path, 'states' + str(self.index) + '.npy'), temp)
                np.save(os.path.join(path, 'plan' + str(self.index) + '.npy'), self.joint_values[i + 1])
                self.index += 1
            temp = np.array([self.joint_values[-1], self.joint_goal]).reshape(-1)
            np.save(os.path.join(path, 'states' + str(self.index) + '.npy'), temp)
            np.save(os.path.join(path, 'plan' + str(self.index) + '.npy'), self.joint_values[-1])
            self.index += 1

    def get_obstacle_data(self, path, step_size):
        lis = []
        for name in self.box_name:
            x, y, z, _ = self.box_position[name]
            xd, yd, zd = self.box_size[name]

            for i in np.arange(x - xd / 2, x + xd / 2, step_size):
                for j in np.arange(y - yd / 2, y + yd / 2, step_size):
                    lis.append(np.array([i, j, z - zd / 2]))
                    lis.append(np.array([i, j, z + zd / 2]))

            for i in np.arange(x - xd / 2, x + xd / 2, step_size):
                for j in np.arange(z - zd / 2, z + zd / 2, step_size):
                    lis.append(np.array([i, y - yd / 2, j]))
                    lis.append(np.array([i, y + yd / 2, j]))

            for i in np.arange(y - yd / 2, y + yd / 2, step_size):
                for j in np.arange(z - zd / 2, z + zd / 2, step_size):
                    lis.append(np.array([x - xd / 2, i, j]))
                    lis.append(np.array([x + xd / 2, i, j]))

        np.save(os.path.join(path, 'point_cloud.npy'), np.array(lis).reshape(-1))
