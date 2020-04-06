#!/usr/bin/env python
import copy
import sys
from math import pi

import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
import numpy as np
import openpyxl
import rospy
from moveit_commander.conversions import pose_to_list
from moveit_msgs.msg import CollisionObject
from std_msgs.msg import String

ws_location = '/home/rajath/project_workspace/src/panda_arm_motion_planning/dataset'

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


class Movearmtopose(object):
    """
    Panda arm class
    """
    def __init__(self):
        """
        Contructor for the Movearmtopose
        initialized the following parameters:
        move_it commander Robot Commander
        move_it commander Planning Scene Interface
        Move_it Move Group Commander
        obstacle data(box_name, box position box_size)
        step size
        workspace
        """
        super(Movearmtopose, self).__init__()  # initialize the object

        # initialize the Node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_panda_arm_to_pose', anonymous=True, disable_signals=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("panda_arm")
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)
        self.box_name = {}
        self.box_position = {}
        self.box_size = {}
        self.planning_frame = self.group.get_planning_frame()
        self.eef_link = self.group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        self.step_size = 0.01
        self.group.set_planner_id("RRTstarConnectkConfigDefault")
        # self.point_cloud_data = []
        self.xrange = (-0.5, 0.5)
        self.yrange = (-0.5, 0.5)
        self.zrange = (0, 1)
        self.index = 0

    def go_to_joint_state(self):
        """
        class function to move the manipulator from present state to the desired joint state
        :return: bool, if possible to reach goal
        """
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
        """
        class function to plan the path from the current state to the desired cartesian position using an RRT* planner
        and move the manipulator the desired cartesian pose.
        :return: if it is possible to reach the desired state.
        """
        # plan = None
        # while (not plan):
        pose_goal = geometry_msgs.msg.Pose()
        goalx = np.around(((self.xrange[1] - self.xrange[0]) * np.random.random_sample()) + self.xrange[0],
                          decimals=3)
        goaly = np.around(((self.yrange[1] - self.yrange[0]) * np.random.random_sample()) + self.yrange[0],
                          decimals=3)
        goalz = np.around(((self.zrange[1] - self.zrange[0]) * np.random.random_sample()) + self.zrange[0],
                          decimals=3)
        pose_goal.position.x = goalx
        pose_goal.position.y = goaly
        pose_goal.position.z = goalz
        self.group.set_pose_target(pose_goal)
        current_pose = self.group.get_current_pose().pose
        plan = self.group.plan()
        self.positions = np.array([list(plan.joint_trajectory.points[i].positions) for i in range(len(plan.joint_trajectory.points))])
        # print(self.positions)
        self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def add_box(self, i, pose, dimension, timeout=4):
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "world"
        box_pose.pose.orientation.w = pose[3]
        box_pose.pose.position.x = pose[0]
        box_pose.pose.position.y = pose[1]
        box_pose.pose.position.z = pose[2]
        self.box_name[i - 1] = ("box" + str(i))
        self.box_position[i - 1] = pose
        self.box_size[i - 1] = dimension
        self.scene.add_box(self.box_name[i - 1], box_pose, size=tuple(dimension))
        return self.wait_for_state_update(i - 1, box_is_known=True, timeout=timeout)

    def remove_box(self, i, timeout=4):
        self.scene.remove_world_object(self.box_name[i - 1])
        return self.wait_for_state_update(i - 1, box_is_attached=False, box_is_known=False, timeout=timeout)

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

    def get_plan(self):
        if(len(self.positions)):
            np.save('plan_'+str(self.index)+'.npy', self.positions)
            # print("Got plan {0}".format(self.index))
            self.index+=1

    def get_obstacle_data(self):
        lis = []
        for i in self.box_name:
            x, y, z, _ = self.box_position[i]
            xd, yd, zd = self.box_size[i]

            for i in np.arange(x - xd / 2, x + xd / 2, self.step_size):
                for j in np.arange(y - yd / 2, y + yd / 2, self.step_size):
                    lis.append(np.array([i,j,z-zd/2]))
                    lis.append(np.array([i,j,z+zd/2]))

            for i in np.arange(x - xd / 2, x + xd / 2, self.step_size):
                for j in np.arange(z - zd / 2, z + zd / 2, self.step_size):
                    lis.append(np.array([i, y - yd / 2, j]))
                    lis.append(np.array([i, y + yd / 2, j]))

            for i in np.arange(y - yd / 2, y + yd / 2, self.step_size):
                for j in np.arange(z - zd / 2, z + zd / 2, self.step_size):
                    lis.append(np.array([x - xd / 2, i, j]))
                    lis.append(np.array([x + xd / 2, i, j]))

        np.save('pcloud_1.npy', np.array(lis))

def main():
    try:
        arm = Movearmtopose()
        while (not arm.add_box(1, [0.7, 0, 0.1, 0], [0.4, 0.4, .1])): continue
        while (not arm.add_box(2, [0.9, 0., 0.4, 0], [0.4, 0.4, .1])): continue
        while (not arm.add_box(3, [1.1, 0., 0.7, 0], [0.4, 0.4, .1])): continue
        while (not arm.add_box(4, [0., 0.7, 0.1, 0], [0.4, 0.4, .1])): continue
        while (not arm.add_box(5, [0., 0.9, 0.4, 0], [0.4, 0.4, .1])): continue
        while (not arm.add_box(6, [0., 1.1, 0.7, 0], [0.4, 0.4, .1])): continue

        arm.get_obstacle_data()

        while (1):
            arm.go_to_desired_pose()
            arm.get_plan()

    except rospy.ROSInterruptException:
        arm.remove_box(1)
        arm.remove_box(2)
        arm.remove_box(3)
        arm.remove_box(4)
        arm.remove_box(5)
        arm.remove_box(6)

    except KeyboardInterrupt:
        arm.remove_box(1)
        arm.remove_box(2)
        arm.remove_box(3)
        arm.remove_box(4)
        arm.remove_box(5)
        arm.remove_box(6)


if __name__ == '__main__':
    main()