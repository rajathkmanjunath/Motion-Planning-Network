#!/usr/bin/env python
import argparse

import rospy

from utils.panda_planner import PandaArmPlanner


def main(args):
    try:
        arm = PandaArmPlanner()
        while (not arm.add_box(1, [0.7, 0, 0.1, 0], [0.4, 0.4, .1])): continue
        while (not arm.add_box(2, [0.9, 0., 0.4, 0], [0.4, 0.4, .1])): continue
        while (not arm.add_box(3, [1.1, 0., 0.7, 0], [0.4, 0.4, .1])): continue
        while (not arm.add_box(4, [0., 0.7, 0.1, 0], [0.4, 0.4, .1])): continue
        while (not arm.add_box(5, [0., 0.9, 0.4, 0], [0.4, 0.4, .1])): continue
        while (not arm.add_box(6, [0., 1.1, 0.7, 0], [0.4, 0.4, .1])): continue

        arm.get_obstacle_data(args.path, args.step_size)

        while (arm.index <= args.num_files):
            arm.go_to_joint_state()
            arm.get_plan(args.path)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../dataset', help='location of dataset directory')
    parser.add_argument('--num_files', type=int, default=10000, help='num of files')
    parser.add_argument('--step_size', type=float, default=0.01, help='The step size of the point cloud data')
    args = parser.parse_args()
    main(args)

