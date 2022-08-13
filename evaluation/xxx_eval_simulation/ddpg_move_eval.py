import rospy
import time
import copy
import torch
from simple_laserscan.msg import SimpleScan
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
import sys

sys.path.append('../../')
from training.utils import *
from trajectory.trajectory_prediction import KFPredictor


class DDPGMoveEval:
    """ Perform Evaluation """

    def __init__(self,
                 controller_net,
                 robot_init_pose_list,
                 target_paths_list,
                 obstacle_poly_list,
                 ros_rate=10,
                 max_steps=1000,
                 min_spd=0.05,
                 max_spd=0.5,
                 batch_window=50,
                 action_rand=0.05,
                 scan_half_num=9,
                 goal_th=0.5,
                 obs_near_th=0.18,
                 use_cuda=True,
                 is_record=False):
        self.controller_net = controller_net
        self.robot_init_pose_list = robot_init_pose_list
        self.target_paths_list = target_paths_list
        self.target_position = None
        self.obstacle_poly_list = obstacle_poly_list
        self.ros_rate = ros_rate
        self.max_steps = max_steps
        self.min_spd = min_spd
        self.max_spd = max_spd
        self.batch_window = batch_window
        self.action_rand = action_rand
        self.scan_half_num = scan_half_num
        self.goal_th = goal_th
        self.obs_near_th = obs_near_th
        self.use_cuda = use_cuda
        self.is_record = is_record
        self.record_data = []
        self.done = False
        # Put network to device
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.controller_net.to(self.device)
        # Robot State
        self.robot_state_init = False
        self.robot_scan_init = False
        self.robot_pose = [0., 0., 0.]
        self.robot_spd = [0., 0.]
        self.robot_scan = np.zeros(2 * scan_half_num)
        # Subscriber
        rospy.Subscriber('gazebo/model_states', ModelStates, self._robot_state_cb)
        rospy.Subscriber('simplescan', SimpleScan, self._robot_scan_cb)
        # Publisher
        self.pub_action = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
        # Service
        self.set_model_target = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        # Init Subscriber
        while not self.robot_state_init:
            continue
        while not self.robot_scan_init:
            continue
        rospy.loginfo("Finish Subscriber Init...")

    def run_ros(self):
        """
        ROS ROS Node
        :return: run_data
        """
        run_num = len(self.robot_init_pose_list)
        run_data = {"final_state": np.zeros(run_num),
                    "time": np.zeros(run_num),
                    "robot_path": [],
                    "target_path": []}
        rate = rospy.Rate(self.ros_rate)
        target_ita = 0
        ita_in_episode = 0
        failure_case = 0
        robot_path = []
        record_target_path = []
        # target trajectory
        target_path = self.target_paths_list[0]
        print("Finish loading target's trajectory...")
        self.target_position = [target_path[0][0], target_path[0][1]]
        self._set_new_target(target_ita)
        self._set_target_pos([-100, -100], 'target_predicted')
        self._set_target_pos([-100, -100], 'pred_traj_point1')
        self._set_target_pos([-100, -100], 'pred_traj_point2')
        self._set_target_pos([-100, -100], 'pred_traj_point3')

        print("Test: ", target_ita)
        print("Start Robot Pose: (%.3f, %.3f, %.3f) Goal: (%.3f, %.3f)" %
              (self.robot_init_pose_list[target_ita][0], self.robot_init_pose_list[target_ita][1],
               self.robot_init_pose_list[target_ita][2],
               self.target_position[0], self.target_position[1]))

        episode_start_time = time.time()
        robot_state = self._get_robot_state()
        scaled_state = state_scale(robot_state)
        while not rospy.is_shutdown() and target_ita < run_num:
            if self.done:
                self.done = False
            while not self.done:
                goal_dis, goal_dir = robot_2_goal_dis_dir(self.target_position, robot_state[0])
                scaled_dis = goal_dis if goal_dis != 0 else 0.3
                scaled_dis = 0.3 / scaled_dis
                scaled_dis = scaled_dis if scaled_dis <= 1 else 1
                scaled_dir = goal_dir / math.pi
                encoded_goal = np.array([scaled_dir, scaled_dis])
                '''
                Perform action according to navigation goal and state
                '''
                joint_state_goal = np.concatenate((encoded_goal, scaled_state), axis=0)
                raw_action = self.controller_act(joint_state_goal)
                action = wheeled_network_2_robot_action(
                    raw_action, self.max_spd, self.min_spd
                )
                move_cmd = Twist()
                move_cmd.linear.x = action[0]
                move_cmd.angular.z = action[1]
                self.pub_action.publish(move_cmd)
                ita_in_episode += 1
                rate.sleep()
                '''
                Motion of target
                '''
                self.target_position = target_path[ita_in_episode]
                record_target_path.append(self.target_position)
                self._set_target_pos(self.target_position, 'target')
                '''
                Acquire State Information
                '''
                robot_state = self._get_robot_state()
                scaled_state = state_scale(robot_state)

                goal_dis, goal_dir = robot_2_goal_dis_dir(self.target_position, robot_state[0])
                is_near_obs = self._near_obstacle(robot_state[0])
                robot_path.append(robot_state[0])
                '''
                Set new test target
                '''
                if goal_dis < self.goal_th or is_near_obs or ita_in_episode == self.max_steps:
                    episode_end_time = time.time()
                    run_data['time'][target_ita] = episode_end_time - episode_start_time
                    self.done = True
                    if goal_dis < self.goal_th:
                        print("End: Success")
                        run_data['final_state'][target_ita] = 1
                    elif is_near_obs:
                        failure_case += 1
                        print("End: Obstacle Collision")
                        run_data['final_state'][target_ita] = 2
                    elif ita_in_episode == self.max_steps:
                        failure_case += 1
                        print("End: Out of steps")
                        run_data['final_state'][target_ita] = 3
                    print("Up to step failure number: ", failure_case)
                    run_data['robot_path'].append(robot_path)
                    run_data['target_path'].append(record_target_path)
                    target_ita += 1
                    if target_ita == run_num:
                        break
                    ita_in_episode = 0
                    robot_path = []
                    record_target_path = []
                    # generate next iteration's track
                    target_path = self.target_paths_list[target_ita]
                    print("Finish loading target's trajectory...")
                    self.target_position = [target_path[0][0], target_path[0][1]]
                    self._set_new_target(target_ita)
                    print("Test: ", target_ita)
                    print("Start Robot Pose: (%.3f, %.3f, %.3f) Goal: (%.3f, %.3f)" %
                          (self.robot_init_pose_list[target_ita][0], self.robot_init_pose_list[target_ita][1],
                           self.robot_init_pose_list[target_ita][2],
                           target_path[0][0], target_path[0][1]))

                    robot_state = self._get_robot_state()
                    scaled_state = state_scale(robot_state)

                    episode_start_time = time.time()

        suc_num = np.sum(run_data["final_state"] == 1)
        obs_num = np.sum(run_data["final_state"] == 2)
        out_num = np.sum(run_data["final_state"] == 3)
        print("Success: ", suc_num, " Obstacle Collision: ", obs_num, " Over Steps: ", out_num)
        print("Success Rate: ", suc_num / run_num)
        return run_data

    def controller_act(self, state):
        state = np.array(state).reshape((1, -1))
        state = torch.Tensor(state).to(self.device)
        action = self.controller_net(state).to('cpu')
        action = action.detach().numpy().squeeze()
        noise = np.random.randn(2) * self.action_rand
        action = noise + (1 - self.action_rand) * action
        action = np.clip(action, [0., 0.], [1., 1.])
        return action

    def _set_new_target(self, ita):
        """
        Set new robot pose and goal position
        :param ita: goal ita
        """
        goal_position = self.target_position
        target_msg = ModelState()
        target_msg.model_name = 'target'
        target_msg.pose.position.x = goal_position[0]
        target_msg.pose.position.y = goal_position[1]

        rospy.wait_for_service('gazebo/set_model_state')
        try:
            resp = self.set_model_target(target_msg)
        except rospy.ServiceException as e:
            print("Set Target Service Failed: %s" % e)
        self.pub_action.publish(Twist())
        robot_init_pose = self.robot_init_pose_list[ita]
        robot_init_quat = euler_2_quat(yaw=robot_init_pose[2])
        robot_msg = ModelState()
        robot_msg.model_name = 'mobile_base'
        robot_msg.pose.position.x = robot_init_pose[0]
        robot_msg.pose.position.y = robot_init_pose[1]
        robot_msg.pose.orientation.x = robot_init_quat[1]
        robot_msg.pose.orientation.y = robot_init_quat[2]
        robot_msg.pose.orientation.z = robot_init_quat[3]
        robot_msg.pose.orientation.w = robot_init_quat[0]
        rospy.wait_for_service('gazebo/set_model_state')
        try:
            resp = self.set_model_target(robot_msg)
        except rospy.ServiceException as e:
            print("Set Target Service Failed: %s" % e)
        rospy.sleep(0.5)

    def _set_target_pos(self, goal_position, model_name):
        """
        Set goal position
        """
        target_msg = ModelState()
        target_msg.model_name = model_name
        target_msg.pose.position.x = goal_position[0]
        target_msg.pose.position.y = goal_position[1]

        rospy.wait_for_service('gazebo/set_model_state')
        try:
            resp = self.set_model_target(target_msg)
        except rospy.ServiceException as e:
            print("Set Target Service Failed: %s" % e)

    def _near_obstacle(self, pos):
        """
        Test if robot is near obstacle
        :param pos: robot position
        :return: done
        """
        done = False
        robot_point = Point(pos[0], pos[1])
        for poly in self.obstacle_poly_list:
            tmp_dis = robot_point.distance(poly)
            if tmp_dis < self.obs_near_th:
                done = True
                break
        return done

    def _get_robot_state(self):
        """
        Get the combination of state after execute the action for a certain time

        State will be: [robot_pose, robot_spd, scan]
        :return: state
        """
        tmp_robot_pose = copy.deepcopy(self.robot_pose)
        tmp_robot_spd = copy.deepcopy(self.robot_spd)
        tmp_robot_scan = copy.deepcopy(self.robot_scan)
        state = [tmp_robot_pose, tmp_robot_spd, tmp_robot_scan]
        return state

    def _robot_state_cb(self, msg):
        """
        Callback function for robot state
        :param msg: message
        """
        if self.robot_state_init is False:
            self.robot_state_init = True
        quat = [msg.pose[-1].orientation.x,
                msg.pose[-1].orientation.y,
                msg.pose[-1].orientation.z,
                msg.pose[-1].orientation.w]
        siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
        cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        linear_spd = math.sqrt(msg.twist[-1].linear.x ** 2 + msg.twist[-1].linear.y ** 2)
        self.robot_pose = [msg.pose[-1].position.x, msg.pose[-1].position.y, yaw]
        self.robot_spd = [linear_spd, msg.twist[-1].angular.z]

    def _robot_scan_cb(self, msg):
        """
        Callback function for robot laser scan
        :param msg: message
        """
        if self.robot_scan_init is False:
            self.robot_scan_init = True
        tmp_robot_scan_ita = 0
        for num in range(self.scan_half_num):
            ita = self.scan_half_num - num - 1
            self.robot_scan[tmp_robot_scan_ita] = msg.data[ita]
            tmp_robot_scan_ita += 1
        for num in range(self.scan_half_num):
            ita = len(msg.data) - num - 1
            self.robot_scan[tmp_robot_scan_ita] = msg.data[ita]
            tmp_robot_scan_ita += 1
