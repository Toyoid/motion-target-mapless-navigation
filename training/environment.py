import rospy
import math
import copy
import numpy as np
from shapely.geometry import Point
from simple_laserscan.msg import SimpleScan
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from training.utils import robot_2_goal_dis_dir, euler_2_quat
import sys
sys.path.append('../../')


class GazeboEnvironment:
    """
    Class for Gazebo Environment

    Main Function:
        1. Reset: Rest environment at the end of each episode
        and generate new goal position for next episode

        2. Step: Execute new action and return state
     """
    def __init__(self,
                 laser_scan_half_num=9,
                 obs_near_th=0.35,
                 goal_near_th=0.5,
                 goal_reward=10,
                 obs_reward=-5,
                 goal_dis_amp=5,
                 step_time=0.1):
        """

        :param laser_scan_half_num: half number of scan points
        :param laser_scan_min_dis: Min laser scan distance
        :param laser_scan_scale: laser scan scale
        :param scan_dir_num: number of directions in laser scan
        :param goal_dis_min_dis: minimal distance of goal distance
        :param goal_dis_scale: goal distance scale
        :param obs_near_th: Threshold for near an obstacle
        :param goal_near_th: Threshold for near an goal
        :param goal_reward: reward for reaching goal
        :param obs_reward: reward for reaching obstacle
        :param goal_dis_amp: amplifier for goal distance change
        :param step_time: time for a single step (DEFAULT: 0.1 seconds)
        """
        self.target_init_pos_list = None
        self.env_range = None
        self.obstacle_poly_list = None
        self.robot_init_pose_list = None
        self.scan_half_num = laser_scan_half_num
        self.obs_near_th = obs_near_th
        self.goal_near_th = goal_near_th
        self.goal_reward = goal_reward
        self.obs_reward = obs_reward
        self.goal_dis_amp = goal_dis_amp
        self.step_time = step_time
        # Robot State
        self.robot_pose = [0., 0., 0.]
        self.robot_speed = [0., 0.]
        self.robot_scan = np.zeros(2 * self.scan_half_num)
        self.robot_state_init = False
        self.robot_scan_init = False
        # Goal Position
        self.target_position = [0., 0.]
        self.target_dis_dir_pre = [0., 0.]  # Last step goal distance and direction
        self.target_dis_dir_cur = [0., 0.]  # Current step goal distance and direction
        self.ctrl_goal_pose_pre = [0., 0.]  # Last step controller's goal distance and direction
        self.ctrl_goal_pose_cur = [0., 0.]  # Current step controller's goal distance and direction
        # Path of motion target
        self.target_path = None
        # Subscriber
        rospy.Subscriber('gazebo/model_states', ModelStates, self._robot_state_cb)
        rospy.Subscriber('simplescan', SimpleScan, self._robot_scan_cb)
        # Publisher
        self.pub_action = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=5)
        # Service
        self.pause_gazebo = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause_gazebo = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.set_model_target = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.reset_simulation = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        # Init Subscriber
        while not self.robot_state_init:
            continue
        while not self.robot_scan_init:
            continue
        rospy.loginfo("Finish Subscriber Init...")

    def step(self, action, goal, pred_traj_point1, pred_traj_point2, pred_traj_point3, ita_in_episode):
        """
        Step Function for the Environment

        Take a action for the robot and return the updated state
        :param action: action taken
        :return: state, reward, done
        """
        assert self.target_init_pos_list is not None
        assert self.obstacle_poly_list is not None
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        '''
        First give action to robot and let robot execute and get next state
        '''
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.angular.z = action[1]
        self.pub_action.publish(move_cmd)
        rospy.sleep(self.step_time)
        next_rob_state = self._get_next_robot_state()
        # get next target state
        if self.target_path:
            self.target_position = self.target_path[ita_in_episode]
        self._set_target_pos(self.target_position, 'target')
        self._set_target_pos(goal, 'target_predicted')
        self._set_target_pos(pred_traj_point1, 'pred_traj_point1')
        self._set_target_pos(pred_traj_point2, 'pred_traj_point2')
        self._set_target_pos(pred_traj_point3, 'pred_traj_point3')

        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        '''
        Then stop the simulation
        1. Transform Robot State to DDPG State
        2. Compute Reward of the action
        3. Compute if the episode is ended
        '''
        target_dis, target_dir = robot_2_goal_dis_dir(self.target_position, next_rob_state[0])
        self.target_dis_dir_cur = [target_dis, target_dir]

        ctrl_goal_dis, ctrl_goal_dir = robot_2_goal_dis_dir(goal, next_rob_state[0])
        self.ctrl_goal_pose_cur = [ctrl_goal_dis, ctrl_goal_dir]

        extrinsic_reward, done, intrinsic_reward, intrinsic_done = self._compute_reward(next_rob_state)

        self.target_dis_dir_pre = [self.target_dis_dir_cur[0], self.target_dis_dir_cur[1]]
        self.ctrl_goal_pose_pre = [self.ctrl_goal_pose_cur[0], self.ctrl_goal_pose_cur[1]]
        return next_rob_state, extrinsic_reward, done, intrinsic_reward, intrinsic_done

    def reset(self, ita, new_target_path):
        """
        Reset Function to reset simulation at start of each episode

        Return the initial state after reset
        :param ita: number of route to reset to
        :return: state
        """
        assert self.target_init_pos_list is not None
        assert self.obstacle_poly_list is not None
        assert self.env_range is not None
        assert self.robot_init_pose_list is not None
        assert ita < len(self.target_init_pos_list)
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        '''
        First choose new goal position and set target model to goal,
        '''
        self.target_position = self.target_init_pos_list[ita]
        self._set_target_pos(self.target_position, 'target')
        self._set_target_pos(self.target_position, 'target_predicted')
        self._set_target_pos(self.target_position, 'pred_traj_point1')
        self._set_target_pos(self.target_position, 'pred_traj_point2')
        self._set_target_pos(self.target_position, 'pred_traj_point3')
        '''
        Then reset robot state and get initial state
        '''
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
        '''
        New motion trajectory of target
        '''
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)

        self.target_path = new_target_path
        '''
        Generate initial robot state
        '''
        rob_state = self._get_next_robot_state()
        target_dis, target_dir = robot_2_goal_dis_dir(self.target_position, rob_state[0])
        self.target_dis_dir_pre = [target_dis, target_dir]
        self.target_dis_dir_cur = [target_dis, target_dir]
        self.ctrl_goal_pose_pre = [target_dis, target_dir]
        self.ctrl_goal_pose_cur = [target_dis, target_dir]
        return rob_state

    def set_new_environment(self, init_pose_list, goal_list, obstacle_list, env_range):
        """
        Set New Environment for training
        :param init_pose_list: init pose list of robot
        :param goal_list: goal position list
        :param obstacle_list: obstacle list
        """
        self.robot_init_pose_list = init_pose_list
        self.target_init_pos_list = goal_list
        self.obstacle_poly_list = obstacle_list
        self.env_range = env_range

    def _get_next_robot_state(self):
        """
        Get the combination of state after execute the action for a certain time

        State will be: [robot_pose, robot_spd, scan]
        :return: state
        """
        tmp_robot_pose = copy.deepcopy(self.robot_pose)
        tmp_robot_spd = copy.deepcopy(self.robot_speed)
        tmp_robot_scan = copy.deepcopy(self.robot_scan)
        state = [tmp_robot_pose, tmp_robot_spd, tmp_robot_scan]
        return state

    def _compute_reward(self, state):
        """
        Compute Reward of the action base on current DDPG state and last step goal distance and direction

        Reward:
            1. R_Arrive If Distance to Goal is smaller than D_goal
            2. R_Collision If Distance to Obstacle is smaller than D_obs
            3. a * (Last step distance to goal - current step distance to goal)

        If robot near obstacle then done
        :param state: DDPG state
        :return: reward, done
        """
        done = False
        intrinsic_done = False
        '''
        First compute distance to all obstacles
        '''
        near_obstacle = False
        robot_point = Point(state[0][0], state[0][1])
        for poly in self.obstacle_poly_list:
            tmp_dis = robot_point.distance(poly)
            if tmp_dis < self.obs_near_th:
                near_obstacle = True
                break
        '''
        Assign Rewards
        '''
        if self.target_dis_dir_cur[0] < self.goal_near_th:
            extrinsic_reward = self.goal_reward
            done = True
        elif near_obstacle:
            extrinsic_reward = self.obs_reward
            done = True
        else:
            extrinsic_reward = 0

        if self.ctrl_goal_pose_cur[0] < self.goal_near_th:
            intrinsic_reward = self.goal_reward
            intrinsic_done = True
        elif near_obstacle:
            intrinsic_reward = self.obs_reward
            intrinsic_done = True
        else:
            intrinsic_reward = self.goal_dis_amp * (self.ctrl_goal_pose_pre[0] - self.ctrl_goal_pose_cur[0])
        return extrinsic_reward, done, intrinsic_reward, intrinsic_done

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
        linear_spd = math.sqrt(msg.twist[-1].linear.x**2 + msg.twist[-1].linear.y**2)
        self.robot_pose = [msg.pose[-1].position.x, msg.pose[-1].position.y, yaw]
        self.robot_speed = [linear_spd, msg.twist[-1].angular.z]

    def _robot_scan_cb(self, msg):
        """
        Callback function for robot scan
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

