import math
import random
import numpy as np
from shapely.geometry import Point, Polygon


def gen_rand_list_env1(size, robot_goal_diff=1.0):
    """
    Generate Random Training Lists for Environment 1
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    robot_init_pose = [8, 8]
    env_range = ((4, 12), (4, 12))
    poly_raw_list = [[(2, 14), (14, 14), (14, 2), (2, 2)]]
    goal_raw_list = [[(4, 12), (12, 12), (12, 4), (4, 4)]]
    '''
    Generate poly_list and goal_list
    '''
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            goal = random.choice(goal_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return env_range, poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list


def gen_rand_list_env2(size, robot_goal_diff=3.0):
    """
    Generate Random Training Lists for Environment 2
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    robot_init_pose_range = ((5.75, 10.25), (-11, -5))
    robot_init_pose_list = []
    init_x, init_y = np.mgrid[robot_init_pose_range[0][0]:robot_init_pose_range[0][1]:0.1,
                     robot_init_pose_range[1][0]:robot_init_pose_range[1][1]:0.1]
    for x in range(init_x.shape[0]):
        for y in range(init_y.shape[1]):
            robot_init_pose_list.append([init_x[x, y], init_y[x, y]])
    env_range = ((2, 14), (-14, -2))
    poly_raw_list = [[(2, -14), (14, -14), (14, -2), (2, -2)],
                     [(5, -4.5), (5, -4), (11, -4), (11, -4.5)],
                     [(5, -11.5), (5, -12), (11, -12), (11, -11.5)],
                     [(5.25, -6), (4.75, -6), (4.75, -10), (5.25, -10)],
                     [(10.75, -6), (11.25, -6), (11.25, -10), (10.75, -10)]]
    goal_raw_list = [[(2, -14), (14, -14), (14, -2), (2, -2)],
                     [(10.75, -11.5), (10.75, -4), (5.25, -4), (5.25, -11.5)],
                     [(5, -4.5), (5, -4), (11, -4), (11, -4.5)],
                     [(5, -11.5), (5, -12), (11, -12), (11, -11.5)],
                     [(5.25, -6), (4.75, -6), (4.75, -10), (5.25, -10)],
                     [(10.75, -6), (11.25, -6), (11.25, -10), (10.75, -10)],
                     [(6, -4.5), (6, -2), (10, -2), (10, -4.5)],
                     [(6, -11.5), (6, -14), (10, -14), (10, -11.5)],
                     [(5.25, -7), (2, -7), (2, -9), (5.25, -9)],
                     [(10.75, -7), (14, -7), (14, -9), (10.75, -9)]]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        robot_init_pose = random.choice(robot_init_pose_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(robot_init_pose_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return env_range, poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list


def gen_rand_list_env3(size, robot_goal_diff=1.0):
    """
    Generate Random Training Lists for Environment 3
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    robot_init_pose_range = ((-9.75, -6.25), (-9, -7))
    robot_init_pose_list = []
    init_x, init_y = np.mgrid[robot_init_pose_range[0][0]:robot_init_pose_range[0][1]:0.1,
                     robot_init_pose_range[1][0]:robot_init_pose_range[1][1]:0.1]
    for x in range(init_x.shape[0]):
        for y in range(init_y.shape[1]):
            robot_init_pose_list.append([init_x[x, y], init_y[x, y]])
    env_range = ((-14, -2), (-14, -2))
    poly_raw_list = [[(-2, -14), (-14, -14), (-14, -2), (-2, -2)],
                     [(-6.5, -6.5), (-6.5, -6), (-9.5, -6), (-9.5, -6.5)],
                     [(-6.5, -9.5), (-6.5, -10), (-9.5, -10), (-9.5, -9.5)],
                     [(-5.25, -6.5), (-4.75, -6.5), (-4.75, -9.5), (-5.25, -9.5)],
                     [(-10.75, -6.5), (-11.25, -6.5), (-11.25, -9.5), (-10.75, -9.5)]]
    goal_raw_list = [[(-2, -14), (-14, -14), (-14, -2), (-2, -2)],
                     [(-5.25, -9.5), (-5.25, -6.5), (-10.75, -6.5), (-10.75, -9.5)],
                     [(-9.5, -6.5), (-9.5, -2), (-14, -2), (-14, -6.5)],
                     [(-6.5, -6.5), (-2, -6.5), (-2, -2), (-6.5, -2)],
                     [(-6.5, -9.5), (-6.5, -14), (-2, -14), (-2, -9.5)],
                     [(-9.5, -9.5), (-14, -9.5), (-14, -14), (-9.5, -14)],
                     [(-6.5, -6.5), (-6.5, -6), (-9.5, -6), (-9.5, -6.5)],
                     [(-6.5, -9.5), (-6.5, -10), (-9.5, -10), (-9.5, -9.5)],
                     [(-5.25, -6.5), (-4.75, -6.5), (-4.75, -9.5), (-5.25, -9.5)],
                     [(-10.75, -6.5), (-11.25, -6.5), (-11.25, -9.5), (-10.75, -9.5)]]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        robot_init_pose = random.choice(robot_init_pose_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(robot_init_pose_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return env_range, poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list


def gen_rand_list_env4(size, robot_goal_diff=5.0):
    """
    Generate Random Training Lists for Environment 4
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    env_range = ((-14, -2), (2, 14))
    poly_raw_list = [[(-2, 14), (-14, 14), (-14, 2), (-2, 2)],
                     [(-6, 9), (-6, 9.5), (-9, 9.5), (-9, 9)],
                     [(-6, 7), (-6, 6.5), (-10, 6.5), (-10, 7)],
                     [(-4, 11.5), (-4, 12), (-8, 12), (-8, 11.5)],
                     [(-11.5, 9), (-11.5, 12), (-12, 12), (-12, 9)],
                     [(-3.75, 4), (-3.75, 7), (-4.25, 7), (-4.25, 4)],
                     [(-8, 4), (-8, 4.5), (-12, 4.5), (-12, 4)]]
    goal_raw_list = [[(-2, 14), (-14, 14), (-14, 2), (-2, 2)],
                     [(-6, 9), (-6, 9.5), (-9, 9.5), (-9, 9)],
                     [(-6, 7), (-6, 6.5), (-10, 6.5), (-10, 7)],
                     [(-4, 11.5), (-4, 12), (-8, 12), (-8, 11.5)],
                     [(-11.5, 9), (-11.5, 12), (-12, 12), (-12, 9)],
                     [(-3.75, 4), (-3.75, 7), (-4.25, 7), (-4.25, 4)],
                     [(-8, 4), (-8, 4.5), (-12, 4.5), (-12, 4)]]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        robot_init_pose = random.choice(goal_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(goal_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return env_range, poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list


def gen_poly_list_env5():
    """
    Generate obstacles of Environment 5
    """
    env_range = ((-12, 12), (18, 42))
    poly_raw_list = [[[-12, 18], [-12, 42], [12, 42], [12, 18]],
                     [[-4.5, 28.5], [-4.5, 29.], [-7.5, 29.], [-7.5, 28.5]],
                     [[-4.5, 25.5], [-4.5, 25.], [-7.5, 25.], [-7.5, 25.5]],
                     [[-3.25, 28.5], [-2.75, 28.5], [-2.75, 25.5], [-3.25, 25.5]],
                     [[-8.75, 28.5], [-9.25, 28.5], [-9.25, 25.5], [-8.75, 25.5]],
                     [[3., 33.5], [3., 34.], [9., 34.], [9., 33.5]],
                     [[3., 26.5], [3., 26.], [9., 26.], [9., 26.5]],
                     [[3.25, 32.], [2.75, 32.], [2.75, 28.], [3.25, 28.]],
                     [[8.75, 32.], [9.25, 32.], [9.25, 28.], [8.75, 28.]],
                     [[-4., 37.], [-4., 37.5], [-7., 37.5], [-7., 37.]],
                     [[-4., 35.], [-4., 34.5], [-8., 34.5], [-8., 35.]],
                     [[-2., 39.5], [-2., 40.], [-6., 40.], [-6., 39.5]],
                     [[-9.5, 37.], [-9.5, 40.], [-10., 40.], [-10., 37.]],
                     [[-1.75, 32.], [-1.75, 35.], [-2.25, 35.], [-2.25, 32.]],
                     [[-6., 32.], [-6., 32.5], [-10., 32.5], [-10., 32.]]]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    return env_range, poly_list, poly_raw_list



def gen_goal_position_list(poly_list, env_size=((-6, 6), (-6, 6)), obs_near_th=0.5, sample_step=0.1):
    """
    Generate list of goal positions
    :param poly_list: list of obstacle polygon
    :param env_size: size of the environment
    :param obs_near_th: Threshold for near an obstacle
    :param sample_step: sample step for goal generation
    :return: goal position list
    """
    goal_pos_list = []
    x_pos, y_pos = np.mgrid[env_size[0][0]:env_size[0][1]:sample_step, env_size[1][0]:env_size[1][1]:sample_step]
    for x in range(x_pos.shape[0]):
        for y in range(x_pos.shape[1]):
            tmp_pos = [x_pos[x, y], y_pos[x, y]]
            tmp_point = Point(tmp_pos[0], tmp_pos[1])
            near_obstacle = False
            for poly in poly_list:
                tmp_dis = tmp_point.distance(poly)
                if tmp_dis < obs_near_th:
                    near_obstacle = True
            if near_obstacle is False:
                goal_pos_list.append(tmp_pos)
    return goal_pos_list


def gen_polygon_exterior_list(poly_point_list):
    """
    Generate list of obstacle in the environment as polygon exterior list
    :param poly_point_list: list of points of polygon (with first always be the out wall)
    :return: polygon exterior list
    """
    poly_list = []
    for i, points in enumerate(poly_point_list, 0):
        tmp_poly = Polygon(points)
        if i > 0:
            poly_list.append(tmp_poly)
        else:
            poly_list.append(tmp_poly.exterior)
    return poly_list


def trajectory_encoder(traj_predict, robot_pose,
                       goal_dis_min_dis=0.3, goal_dis_scale=1.0, goal_dir_range=math.pi):
    '''
    Compuate relative distance and direction between robot and
    predicted trajectory of motion target
    '''
    # relative distance
    trajectory = np.array(traj_predict)
    delta_x = trajectory[:, 0] - robot_pose[0]
    delta_y = trajectory[:, 1] - robot_pose[1]
    dist = np.sqrt(delta_x ** 2 + delta_y ** 2)
    # scale distance
    scaled_dist = np.where(dist != 0, dist, goal_dis_min_dis)
    scaled_dist = goal_dis_min_dis / scaled_dist
    scaled_dist = np.where(scaled_dist <= 1, scaled_dist * goal_dis_scale, goal_dis_scale)
    # relative direction
    ego_direction = np.arctan2(delta_y, delta_x)
    robot_direction = robot_pose[2]
    theta = ego_direction - robot_direction
    direction = np.arctan2(np.sin(theta), np.cos(theta))
    # scale direction
    scaled_dir = direction / goal_dir_range
    # No Flatting
    encoded_trajectory = np.column_stack((scaled_dir, scaled_dist))
    # Flat
    faltted_trajectory = np.concatenate((scaled_dir, scaled_dist), axis=0)
    return encoded_trajectory, faltted_trajectory


def robot_2_goal_dis_dir(goal_pos, robot_pose):
    """
    Compute Relative Distance and Direction between robot and goal
    :param robot_pose: robot pose
    :param goal_pos: goal position
    :return: distance, direction
    """
    delta_x = goal_pos[0] - robot_pose[0]
    delta_y = goal_pos[1] - robot_pose[1]
    distance = np.sqrt(delta_x ** 2 + delta_y ** 2)

    ego_direction = np.arctan2(delta_y, delta_x)
    robot_direction = robot_pose[2]
    theta = ego_direction - robot_direction
    direction = np.arctan2(np.sin(theta), np.cos(theta))
    return distance, direction


def state_scale(state, linear_spd_range=0.5, angular_spd_range=2,
                laser_scan_scale=1.0, scan_min_dis=0.35):
    # scale robot speed
    scaled_state = np.zeros(2 + len(state[2]))
    scaled_state[0] = state[1][0] / linear_spd_range
    scaled_state[1] = state[1][1] / angular_spd_range

    # Transform distance in laser scan to [0, scale]
    tmp_laser_scan = state[2]
    tmp_laser_scan[tmp_laser_scan == 0] = 0.001
    tmp_laser_scan = laser_scan_scale * (scan_min_dis / tmp_laser_scan)
    tmp_laser_scan = np.clip(tmp_laser_scan, 0, laser_scan_scale)
    scaled_state[2:] = tmp_laser_scan
    return scaled_state


def wheeled_network_2_robot_action(action, wheel_max, wheel_min, diff=0.25):
    """
    Decode wheeled action from network to linear and angular speed for the robot
    :param action: action for wheel spd
    :param wheel_max: max wheel spd
    :param wheel_min: min wheel spd
    :param diff: diff of wheel for decoding angular spd
    :return: robot_action
    """
    l_spd = action[0] * (wheel_max - wheel_min) + wheel_min
    r_spd = action[1] * (wheel_max - wheel_min) + wheel_min
    linear = (l_spd + r_spd) / 2
    angular = (r_spd - l_spd) / diff
    return [linear, angular]


def euler_2_quat(yaw=0, pitch=0, roll=0):
    """
    Transform euler angule to quaternion
    :param yaw: z
    :param pitch: y
    :param roll: x
    :return: quaternion
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return [w, x, y, z]
