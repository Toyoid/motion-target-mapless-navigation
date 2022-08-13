import rospy
import sys
sys.path.append('../../')
from evaluation.xxx_eval_simulation.xxx_evaluation import XxxEvaluation
from evaluation.xxx_eval_simulation.utils import *


def evaluate_xxx(pos_start=0, pos_end=199, is_save_result=False, use_cuda=True):
    """
    Evaluate XXX in Simulated Environment

    :param pos_start: Start index position for evaluation
    :param pos_end: End index position for evaluation
    :param model_name: name of the saved model
    :param save_dir: directory to the saved model
    :param is_save_result: if true save the evaluation result
    :param use_cuda: if true use gpu
    """

    rospy.init_node('navi_eval')
    poly_list, raw_poly_list = gen_test_env_poly_list_env()
    rand_paths_robot_list = pickle.load(open('eval_paths_robot_pose.p', 'rb'))
    robot_init_list = rand_paths_robot_list[0][:]
    target_paths_list = rand_paths_robot_list[1][:]
    # prediction parameter
    pred_tau = 30
    pred_length = 150

    net_dir = '../../training/save_model_weights/save_meta_controller/'\
              + 'Meta_Controller_actor_network_param3' + '.pt'
    meta_state_num = pred_length * 2 + 20
    meta_action_num = 1
    meta_controller_net = load_actor_net(net_dir, state_num=meta_state_num, action_num=meta_action_num)
    net_dir = '../saved_model/' + 'ddpg' + '.pt'
    # net_dir = '../../training/save_model_weights/save_controller/' \
    #           + 'HiDDPG_ctrl_actor_network_s0' + '.pt'
    ctrl_state_num = 22
    ctrl_action_num = 2
    controller_net = load_actor_net(net_dir, state_num=ctrl_state_num, action_num=ctrl_action_num)
    eval = XxxEvaluation(meta_controller_net, controller_net, robot_init_list, target_paths_list, poly_list,
                         pred_tau=pred_tau, pred_length=pred_length, max_steps=1000,
                         action_rand=0.01, use_cuda=use_cuda)
    data = eval.run_ros()
    if is_save_result:
        pickle.dump(data,
                    open('../record_data/' + 'xxx_hyperparam3_try_againkf' + str(pos_start) + '_' + str(pos_end) + '.p', 'wb+'))
    model_name = 'XXX'
    print(str(model_name) + " Evaluation Finished ...")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False

    SAVE_RESULT = False
    if args.save == 1:
        SAVE_RESULT = True
    evaluate_xxx(use_cuda=USE_CUDA,
                 is_save_result=SAVE_RESULT)
