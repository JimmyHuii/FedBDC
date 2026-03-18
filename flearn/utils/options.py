import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rate', type=float, default=50.0, help='hypernet_penalty_rate')
    parser.add_argument('--lambda_reg', type=float, default=0.05, help='hypernet_lambda_reg')
    parser.add_argument("--is_qsgd", type=bool, default=False, help="whether to use qsgd")
    parser.add_argument("--quan_bits", type=int, default=4, help="level qsgd")
    parser.add_argument("--hypernet_comm", type=bool, default=False, help="whether to use hypernet")
    parser.add_argument('--B', type=int, default=20000000, help='number of layers')
    parser.add_argument('--number_of_layer', type=int, default=0, help='number of layers')
    parser.add_argument('--active_hybrid_down', type=bool, default=False,
                        help="whether active hybrid down")
    parser.add_argument('--start_epoch_layer_down', type=int, default=1000,
                        help="start_epoch_layer_down")
    parser.add_argument('--layer_wise_up', type=bool, default=False,
                        help="is_or_not_layer_up_wise")
    parser.add_argument('--layer_wise_down', type=bool, default=False,
                        help="is_or_not_layer_wise_down")
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--priority_list', type=list, default=None,
                        help="priority of the layers")
    parser.add_argument('--layer_wise', type=bool, default=False,
                        help="is_or_not_layer_wise")
    parser.add_argument('--freeze_except_last', type=bool, default=False,
                        help="is_or_not_freeze_except_last")
    parser.add_argument('--is_downtopk', type=bool, default=False,
                        help="is_or_not_downtopk")
    parser.add_argument('--is_uptopk', type=bool, default=False,
                        help="is_or_not_uptopk")
    parser.add_argument('--is_FedAvg', type=bool, default=False,
                        help="is_or_not_FedAvg")
    parser.add_argument('--l', type=int, default=2,
                        help="level of non-iid")
    parser.add_argument('--down_ratio', type=float, default=1,
                        help="ratio of downloaded model")
    parser.add_argument('--final_ratio', type=float, default=1,
                        help="ratio of final uploaded model")
    parser.add_argument('--epochs', type=int, default=500,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--global_lr', type=float, default=1.0,
                        help='global learning rate')
    parser.add_argument('--local_iter', type=float, default=250,
                        help='local iterations')

    parser.add_argument('--global_mask_frozen', type=int, default=0,
                        help="1: use global mask frozen")
    parser.add_argument("--upload_stable_params", type=float, default=1,
                        help="0: don't upload stable params")
    parser.add_argument("--upload_iter", type=int, default=100,
                        help="iter for upload stable params")
    parser.add_argument("--double_check", type=int, default=0,
                        help="1: use double check for stable params")

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optim', type=str, default='sgd', help="type \
                        of optim")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument("--result_dir", type=str, default='test', help="dir name for save result")

    args = parser.parse_args()
    return args