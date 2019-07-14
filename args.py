import argparse

NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
LAYERS = ["ignore", "concat", "concat_v2", "squash", "concatsquash", "scale", "concatscale"]


def add_args(parser):
    # model architecture options
    parser.add_argument('--input_dim', type=int, default=3,
                        help='Number of input dimensions (3 for 3D point clouds)')
    parser.add_argument('--dims', type=str, default='256')
    parser.add_argument('--latent_dims', type=str, default='256')
    parser.add_argument("--num_blocks", type=int, default=1,
                        help='Number of stacked CNFs.')
    parser.add_argument("--latent_num_blocks", type=int, default=1,
                        help='Number of stacked CNFs.')
    parser.add_argument("--layer_type", type=str, default="concatsquash", choices=LAYERS)
    parser.add_argument('--time_length', type=float, default=0.5)
    parser.add_argument('--train_T', type=eval, default=True, choices=[True, False])
    parser.add_argument("--nonlinearity", type=str, default="tanh", choices=NONLINEARITIES)
    parser.add_argument('--use_adjoint', type=eval, default=True, choices=[True, False])
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--sync_bn', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)

    # training options
    parser.add_argument('--use_latent_flow', action='store_true',
                        help='Whether to use the latent flow to model the prior.')
    parser.add_argument('--use_deterministic_encoder', action='store_true',
                        help='Whether to use a deterministic encoder.')
    parser.add_argument('--zdim', type=int, default=128,
                        help='Dimension of the shape code')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use', choices=['adam', 'adamax', 'sgd'])
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size (of datasets) for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the Adam optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam.')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training (default: 100)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for initializing training. ')
    parser.add_argument('--recon_weight', type=float, default=1.,
                        help='Weight for the reconstruction loss.')
    parser.add_argument('--prior_weight', type=float, default=1.,
                        help='Weight for the prior loss.')
    parser.add_argument('--entropy_weight', type=float, default=1.,
                        help='Weight for the entropy loss.')
    parser.add_argument('--scheduler', type=str, default='linear',
                        help='Type of learning rate schedule')
    parser.add_argument('--exp_decay', type=float, default=1.,
                        help='Learning rate schedule exponential decay rate')
    parser.add_argument('--exp_decay_freq', type=int, default=1,
                        help='Learning rate exponential decay frequency')

    # data options
    parser.add_argument('--dataset_type', type=str, default="shapenet15k",
                        help="Dataset types.", choices=['shapenet15k', 'modelnet40_15k', 'modelnet10_15k'])
    parser.add_argument('--cates', type=str, nargs='+', default=["airplane"],
                        help="Categories to be trained (useful only if 'shapenet' is selected)")
    parser.add_argument('--data_dir', type=str, default="data/ShapeNetCore.v2.PC15k",
                        help="Path to the training data")
    parser.add_argument('--mn40_data_dir', type=str, default="data/ModelNet40.PC15k",
                        help="Path to ModelNet40")
    parser.add_argument('--mn10_data_dir', type=str, default="data/ModelNet10.PC15k",
                        help="Path to ModelNet10")
    parser.add_argument('--dataset_scale', type=float, default=1.,
                        help='Scale of the dataset (x,y,z * scale = real output, default=1).')
    parser.add_argument('--random_rotate', action='store_true',
                        help='Whether to randomly rotate each shape.')
    parser.add_argument('--normalize_per_shape', action='store_true',
                        help='Whether to perform normalization per shape.')
    parser.add_argument('--normalize_std_per_axis', action='store_true',
                        help='Whether to perform normalization per axis.')
    parser.add_argument("--tr_max_sample_points", type=int, default=2048,
                        help='Max number of sampled points (train)')
    parser.add_argument("--te_max_sample_points", type=int, default=2048,
                        help='Max number of sampled points (test)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading threads')

    # logging and saving frequency
    parser.add_argument('--log_name', type=str, default=None, help="Name for the log dir")
    parser.add_argument('--viz_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)

    # validation options
    parser.add_argument('--no_validation', action='store_true',
                        help='Whether to disable validation altogether.')
    parser.add_argument('--save_val_results', action='store_true',
                        help='Whether to save the validation results.')
    parser.add_argument('--eval_classification', action='store_true',
                        help='Whether to evaluate classification accuracy on MN40 and MN10.')
    parser.add_argument('--no_eval_sampling', action='store_true',
                        help='Whether to evaluate sampling.')
    parser.add_argument('--max_validate_shapes', type=int, default=None,
                        help='Max number of shapes used for validation pass.')

    # resuming
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to the checkpoint to be loaded.')
    parser.add_argument('--resume_optimizer', action='store_true',
                        help='Whether to resume the optimizer when resumed training.')
    parser.add_argument('--resume_non_strict', action='store_true',
                        help='Whether to resume in none-strict mode.')
    parser.add_argument('--resume_dataset_mean', type=str, default=None,
                        help='Path to the file storing the dataset mean.')
    parser.add_argument('--resume_dataset_std', type=str, default=None,
                        help='Path to the file storing the dataset std.')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    # Evaluation options
    parser.add_argument('--evaluate_recon', default=False, action='store_true',
                        help='Whether set to the evaluation for reconstruction.')
    parser.add_argument('--num_sample_shapes', default=10, type=int,
                        help='Number of shapes to be sampled (for demo.py).')
    parser.add_argument('--num_sample_points', default=2048, type=int,
                        help='Number of points (per-shape) to be sampled (for demo.py).')

    return parser


def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='Flow-based Point Cloud Generation Experiment')
    parser = add_args(parser)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
