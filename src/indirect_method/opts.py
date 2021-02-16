"""User configuration file

File organizing all configurations that may be set by user when running the 
train.py script. 
Call python -m src.train --help for a complete and formatted list of available user options.
"""

import argparse
import time
from random import randint
import os
import socket
import numpy as np

#convert a few possibilities of ways of inputing boolean values to a python boolean
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def get_opt():
    parser = argparse.ArgumentParser(description='Configuration for running VRGAN code')
    
    parser.add_argument('--skip_train', type=str2bool, nargs='?', default='false',
                        help='If you just want to run validation, set this value to true.')
    parser.add_argument('--batch_size_train', type=int, nargs='?', default=12,
                            help='Batch size for training the toy dataset.')
    parser.add_argument('--batch_size_val', type=int, nargs='?', default=12,
                            help='Batch size for training the toy dataset.')
    parser.add_argument('--save_folder', type=str, nargs='?', default='./runs',
                                help='If you want to save files and outputs in a folder other than \'./runs\', change this variable.')
    parser.add_argument('--gpus', type=str, nargs='?', default=None,
                                help='Set the gpus to use, using CUDA_VISIBLE_DEVICES syntax.')
    parser.add_argument('--experiment', type=str, nargs='?', default='',
                                help='Set the name of the folder where to save the run.')
    parser.add_argument('--nepochs', type=int, nargs='?', default=30,
                                help='Number of epochs to run training and validation')
    parser.add_argument('--split_validation', type=str, nargs='?', default='val',
                                    help='Use \'val\' to use the validation set for calculating scores every epoch. Use \'test\' for using the test set during scoring.')
    parser.add_argument('--load_checkpoint_g', type=str, nargs='?', default=None,
                                    help='If you want to start from a previously trained generator, set a filepath locating a model checkpoint that you want to load')
    parser.add_argument('--load_checkpoint_d', type=str, nargs='?', default=None,
                                        help='If you want to start from a previously trained classifier, set a filepath locating a model checkpoint that you want to load')
    parser.add_argument('--folder_dataset', type=str, nargs='?', default='./',
                                    help='If you want to load/save toy dataset files in a folder other than the local folder, change this variable.')
    parser.add_argument('--dataset_to_use', type=str, nargs='?', default='squares',
                                    help='Select the dataset to load. Options are "squares", "mnist", and "cifar".')
    parser.add_argument('--total_iters_per_epoch', type=int, nargs='?', default=200,
                                    help='Set the number of batches that are loaded in each epoch.')
    parser.add_argument('--cosine_penalty', type=str2bool, nargs='?', default='false',
                                    help='If true, applies the alignment penalty from equation (9) in the paper.')
    parser.add_argument('--lambda_penalty', type=float, nargs='?', default=0.1,
                                help='The constant that multiplies the alignment penalty, as defined in equation (9) in the paper.')
    parser.add_argument('--epsilons_val_attack', type=float, nargs='*', default=[],
                                help='List of epsilons (check equation (17) in the paper) to be used for validation adversarial attacks.')
    parser.add_argument('--epsilon_attack_training', type=float, nargs='?', default=0.3,
                                    help='Epsilon of equation (17) in the paper, to be used during adversarial robustness training with PGD.') 
    parser.add_argument('--attack_to_use_training', type=str, nargs='?', default='none',
                                    help="Defines the kind of atttack to use in validation, defining the order p of the norm used to limit attacks. For p=inf, use \'inf\', and for p=2, use \'l2\'.")
    parser.add_argument('--attack_to_use_val', type=str, nargs='?', default='inf',
                                    help="Defines the kind of attack to use in validation, defining the order p of the norm used to limit attacks. For p=inf, use \'inf\', and for p=2, use \'l2\'.")
    parser.add_argument('--alpha_multiplier', type=float, nargs='?', default=1.,
                                    help='The default alpha (as defined by the step size \eta of gradient updates in the paper) used for adversarial attacks is 1e-2. The value of this config optionmultiplies the default alpha.')
    parser.add_argument('--save_models', type=str2bool, nargs='?', default='true',
                                    help='If true, will save model every epoch.')
    parser.add_argument('--save_best_model', type=str2bool, nargs='?', default='true',
                                    help="If true, will save the model with the best score, using the checkpoint name \'state_dict_<g/d>_best_epoch\'.")
    parser.add_argument('--blackbox_attack', type=str2bool, nargs='?', default='false',
                                    help='If true, adversarial examples will be calculated using a black-box attack instead of PGD')
    parser.add_argument('--skip_validation', type=str2bool, nargs='?', default='false',
                                    help='If true, skips the whole validation.')
    parser.add_argument('--resnet_n_layers', type=int, nargs='?', default = 18,
                                    help='Set the number of layers in the Resnet used as classifier. Only accepts the subset defined by torchvision.')
    parser.add_argument('--learning_rate_d', type=float, nargs='?', default = 1e-4,
                                    help='Set the learning rate for learning the classifier')
    parser.add_argument('--skip_long_validation', type=str2bool, nargs='?', default='false',
                                    help='Set to true to skip the validation that takes longer. If true, only save images related to the separated fixed set of validation.')
    parser.add_argument('--get_z_lr', type=float, nargs='?', default=0.2,
                                    help='Learning rate for the second phase of the optimization of latent space z.')
    parser.add_argument('--get_z_penalty', type=float, nargs='?', default=3e-2,
                                    help='Weight for the penalty over the norm of latent space z when optimizing over z.')
    parser.add_argument('--get_z_iter', type=int, nargs='?', default=150,
                                    help='Number of iterations for the second phase optimization of latent space z where penalty on the norm is set to get_z_penalty.')
    parser.add_argument('--get_z_init_lr', type=float, nargs='?', default=0.2,
                                    help='Learning rate for the first phase of the optimization of latent space z.')
    parser.add_argument('--get_z_init_iter', type=int, nargs='?', default=600,
                                    help='Number of iterations for the first phase optimization of latent space z where penalty on the norm is set to 0.')
    parser.add_argument('--optimizer_type', type=str, nargs='?', default='adam',
                                    help='Optimizer type for the classifier. Accepts adam and sgd.')
    parser.add_argument('--optimizer_momentum', type=float, nargs='?', default=0.9,
                                    help='Sets the momentum of the optimizer of the classifier, in case SGD is used as optimizer.')
    parser.add_argument('--deactivate_bn', type=str2bool, nargs='?', default='true',
                                    help='If true, it will set the BatchNormalization modules from the classifier to eval mode.')
    parser.add_argument('--index_produce_val_image', type=int, nargs='*', default=[],
                                help='Sets a list of image indices that will be used to limit the iages loaded for the validation/test set.')
    parser.add_argument('--force_closest_class', type=int, nargs='*', default=None,
                                help='Forces a destination class for the gradients calculated during validation of the fixed set of images.')
    args = parser.parse_args()
    
    #sets of configs that will likely not need to be modified
    #defining how the "best model" is chosen from validation scores
    args.metric_to_validate = 'epsilon_0.5'
    args.function_to_compare_validation_metric = lambda x,y:x>=y
    args.initialization_comparison = float('-inf')
    #number of update steps during calculation of PGD attacks, for both training and validation
    args.k_pgd_training = 40
    
    #gets the current time of the run, and adds a four digit number for getting
    #different folder name for experiments run at the exact same time.
    timestamp = time.strftime("%Y%m%d-%H%M%S") + '-' + str(randint(1000,9999))
    args.timestamp = timestamp
    
    #register a few values that might be important for reproducibility
    args.screen_name = os.getenv('STY')
    args.hostname = socket.gethostname()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    else:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES']
    
    if args.dataset_to_use=='mnist':
        args.n_classes = 10
        args.n_channels = 1
        args.im_size = 28
        args.im_size_g = 32
    elif args.dataset_to_use=='squares':
        args.n_classes = 2
        args.n_channels = 1
        args.im_size_g = 32
    elif args.dataset_to_use=='cifar':
        args.n_classes = 10
        args.n_channels = 3
        args.im_size = 32
        args.im_size_g = 32
    args.list_of_class = range(args.n_classes)
    import platform
    args.python_version = platform.python_version()
    import torch
    args.pytorch_version = torch.__version__ 
    import torchvision
    args.torchvision_version = torchvision.__version__
    args.numpy_version = np.__version__
    return args
