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
    parser.add_argument('--lambda_regg', type=float, nargs='?', default=0.5,
                    help='Multiplier for the generator regularization loss L_{RegG}. Appears on Eq. 13 on the paper.')
    parser.add_argument('--lambda_g', type=float, nargs='?', default=0.3,
                        help='Multiplier for the generator loss L_{G}. Appears on Eq. 13 on the paper.')
    parser.add_argument('--lambda_dx', type=float, nargs='?', default=1.0,
                        help='Multiplier for the discriminator loss L_{Dx}. Appears on Eq. 13 on the paper.')
    parser.add_argument('--lambda_dxprime', type=float, nargs='?', default=0.01,
                            help='Multiplier for the discriminator loss L_{Dxhat\'}. Appears on Eq. 13 on the paper.')
    parser.add_argument('--batch_size', type=int, nargs='?', default=12,
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
                                    help='Select the dataset to load. Options are "spheres", "squares", "mnist", and "copd".')
    parser.add_argument('--total_iters_per_epoch', type=int, nargs='?', default=200,
                                    help='Set the number of batches that are loaded in each epoch.')
    parser.add_argument('--cosine_penalty', type=str2bool, nargs='?', default='false',
                                    help='If true, applies the alignment penalty from equation (9) in the paper.')
    parser.add_argument('--vrgan_training', type=str2bool, nargs='?', default='false',
                                    help='If true, trains the generative model for estimating the closest example of the opposite class using the modified VRGAN method (section 2.2).')
    parser.add_argument('--lambda_penalty', type=float, nargs='?', default=0.1,
                                help='The constant that multiplies the alignment penalty, as defined in equation (9) in the paper.')
    parser.add_argument('--epsilons_val_attack', type=float, nargs='*', default=[],
                                help='List of epsilons (check equation (17) in the paper) to be used for validation adversarial attacks.')
    parser.add_argument('--epsilon_attack_training', type=float, nargs='?', default=0.3,
                                    help='Epsilon of equation (17) in the paper, to be used during adversarial robustness training with PGD.') 
    parser.add_argument('--attack_to_use_training', type=str, nargs='?', default='none',
                                    help="Defines the kind of atttack to use in validation, defining the order p of the norm used to limit attacks. For p=inf, use \'inf\', and for p=2, use \'l2\'.")
    parser.add_argument('--attack_to_use_val', type=str, nargs='?', default='inf',
                                    help="Defines the kind of atttack to use in validation, defining the order p of the norm used to limit attacks. For p=inf, use \'inf\', and for p=2, use \'l2\'.")
    parser.add_argument('--alpha_multiplier', type=float, nargs='?', default=1.,
                                    help='The default alpha (as defined by the step size \eta of gradient updates in the paper) used for adversarial attacks is 1e-2. The value of this config optionmultiplies the default alpha.')
    parser.add_argument('--unet_downsamplings', type=int, nargs='?', default=4,
                                    help='number of downsampling operations performed by unet. A smaller number here causes the generator to use less memory but it also becomes less powerful.')
    parser.add_argument('--save_models', type=str2bool, nargs='?', default='true',
                                    help='If true, will save model every epoch.')
    parser.add_argument('--save_best_model', type=str2bool, nargs='?', default='true',
                                    help="If true, will save the model with the best score, using the checkpoint name \'state_dict_<g/d>_best_epoch\'.")
    parser.add_argument('--blackbox_attack', type=str2bool, nargs='?', default='false',
                                    help='If true, adversarial examples will be calculated using a black-box attack instead of PGD')
    
    args = parser.parse_args()
    
    #sets of congifs that will likely not need to be modified
    #mnist classes used for after binarizing the dataset
    args.class_0_mnist = 3
    args.class_1_mnist = 5
    #number of dimensions for the spheres dataset
    args.n_dimensions_sphere = 500
    #number of hidden neurons for both classifier and generator for the sphere dataset.
    args.sphere_hidden_neurons_per_layer=1000
    #number of scoring examples for the spheres and the squares dataset
    if args.split_validation=='test':
        args.validation_examples_spheres = 1000
    else:
        args.validation_examples_spheres = 200
    #defining how the "best model" is chosen from validation scores
    if args.vrgan_training:
        args.metric_to_validate = 'total_loss'
        args.function_to_compare_validation_metric = lambda x,y:x<=y
        args.initialization_comparison = float('inf')
    else:
        args.metric_to_validate = 'epsilon_0.5'
        args.function_to_compare_validation_metric = lambda x,y:x>=y
        args.initialization_comparison = float('-inf')
    #number of update steps during calculation of PGD attacks, for both training and validation
    args.k_pgd_training = 40
    #learning rates values for generator and critic
    args.learning_rate_g = 1e-4
    args.learning_rate_d = 1e-4
    #If true, copd dataset is completely loaded to memory, without a need to read the disk during training
    args.load_copd_dataset_to_memory = True
    #Defines folder where the files images2012-2016.txt, images2017.txt,  all_subjects_more_than_one_image.pkl and valids_all_subjects_more_than_one_image.pkl, containing a list of the location of every image in the dataset and a list of the subjects used for validation and for test.
    args.COPD_lists_location = './'
    #Defines the location of file containing the metadata for the COPD dataset."
    args.COPD_labels_location = './Chest_Xray_Main_TJ_clean_ResearchID_PFTValuesAndInfo_WithDate_NoPHI.csv'
    
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
    if args.attack_to_use_val=='cwl2':
        args.epsilons_val_attack = [0]
    import platform
    args.python_version = platform.python_version()
    import torch
    args.pytorch_version = torch.__version__ 
    import torchvision
    args.torchvision_version = torchvision.__version__
    import numpy as np
    args.numpy_version = np.__version__
    return args
