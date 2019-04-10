# Reference from https://blog.csdn.net/u010895119/article/details/79470443

import logging
import argparse
import ast
from datetime import datetime

def mylogger(logpath='./param.log'):
    logger = logging.getLogger('mylogger')
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    fhlr = logging.FileHandler(logpath) # 
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)
    str_sharp = '#####################################################################'
    logger.info('Record Experiment Information and Conditions\n'+str_sharp+'\n\n\n'+str_sharp+'\n')
    logger.info('  Experiment Setting and Running Logs\n\n')

    chlr = logging.StreamHandler() # 
    chlr.setFormatter(formatter)
    chlr.setLevel('DEBUG')  # 
    logger.addHandler(chlr)


def parseArg ():
    parseArgs = argparse.ArgumentParser(description='Arguments for project.')
    parseArgs.add_argument('--att',type=ast.literal_eval, default= 'False', help='add attention layer or not') 
    parseArgs.add_argument('--batch',type=int, default= 200, help='Number of batch size') 
    parseArgs.add_argument('--dataset',type=str,default = 'cifar', help='specify the training dataset')
    parseArgs.add_argument('--dim',type=int, default= 512, help='the dimension of z') 
    parseArgs.add_argument('--DNN',type=str, choices= ['vgg19','vgg19_bn','lenet', 'cifarnet', 'celebANet'], default= 'cifarnet', help='the name of DNN for extracting features') 
    parseArgs.add_argument('--epoch',type=int, default= 0, help='the num of training epoch for restore,0 means training from scrach')
    parseArgs.add_argument('--gpu_ids',type=str, default= '', help='the ids of GPUs')
    parseArgs.add_argument('--iteration',type=int, default= 1000, help='the num of max iteration')
    parseArgs.add_argument('--layer',type=int, default= 3, help='the layer of extracting features in classifier module')
    parseArgs.add_argument('--nb_channels_first_layer',type=int, default= 512, help='the num of channels in first layer') 
    parseArgs.add_argument('--num_classes',type=int, default= 1, help='the num of classes in celebA dataset')    
    parseArgs.add_argument('--phase',type=str, default= 'test', help='train or test') 
    parseArgs.add_argument('--save_freq',type=int, default= 20, help='the num of training epoch for restore')
    parseArgs.add_argument('--train_attribute',type=str, default= 'train', help='the name of training path')
    parseArgs.add_argument('--test_attribute',type=str, default= 'test', help='the name of test path')
    parseArgs.add_argument('--ite_max_eps',type=int, default= 300, help='the num of iteration for finding epsilon')
    parseArgs.add_argument('--eps_step',type=float, default= 0.1, help='the step size for updating epsilon')
    parseArgs.add_argument('--eps_max',type=int, default= 21, help='the maximum of epsilon')
    parseArgs.add_argument('--img_num',type=int, default= 20, help='the maximum of images to generate adversatial images')
    parseArgs.add_argument('--eps_list',type=list, default= [0.1,1.0, 2.0,3.0,4.0, 5.0, 7.0,9.0,11.0,13.0, 15.0,17.0, 20.0], nargs='+', help='the list of epsilon to save check attack rate') # Thsi method does not work in commadline, we should give the values in this file.
    parseArgs.add_argument('--v',type=ast.literal_eval, default = True, help='display the debug info or not')
    parseArgs.add_argument('--info', '-I', type=str, default='Info for running program',
                        help='This info is used to record the running conditions for the current program, which is stored in param.log')
    parseArgs.add_argument('--direction_type',type=str, default= 'cav', help='the name of direction type')
    parseArgs.add_argument('--direction_model',type=str, default= 'att_svm', choices=['linear','att_svm','logistic', 'max_dis_svm', 'att_neighbor_svm'], help='the name of model for training direction')
    parseArgs.add_argument('--cav_imgnum',type=int, default= 200, help='the num of examples for training svm')
    parseArgs.add_argument('--exp_path', type=str, default='./experiments', help='the path for saving experiments results.')
    parseArgs.add_argument('--exp_name',type=str, default= 'exp_param', help='the name of experiments')

    
    #parseArgs.add_argument('--train_num',type=int, default= 0, help='0 means using all the image in train dir')
    return parseArgs.parse_args()

def time_stamp():
  TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
  return TIMESTAMP
