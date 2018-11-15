# Reference from https://blog.csdn.net/u010895119/article/details/79470443

import logging
import argparse
import ast


def mylogger():
    logger = logging.getLogger('mylogger')
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel('DEBUG')  # 也可以不设置，不设置就默认用logger的level
    fhlr = logging.FileHandler('param.log') # 输出到文件的handler
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

def parseArg ():
    parseArgs = argparse.ArgumentParser(description='Arguments for project.')
    parseArgs.add_argument('--v',type=ast.literal_eval, default = True, help='display the debug info or not')
    parseArgs.add_argument('--dataset',type=str,default = 'celebA', help='specify the training dataset')
    parseArgs.add_argument('--train_attribute',type=str, default= 'celebA_train', help='the name of training path')
    parseArgs.add_argument('--test_attribute',type=str, default= 'celebA_test', help='the name of test path')
    parseArgs.add_argument('--dim',type=int, default= 4096, help='the dimension of z') 
    parseArgs.add_argument('--DNN',type=str, choices= ['vgg19','vgg19_bn'], default= 'vgg19', help='the name of DNN for extracting features') 
    parseArgs.add_argument('--layer',type=int, default= 3, help='the layer of extracting features in classifier module')
    parseArgs.add_argument('--nb_channels_first_layer',type=int, default= 32, help='the num of channels in first layer')    
    parseArgs.add_argument('--iteration',type=int, default= 10000, help='the num of max iteration')
    parseArgs.add_argument('--epoch',type=int, default= 0, help='the num of training epoch for restore')
    parseArgs.add_argument('--save_freq',type=int, default= 200, help='the num of training epoch for restore')
    parseArgs.add_argument('--phase',type=str, default= 'test', help='train or test') 
    return parseArgs.parse_args()

