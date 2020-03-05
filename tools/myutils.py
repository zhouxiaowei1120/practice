import logging
import argparse
import ast
from collections import OrderedDict
import torch
import os
import scipy
from datetime import datetime
import time
import math
import random
import numpy as np
import sys
import cv2
irange = range

def mylogger(logpath='./param.log'):
    logger = logging.getLogger('mylogger')
    logger.setLevel('DEBUG')
    logger.propagate = False
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    fhlr = logging.FileHandler(logpath) # 
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)
    str_sharp = '#####################################################################'
    logger.info(str_sharp +'Record Experiment Information and Conditions\n')
    # logger.info('  Experiment Setting and Running Logs\n\n')

    chlr = logging.StreamHandler() # 
    chlr.setFormatter(formatter)
    chlr.setLevel('DEBUG')  # 
    logger.addHandler(chlr)

def parseArg ():
    parseArgs = argparse.ArgumentParser(description='Arguments for project.')
    # parseArgs.add_argument('--attr_loss_type',type=str, default='triplet', help='the type of loss function for attributes')
    parseArgs.add_argument('--batch',type=int, default= 200, help='Number of batch size') 
    parseArgs.add_argument('--baseline', action='store_true', default=False, help='the mode of training baseline or our model')
    parseArgs.add_argument('--dataset',type=str,default = 'CUB_200_2011', help='specify the training dataset')
    parseArgs.add_argument('--epoch',type=int, default= 0, help='the num of training epoch for restore,0 means training from scrach')
    parseArgs.add_argument('--exp_att',type=str, default='CUB_test', help='the name of current experiment')
    parseArgs.add_argument('--gpu_ids',type=str, default= '', help='the ids of GPUs')
    parseArgs.add_argument('--gamma',type=float, default=1.0, help='the factor of MSE Loss')
    parseArgs.add_argument('--img_size',type=int, default=299, help='the size of image (299,299)')
    parseArgs.add_argument('--info', '-I', type=str, default='Info for running program', help='This info is used to record the running conditions for the current program, which is stored in param.log')
    parseArgs.add_argument('--iteration',type=int, default= 1000, help='the num of max iteration')
    parseArgs.add_argument('--logfile',type=str, default= './param.log', help='the name of log file')
    parseArgs.add_argument('--loss_weights', type=float, nargs='+', default=[1,0,1,0], help='whether to add the position loss')
    parseArgs.add_argument('--log_interval',type=int, default= 200, help='the interval of training epoch for logging')
    parseArgs.add_argument('--modelname',type=str, default= 'inceptionV3', help='the name of DNN for explaining') 
    parseArgs.add_argument('--num_classes',type=int, default= 200, help='the num of classes in celebA dataset')    
    parseArgs.add_argument('--num_workers',type=int, default= 8, help='the num of strides for loading data')    
    parseArgs.add_argument('--num_experts',type=int, default= 4, help='the num of experts')
    parseArgs.add_argument('--pertur',type=str, default= 'line', help='the method to perturbate the image') 
    parseArgs.add_argument('--path_conflict',type=str, default= '', help='Use for path conflict') 
    parseArgs.add_argument('--phase',type=str, default= 'test', help='train or test')
    # parseArgs.add_argument('--region',type=list, default= [[120,60],[50,50],[120,60],[60,60],[26,26],[26,26],[26,26],[60,60],[160,160],[50,50],[26,26],[60,60],[160,160],[60,60],[26,26]], help='the regions for different positions') # w,h
    # parseArgs.add_argument('--region',type=list, default= [[28,10],[10,16],[28,10],[16,10],[36,36]], help='the regions for different positions') #w,h
    # parseArgs.add_argument('--region',type=list, default= [[76,28],[28,44],[76,28],[44,28],[98,98]], help='the regions for different positions') #w,h
    parseArgs.add_argument('--region',type=list, default= [[76,28],[76,76],[76,28],[44,28],[98,98]], help='the regions for different positions') #w,h the second means nose and mouth
    parseArgs.add_argument('--res_dir',type=str, default='./experiments', help='the path for saving results')
    parseArgs.add_argument('--restore_file',type=str, default='experiments/pre_trained_models/inceptonV3.pth', help='the path/file for restore models')
    parseArgs.add_argument('--seed', type=int, default= 1, help='the seed for random selection')
    parseArgs.add_argument('--test_batch',type=int, default= 300, help='Number of test batch size') 
    parseArgs.add_argument('--v', type=ast.literal_eval, default = False, help='display the debug info or not')
    parseArgs.add_argument('--weight_attr', type=str, default = 'continuous', help='The method of using attributes')

    
    #parseArgs.add_argument('--train_num',type=int, default= 0, help='0 means using all the image in train dir')
    return parseArgs.parse_args()

def time_stamp():
  TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
  return TIMESTAMP

def create_name_experiment(parameters, attribute_experiment):
    name_experiment = '{}/{}'.format(parameters['dataset'], attribute_experiment)

    print('Name experiment: {}'.format(name_experiment))

    return name_experiment

def create_folder(folder, force=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        if force:
            folder = folder+str(np.random.randint(100))
            os.makedirs(folder)
    return folder

def normalize_range(t, range=None):
    '''
    @Description: Normlize the input t into [0,1]
    @param {type} {t: tensor}
    @return: 
    '''
    def norm_ip1(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)
        return img
    
    if range is not None:
        output = norm_ip1(t, range[0], range[1])
    else:
        output = norm_ip1(t, float(t.min()), float(t.max()))
    return output

def loadweights(model, filename_model, gpu_ids=''):
    '''
    @Description: Load weights for pytorch model in different hardware environments
    @param {type} : {model: pytorch model, model that waits for loading weights
                     filename_model: str, name of pretrained weights
                     gpu_ids: list, available gpu list}
    @return: 
    '''
    if filename_model != '' and os.path.exists(filename_model):
        if len(gpu_ids) == 0:
            # load weights to cpu
            state_dict = torch.load(filename_model, map_location=lambda storage, loc: storage)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.','') # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif len(gpu_ids) == 1:
            state_dict = torch.load(filename_model)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.','') # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = torch.load(filename_model)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' not in k:
                    name = ''.join(['module.',k]) # add `module.`
                    new_state_dict[name] = v
            if new_state_dict:
                state_dict = new_state_dict
    else:
         state_dict = ''

    if len(gpu_ids) > 1:
         model = torch.nn.DataParallel(model,device_ids=gpu_ids)
    return model, state_dict

def cover_pytorch_tf(pytorch_weights, tf_model_var, sess, match_dict):
    '''
    @Description: This function is used to copy trained weights from pytorch to tensorflow.
    @param {type} : {pytorch_weights: OrderDict, save the weights of one model
                     tf_model_var: tf variable list, save the variable list in tf model
                     sess: tf.Session()
                     match_dict: dic, the match relationship between pytorch weights and tf weiths}
    @return: copied weights file name for tf
    '''
    import tensorflow as tf
    # py_weights_name = ['num_batches_tracked']
    tf_py_weights_name = {'kernel':'weight', 'bias':'bias', 'gamma':'weight', 'beta':'bias', 'moving_mean':'running_mean', 'moving_variance':'running_var'}
    for tf_v in tf_model_var:
        tf_names = tf_v.name.split('/')
        tf_layer_name = '/'.join(tf_names[1:3]) # used for confirm the layer relationship
        
        py_weight_name = tf_py_weights_name.get(tf_names[3].split(':')[0]) # used for confirming the weight or bias relationship
        py_layer_name = match_dict.get(tf_layer_name)
        if py_layer_name == None:
            continue
        py_name = '.'.join([py_layer_name, py_weight_name])
        py_w = pytorch_weights.get(py_name)
        if len(py_w.shape) == 4:
            py_w = py_w.permute(3,2,1,0) # [64, 3, 3, 3] => [3, 3, 3, 64]
        elif py_w.dim() == 2:
            py_w = py_w.permute(1,0)
        assign_op = tf.assign(tf_v, py_w.cpu().detach().numpy())
        sess.run(assign_op)
    return tf_model_var


def make_single_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x 1 x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(tensor.size(1), height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


TOTAL_BAR_LENGTH = 55.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def gauss(x,a,b,c):
    return torch.exp(-torch.pow(torch.add(x,-b),2).div(2*c*c)).mul(a)

def colorize(x):
     ''' Converts a one-channel grayscale image to a color heatmap image '''
     if x.dim() == 2:
         torch.unsqueeze(x, 0, out=x)
     if x.dim() == 3:
         cl = torch.zeros([3, x.size(1), x.size(2)])
         cl[0] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
         cl[1] = gauss(x,1,.5,.3)
         cl[2] = gauss(x,1,.2,.3)
         cl[cl.gt(1)] = 1
     elif x.dim() == 4:
         cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
         cl[:,0,:,:] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
         cl[:,1,:,:] = gauss(x,1,.5,.3)
         cl[:,2,:,:] = gauss(x,1,.2,.3)
     return cl

def visualize(logpath, d_inputs, c_att):
    in_c, in_y, in_x = d_inputs.shape
    # for item_img, item_att in zip(d_inputs, c_att):
    v_img = d_inputs.transpose(1,2,0)* 255. # change to h*w*c
    v_img = v_img[:, :, ::-1] # change to bgr
    resize_att = cv2.resize(c_att[0], (in_x, in_y))
    resize_att *= 255.

    # cv2.imwrite(os.path.join(logpath, 'CV_oriImg.png'), v_img)
    # cv2.imwrite(os.path.join(logpath, 'CV_attnImg.png'), resize_att)
    # v_img = cv2.imread(os.path.join(logpath, 'CV_oriImg.png'))
    # vis_map = cv2.imread(os.path.join(logpath, 'CV_attnImg.png'), 0)
    jet_map = cv2.applyColorMap(resize_att.astype(np.uint8), cv2.COLORMAP_JET)
    jet_map = cv2.add(0.6*v_img.astype(np.uint8), 0.4*jet_map)

    # out_path = os.path.join(logpath, 'attention_combine.png')
    # cv2.imwrite(out_path, jet_map)
    # out_path = os.path.join(logpath, 'rawImage.png')
    # cv2.imwrite(out_path, v_img)
    # count += 1
    return jet_map
