from collections import OrderedDict
import torch
import torch.utils.data as data
import tensorflow as tf

def loadweights(model, filename_model, gpu_ids=''):
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
        model = torch.nn.DataParallel(model,device_ids=gpu_ids)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                name = ''.join(['module.',k]) # add `module.`
                new_state_dict[name] = v
        if new_state_dict:
            state_dict = new_state_dict
    return model, state_dict

class celebA_dataset(data.Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.filenames = []
        self.train = train
        if train == True:
            data_dir = os.path.join(data_dir, 'train')
        else:
            data_dir = os.path.join(data_dir, 'test')
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename[-4:] not in ['.png','.jpg']:
                    continue
                self.filenames.append(os.path.join(root, filename))
        self.transform = transform
        with open('./datasets/celebA/labels/list_attr_celeba.txt') as labels:
            self.labelsList = labels.readlines()
        self.labelsList = self.labelsList[2:]

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename)
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(np.array(img))

        idx = int(filename.split('/')[-1].split('.')[0])
        labelLine = self.labelsList[idx-1]
        labelLine = labelLine.rstrip('\n')
        labelLine = labelLine.split()
        label = int(labelLine[32])
        if label == -1:
            label = 0

        return img, label
    
    def __len__(self):
        return len(self.filenames)

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


kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
        celebA_dataset('./datasets/celebA_post/', train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
