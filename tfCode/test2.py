import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow  

model_dir = '/home/xiaozhou/Data/research/dataGenaration/tcav/'
checkpoint_path = os.path.join(model_dir, "inception_v1.ckpt")  
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)