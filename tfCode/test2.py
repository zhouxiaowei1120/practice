import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow  

model_dir = '/media/uts/Windows/Users/12882357/OneDrive - University of Technology Sydney/research/R_interFS/'
checkpoint_path = os.path.join(model_dir, "vgg_19.ckpt")  
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map() 
for key in var_to_shape_map:  
    print("tensor_name: ", key)  
    print(reader.get_tensor(key).shape) 

input('Input')