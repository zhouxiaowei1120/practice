# -*- coding: utf-8 -*-
def im_crop(ori_path,des_path,width):
"""
Spyder Editor

This is a temporary script file.
This is used for crop images.
"""

import os
from PIL import Image


#pathname = argv[1]
pathname = des_path
for root,dirs,filenames in os.walk(ori_path):
    for imname in filenames:
        im_tmp = Image.open(os.path.join(root,imname))
        (x,y) = im_tmp.size
        print x,y
        y = y * width // x
        im_resize = im_tmp.resize((width,y),Image.ANTIALIAS)
        new_name = root +'new_'+imname 
        im_resize.save(new_name)