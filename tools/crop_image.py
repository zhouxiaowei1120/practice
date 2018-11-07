# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
This is used for crop images.
"""

import os
from PIL import Image


#pathname = argv[1]
pathname = "E:/Desktop/gaojian_highR/"
box = (0,230,400,1080)
for root,dirs,filenames in os.walk(pathname):
    for imname in filenames:
        im_tmp = Image.open(os.path.join(root,imname))
        #(x,y) = im_tmp.size
        #print x,y
        #y = y * width // x
        im_crop = im_tmp.crop(box)
        new_name = root +'new_'+imname 
        im_crop.save(new_name)