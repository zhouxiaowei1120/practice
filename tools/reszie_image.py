# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
This is used for crop images.
"""

import os
from PIL import Image


#pathname = argv[1]
pathname = "./"
width = 800
for root,dirs,filenames in os.walk(pathname):
    for imname in filenames:
        if not 'png' in imname and not 'jpg' in imname:
            continue
        im_tmp = Image.open(os.path.join(root,imname))
        (x,y) = im_tmp.size
        print (x,y)
        y = y * width // x
        im_resize = im_tmp.resize((width,y),Image.ANTIALIAS)
        new_name = os.path.join(root, 'new_'+imname)
        im_resize.save(new_name)
