# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
This is used for crop images.
"""

import os
from PIL import Image
import numpy as np
import dlib

def getface(rgbImg):  
    detector=dlib.get_frontal_face_detector()  
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    #img = io.imread('1.jpg')  
    faces = detector(rgbImg, 1)  
    if len(faces) > 0:  
      face=max(faces, key=lambda rect: rect.width() * rect.height())  
    [x1,x2,y1,y2]=[face.left(),face.right(),face.top(),face.bottom()]
    fshape = landmark_predictor(rgbImg,face)
    return [x1,x2,y1,y2,fshape]

#pathname = argv[1]
pathname = "E:/Desktop/test/"
path_post = "E:/Desktop/test_post/"
width = 256
height = 256
w_min = 141
h_min =175
w_max =800
h_max =1000
for root,dirs,filenames in os.walk(pathname):
    for imname in filenames:
        im_tmp = Image.open(os.path.join(root,imname))
        (x,y) = im_tmp.size
        im = np.array(im_tmp)
        x1,x2,y1,y2,fshape=getface(im_tmp) # get landmarks and face locations
        l_len = fshape.part(27).y - fshape.part(36).y
        l_pad = abs(fshape.part(36).y - 2*l_len)
        r_len = fshape.part(45).y - fshape.part(27).y
        r_pad = abs(x - fshape.part(45).y - 2*r_len)
        eye_len = abs(fshape.part(45).y - fshape.part(36).y)
        up_len = fshape.part(30).x - fshape.part(27).x
        up_pad = abs(fshape.part(27).x - 2*up_len)
        down_len = fshape.part(8).x - fshape.part(27).x
        down_pad = abs(y - fshape.part(8).x - 2*down_len)
        f_width = 2*l_len + 2*r_len
        f_height = 2*up_len + 2*down_len
        if fshape.part(36).y < 2*l_len:
           im = np.lib.pad(im_tmp,(l_pad,0),'constant',constant_values=255)
        else:
           im = im[:,l_pad:-1,:]
           
        if (x - fshape.part(45).y) < 2*r_len:
           im = np.lib.pad(im_tmp,(0,r_pad),'constant',constant_values=255)
        else:
           im = im[:,:(x-r_pad),:]
        
        if fshape.part(27).x < 2*up_len:
           im = np.lib.pad(im_tmp,((0,0),(up_pad,0)),'constant',constant_values=255)
        else:
           im = im[up_pad:-1,:,:]
          
        if y - fshape.part(8).x < 2*down_len:
           im = np.lib.pad(im_tmp,((0,0),(0,down_pad)),'constant',constant_values=255)
        else:
           im = im[:(y-down_pad),:,:]
        postprocess_name = os.path.join(path_post,imname)
        im_tmp = Image.fromarray(im)
        im_tmp.save(postprocess_name)
        (x,y) = im_tmp.size
        print "Saving image as:{},width:{};height:{}".format(postprocess_name,x,y)
#        if x < width:
#            w_pad = width-x
#            if y < height:
#                h_pad = height - y
#                im = np.lib.pad(im_tmp,((w_pad//2,w_pad-w_pad//2),(h_pad//2,h_pad-h_pad//2)),'constant',constant_values=255)
#            elif y > height:
#                im = np.lib.pad(im_tmp,(w_pad//2,w_pad-w_pad//2),'constant',constant_values=255)
#        y = y * width // x
#        im_resize = im_tmp.resize((width,y),Image.ANTIALIAS)
#        new_name = root +'new_'+imname 
#        im_resize.save(new_name)