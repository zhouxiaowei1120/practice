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
import imageutils
import argv

def getface(rgbImg):
    #print (rgbImg.mode)
    facefound = True
    detector=dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    #img = io.imread('1.jpg')
    faces = detector(rgbImg, 1)
    if len(faces) > 0:
      face=max(faces, key=lambda rect: rect.width() * rect.height())
      [x1,x2,y1,y2]=[face.left(),face.right(),face.top(),face.bottom()]
      fshape = landmark_predictor(rgbImg,face)
    else:
      x1=x2=y1=y2=0
      fshape = 0
      facefound = False
    return [facefound,x1,x2,y1,y2,fshape]

#pathname = argv[1]
# pathname = argv[1]
pathname = ''pure/"
path_post = argv[2]
#"/data/zxw/xiaolianer/deepfeatinterp-master/images/exp_pure_post/"
w_max =64
h_max =64
for root,dirs,filenames in os.walk(pathname):
    for imname in filenames:
        post_dir = path_post+root.split('/')[-1]
        imgname = imname.split('.')[0] + '.png'
        postprocess_name = os.path.join(post_dir,imgname)
        if os.path.exists(postprocess_name):
            print("File exists! Skipping image:{}".format(os.path.join(root,imname)))
            continue
        if not os.path.exists(post_dir):
            os.mkdir(post_dir)
        print ("Processing image:{}".format(os.path.join(root,imname)))
        im_tmp = Image.open(os.path.join(root,imname))
        (x,y) = im_tmp.size
        im_tmp = im_tmp.convert('RGB')
        im = np.array(im_tmp)
        facefound,x1,x2,y1,y2,fshape=getface(im) # get landmarks and face locations
        if facefound == False:
            print("No faces found.")
            continue
        landmarks = np.array([[fshape.part(i).x,fshape.part(i).y] for i in range(68)])
        #print (landmarks)
        eye_dis = int(np.linalg.norm(landmarks[42] - landmarks[39]))
        pad_dis = 2 * (landmarks[33] - landmarks[27])
        landmarks[19] = landmarks[19]-pad_dis
        landmarks[24] = landmarks[24]+pad_dis
        print ("eye_dis:{},pad_dis:{}".format(eye_dis,pad_dis))
        left = min(landmarks[...,1])-eye_dis
        right = max(landmarks[...,1])+eye_dis
        up = min(landmarks[...,0])-eye_dis
        down = max(landmarks[...,0])+eye_dis
        #print(im.shape)
        rect = im[max(0,left):right,max(0,up):down]
        #print (rect.shape)
        if rect.shape[0]>h_max or rect.shape[1]>w_max:
            rect = rect / 255.0
            scale = min(float(h_max)/rect.shape[0],float(w_max)/rect.shape[1])
            print (scale)
            rect = imageutils.scale(rect,scale)
            imageutils.write(postprocess_name,rect)
        else:
            rect = Image.fromarray(rect)
            rect.save(postprocess_name)
        #(x_p,y_p) = rect.size
        print ("Saving image as:{}".format(postprocess_name))
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
