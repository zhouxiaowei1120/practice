# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:55:51 2017

@author: Administrator
"""
import os
import dlib
import cv2

def getface(rgbImg):
    #print (rgbImg.mode)
    facefound = True
    detector=dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('/data/zxw/xiaolianer/deepfeatinterp-master/shape_predictor_68_face_landmarks.dat')
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
pathname ="juezhan_test/"
path_post = "juezhan/"
filew = open('1.txt','w')
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
        im_tmp = cv2.imread(os.path.join(root,imname))
        facefound,x1,x2,y1,y2,fshape=getface(im_tmp) # get landmarks and face locations
        if facefound == False:
            print("No faces found.")
            continue
        rect = im_tmp[y1:y2,x1:x2]
        print (rect.shape)
        res = cv2.resize(rect,(256,256))
        cv2.imwrite(postprocess_name,res)
        bu_tmp = [imgname,x1,y1,x2,y2,rect.shape[0],rect.shape[1]]
        bu_tmp = str(bu_tmp)
        filew.writelines(bu_tmp)
        filew.writelines('\n')
        #(x_p,y_p) = rect.size
        print ("Saving image as:{}".format(postprocess_name))
filew.close()
