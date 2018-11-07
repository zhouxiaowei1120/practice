# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:55:51 2017

@author: Administrator
"""
import os
import cv2
import re

#pathname = argv[1]
repathname ="juezhan_cropped_res/HRGaoLi_cycleGAN/test_latest/images/"
impath = "juezhan/"
path_post = "juezhan_post/"
filer = open('1.txt','r')
for res_line in filer.readlines():
    eachline = re.sub('[\[\]\n\ \']','',res_line)
    print (eachline)
    eachline = eachline.split(',')
    print (eachline)
    imname_s = eachline[0].split('.')[0]
    imname = os.path.join(repathname,imname_s)
    imname = imname + '_fake_B.png'
    #print (imname)
    if not os.path.exists(imname):
        print("No file {} exists.".format(imname))
        continue
    im_tmp = cv2.imread(imname)
    #print(im_tmp)
    print (int(eachline[-1]),int(eachline[-2]))
    im_res = cv2.resize(im_tmp,(int(eachline[-1]),int(eachline[-2])))
    print ("The resized size is:{}".format(im_res.shape))
    print (os.path.join(impath,eachline[0]))
    im_tmp = cv2.imread(os.path.join(impath,eachline[0]))
    print (im_tmp.shape)
    print (int(eachline[2]),int(eachline[4]),int(eachline[1]),int(eachline[3]))
    im_tmp[int(eachline[2]):int(eachline[4]),int(eachline[1]):int(eachline[3])]=im_res
    cv2.imwrite(os.path.join(path_post,eachline[0]),im_tmp)
    print("Image {} is saved.".format(eachline[0]))
filer.close()
