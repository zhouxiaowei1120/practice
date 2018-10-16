# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 15:21:19 2017

@author: Administrator
"""

from PIL import Image
import os  
import numpy as np

def mergeReport(files,img_name):
    baseimg=Image.open(files[0])
    sz = baseimg.size
    basemat=np.atleast_2d(baseimg)
    for file in files[1:]:
        im=Image.open(file)
    #resize to same width
        sz2 = im.size
        if sz2[0]!=sz[0]:
            print ("tar image width is:%d, ori image width is:%d" %(sz2[0],sz[0]))
            im=im.resize((sz[0],round(sz2[0] / sz[0] * sz2[1])),Image.ANTIALIAS)
        mat=np.atleast_2d(im)
        basemat=np.append(basemat,mat,axis=1)
    report_img=Image.fromarray(basemat)
    report_img.save(img_name)


path1 = "E:/Desktop/neutral_test1/neutral_test/"              #文件夹目录  
for root,dirs,files1 in os.walk(path1):       #得到文件夹下的所有文件名称  
    print files1      

print "============================"
print files1

print "============================"

# path2 = "/home/xbsj/Desktop/1/"
# filesname2=[]
# for root,dirs,files2 in os.walk(path2):
    # print files2
# files2.sort(key= lambda files2:int(files2[11:-4]))
# #files2.sort()
# print "============================"
# print files2
outpath = "E:/Desktop/neutral_test1/1/"
num_file = len(files1)
m = 0

for i in range(num_file/13): #遍历文件夹 
    files=[]
    tempfile=path1+files1[i*13]
    files.append(tempfile)
    n=1
    m = m+1
    for j in range(4):
        for k in range(3):
            img1=path1+files1[n+i*13]
            print n+i*13
            n = n+1
            files.append(img1)
        if len(files)==4:
            tempfile = files[3]
            tempfile = tempfile.split('_')
            exptype=''.join(tempfile[-2])
            img_name = outpath + "%04d_" %m +exptype+".jpg"
            mergeReport(files,img_name)
            tempfile = files[0]
            files=[]
            files.append(tempfile)
print ("Finished!")
