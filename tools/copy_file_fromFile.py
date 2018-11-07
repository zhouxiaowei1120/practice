# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 15:26:49 2017

@author: Administrator
"""

import os
import shutil
from PIL import Image

filenamepath = "E:/Desktop/liangjian_test.txt"
spath = "E:/Desktop/liangjian_test/"
tpath = "E:/Desktop/liangjian_test_select/"

readFile = open(filenamepath)
for filename in readFile.readlines():
    print filename
    filename = filename.replace("\n","")
    shutil.copyfile(os.path.join(spath,filename),os.path.join(tpath,filename))