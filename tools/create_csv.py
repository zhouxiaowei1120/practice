# This file is used for generating csv file that is used in training 3D face swap model.
# Written by Dave Zhou
# 2017.12.25

import os
import sys

#sourcefile = "Desktop/DNA/images/0055.png"
#pathname = "Desktop/DNA/liangjian_test_select/"
sourcefile = sys.argv[1]
pathname = sys.argv[2]
csvname = sys.argv[3]

filew = open(csvname,'w')
filelist = os.listdir(pathname)
filelist.sort()
for filename in filelist:
    wtmp = sourcefile + ',' + pathname + filename
    filew.writelines(wtmp)
    filew.writelines('\n')
filew.close()
