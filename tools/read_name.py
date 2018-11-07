import os

file_w=open('1.txt','w')
path1='../data/person2dingdang/dingdang/'
for root,dirs,files1 in os.walk(path1): 
    print files1
for filename in files1:
    print filename
    file_w.writelines(filename)
    file_w.writelines('\n')
file_w.close()
