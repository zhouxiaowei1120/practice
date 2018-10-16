import os

file_w=open('1.sh','w')
listFileName='/data/xiaozhou/Downloads/datasets/imagenet/caffe_ilsvrc12/val.txt'
cpLabel = {'340':'zebra','603':'horse_cart','254':'pug'}
path2='/data/xiaozhou/Downloads/datasets/imagenet/imagenet12-val/'
path3='/data/xiaozhou/Downloads/datasets/imagenet/testimages/'

with open(listFileName) as f:
    listfiles = f.readlines()

    
for filename in listfiles:
    filename = filename.strip('\n')
    filename = filename.split(' ')
    print(filename)
    if filename[-1] == '340':
        file_w.writelines("cp -as "+path2+filename[0]+" "+path3+cpLabel['340']+'/')
        file_w.writelines('\n')
    elif filename[-1] == '603':
        file_w.writelines("cp -as "+path2+filename[0]+" "+path3+cpLabel['603']+'/')
        file_w.writelines('\n')
    elif filename[-1] == '254':
        file_w.writelines("cp -as "+path2+filename[0]+" "+path3+cpLabel['254']+'/')
        file_w.writelines('\n')

file_w.close()
val = os.system("sh 1.sh")
print (val)
