import os

file_w=open('1.sh','w')
path1='/data/zxw/xiaolianer/pix2pix/data/B/val/'
path2='/data/zxw/xiaolianer/deepfeatinterp-master/images/p2p_data/'
path3='/data/zxw/xiaolianer/pix2pix/data/A/val/'
for root,dirs,files1 in os.walk(path1):
    for filename in files1:
       print filename
       filename = filename.split('/')
       file_w.writelines("cp -as "+path2+filename[-1]+" "+path3)
       file_w.writelines('\n')
file_w.close()
val = os.system("sh 1.sh")
print val
