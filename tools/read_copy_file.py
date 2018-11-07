import os

file_w=open('1.sh','w')
listFileName='/data/xiaozhou/Downloads/datasets/imagenet/caffe_ilsvrc12/val.txt'
cpLabel = {'242':'random500_1','309':'random500_2','310':'random500_3','347':'random500_4','348':'random500_5','349':'random500_6','357':'random500_7','358':'random500_8','359':'random500_9'}
path2='/data/xiaozhou/Downloads/datasets/imagenet/imagenet12-val/'
path3='/data/xiaozhou/Downloads/datasets/imagenet/testimages/'

with open(listFileName) as f:
    listfiles = f.readlines()

    
for filename in listfiles:
    filename = filename.strip('\n')
    filename = filename.split(' ')
    print(filename)
    for key,value in cpLabel.items():
        if filename[-1] == key:
           desDir = path3+value+'/'
           if not os.path.exists(desDir):
               os.mkdir(desDir)
           file_w.writelines("cp "+path2+filename[0]+" "+path3+value+'/')
           file_w.writelines('\n')

file_w.close()
val = os.system("sh 1.sh")
print (val)
