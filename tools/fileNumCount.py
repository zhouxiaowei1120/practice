import os
import subprocess

fw=open('1.txt', 'w')
for root, dirs, files in os.walk('./datasets/ADE20K_2016_07_26/images/training'):
    for eachDir in dirs:
        fw.writelines(os.path.join(root,eachDir))
        num=subprocess.Popen("ls -l "+os.path.join(root,eachDir)+" | grep jpg | wc -l", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        num = str(num.communicate()[0])
        fw.writelines('\t'+num+'\n')
        print(os.path.join(root,eachDir), num)
fw.close()
