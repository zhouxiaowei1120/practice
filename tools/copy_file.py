import os

file_w=open('1.sh','w')
path1='/home/uts/data/research/R_generation/experiments/gsn_hf/mnist_train_test_lenet_3_1norm_ncfl50_NormL2/transimg/att_svm/'
path2='/data/zxw/xiaolianer/deepfeatinterp-master/images/p2p_data/'
path3='/home/uts/data/research/R_generation/experiments/gsn_hf/mnist_train_test_lenet_3_1norm_ncfl50_NormL2/transimg/att_svm1/'
for root,dirs,files1 in os.walk(path1):
    for filename in files1:
       dirnames = root.split('/')
       if not os.path.exists(os.path.join(path3, dirnames[-2])):
              os.makedirs(os.path.join(path3, dirnames[-2]))
       filenames = filename.split('_')
       if filenames[2] == 'ori.png':
              continue
       if int(filenames[3][3]) != int(dirnames[-2][0]):
              file_w.writelines("cp "+os.path.join(root, filename)+" "+os.path.join(path3, dirnames[-2]))
              file_w.writelines('\n')
              print (filename)
              
file_w.close()
val = os.system("sh 1.sh")
print (val)
