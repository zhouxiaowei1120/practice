import os

filelist = os.listdir('./train/')

filelist = filelist[:100]

file_w = open('1.sh','w')

for i in filelist:
   print(i)
   file_w.writelines("mv ./train/"+i+" ./test/")
   file_w.writelines("\n")

file_w.close()
val = os.system("sh 1.sh")
print(val)
