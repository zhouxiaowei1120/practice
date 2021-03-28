import numpy as np
import os

count = 0
fw = open('summary.txt','w')
for root, dirs, files in os.walk('1/'):
    for fname in files:
        if 'log' in fname:
            fr = open(os.path.join(root,fname), 'r')
            print(os.path.join(root,fname))
            ft = fr.readlines()
            fw.writelines(os.path.join(root,fname)+'\t')
            count += 1
            accList = []
            for eachline in ft:
                if 'Current test acc:' in eachline:
                    accList.append(eachline.split(']')[-1])
            print(accList)
            if accList:
                acc = accList[-1]
            else:
                acc='\n'
            fw.writelines(acc)
fw.close()
print('Number of files: ', count)