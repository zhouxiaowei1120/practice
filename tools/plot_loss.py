# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:54:19 2017

Description: This code is usde for visualizing the loss curve.

Input: loss_log.txt The file generated from the code train.py
Output: loss_log.png The file of loss curve.
@author: zxw
"""

import matplotlib.pyplot as plt
import sys

filename = input("Input:")
#filename = "E:\Desktop\Cycle_GAN\CycleGAN-and-pix2pix-master\Experiments\horse2zebra\result\loss_log.txt"
#filename = "/home/uts/param.log"
show_freq = 1  # control whether print every loss on the figure. show_freq for print every 'show_freq'

loss = []
itera = []
flag = False
i = -1

try:
    f = open(filename,'r')
except IOError:
    print ("Open file failed or no such file!")
else:  
    filelines = f.readlines()
    for fileline in filelines:
        # print (fileline[25:30])
        if fileline[25:30] == 'epoch':
            i += 1  
            if i % show_freq == 0:
                fileline = fileline[25:-1]
                fileline = fileline.split(',')
                for eachline in fileline:
                    eachline = eachline.replace(' ','').split(':')
                    if flag:
                        loss.append(float(eachline[1]))
                        flag = False
                    else:
                        itera.append(int(eachline[1])) 
                        flag = True
    f.close()
        
    
    fig1 = plt.figure(figsize=(14.5,10.5))

    # plt.subplot(2,1,2)
    # plt.plot(itera,D_B,'r')
    # plt.plot(itera,G_B,'g')
    # plt.plot(itera,Cyc_B,'b')  
    # label = ["D_B","G_B","Cyc_B"]
    # plt.legend(label,loc=0,ncol=3)
        
    # plt.subplot(2,1,1)
    # plt.plot(itera,D_A,'r')
    # plt.plot(itera,G_A,'g')
    # plt.plot(itera,Cyc_A,'b')
    # label = ["D_A","G_A","Cyc_A"]
    plt.plot(itera,loss,'g')
    label = ['training loss']
    plt.legend(label,loc=0,ncol=3)
    
    title = 'Generate data on the boundary: dataset=\'imagenet\', iteration=500'
    plt.title(title)
    plt.savefig("loss_log.png")
    plt.show()
    
