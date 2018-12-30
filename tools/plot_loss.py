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
# filename = '/home/uts/Desktop/param.log'
savepath = filename.split('.')[0]
show_freq = 1  # control whether print every loss on the figure. show_freq for print every 'show_freq'

itera = []
grad = []
aeLoss = []
gLoss = []
Dloss = []
flag = False
i = j = k = m = -1

try:
    f = open(filename,'r')
except IOError:
    print ("Open file failed or no such file!")
else:  
    filelines = f.readlines()
    for fileline in filelines:
        # print (fileline[25:30])
        if 'Generator Gradient:' in fileline:
            i += 1  
            if i % show_freq == 0:
                fileline = fileline.split(':')
                grad.append(float(fileline[-1].strip()))
        elif 'Autoencoder Loss:' in fileline:
            j += 1  
            if j % show_freq == 0:
                fileline = fileline.split(':')
                aeLoss.append(float(fileline[-1].strip()))
        elif 'Discriminator Loss:' in fileline:
            k += 1  
            if k % show_freq == 0:
                fileline = fileline.split(':')
                Dloss.append(float(fileline[-1].strip()))
        elif 'Generator Loss:' in fileline:
            m += 1  
            if m % show_freq == 0:
                fileline = fileline.split(':')
                gLoss.append(float(fileline[-1].strip()))
        else:
            pass
        
    f.close()
        
    
    fig1 = plt.figure(figsize=(14.5,10.5))

    plt.subplot(2,2,1)
    plt.plot(list(range(len(grad))),grad,'r')
    label = ["Gradient of generator"]
    plt.legend(label,loc=0)
        
    plt.subplot(2,2,2)
    plt.plot(list(range(len(aeLoss))),aeLoss,'r')
    label = ["Autoencoder loss"]
    plt.legend(label,loc=0)
    
    plt.subplot(2,2,3)
    plt.plot(list(range(len(Dloss))),Dloss,'g')
    label = ['Discriminator loss']
    plt.legend(label,loc=0)

    plt.subplot(2,2,4)
    plt.plot(list(range(len(gLoss))),gLoss,'g')
    label = ['Generator loss']
    plt.legend(label,loc=0)
    
    title = 'Gradients and loss of interpretable feature selection'
    plt.suptitle(title)
    plt.savefig(savepath+".png")
    plt.show()

    # plt.hist(grad)
    # plt.show()
    
