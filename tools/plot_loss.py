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

#filename = input("Input:")
#filename = "E:\Desktop\Cycle_GAN\CycleGAN-and-pix2pix-master\Experiments\horse2zebra\result\loss_log.txt"
filename = "E:/Desktop/loss_log.txt"
open_sta = 1  # control whether print every loss on the figure. 0 for print all, 1 for print every 10

D_A = []
G_A = []
Cyc_A = []
D_B = []
G_B = []
Cyc_B = []
i = 0

try:
    f = open(filename,'r')
except IOError:
    print "Open file failed or no such file!"
else:  
    filelines = f.readlines()
    for fileline in filelines:
        if fileline[0] == "=":
            title_1 = fileline[17:31]
            print title_1
        elif fileline[0] == "(":
            i += 1
            if (open_sta):
                if (i == 1 or i % 10 == 0):
                    fileline = fileline.split(' ')
                    D_A.append(float(fileline[7]))
                    G_A.append(float(fileline[9]))
                    Cyc_A.append(float(fileline[11]))
                    D_B.append(float(fileline[13]))
                    G_B.append(float(fileline[15]))
                    Cyc_B.append(float(fileline[17]))
            else:
                fileline = fileline.split(' ')
                D_A.append(float(fileline[7]))
                G_A.append(float(fileline[9]))
                Cyc_A.append(float(fileline[11]))
                D_B.append(float(fileline[13]))
                G_B.append(float(fileline[15]))
                Cyc_B.append(float(fileline[17]))
    f.close()
        
    itera = [n for n in range(1,len(D_A)+1)]

    fig1 = plt.figure(figsize=(14.5,10.5))

    plt.subplot(2,1,2)
    plt.plot(itera,D_B,'r')
    plt.plot(itera,G_B,'g')
    plt.plot(itera,Cyc_B,'b')  
    label = ["D_B","G_B","Cyc_B"]
    plt.legend(label,loc=0,ncol=3)
        
    plt.subplot(2,1,1)
    plt.plot(itera,D_A,'r')
    plt.plot(itera,G_A,'g')
    plt.plot(itera,Cyc_A,'b')
    label = ["D_A","G_A","Cyc_A"]
    plt.legend(label,loc=0,ncol=3)
    
    plt.title(title_1)
    plt.savefig("E:/Desktop/loss_log.png")
    plt.show()
    