from numpy import *
import matplotlib
import matplotlib.pyplot as plt
x = random.rand(100,30)
 
#basic
f1 = plt.figure(1)
plt.subplot(211)
plt.scatter(x[:,1],x[:,0])
 
# with label
plt.subplot(212)
label = list(ones(30))+list(2*ones(25))+list(3*ones(25)) + list(4*ones(20))
label = array(label)
# plt.scatter(x[:,1],x[:,0],15.0*label,15.0*label)
# plt.show() 

# with legend
f2 = plt.figure(2)
idx_1 = where(label==1)
p1 = plt.scatter(x[idx_1,1], x[idx_1,0], marker = 'x', color = 'm', label='1', s = 30)
idx_2 = where(label==2)
p2 = plt.scatter(x[idx_2,1], x[idx_2,0], marker = '+', color = 'c', label='2', s = 50)
idx_3 = where(label==3)
p3 = plt.scatter(x[idx_3,1], x[idx_3,0], marker = 'o', color = 'r', label='3', s = 15)
idx_3 = where(label==4)
p4 = plt.scatter(x[idx_3,1], x[idx_3,0], marker = '^', color = 'g', label='4', s = 20)
# plt.legend(loc = 'upper right')
plt.show() 
