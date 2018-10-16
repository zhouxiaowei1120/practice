import tensorflow as tf
import os

pathname = 'C:/Users/12882357/Desktop/personal/Phone Plan/'
shape = [224,224]

for root, dirs, filenames in os.walk(pathname):
    for filename in filenames:
        print(filename)