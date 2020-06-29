import os
from sys import argv

path = argv[1]
for root, dirs, filename in os.walk(path):
    for fn in filename:
        if '2.000000' in fn:
           print(fn)
