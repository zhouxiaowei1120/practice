import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

for root, dirs, filenames in os.walk("p2p_data/"):
    for filename in filenames:
        im = Image.open(os.path.join(root,filename))
        img = im.convert('RGB')
        im_name = os.path.join(root,os.path.splitext(filename)[0])
        img.save(im_name+'.png')
        print ("Saving image:{}".format(im_name))
