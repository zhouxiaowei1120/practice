# randomly pick and show the images in html format

import dominate
from dominate.tags import meta, h3, h6, table, tr, td, p, a, img, br
from dominate.util import text
import os
import numpy as np
import random
import argparse

class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.
     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes
        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            reflect (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def get_attrs_dir(self):
        """Return the directory that stores images"""
        return self.attrs_dir

    def add_header(self, text):
        """Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)


    def add_test_images(self, ims, names, txts, links, width=400):
        """add images to the HTML file
        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, name, txt, link in zip(ims, names, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            attrs_path = os.path.join(self.web_dir,'attributes', txt)
                            with open(attrs_path, "r") as attr_log_file:
                                attrs_str =attr_log_file.read()  # save the message
                            attrs_str0 = name + attrs_str
                            attrs_str0.replace('\n','<br>')
                            text(attrs_str0)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file
        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href= link):
                                img(style="width:%dpx" % width, src= im)
                            br()
#                            attrs_path = os.path.join(self.web_dir,'attributes', txt)
#                            with open(attrs_path, "r") as attr_log_file:
#                                attrs_str =attr_log_file.read()  # save the message
#                            attrs_str.replace('\n','<br>')
                            text(txt)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = os.path.join(self.web_dir, 'index.html')
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':  # we show an example usage here.
    parseargs = argparse.ArgumentParser(description='Arguments for running')
    parseargs.add_argument('img_dir', type=str, default='./', help='path of storing the images')
    parseargs.add_argument('--img_size', type=int, default=256, help='the size of display image')
    parseargs.add_argument('--seed', type=int, default=1, help='seed for the program')
    parseargs.add_argument('--prob', type=float, default=0.1, help='probability to show one image')
    parseargs.add_argument('--info', type=str, default='hello world', help='info about the images')
   
    args = parseargs.parse_args()
    if not args.img_dir.endswith('/'):
        args.img_dir = args.img_dir + '/'
    html = HTML(args.img_dir, args.info)
    html.add_header('hello world')
    np.random.seed = args.seed
    random.seed = args.seed
    
    prob = np.array([args.prob,1-args.prob]) # 10% probability to show the images in that subfolder
    ims, txts, links = [], [], []
    for root, dirs, files in os.walk(args.img_dir):
        for each_dir in dirs:
            if each_dir == 'assignments':
                continue
            choose_flag = np.random.choice([1,0], p=prob)
            if not choose_flag:
                continue
            flist = list(sorted(os.listdir(os.path.join(root,each_dir))))
            for fname in flist:
                if fname[-4:] in ['.png', 'jpg']:
                    full_path = os.path.join(root, each_dir, fname)
                    full_path = full_path.replace(args.img_dir, '')
                    ims.append(full_path)
                    txts.append(full_path)
                    links.append(full_path)
                elif 'assignments' in fname:
                    tmp_flist = list(sorted(os.listdir(os.path.join(root,each_dir, fname))))
                    for tmp_name in tmp_flist:
                        full_path = os.path.join(root, each_dir, fname, tmp_name)
                        full_path = full_path.replace(args.img_dir, '')
                        ims.append(full_path)
                        txts.append(full_path)
                        links.append(full_path)
            html.add_images(ims, txts, links, args.img_size)
            html.save()
            ims, txts, links = [], [], []
