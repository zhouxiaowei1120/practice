import os
from skimage import draw
import matplotlib.pyplot as plt
from PIL import Image

video_file = '/data/xiaozhou/project/supervision-by-registration/cache_data/cache/out_our.mp4'
image_path = './datasets/ourVideo/im_video/'
pre_image_path = './datasets/ourVideo/pre_im_video/'
pre_video_path = './datasets/ourVideo/'
testFile = './datasets/ourVideo/test_video.txt'

if os.path.exists(image_path):
    if os.path.isdir(image_path):
        if len(os.listdir(image_path)) == 0:
            print('ffmpeg -i '+video_file+' -qscale 1 '+image_path+'\%04d.png')
            os.system('ffmpeg -i '+video_file+' -qscale 1 '+image_path+'%04d.png')
        else:
            print('There are extracted images in the path:%s.',image_path)
    else:
        print('Error! %s is not a directory.',image_path)
else:
    os.mkdir(image_path)
    print('ffmpeg -i '+video_file+' -qscale 1 '+image_path+'\%04d.png')
    os.system('ffmpeg -i '+video_file+' -qscale 1 '+image_path+'%04d.png')

fileNameList = os.listdir(image_path)
coor = [0]*196
txtW = open(testFile)
for fileName in fileNameList:
    txtW.writelines(coor+''+fileName+'\n')
txtW.close()

print('./build/tools/alignment_tools run_test_on_wflw \
--input_file_1='+testFile+' '\
'--input_file_2=./meanpose/meanpose_71pt.txt \
--input_folder=./datasets/WFLW/WFLW_images/ \
--model_path=./models/WFLW/WFLW_final/ \
--output_file_1=./datasets/ourVideo/pred_98pt_largepose.txt \
--label_num=196 --thread_num=4')
os.system('./build/tools/alignment_tools run_test_on_wflw \
--input_file_1='+testFile+' '\
'--input_file_2=./meanpose/meanpose_71pt.txt \
--input_folder=./datasets/WFLW/WFLW_images/ \
--model_path=./models/WFLW/WFLW_final/ \
--output_file_1=./datasets/ourVideo/pred_98pt_largepose.txt \
--label_num=196 --thread_num=4' )

if not os._exists(pre_image_path):
    os.mkdir(pre_image_path)
txtR = open(testFile)
for fileName in fileNameList:
    pre_landmark = txtR.readline()
    pre_landmark = pre_landmark.split(' ')
    landmark = pre_landmark[0:-2]
    if fileName == pre_landmark[-1]:
        img = Image.open(fileName)
        for i in range(len(landmark)/2):
            rr,cc = draw.circle(landmark[i],landmark[i+1],5)
            draw.set_color(img,[rr,cc],[0,255,0])
        img.save(pre_image_path+fileName)

print('ffmpeg -i '+pre_image_path+'\%04d.png'+' -vcodec mpeg4 '+pre_video_path+'output.mp4')
os.system('ffmpeg -i '+pre_image_path+'%04d.png'+' -vcodec mpeg4 '+pre_video_path+'output.mp4')