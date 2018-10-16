#!/bin/bash

for names in /data/zxw/xiaolianer/deepfeatinterp-master/results/*
do
  #echo $names
  i=$(($i+1))
  echo $i
  if [ $i -le 500 ]
  then
      echo cp -as "$names" "../pix2pix/data/B/train/"
      cp -as "$names" "../pix2pix/data/B/train/"
      continue
  elif [ $i -le 600 ];then
      cp -as "$names" "../pix2pix/data/B/test/"
      continue
  else
      #echo cp -as "$names" "../pix2pix/data/B/val/"
      cp -as "$names" "../pix2pix/data/B/val/"
  fi
done
