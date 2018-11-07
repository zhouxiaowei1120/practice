#!/bin/bash

for names in ./1/*.jpg
do
  i=$(($i+1))
  new_name=`printf "./1/%04d.jpg" $i`
  echo $new_name
  echo mv "$names" "$new_name"
  mv "$names" "$new_name"
done 
