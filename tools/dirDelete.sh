#!/bin/bash
#遍历文件夹、删除指定名字的文件夹
#问题：受保护的文件删除不了
 
function scandir() {
    local cur_dir parent_dir workdir
    workdir=$1
    cd ${workdir}
    if [ ${workdir} = "/" ]
    then
        cur_dir=""
    else
        cur_dir=$(pwd)
    fi
 
    for dirlist in $(ls ${cur_dir})
    do
        if test -d ${dirlist};then
            if [[ ${dirlist} = $2 ]]
            then
                echo 'delete fold: '${dirlist}
                rm -rf ${dirlist}
            else
                cd ${dirlist}
                scandir ${cur_dir}/${dirlist} $2
                cd ..
            fi
        fi
    done
}
 
if test -d $1
then
    scandir $1 $2
elif test -f $1
then
    echo "you input a file but not a directory,pls reinput and try again"
    exit 1
else
    echo "the Directory isn't exist which you input,pls input a new one!!"
    exit 1
fi
