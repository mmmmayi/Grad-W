# Masking network

用来训练masking network，先分开训练，再进行end-to-end训练

## Generate mixed speech
数据库：DNS-challenge 2020 interspeech，安装git-lfs后直接clone就行了

调用 lst.py 中的 list_wav() 写lst文件，包含所有使用到的clean和noise utterance的路径

调用add_noise.sh 生成mixed speech

调用 lst.py 中的 get_list() 进行训练集和验证集划分，将路径分别写入lst文件

## Training 
使用可以处理变长的dataset的话，先用preprocessor生成json文件，包含路径和长度，然后运行train_new.py

使用默认的dataset的话直接运行train.py就可以了
