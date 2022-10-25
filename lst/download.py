import os
import shutil

file = open('test.lst','r')
line = file.readline()
download_path = '/data07/mayi/download'
while line:
    line = line.strip('\n')
    name = line.split('/')[-1]
    shutil.copyfile(line,os.path.join(download_path,name))
    line = file.readline()
