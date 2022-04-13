import os 
import shutil

directory = "../assets/summit_xl_description"

for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename[:13] == 'AnyConv.com__':
            newFilename = filename[13:].split('.obj')[0]+'_1'+'.obj'
            shutil.copyfile(os.path.join(root, filename), os.path.join(root, newFilename))