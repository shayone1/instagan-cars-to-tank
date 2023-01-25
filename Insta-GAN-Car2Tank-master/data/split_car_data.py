import os
import random
import shutil
files = os.listdir("../datasets\car2tank/trainA_old")

sorted(files)
save_files = []
for ind in range(0, len(files), 16):
    idx = random.sample(range(15), 2)
    if idx[0] == idx[1]:
        idx[1] = 3 if idx[0] != 3 else 4
    save_files.append(files[ind + idx[0]])
    save_files.append(files[ind + idx[1]])

print(len(save_files))
for file in save_files:
    source="../datasets/car2tank/trainA_old/"+file
    destination="../datasets/car2tank/trainA/"+file
    dest = shutil.copyfile(source, destination)
    mask_name=file.replace(".png","_0.png")
    source="../datasets/car2tank/trainA_seg_old/"+mask_name
    destination="../datasets/car2tank/trainA_seg/"+mask_name
    dest = shutil.copyfile(source, destination)