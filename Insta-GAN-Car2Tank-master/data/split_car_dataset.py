import os
from PIL import Image
import numpy as np

# path = 'data/cityscapes_data/'
# os.makedirs(path + '/seg_img/train', exist_ok=True)
# os.makedirs(path + '/seg_img/val', exist_ok=True)
# os.makedirs(path + '/ori_img/train', exist_ok=True)
# os.makedirs(path + '/ori_img/val', exist_ok=True)
#
# # split original images to origin and segmentation
# for data_type in ['train', 'val']:
#     all_files = os.listdir(path + data_type)
#     for f in all_files:
#         im = Image.open(path + data_type + '/' + f)
#         seg_array_image = np.array(im)[:, im.size[0] // 2:]
#         seg_image = Image.fromarray(np.uint8(seg_array_image))
#         seg_image.save(path + f'/seg_img/{data_type}/seg_val_{f}')
#
#         ori_array_image = np.array(im)[:, :im.size[0] // 2]
#         ori_image = Image.fromarray(np.uint8(ori_array_image))
#         ori_image.save(path + f'/ori_img/{data_type}/ori_val_{f}')
#
# # get mask from segmentations
# path = 'C:/Users\project26\PycharmProjects\pytorch-CycleGAN-and-pix2pix\datasets\cityscapes/'
# # os.makedirs('data/mask', exist_ok=True)
# all_seg_imgs = os.listdir(path + "trainB")
# for f in all_seg_imgs[:10]:
#     print()
#     seg = Image.open(path + "/trainB/" + f)
#     seg_array_image = np.array(seg)
#     mask = Image.fromarray(np.uint8((seg_array_image[:, :, 2] > 80)
#                                     & (seg_array_image[:, :, 1] < 20)
#                                     & (seg_array_image[:, :, 0] < 30)) * 255)
#     filename = f.split("_")[0]
#     print(filename,np.array(mask).sum()/255)
#
#     mask.save(f'../car_data/sampleA_seg_gif/{filename+".jpg"}')
#
#     img = Image.open(path + "/trainA/" + filename + "_A.jpg")
#     img.save(f'../car_data/sampleA/{filename + ".jpg"}')


### gif to jpg
# from PIL import Image
# import os
#
# files=os.listdir('../datasets/car2tank/sampleA_seg_gif/')
# for file in files:
#     file_name=file[:-9]
#     Image.open(f'../datasets/car2tank/sampleA_seg_gif/{file}').convert('RGB').save(f'../datasets/car2tank/sampleA_seg/{file_name}_0.jpg')
# img.save(f'../car_data/sampleA/{filename + ".jpg"}')

#### change name of file to XXXX_0.jpg


from PIL import Image
import os

files=os.listdir('../datasets/car2tank/trainA_seg/')
for i,file in enumerate(files):
    if file[-3:]== "jpg":
        print(i,file,end="\r")
    #     file_name=file.split(".")[0]
        # Image.open(f'../datasets/car2tank/trainB_seg/{file}').save(f'../datasets/car2tank/trainB_seg/{file_name}_0.jpg')
        # os.remove(f'../datasets/car2tank/trainB_seg/{file}')
        Image.open(f'../datasets/car2tank/trainA_seg/{file}').save(f'../datasets/car2tank/trainA_seg/{file[:-3]}png')
        os.remove(f'../datasets/car2tank/trainA_seg/{file}')
    #
    else:
        print(file)
