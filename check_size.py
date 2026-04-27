from PIL import Image
import os

os.chdir(r'd:\Projects\0_un_finish\123python123')

files = ['image.jpg', 'image_gt.jpg', 'image_lr.jpg']
for f in files:
    if os.path.exists(f):
        img = Image.open(f)
        print(f'{f}: {img.size[0]}x{img.size[1]}')
