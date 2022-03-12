from PIL import Image
import os
from pylab import *
import re
from PIL import Image, ImageChops, ImageEnhance
import split_folders  # or import split_folders
from tqdm import tqdm


def get_imlist(path):
    path = "/home1"
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.tif')]

def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'
    
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    
    return ela_im


real_images_list=get_imlist("./Mediaeval2016") ## real image list
#fake_images_list=get_imlist("./DATASET4/fake") ## fake image list

quality=90 ## Quality of Image

print("Pre Processing ELM for Real Images")
for img in tqdm(real_images_list):
    elm_img=convert_to_ela_image(img, quality) ## get ELM Image (REAL)
    elm_img.save("./DATASET_ELM/real/"+img.split("/")[-1], 'JPEG', quality=quality)

print("Pre Processing ELM for Fake Images")
#for img in tqdm(fake_images_list):
 #   elm_img=convert_to_ela_image(img, quality) ## get ELM Image  (FAKE)
 #   elm_img.save("./DATASET_ELM/fake/"+img.split("/")[-1], 'JPEG', quality=quality)


# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#print("Splitting into Train/Validation Split")
#split_folders.ratio("./DATASET_ELM", output="./DATASET_FINAL", seed=1337, ratio=(.8, .2)) # default values
