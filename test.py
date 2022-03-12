import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from argparse import ArgumentParser
import os.path
import json
from PIL import Image, ImageChops, ImageEnhance

input_path = "./DATASET_FINAL/"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

## Convert to ELM image
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


## Generators
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'val':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}



### network
model = models.wide_resnet50_2(pretrained=False).to(device)
    
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)
model.load_state_dict(torch.load("./models/weights.h5",map_location=torch.device('cpu')), strict=False)
model.eval()


if __name__ == "__main__":
    # execute only if run as a script
    parser = ArgumentParser(description="File Parser")
    parser.add_argument("-i", dest="filename", required=True,
                        help="Input File Path", metavar="FILE")
    args = parser.parse_args()

    arg=args.filename
    path_type=['png','jpg','jpeg','tif']
    if arg.split('.')[-1] not in path_type:
        parser.error("The file %s is not a valid image file!" % arg)


    # make predictions
    validation_img_paths = [arg]
    img_list = [convert_to_ela_image(img_path,90) for img_path in validation_img_paths]

    validation_batch = torch.stack([data_transforms['val'](img).to(device)
                                    for img in img_list])


    pred_logits_tensor = model(validation_batch)
    # print(pred_logits_tensor)

    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    # print(pred_probs)
 
    ##JSON
    jss={"FAKE":str(100*pred_probs[0,0]), 
          "REAL":str(100*pred_probs[0,1])
        }
    print(jss)