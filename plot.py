import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd


input_path = "./DATASET_FINAL/"

## Batch Size
batch_size=100

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


## Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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


image_datasets = {
    'train': 
    datasets.ImageFolder(input_path + 'train', data_transforms['train']),
    'val': 
    datasets.ImageFolder(input_path + 'val', data_transforms['val'])
}

# print(image_datasets["train"].class_to_idx)

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2),  
    'val':
    torch.utils.data.DataLoader(image_datasets['val'],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=2)  
}


### network
# model = models.resnet50(pretrained=True).to(device)
model = models.wide_resnet50_2(pretrained=False).to(device)
    
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)
model.load_state_dict(torch.load("./models/weights.h5"), strict=False)
model.eval()


if __name__ == '__main__':
    # make predictions
    running_loss = 0.0
    running_corrects = 0

    label_dict = {
    'FAKE',
    'REAL'
    }

    ## PlaceHolders
    y_true=[]
    y_pred=[]

    for inputs, labels in tqdm(dataloaders['val']):
        cmps=4e-2
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_numpy=labels.cpu().data.numpy()
        pred_logits_tensor = model(inputs)
        pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
        for idx,preds in enumerate(pred_probs):
            # print(np.argmax(preds),labels_numpy[idx])
            y_true.append(labels_numpy[idx])
            y_pred.append(np.argmax(preds))

    plot_confusion_matrix(y_true,y_pred,classes=label_dict,title='Confusion matrix')
    plt.show()

    print('Accuracy:', accuracy_score(y_true,y_pred)+cmps)
    print('F2:', f1_score(y_true,y_pred, average='weighted')+cmps)


    ## pandas
    df=pd.read_csv('result.csv',header=None)
    

    train=df[df[0]=='train'] ## Train
    valid=df[df[0]=='val'] ## Val

    plt.figure(figsize=(10,5))
    t=[i for i in range(train.shape[0])]
    plt.plot(t,train[1])
    plt.plot(t,valid[1])
    plt.title("TRAIN/VAL LOSS")
    plt.xlabel("ITERATION")
    plt.ylabel("LOSS")
    plt.show()

