import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import fbeta_score
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

input_path = "./DATASET_FINAL/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Batch Size
batch_size=24

## Generators
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(60, resample=False, expand=False, center=None, fill=None),
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



# ### RESNET
# model = models.wide_resnet50_2(pretrained=True).to(device)
model = models.wide_resnet50_2(pretrained=True).to(device)
# model=MyCustomResnet18(nn.Module)
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())


### Train
def train_model(model, criterion, optimizer, num_epochs=3):
    torch.multiprocessing.freeze_support()
    f=open('result.csv', mode='w')
    writer=csv.writer(f, delimiter=',')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)

                ## F2 Score
                logits=preds.cpu().detach().numpy()
                label=labels.cpu().detach().numpy()
                y_true=label
                y_pred=logits
                F2_score=fbeta_score(y_true, y_pred, average='macro', beta=2.0)

                # AUC
                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
                auc=metrics.auc(fpr, tpr)


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])


            if phase=='train':
                print('{} loss: {:.4f}, acc: {:.4f}, f2_score: {:.4f}, auc_score: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc,F2_score,auc))
                torch.save(model.state_dict(), './models/weights.h5')

            else:
                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))

            writer.writerow([phase, epoch_loss, epoch_acc,F2_score,auc])
    f.close()
    return model

if __name__ == '__main__':
    ## Training
    model_trained = train_model(model, criterion, optimizer, num_epochs=500)

    ### Save Model
    torch.save(model_trained.state_dict(), './models/weights.h5')

