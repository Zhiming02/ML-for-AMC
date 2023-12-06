# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:19:21 2023

@author: wang  zhiming
"""

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

image_folder = 'D:\RF_ML\data\constellation_data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4
n_epochs = 20

weight_decay = 0.0001
beta1 = 0.9
beta2 = 0.999
CNN_checkpoint_path = 'D:\RF_ML\saved_model\CNN_checkpoint.pth'
CNN_final_path = 'D:\RF_ML\saved_model\CNN_final.pth'



class ConstellationDataset(Dataset):
    labels_map = {
        0: "BPSK",
        1: "4PSK",
        2: "8PSK",
        3: "16QAM",
        4: "64QAM",
        5: "256QAM"
    }

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.reverse_label_map = {v: k for k, v in self.labels_map.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        #filename format is "scheme_snr_number.png"
        parts = img_name.split('_')
        modulation_scheme = parts[0]
        modulate_label = self.reverse_label_map[modulation_scheme]
        snr = parts[1]
        if self.transform:
            image = self.transform(image)
        return image, modulate_label #snr


transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
whole_dataset = ConstellationDataset(image_dir=image_folder, 
                               transform=transformations)
train_size,val_size,test_size = int(0.8*len(whole_dataset)),int(0.1*len(whole_dataset)),int(0.1*len(whole_dataset))
train_dataset,val_dataset,test_dataset =torch.utils.data.random_split(whole_dataset,[train_size,val_size,test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)



class EarlyStopping:
    def __init__(self, path, patience=5, delta=0):
        super(EarlyStopping, self).__init__()
        self.path = path
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.flag_i = True

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.flag_i:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.flag_i = False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        print("Checkpoint saving...\n")


def plot_confusion_matrix(cm):
    plt.figure()
    labels_map = {
        0: "BPSK",
        1: "4PSK",
        2: "8PSK",
        3: "16QAM",
        4: "64QAM",
        5: "256QAM"
    } 
    class_names = [labels_map[i] for i in range(len(labels_map))]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,4,kernel_size=3,stride=1, padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(2,2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4,8,kernel_size=3,stride=1, padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            nn.Flatten(), 
            nn.Linear(8*56*56, 64), 
            nn.Linear(64, 6),
            # nn.Softmax(dim=1),
        )

    def forward(self, x):
        logits = self.cnn(x)
        return logits

model = CNN().to(device)
early_stopping = EarlyStopping(CNN_checkpoint_path)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), amsgrad=True)
loss_fn = nn.CrossEntropyLoss()
L_val = []
L_train = []
time_start = time.time()

for t in range(n_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    model.train()
    total_train_loss = 0.0
    for batch, (img, gt_label) in tqdm(enumerate(train_dataloader), desc='Processing', total=len(train_dataloader)):
        img = img.to(device)
        gt_label = gt_label.to(device)
        pred_label = model(img)
        loss = loss_fn(pred_label, gt_label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_train_loss += loss.item()
    total_train_loss = total_train_loss / len(train_dataloader)
    print(f" Train loss: {total_train_loss:>8f} \n")
    with torch.no_grad():
        model.eval()
        total_val_loss = 0.0
        for batch, (val_img, val_label) in tqdm(enumerate(val_dataloader), desc='Processing', total=len(val_dataloader)):
            val_img = val_img.to(device)
            val_label = val_label.to(device)
            pred_val_label = model(val_img)
            val_loss = loss_fn(pred_val_label, val_label)
            total_val_loss += val_loss.item()
        total_val_loss = total_val_loss / len(val_dataloader)
        print("[Epoch %d/%d] [validation loss: %f]" %(t+1, n_epochs, total_val_loss))
    L_val.append(total_val_loss)
    L_train.append(total_train_loss)
    early_stopping(total_val_loss, model)
time_end = time.time()
print(f"Train time is {time_end-time_start}")
torch.save(model.state_dict(), CNN_final_path)
print("Saving final model...\n")


def pltloss():
    plt.figure(1)
    plt.plot(L_val, color='red', label='val')
    plt.plot(L_train, color='blue', label='train')
    plt.legend(loc='best')  # This line will show the legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Trend')
    plt.show()
pltloss()





model.eval()
all_preds = []
all_labels = []
test_loss, correct = 0, 0
with torch.no_grad():
    for batch, (test_img, test_label) in tqdm(enumerate(test_dataloader), desc='Processing', total=len(test_dataloader)):
        test_img = test_img.to(device)
        test_label = test_label.to(device)
        test_pred = model(test_img)
        all_preds.extend(test_pred.argmax(1).cpu().numpy())
        all_labels.extend(test_label.cpu().numpy())
        test_loss += loss_fn(test_pred, test_label).item()
        correct += (test_pred.argmax(1) == test_label).type(torch.float).sum().item()
test_loss /= len(test_dataloader)
correct /= test_size
print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm)






#%% analysis
labels_map = {
    0: "BPSK",
    1: "4PSK",
    2: "8PSK",
    3: "16QAM",
    4: "64QAM",
    5: "256QAM"
}
mismatch = []
for i in range(len(all_preds)):
    if all_preds[i] != all_labels[i]:
        mismatch.append(i)
        
for i in mismatch:
    aly_img,_= test_dataset[i]
    pred_label = all_preds[i]
    gt= all_labels[i]
    aly_img=aly_img.cpu().numpy().squeeze(0)
    plt.imshow(aly_img, cmap='gray')
    plt.title(f"Predicted: {labels_map[pred_label]}, Ground Truth: {labels_map[gt]}")
    plt.axis('off')
    plt.show()

    
    
    

#%% test sth.


test_image_folder = 'D:\RF_ML\data\imgtestfile'
model.load_state_dict(torch.load(CNN_final_path))
class testdata(Dataset):
    labels_map = {
        0: "BPSK",
        1: "4PSK",
        2: "8PSK",
        3: "16QAM",
        4: "64QAM",
        5: "256QAM"
    }

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.reverse_label_map = {v: k for k, v in self.labels_map.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        #filename format is "scheme_snr_number.png"
        parts = img_name.split('_')
        modulation_scheme = parts[0]
        modulate_label = self.reverse_label_map[modulation_scheme]
        snr = parts[1]
        if self.transform:
            image = self.transform(image)
        return image, modulate_label #snr


transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
whole_dataset = testdata(image_dir=test_image_folder, transform=transformations)

test_dataloader = DataLoader(whole_dataset, batch_size=1, shuffle=False)


model.eval()
all_preds = []
all_labels = []
test_loss, correct = 0, 0
with torch.no_grad():
    for batch, (test_img, test_label) in tqdm(enumerate(test_dataloader), desc='Processing', total=len(test_dataloader)):
        test_img = test_img.to(device)
        test_label = test_label.to(device)
        test_pred = model(test_img)
        all_preds.extend(test_pred.argmax(1).cpu().numpy())
        all_labels.extend(test_label.cpu().numpy())
        test_loss += loss_fn(test_pred, test_label).item()
        correct += (test_pred.argmax(1) == test_label).type(torch.float).sum().item()
test_loss /= len(test_dataloader)
correct /= len(test_dataloader)
print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm)
