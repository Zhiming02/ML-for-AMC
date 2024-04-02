# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:13:17 2024

@author: www02
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
from torch.quantization import QuantStub, DeQuantStub
import numpy as np

image_folder = 'D:\RF_ML\data\dataset_3'
device = torch.device("cpu")
learning_rate = 1e-3
weight_decay = 0.001
beta1 = 0.9
beta2 = 0.999
CNN_checkpoint_path = 'D:\RF_ML\Final_model\dataset3_checkpoint.pth'
CNN_final_path = 'D:\RF_ML\Final_model\dataset3_final.pth'
CNN_Dynamic_quantize ='D:\RF_ML\Final_model\dataset3_Dq.pth'

class ConstellationDataset(Dataset):
    labels_map = {
        0: "BPSK",
        1: "4PSK",
        2: "8PSK",
        3: "16QAM",
        4: "64QAM",
        5: "256QAM",
        6: "16APSK",
        7: "32APSK",
        8: "64APSK",
        9: "4PAM",
        10:"8PAM"
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
        #filename format is "scheme_number.png"
        parts = img_name.split('_')
        modulation_scheme = parts[0]
        modulate_label = self.reverse_label_map[modulation_scheme]
        img_id = parts[1]
        if self.transform:
            image = self.transform(image)
        return image, modulate_label, img_id


train_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0212],std=[0.1052]) # img_norm_calculator
])
test_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0212],std=[0.1052])
])

train_dataset = ConstellationDataset(image_dir=image_folder, transform=train_transformations)
test_dataset = ConstellationDataset(image_dir=image_folder, transform=test_transformations)

train_size,val_size,test_size = int(0.8*len(train_dataset)),int(0.1*len(train_dataset)),int(0.1*len(train_dataset))
train_dataset,_,_ =torch.utils.data.random_split(train_dataset,[train_size,val_size,test_size], torch.Generator().manual_seed(2024))
_,val_dataset,test_dataset =torch.utils.data.random_split(test_dataset,[train_size,val_size,test_size], torch.Generator().manual_seed(2024))

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)    

def flip_example(train_dataset):
    img,_,_=train_dataset[143]    
    img_np = img.numpy() 
    plt.axis('off')
    plt.imshow(img_np.squeeze(0))
    plt.show()
    img_hflip = torch.flip(img, [2])
    plt.axis('off')
    plt.imshow(img_hflip.squeeze(0))
    plt.show()
    img_vflip = torch.flip(img, [1])
    plt.axis('off')
    plt.imshow(img_vflip.squeeze(0))
    plt.show()

# flip_example(train_dataset)

def img_norm_calculator(train_dataset):
    pixel_sum = torch.zeros(1,device='cuda')
    pixel_squared_sum = torch.zeros(1, device='cuda')

    # Iterate over the dataset and accumulate pixel values
    for (img,_,_) in tqdm((train_dataset),desc='Processing',total=len(train_dataset)):
    # for (img) in tqdm((training_data),desc='Processing',total=len(training_data)):
        img = img.to('cuda')
        pixel_sum += torch.sum(img)
        pixel_squared_sum += torch.sum(img ** 2)

    # Calculate the mean and standard deviation
    num_pixels = len(train_dataset) * img.shape[1] * img.shape[2]
    mean = pixel_sum / num_pixels
    std = torch.sqrt(pixel_squared_sum / num_pixels - mean ** 2)

    print("Mean:", mean)
    print("Standard Deviation:", std)

# img_norm_calculator(train_dataset)

class EarlyStopping:
    def __init__(self, path, patience=10, delta=0):
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
        5: "256QAM",
        6: "16APSK",
        7: "32APSK",
        8: "64APSK",
        9: "4PAM",
        10:"8PAM"
    }
    class_names = [labels_map[i] for i in range(len(labels_map))]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.xticks(rotation=45)  
    plt.yticks(rotation=0)   
    plt.show()

def plot_type_cm(cm):
    
    psk_indices = [0, 1, 2]
    qam_indices = [3, 4, 5]
    apsk_indices = [6,7,8]
    pam_indices = [9,10]
    
    type_cm = np.zeros((4, 4), dtype=int)
    type_cm[0, 0] = cm[psk_indices][:, psk_indices].sum()
    type_cm[0, 1] = cm[psk_indices][:, qam_indices].sum()
    type_cm[0, 2] = cm[psk_indices][:, apsk_indices].sum()
    type_cm[0, 3] = cm[psk_indices][:, pam_indices].sum()
    
    type_cm[1, 0] = cm[qam_indices][:, psk_indices].sum()
    type_cm[1, 1] = cm[qam_indices][:, qam_indices].sum()
    type_cm[1, 2] = cm[qam_indices][:, apsk_indices].sum()
    type_cm[1, 3] = cm[qam_indices][:, pam_indices].sum()
    
    type_cm[2, 0] = cm[apsk_indices][:, psk_indices].sum()
    type_cm[2, 1] = cm[apsk_indices][:, qam_indices].sum()
    type_cm[2, 2] = cm[apsk_indices][:, apsk_indices].sum()
    type_cm[2, 3] = cm[apsk_indices][:, pam_indices].sum()
    
    type_cm[3, 0] = cm[pam_indices][:, psk_indices].sum()
    type_cm[3, 1] = cm[pam_indices][:, qam_indices].sum()
    type_cm[3, 2] = cm[pam_indices][:, apsk_indices].sum()
    type_cm[3, 3] = cm[pam_indices][:, pam_indices].sum()
    
    
    plt.figure()
    type_map = {
        0: "PSK",
        1: "QAM",
        2: "APSK",
        3: "PAM"
    }
    
    type_accuracy = np.trace(type_cm) / np.sum(type_cm)
    
    type_names = [type_map[i] for i in range(len(type_map))]
    sns.heatmap(type_cm, annot=True, fmt="d", cmap="Blues", xticklabels=type_names, yticklabels=type_names)
    plt.tight_layout()
    plt.ylabel('Predicted type')
    plt.xlabel('True type')
    plt.show()

    print(f"Accuracy: {(100*type_accuracy):>0.1f}% \n")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,4,kernel_size=3,stride=2, padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(4),
            # nn.MaxPool2d(2,2),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
            
            nn.Conv2d(4,8,kernel_size=3,stride=2, padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
            
            nn.Conv2d(8,16,kernel_size=7,stride=2, padding=3,padding_mode='reflect'),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),

            nn.Flatten(), 
            nn.Linear(16*28*28, 256),
            nn.Linear(256, 11),
        )

    def forward(self, x):
        logits = self.cnn(x)
        return logits

model = CNN().to(device)
model.load_state_dict(torch.load(CNN_checkpoint_path))
model.eval()


loss_fn = nn.CrossEntropyLoss()
fp_test_loss, fp_correct = 0, 0
fp_time_start = time.time()
with torch.no_grad():
    for batch, (test_img, test_label,_) in tqdm(enumerate(test_dataloader), desc='Processing', total=len(test_dataloader)):
        test_img = test_img.to(device)
        test_label = test_label.to(device)
        test_pred = model(test_img)
        fp_test_loss += loss_fn(test_pred, test_label).item()
        fp_correct += (test_pred.argmax(1) == test_label).type(torch.float).sum().item()
        
fp_end_time = time.time()
fp_test_loss /= len(test_dataloader)
fp_correct /= test_size
fp_time = fp_end_time -fp_time_start
print("fp total time (seconds): {0:.1f}".format(fp_time))
print(f"Fp Accuracy: {(100*fp_correct):>0.1f}%, Avg loss: {fp_test_loss:.16f} \n")



quantized_model = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)
torch.save(quantized_model.state_dict(), CNN_Dynamic_quantize)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)

quantized_model.eval()
all_preds = []
all_labels = []
test_loss, correct = 0, 0
int_time_start = time.time()
with torch.no_grad():
    for batch, (test_img, test_label,_) in tqdm(enumerate(test_dataloader), desc='Processing', total=len(test_dataloader)):
        test_img = test_img.to(device)
        test_label = test_label.to(device)
        test_pred = quantized_model(test_img)
        all_preds.extend(test_pred.argmax(1).cpu().numpy())
        all_labels.extend(test_label.cpu().numpy())
        test_loss += loss_fn(test_pred, test_label).item()
        correct += (test_pred.argmax(1) == test_label).type(torch.float).sum().item()
int_end_time = time.time()
int_time =  int_end_time -int_time_start
test_loss /= len(test_dataloader)
correct /= test_size
print("int total time (seconds): {0:.1f}".format(int_time))
print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:.16f} \n")


cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm)
plot_type_cm(cm)
