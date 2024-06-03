import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, CLAHE, RandomBrightnessContrast, ColorJitter, RGBShift, RandomSnow, RandomResizedCrop, ShiftScaleRotate, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90, Normalize
from albumentations.pytorch import ToTensorV2
import cv2
import json
import torch
from sklearn.model_selection import train_test_split
import glob
from src.utils import label_description



# 이미지 증강
def get_train_transforms(height: int, width: int):
    return Compose([
        CLAHE(p=0.3),
        RandomBrightnessContrast(p=0.3),
        RGBShift(p=0.2),
        RandomSnow(p=0.2),
        RandomResizedCrop(height=height, width=width, p=0.4),
        ShiftScaleRotate(scale_limit=0.1, rotate_limit=20, p=0.4),
        HorizontalFlip(p=0.2),
        VerticalFlip(p=0.2),
        Rotate(p=0.2),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])



class CustomDataset(Dataset):
    def __init__(self, files, labels=None, mode='train', label_encoder=None, transform=None):
        self.mode = mode
        self.files = files
        self.transform = transform
        self.labels = labels  # 레이블이 필요한 경우, 사전에 매핑된 레이블을 저장
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('/')[-1]

 #image
        image_path = f'{file}/{file_name}.jpg'
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255
        img = np.transpose(img,(2,0,1))

        if self.mode in ['train', 'val']:
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)

            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'

            return{
                'img' : torch.tensor(img, dtype = torch.float32),
                'label': torch.tensor(self.label_encoder[label], dtype=torch.long)
            }
        else:
            return{
                'img' : torch.tensor(img, dtype=torch.float32),

            }
        



class CustomDataModule:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.X_train, self.X_test, self.Y_train, self.Y_test= self.split_dataset()
        label_decoder, label_encoder,_ = label_description()
        self.label_decoder = label_decoder
        self.label_encoder = label_encoder
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.prepare_data()

    def split_dataset(self):
        train_files = sorted(glob.glob('data/train/train/*'))
        labels_df = pd.read_csv('data/train.csv')
        labels = labels_df['label'].values
        X_train, X_test, Y_train, Y_test= train_test_split(train_files, labels, test_size=0.2, stratify=labels, random_state=42)
        return X_train, X_test, Y_train, Y_test
    

    def prepare_data(self):
        val_transforms = Compose([
            Resize(height=256, width=256),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        train_transforms = get_train_transforms(height=256, width=256)

        test = sorted(glob.glob('data/test/test/*'))

        X_train_dataset = CustomDataset(self.X_train, mode='train', 
                                      label_encoder=self.label_encoder, transform=train_transforms)
        X_test_dataset = CustomDataset(self.X_test, mode='val', 
                                    label_encoder=self.label_encoder, transform=val_transforms)
        test_dataset = CustomDataset(test, mode='test', 
                                     label_encoder=self.label_encoder, transform=val_transforms)


        train_dataloader = DataLoader(X_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        val_dataloader = DataLoader(X_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)


        return train_dataloader, val_dataloader, test_dataloader

CustomDataModule()