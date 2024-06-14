import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, RandomBrightnessContrast, RandomResizedCrop, ShiftScaleRotate, HorizontalFlip, Rotate, Normalize
from albumentations.pytorch import ToTensorV2
import cv2
import json
import torch
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from collections import Counter
from src.utils import label_description

# 폰트 경로 설정
font_path = "fonts/NanumGothic.ttf"  # 프로젝트 폴더 내에 저장된 폰트 파일 경로
fontprop = font_manager.FontProperties(fname=font_path)
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fontprop.get_name()

# 이미지 증강
def get_train_transforms(height: int, width: int):
    return Compose([
        RandomBrightnessContrast(p=0.3),
        RandomResizedCrop(height=height, width=width, p=0.4),
        ShiftScaleRotate(scale_limit=0.1, rotate_limit=20, p=0.4),
        HorizontalFlip(p=0.2),
        Rotate(p=0.2),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(height: int, width: int):
    return Compose([
        Resize(height=height, width=width),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def decode_label(label, label_description_dict):
    return label_description_dict.get(label, 'Unknown')

# 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, files, labels=None, mode='train', label_encoder=None, transform=None):
        self.mode = mode
        self.files = files
        self.transform = transform
        self.labels = list(labels) if labels is not None else None
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('/')[-1]
        image_path = f'{file}/{file_name}.jpg'
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        if self.transform:
            img = self.transform(image=img)['image']

        if self.mode in ['train', 'val']:
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'
            return {'img': img, 'label': self.label_encoder[label]}
        else:
            return {'img': img}

# 데이터 모듈
class CustomDataModule:
    def __init__(self, batch_size=32, target_count=200):
        self.batch_size = batch_size
        self.target_count = target_count
        self.label_decoder, self.label_encoder, self.label_description_dict = label_description()
        self.prepare_data()

    def prepare_data(self):
        train_files = sorted(glob.glob('data/train/train/*'))
        test_files = sorted(glob.glob('data/test/test/*'))
        labels_df = pd.read_csv('data/train.csv')
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(train_files, labels_df['label'].values, test_size=0.2, stratify=labels_df['label'].values, random_state=42)
        train_transforms = get_train_transforms(256, 256)
        val_transforms = get_val_transforms(256, 256)

        self.train_dataset = CustomDataset(self.X_train, self.Y_train, 'train', self.label_encoder, train_transforms)
        self.val_dataset = CustomDataset(self.X_test, self.Y_test, 'val', self.label_encoder, val_transforms)
        self.test_dataset = CustomDataset(test_files, mode='test', label_encoder=self.label_encoder, transform=val_transforms)

        self.augment_dataset(self.train_dataset)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

    def augment_dataset(self, dataset):
        exclude_labels = ['1_00_0', '4_00_0', '5_00_0', '6_00_0']
        label_counts = Counter(dataset.labels)
        target_count = label_counts['1_00_0']  # 딸기_정상 이미지 개수를 기준으로 설정

        for label, count in label_counts.items():
            if label not in exclude_labels and count < target_count:
                diff = target_count - count
                indices = [i for i, lbl in enumerate(dataset.labels) if lbl == label]
                for _ in range(diff):
                    idx = np.random.choice(indices)
                    dataset.files.append(dataset.files[idx])
                    dataset.labels.append(dataset.labels[idx])

    def plot_class_distribution(self, dataset, top_n=25):
        label_counts = Counter(dataset.labels)
        most_common_labels = label_counts.most_common(top_n)
        labels, counts = zip(*most_common_labels)
        decoded_labels = [decode_label(label, self.label_description_dict) for label in labels]

        plt.figure(figsize=(12, 8))
        plt.bar(decoded_labels, counts)
        plt.xticks(rotation=90, fontproperties=fontprop)
        plt.xlabel('클래스', fontproperties=fontprop)
        plt.ylabel('샘플 수', fontproperties=fontprop)
        plt.title('상위 25개 클래스 분포', fontproperties=fontprop)
        plt.tight_layout()
        plt.show()



data_module = CustomDataModule(batch_size=32, target_count=200)
data_module.plot_class_distribution(data_module.train_dataset)