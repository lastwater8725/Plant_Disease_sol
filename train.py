import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.models import CNN2RNN
from src.dataset import CustomDataModule

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
parser.add_argument("--epochs", type=int, default=30, help="학습 에포크 수")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="학습률")
parser.add_argument("--save_path", default="model.pth", help="모델 저장 경로")
args = parser.parse_args()

# Set device
device = torch.device(args.device)

# Initialize data module
data_module = CustomDataModule(batch_size=32)

# Initialize model
model = CNN2RNN(max_len=24*6, embedding_dim=100, num_features=len(data_module.csv_feature_dict), class_n=len(data_module.label_encoder), rate=0.1)
model.to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

# Accuracy function
def accuracy_function(real, pred):
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score

# Training step
def train_step(batch_item, training):
    img = batch_item['img'].to(device)
    csv_feature = batch_item['csv_feature'].to(device)
    label = batch_item['label'].to(device)
    
    if training:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img, csv_feature)
            loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            output = model(img, csv_feature)
            loss = criterion(output, label)
    
    score = accuracy_function(label, output)
    return loss, score

# Training and Validation loop
loss_plot, val_loss_plot = [], []
metric_plot, val_metric_plot = [], []

for epoch in range(args.epochs):
    total_loss, total_val_loss = 0, 0
    total_acc, total_val_acc = 0, 0
    
    # Training loop
    tqdm_dataset = tqdm(enumerate(data_module.train_dataloader), total=len(data_module.train_dataloader))
    for batch, batch_item in tqdm_dataset:
        batch_loss, batch_acc = train_step(batch_item, training=True)
        total_loss += batch_loss
        total_acc += batch_acc
        
        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Loss': '{:.6f}'.format(batch_loss.item()),
            'Mean Loss': '{:.6f}'.format(total_loss / (batch + 1)),
            'Mean F-1': '{:.6f}'.format(total_acc / (batch + 1))
        })
    loss_plot.append(total_loss / (batch + 1))
    metric_plot.append(total_acc / (batch + 1))
    
    # Validation loop
    tqdm_dataset = tqdm(enumerate(data_module.val_dataloader), total=len(data_module.val_dataloader))
    for batch, batch_item in tqdm_dataset:
        batch_loss, batch_acc = train_step(batch_item, training=False)
        total_val_loss += batch_loss
        total_val_acc += batch_acc
        
        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Val Loss': '{:.6f}'.format(batch_loss.item()),
            'Mean Val Loss': '{:.6f}'.format(total_val_loss / (batch + 1)),
            'Mean Val F-1': '{:.6f}'.format(total_val_acc / (batch + 1))
        })
    val_loss_plot.append(total_val_loss / (batch + 1))
    val_metric_plot.append(total_val_acc / (batch + 1))
    
    if np.max(val_metric_plot) == val_metric_plot[-1]:
        torch.save(model.state_dict(), args.save_path)

import matplotlib.pyplot as plt

# Save and plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(loss_plot, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig("train_loss_plot.png")

# Save and plot training and validation F1 score
plt.figure(figsize=(10, 5))
plt.plot(metric_plot, label='Train F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Training F1 Score')
plt.legend()
plt.savefig("train_f1_score_plot.png")