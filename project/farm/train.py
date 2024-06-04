import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from src.models import CNN_Encoder
from src.dataset import CustomDataModule
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
parser.add_argument("--epochs", type=int, default=10, help="학습 에포크 수")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="학습률")
parser.add_argument("--save_path", default="model.pth", help="모델 저장 경로")
args = parser.parse_args()

# Set device
device = torch.device(args.device)

# Initialize data module
data_module = CustomDataModule(batch_size=32)

# Initialize model
model = CNN_Encoder(class_n=len(data_module.label_encoder))
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

# Define the train_step function
def train_step(model, batch_item, optimizer, criterion, device, training=True):
    img = batch_item['img'].to(device)
    label = batch_item['label'].to(device)

    if training:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img)
            loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            output = model(img)
            loss = criterion(output, label)

    score = accuracy_function(label, output)
    return loss, score

# Initialize lists to store fold results
all_fold_train_loss = []
all_fold_val_loss = []
all_fold_train_f1 = []
all_fold_val_f1 = []

# Training and Validation loop with 5-fold cross validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(data_module.train, data_module.labels_train)):
    train_dataloader, val_dataloader, _ = data_module.prepare_data()

    fold_train_loss = []
    fold_val_loss = []
    fold_train_f1 = []
    fold_val_f1 = []

    for epoch in range(args.epochs):
        total_loss, total_val_loss = 0, 0
        total_acc, total_val_acc = 0, 0
        
        # Training loop
        tqdm_dataset = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(model, batch_item, optimizer, criterion, device, training=True)
            total_loss += batch_loss
            total_acc += batch_acc
            
            tqdm_dataset.set_postfix({
                'Fold': fold + 1,
                'Epoch': epoch + 1,
                'Loss': '{:.6f}'.format(batch_loss.item()),
                'Mean Loss': '{:.6f}'.format(total_loss / (batch + 1)),
                'Mean F-1': '{:.6f}'.format(total_acc / (batch + 1))
            })
        fold_train_loss.append(total_loss / (batch + 1))
        fold_train_f1.append(total_acc / (batch + 1))
        
        # Validation loop
        tqdm_dataset = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(model, batch_item, optimizer, criterion, device, training=False)
            total_val_loss += batch_loss
            total_val_acc += batch_acc
            
            tqdm_dataset.set_postfix({
                'Fold': fold + 1,
                'Epoch': epoch + 1,
                'Val Loss': '{:.6f}'.format(batch_loss.item()),
                'Mean Val Loss': '{:.6f}'.format(total_val_loss / (batch + 1)),
                'Mean Val F-1': '{:.6f}'.format(total_val_acc / (batch + 1))
            })
        fold_val_loss.append(total_val_loss / (batch + 1))
        fold_val_f1.append(total_val_acc / (batch + 1))
        
        if np.max(fold_val_f1) == fold_val_f1[-1]:
            torch.save(model.state_dict(), f"fold{fold+1}_{args.save_path}")

    # Save fold results
    all_fold_train_loss.append(fold_train_loss)
    all_fold_val_loss.append(fold_val_loss)
    all_fold_train_f1.append(fold_train_f1)
    all_fold_val_f1.append(fold_val_f1)

    # Save and plot training and validation loss for each fold
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 5))
    plt.plot([loss.item() for loss in fold_train_loss], label='Train Loss')
    plt.plot([loss.item() for loss in fold_val_loss], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Fold {fold+1})')
    plt.legend()
    plt.savefig(f"fold{fold+1}_train_val_loss_plot.png")

    # Save and plot training and validation F1 score for each fold
    plt.figure(figsize=(10, 5))
    plt.plot(fold_train_f1, label='Train F1 Score')
    plt.plot(fold_val_f1, label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'Training and Validation F1 Score (Fold {fold+1})')
    plt.legend()
    plt.savefig(f"fold{fold+1}_train_val_f1_score_plot.png")

# Calculate and print overall results
mean_train_loss = np.mean([np.mean([loss.item() for loss in fold]) for fold in all_fold_train_loss])
std_train_loss = np.std([np.mean([loss.item() for loss in fold]) for fold in all_fold_train_loss])
mean_val_loss = np.mean([np.mean([loss.item() for loss in fold]) for fold in all_fold_val_loss])
std_val_loss = np.std([np.mean([loss.item() for loss in fold]) for fold in all_fold_val_loss])
mean_train_f1 = np.mean([np.mean([f1.item() for f1 in fold]) for fold in all_fold_train_f1])
std_train_f1 = np.std([np.mean([f1.item() for f1 in fold]) for fold in all_fold_train_f1])
mean_val_f1 = np.mean([np.mean([f1.item() for f1 in fold]) for fold in all_fold_val_f1])
std_val_f1 = np.std([np.mean([f1.item() for f1 in fold]) for fold in all_fold_val_f1])

print(f"Overall Train Loss: {mean_train_loss:.6f} ± {std_train_loss:.6f}")
print(f"Overall Val Loss: {mean_val_loss:.6f} ± {std_val_loss:.6f}")
print(f"Overall Train F1 Score: {mean_train_f1:.6f} ± {std_train_f1:.6f}")
print(f"Overall Val F1 Score: {mean_val_f1:.6f} ± {std_val_f1:.6f}")
