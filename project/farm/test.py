import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from src.models import CNN_Encoder
from src.dataset import CustomDataModule

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="예측에 사용되는 장치")
parser.add_argument("--model_path", default="fold3_model.pth", help="학습된 모델 경로")
args = parser.parse_args()

# Set device
device = torch.device(args.device)

# Initialize data module
data_module = CustomDataModule(batch_size=32)
test_dataloader = data_module.prepare_test_data()  # 테스트 데이터 로드

# Initialize model
model = CNN_Encoder(class_n=len(data_module.label_encoder))
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# Accuracy function
def accuracy_function(real, pred):
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score

# Test loop
total_test_loss = 0
total_test_acc = 0
criterion = torch.nn.CrossEntropyLoss()

tqdm_dataset = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
for batch, batch_item in tqdm_dataset:
    img = batch_item['img'].to(device)
    label = batch_item['label'].to(device)

    with torch.no_grad():
        output = model(img)
        loss = criterion(output, label)
    
    total_test_loss += loss.item()
    total_test_acc += accuracy_function(label, output)
    
    tqdm_dataset.set_postfix({
        'Loss': '{:.6f}'.format(loss.item()),
        'Mean Loss': '{:.6f}'.format(total_test_loss / (batch + 1)),
        'Mean F-1': '{:.6f}'.format(total_test_acc / (batch + 1))
    })

mean_test_loss = total_test_loss / len(test_dataloader)
mean_test_f1 = total_test_acc / len(test_dataloader)

print(f"Test Loss: {mean_test_loss:.6f}")
print(f"Test F1 Score: {mean_test_f1:.6f}")
