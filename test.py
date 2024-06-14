import pandas as pd
import torch
from src.models import CNN_Encoder
from src.dataset import CustomDataModule
import argparse
from src.utils import label_description


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
parser.add_argument("--epochs", type=int, default=10, help="학습 에포크 수")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="학습률")
parser.add_argument("--save_path", default="model.pth", help="모델 저장 경로")

args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
model = CNN_Encoder(class_n=111)
model.load_state_dict(torch.load(args.save_path))
model.to(device)
model.eval()

# CustomDataModule 인스턴스 생성
data_module = CustomDataModule(batch_size=32)

# 각 데이터 로더에 접근
train_dataloader = data_module.train_dataloader
val_dataloader = data_module.val_dataloader
test_dataloader = data_module.test_dataloader

label_decoder, label_encoder, label_description_dict = label_description()


# Function to perform prediction
def predict(data_loader):
    model.eval()
    total_preds = []
    with torch.no_grad():
        for batch_item in data_loader:
            img = batch_item['img'].to(device)
            output = model(img)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            total_preds.extend(preds)
    return total_preds

# Predict on test data
test_loader = test_dataloader
predictions = predict(test_loader)

labels = [label_decoder[pred] for pred in predictions]  # 수정된 부분

# Load sample submission and fill it
submission = pd.read_csv('data/sample_submission.csv')
submission['label'] = labels
submission.to_csv('baseline_submission.csv', index=False)

print("Submission file has been created: baseline_submission.csv")
