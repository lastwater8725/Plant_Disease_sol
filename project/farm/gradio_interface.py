import torch
import torch.nn as nn
from PIL import Image
import gradio as gr
import torchvision.transforms as transforms
from src.models import CNN_Encoder
from src.dataset import CustomDataModule

# 사람이 읽기 쉬운 형식으로 변환하는 딕셔너리
crop = {'1': '딸기', '2': '토마토', '3': '파프리카', '4': '오이', '5': '고추', '6': '시설포도'}
disease = {
    '1': {'a1': '딸기잿빛곰팡이병', 'a2': '딸기흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
    '2': {'a5': '토마토흰가루병', 'a6': '토마토잿빛곰팡이병', 'b2': '열과', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
    '3': {'a9': '파프리카흰가루병', 'a10': '파프리카잘록병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
    '4': {'a3': '오이노균병', 'a4': '오이흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
    '5': {'a7': '고추탄저병', 'a8': '고추흰가루병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
    '6': {'a11': '시설포도탄저병', 'a12': '시설포도노균병', 'b4': '일소피해', 'b5': '축과병'}
}
risk = {'1': '초기', '2': '중기', '3': '말기'}


data_module = CustomDataModule(batch_size=32)

# 학습된 모델 로드
class_n = len(data_module.label_encoder)
model = CNN_Encoder(class_n=class_n)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 이미지 전처리 함수
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 예측 함수 정의
def predict(img):
    img = preprocess(img)  # 이미지 전처리
    img = img.unsqueeze(0)  # 배치 차원 추가
    with torch.no_grad():
        output = model(img)
    _, predicted = torch.max(output, 1)
    
    # 클래스 레이블 정의
    labels = [f"Class {i}" for i in range(class_n)]  # 여기서 각 클래스에 대한 레이블을 정의합니다.
    
    return {labels[predicted.item()]: float(output[0][predicted].item())}



# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=predict,              # 예측 함수
    inputs=gr.Image(type='pil'),  # 입력: 이미지 (PIL 형식)
    outputs=gr.Label(num_top_classes=1)  # 출력: 레이블 (상위 1개의 클래스)
)

# 인터페이스 실행
iface.launch()



if __name__ == "__main__":
    iface.launch(share=True)

