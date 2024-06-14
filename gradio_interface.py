import torch
import gradio as gr
import torchvision.transforms as transforms
from src.models import CNN_Encoder
from src.dataset import CustomDataModule
from src.utils import label_description



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


label_decoder, label_encoder, label_description_dict = label_description()

# 예측 함수 정의
def predict(img):
    img = preprocess(img)  # 이미지 전처리
    img = img.unsqueeze(0)  # 배치 차원 추가
    with torch.no_grad():
        output = model(img)
    print(f'output: {output}')
    _, predicted = torch.max(output, 1)
    predicted_value =predicted.item() 
    
    # 클래스 레이블 정의
    label = label_decoder[predicted_value]
    state = label_description_dict[label]
    return state



# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=predict,              # 예측 함수
    inputs=gr.Image(type='pil'),  # 입력: 이미지 (PIL 형식)
    outputs=gr.Label(num_top_classes=1)  # 출력: 레이블 (상위 1개의 클래스)
)

# # 인터페이스 실행
# iface.launch()



if __name__ == "__main__":
    iface.launch(share=True)

