from torchvision.models import resnet101, ResNet101_Weights
from torch import nn

class CNN_Encoder(nn.Module):
    def __init__(self, class_n, rate=0.1):
        super(CNN_Encoder, self).__init__()  # 부모 클래스의 생성자 호출
        # ResNet101 모델 불러오기 (이미지넷 가중치 포함)
        self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        # 원래의 분류기를 교체 (1000개 클래스에서 class_n개 클래스로)
        num_ftrs = self.model.fc.in_features  # fc (Fully Connected) 레이어의 입력 특성 수 가져오기
        self.model.fc = nn.Linear(num_ftrs, class_n)  # 새로운 선형 레이어로 교체
        # 드롭아웃 추가
        self.dropout = nn.Dropout(rate)

    def forward(self, inputs):
        x = self.dropout(inputs)  # 입력에 드롭아웃 적용
        output = self.model(x)  # 모델에 입력 전달
        return output
