from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torch import nn

class CNN_Encoder(nn.Module):
    def __init__(self, class_n, rate=0.1):
        super(CNN_Encoder, self).__init__()  # 부모 클래스의 생성자 호출
       
        self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier[1].in_features  # 마지막 레이어의 입력 특성 수 가져오기

        self.model.classifier[1] = nn.Linear(num_ftrs, class_n)  # 새로운 선형 레이어로 교체
        print(self.model.classifier)

    def forward(self, x):
        output = self.model(x)  # 모델에 입력 전달
        return output
    


    

