import torch.nn as nn
import torchvision

class DenseNet121(nn.Module):
    """ multi-label classification을 위해 모델 수정 """
    
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()

        # ImageNet의 Pretrained weights 사용
        self.densenet121 = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = self.densenet121.classifier.in_features

        # Classifier : Multilabel 분류를 위해 Sigmoid 활성화 함수 사용
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x