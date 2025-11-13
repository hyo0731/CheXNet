import torchvision.transforms as transforms
import torch

def get_train_transforms():
    # 정규화 (평균, 표준편차 이용)
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    # 데이터 crop & augmentation
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

def get_test_transforms():
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    # TenCrop을 사용하여 test 시 앙상블 수행
    return transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224), 
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])