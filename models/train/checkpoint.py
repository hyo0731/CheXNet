import os
import torch
from torch.nn.modules import Module

def load_checkpoint(model: Module, ckpt_path: str, device: torch.device):
    """ 저장된 model 파일(checkpoint)에서 가중치 읽어와서 현재 model에 적용 """
    
    # 체크포인트 파일 존재하는지 확인
    if not os.path.isfile(ckpt_path):
        print(f"=> no checkpoint found at specified path: {ckpt_path}. Training will start from randomly initialized classifier.")
        return False

    print("=> loading checkpoint")

    # map_location을 사용하여 device(GPU/CPU)에 맞게 가중치 로드
    checkpoint = torch.load(ckpt_path, map_location=device) 
    state_dict = checkpoint['state_dict']

    # 'module.' prefix 제거 (DataParallel로 저장된 경우 처리)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    # 기존 Classifier 가중치 제거 (새로운 N_CLASSES에 맞추기 위함)
    keys_to_remove = [k for k in state_dict.keys() if k.startswith('densenet121.classifier.')]
    for k in keys_to_remove:
        del state_dict[k]
    print(f"Removed {len(keys_to_remove)} classifier keys for size mismatch.")
    
    # 4. 모델 key 이름 mapping 수정 (torchvision 버전/레이어 이름 변경 대응)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if 'features' in k:
            # DenseNet layer 이름 변경 대응 (예: .norm.1. -> .norm1.)
            new_k = new_k.replace('.norm.1.', '.norm1.').replace('.conv.1.', '.conv1.')
            new_k = new_k.replace('.norm.2.', '.norm2.').replace('.conv.2.', '.conv2.')
        new_state_dict[new_k] = v
    state_dict = new_state_dict

    # 5. 모델 가중치 로드
    try:
        # strict=False로 하여 classifier를 제외한 가중치만 로드 (사전 학습된 모델의 특징 추출기만 사용)
        model.load_state_dict(state_dict, strict=False) 
        print("-> loaded checkpoint (Feature Extractor Only) successfully.")
        return True
    except RuntimeError as e:
        print("CRITICAL ERROR during state_dict loading. Check key mapping.")
        print(e)
        return False