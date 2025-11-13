import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np

# --- module import ------------------------------------------------------

from utils.constants import (
    CKPT_PATH, TRAIN_IMAGE_LIST_CSV, TEST_IMAGE_LIST_CSV, DATA_DIR, 
    N_CLASSES, CLASS_NAMES, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    THRESHOLD 
)
from models.train import trainer 
from utils.device import get_device
from models.model import DenseNet121
from data.dataset import CSVDataset
from data.transforms import get_train_transforms, get_test_transforms
from models.eval.metrics import compute_AUCs, compute_metrics
from models.eval.report import plot_roc_curves, print_results 
from models.train.checkpoint import load_checkpoint

# -----------------------------------------------------------------------


def main(device):
    """ 최종 결과를 출력하는 함수 (전체 워크플로우 제어) """    
    if DEVICE.type == 'cuda':
        cudnn.benchmark = True

    model = DenseNet121(N_CLASSES).to(DEVICE)

    # --- 1. 체크포인트 로딩 ---
    load_checkpoint(model, CKPT_PATH, DEVICE)
    # -----------------------

    # 2. train dataset load 및 훈련 실행
    print("\n--- 1. Training Phase ---")
    try:
        train_dataset = CSVDataset(data_dir=DATA_DIR,
                                   image_list_file_csv=TRAIN_IMAGE_LIST_CSV,
                                   class_names=CLASS_NAMES,
                                   transform=get_train_transforms())
    except (FileNotFoundError, KeyError) as e:
        print(f"Error initializing TRAIN dataset: {e}")
        return

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False)

    # train 함수 호출
    model = trainer.train(model, train_loader, DEVICE, num_epochs=NUM_EPOCHS)


    # 3. test dataset load 및 평가
    print("\n--- 2. Evaluation Phase ---")
    try:
        test_dataset = CSVDataset(data_dir=DATA_DIR,
                                  image_list_file_csv=TEST_IMAGE_LIST_CSV,
                                  class_names=CLASS_NAMES,
                                  transform=get_test_transforms())
    except (FileNotFoundError, KeyError) as e:
        print(f"Error initializing TEST dataset: {e}")
        return

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False)

    # 정답이랑 예측값 모두 초기화
    gt = torch.FloatTensor().to(DEVICE)
    pred = torch.FloatTensor().to(DEVICE)

    # evaluate mode로 전환
    model.eval()

    # --- 평가 루프 ---
    with torch.no_grad():
        for i, (inp, target) in enumerate(test_loader):
            target = target.to(DEVICE)
            inp = inp.to(DEVICE)

            gt = torch.cat((gt, target), 0)

            # TenCrop 변환 처리
            bs, n_crops, c, h, w = inp.size()
            input_var = inp.view(-1, c, h, w)

            output = model(input_var)

            # 10개 crop 결과 평균
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean), 0)

    # --- 지표 계산 및 플로팅 ---
    gt_cpu = gt.cpu()
    pred_cpu = pred.cpu()
    
    # 1. AUROC 계산
    AUROCs = compute_AUCs(gt_cpu, pred_cpu)
    
    # 2. F1 score 및 TP/TN 계산
    metrics_results = compute_metrics(gt_cpu, pred_cpu, THRESHOLD)
    
    # 3. ROC curve plot
    plot_roc_curves(gt_cpu, pred_cpu)

    # 4. 결과 출력
    print_results(AUROCs, metrics_results)



if __name__ == '__main__':

    # CUDA/MPS/CPU
    DEVICE = get_device()
    print(f"Using device: {DEVICE}") 
    main(DEVICE)