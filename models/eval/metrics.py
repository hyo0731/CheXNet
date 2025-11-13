import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, auc

from utils.constants import N_CLASSES, CLASS_NAMES

def compute_AUCs(gt: torch.Tensor, pred: torch.Tensor):
    """Computes Area Under the Curve (AUC) from prediction scores."""
    
    AUROCs = []

    # GPU 텐서가 입력될 경우를 대비해 CPU로 이동 및 Numpy 변환
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        
    return AUROCs

def compute_metrics(gt: torch.Tensor, pred: torch.Tensor, threshold: float):
    """Computes F1 score, TP, TN, FP, FN based on a threshold."""
    
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    # 예측 확률을 binary label 로 변환 (Threshold 기준)
    pred_binary = (pred_np >= threshold).astype(int)
    
    results = {}
    
    for i in range(N_CLASSES):

        # Confusion Matrix 계산 (TN, FP, FN, TP 순서로 반환)
        cm = confusion_matrix(gt_np[:, i], pred_binary[:, i], labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        
        f1 = f1_score(gt_np[:, i], pred_binary[:, i])
        
        results[i] = {
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'F1 Score': f1
        }
    return results