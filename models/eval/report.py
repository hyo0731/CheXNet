import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from eval.metrics import roc_curve, auc
from utils.constants import N_CLASSES, CLASS_NAMES, THRESHOLD

def print_results(aurocs: List[float], metrics_results: Dict[int, Any]):
    """ AUROC 계산 """
    
    AUROC_avg = np.array(aurocs).mean()
    
    # --- AUROC 결과 출력 ---
    print('---' * 15)
    print('Final Evaluation Results')
    print('---' * 15)
    print('## 🥇 AUROC Results')
    print('The average AUROC is **{AUROC_avg:.3f}**'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of **{}** is {:.3f}'.format(CLASS_NAMES[i], aurocs[i]))
    
    # --- Classification Metrics 출력 ---
    print('\n' + '---' * 15)
    print(f'## Classification Metrics (Threshold: {THRESHOLD})')
    for i in range(N_CLASSES):
        name = CLASS_NAMES[i]
        res = metrics_results[i]
        print(f"\n### {name}")
        print(f"  - **F1 Score**: {res['F1 Score']:.3f}")
        print(f"  - True Positives (TP): {res['TP']}")
        print(f"  - True Negatives (TN): {res['TN']}")
        print(f"  - False Positives (FP): {res['FP']}")
        print(f"  - False Negatives (FN): {res['FN']}")
    print('---' * 15)

def plot_roc_curves(gt: torch.Tensor, pred: torch.Tensor, file_name: str = 'roc_curves.png'):
    """ ROC curve plot 하고 PNG file로 저장"""
    
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    
    for i in range(N_CLASSES):

        # ROC curve 계산
        fpr, tpr, _ = roc_curve(gt_np[:, i], pred_np[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(file_name)
    print(f"\n[INFO] ROC curves saved to {file_name}")
    plt.close()