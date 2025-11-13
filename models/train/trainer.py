import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules import Module
import torch.optim
from utils.constants import (N_CLASSES, LEARNING_RATE, NUM_EPOCHS)


def train(model: Module, train_loader: DataLoader, device: torch.device, num_epochs: int = NUM_EPOCHS, lr: float = LEARNING_RATE) -> Module:
    """모델 훈련을 수행하고 최종 가중치를 저장"""
    
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n=> Starting training for {num_epochs} epochs on {device}...")
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

    print("Training complete.")
    
    # 모델 저장
    save_path = 'trained_model_final.pth.tar' 

    # 모델을 state_dict로 저장
    torch.save({'state_dict': model.state_dict()}, save_path)
    print(f"Trained model saved to {save_path}")

    return model