import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image

from utils.constants import CLASS_NAMES

class CSVDataset(Dataset):
    """ CSV file로부터 label 정보 가져오기 """
    def __init__(self, data_dir, image_list_file_csv, class_names, transform=None):
        self.data_dir = data_dir
        self.class_names = class_names
        self.transform = transform
        
        if not os.path.exists(image_list_file_csv):
             raise FileNotFoundError(f"CSV file not found at: {image_list_file_csv}")
             
        # 첫 줄을 헤더로 인식 (데이터셋의 첫행은 데이터가 아닌 class name 이어야함!)
        self.image_list = pd.read_csv(image_list_file_csv).fillna(0)
        
        # 이미지 파일명을 가진 컬럼 이름 (CSV 첫 번째 컬럼 이름)
        IMAGE_ID_COLUMN_NAME = self.image_list.columns[0] # 'Image_Index'

        # 레이블은 CLASS_NAMES 문자열 이름 목록을 사용하여 접근
        self.image_paths = [os.path.join(self.data_dir, row[IMAGE_ID_COLUMN_NAME]) 
                            for index, row in self.image_list.iterrows()]
        
        # 레이블을 CLASS_NAMES 문자열 이름 목록을 사용하여 추출
        self.labels = self.image_list[self.class_names].values.astype(np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        if not os.path.exists(image_path):
             raise FileNotFoundError(f"Image file not found at: {image_path}. Check DATA_DIR setting.")

        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        label_tensor = torch.from_numpy(label)
        
        if self.transform:
            image = self.transform(image)

        return image, label_tensor