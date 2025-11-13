import os

# -------------------------------------------------------------------------------------------
"""경로 설정: 각자의 폴더 경로로 반드시 수정!"""

# 데이터 root directory 경로
DATA_ROOT_DIR = './ChestX-ray14' 

# 모델 가중치 경로
CKPT_PATH = 'model.pth.tar' 

# CSV 파일 경로
TRAIN_IMAGE_LIST_CSV = os.path.join(DATA_ROOT_DIR, 'labels/train_list.csv')
TEST_IMAGE_LIST_CSV = os.path.join(DATA_ROOT_DIR, 'labels/test_list.csv')

# image 데이터 폴더 경로
DATA_DIR = os.path.join(DATA_ROOT_DIR, 'images')

# -------------------------------------------------------------------------------------------

# 우리가 분류하는 5개의 질병 클래스
N_CLASSES = 5
CLASS_NAMES = ['Edema', 'Effusion', 'Mass', 'Nodule', 'Pneumothorax']

# 학습 하이퍼파라미터 (자유롭게 수정 가능!)
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
THRESHOLD = 0.5