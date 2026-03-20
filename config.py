# config.py
import os

class Config:
    DATA_DIR = "/home/redili/TR_project/code_2_4/data" 
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    VAL_CSV = os.path.join(DATA_DIR, "val.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test_external.xlsx")
    SAVE_DIR = "/home/redili/TR_project/code_2_4/code_2class/save_ECHO-RADIL"
    
    IMG_SIZE = 224
    NUM_FRAMES = 32  
    
    COL_FILENAME = 'filename'
    COL_PATH = 'full_path' 
    COL_GENDER = '性别'
    COL_AGE = '年龄'
    
    COL_FIRST_VAL = '三尖瓣反流严重程度'
    COL_LAST_VAL = '末次三尖瓣反流严重程度'
    
    COL_TIME_INTERVAL = '报告时间差值'
    COL_SEVERITY_DIFF = '严重程度差值'  
    COL_LABEL = 'Label'

    LABEL_MAPPING = {'improve + stable': 0,  'worsen': 1}
    NUM_CLASSES = 2
    
    AUX_NUM_CLASSES = 3
    NUM_SEVERITY_LEVELS = 3
    
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    
    LR_BACKBONE = 3e-6
    LR_HEAD = 1e-4     
    
    MAX_EPOCHS = 60
    
    CLASS_WEIGHTS = [1.0, 2.4]
    AUX_LOSS_WEIGHT = 0.516
    HIDDEN_DIM = 64

    LABEL_SMOOTHING = 0.10
    WEIGHT_DECAY = 0.05
    DROPOUT_RATE = 0.5