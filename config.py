# config.py
import os


class Config:
    # ----- Data Paths (modify these for your environment) -----
    DATA_DIR = "./data"
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    VAL_CSV = os.path.join(DATA_DIR, "val.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test_external.xlsx")
    SAVE_DIR = "./save_ECHO-RADIL"

    # ----- Video Preprocessing -----
    IMG_SIZE = 224
    NUM_FRAMES = 32

    # ----- CSV Column Names -----
    COL_FILENAME = 'filename'
    COL_PATH = 'full_path'
    COL_GENDER = '性别'                    # Sex (女=female, 男=male)
    COL_AGE = '年龄'                       # Age (years)
    COL_FIRST_VAL = '三尖瓣反流严重程度'     # Baseline TR severity (1=mild, 2=moderate, 3=severe)
    COL_LAST_VAL = '末次三尖瓣反流严重程度'   # Follow-up TR severity
    COL_TIME_INTERVAL = '报告时间差值'       # Follow-up interval (days)
    COL_SEVERITY_DIFF = '严重程度差值'       # Severity difference (follow-up minus baseline)
    COL_LABEL = 'Label'

    # ----- Label Definition -----
    LABEL_MAPPING = {'improve + stable': 0, 'worsen': 1}
    NUM_CLASSES = 2

    # ----- Auxiliary Task -----
    AUX_NUM_CLASSES = 3       # Follow-up severity: mild / moderate / severe
    NUM_SEVERITY_LEVELS = 3   # Baseline severity levels for embedding

    # ----- Training Hyperparameters -----
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    LR_BACKBONE = 3e-6        # Fine-tuned backbone layers (13-16)
    LR_HEAD = 1e-4            # All other trainable modules
    MAX_EPOCHS = 60
    WEIGHT_DECAY = 0.05

    # ----- Loss Function -----
    CLASS_WEIGHTS = [1.0, 2.4]   # [stable/improved, worsen]
    AUX_LOSS_WEIGHT = 0.516      # L_total = L_main + 0.516 * L_aux
    LABEL_SMOOTHING = 0.10

    # ----- Model Architecture -----
    HIDDEN_DIM = 64
    DROPOUT_RATE = 0.5
