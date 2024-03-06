import os


N_EPISODES = 1
N_EVAL_EPISODES = 1
MAX_N_PARTS = 5

TRAIN = True
EXTENDED = True

# Set low and high values for the actions of the center coordination (X, Y, Z)
ACTION_SPACE_CENTER_COOR_LOW = 0  # unit: mm
ACTION_SPACE_CENTER_COOR_HIGH = 200  # unit: mm

# Set low and high values for the actions of the cutting plane angle (X, Y, Z)
ACTION_SPACE_CUT_PLANE_ANGLE_LOW = 0  # unit: degree
ACTION_SPACE_CUT_PLANE_ANGLE_HIGH = 180  # unit: degree

# Set low and high values for the observation space


# Set the initial model and import&export directory names
INPUT_MODEL = 'StanfordBunny.stl'
IMPORT_DIR = 'models'
EXPORT_DIR = 'results'

# Import and export directories for STL files
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
MESH_PATH = os.path.join(PARENT_DIR, IMPORT_DIR, INPUT_MODEL)
EXPORT_DIR = os.path.join(PARENT_DIR, EXPORT_DIR)


# 학습 로그 및 모델 저장 디렉토리
log_dir = "./logs"

# TensorBoard 실행:
# tensorboard --logdir="C:/tensorboard_logs/"
# http://localhost:6006/
