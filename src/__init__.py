import os

# DIRECTORIES
SOURCE_DIR = str(os.path.dirname(os.path.realpath(__file__)))
if 'nfs' in SOURCE_DIR:
    SOURCE_DIR = SOURCE_DIR.split('nfs')[1]
PROJECT_DIR = str(os.path.abspath(os.path.join(SOURCE_DIR, os.pardir)))
DATA_DIR = str(os.path.join(PROJECT_DIR, 'data'))
MODEL_DIR = str(os.path.join(PROJECT_DIR, 'models'))
RESULT_DIR = str(os.path.join(PROJECT_DIR, 'results'))
SCRIPT_DIR = str(os.path.join(PROJECT_DIR, 'scripts'))
EXTRA_DIR = str(os.path.join(PROJECT_DIR, 'extra'))

FEATURE_DIR = str(os.path.join(DATA_DIR, 'features'))

# SHAPE
VDAO_FRAMES_SHAPE = [720, 1280, 3]
