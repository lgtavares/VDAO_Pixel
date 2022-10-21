import os

# DIRECTORIES
SOURCE_DIR  =  os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR =  os.path.abspath(os.path.join(SOURCE_DIR, os.pardir))
DATA_DIR    =  os.path.join(PROJECT_DIR, 'data')
MODEL_DIR   =  os.path.join(PROJECT_DIR, 'models')
RESULT_DIR  =  os.path.join(PROJECT_DIR, 'results')
SCRIPT_DIR  =  os.path.join(PROJECT_DIR, 'scripts')
EXTRA_DIR   =  os.path.join(PROJECT_DIR, 'extra')

# SHAPE
VDAO_FRAMES_SHAPE = [720, 1280, 3]



