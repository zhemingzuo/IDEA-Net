"""
Manager of parameters
"""

# Number of cycles for a single image
DATASET_TO_SIZES = {'ideanet': 10}

# Image type supported
DATASET_TO_IMAGETYPE = {'ideanet': '.jpg'}

# Path to the input csv file
PATH_TO_CSV = {'ideanet': './input/train.csv'}

BATCH_SIZE = 1
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

POOL_SIZE = 50
ngf = 32
ndf = 64
