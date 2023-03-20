import sys
print(sys.executable)
print(sys.path)

%%capture
!pip install pytorch_lightning

import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import matplotlib as mpl

!python --version
torch.__version__,pl.__version__,np.__version__,pd.__version__,mpl.__version__

import platform
platform.platform()

import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.config.list_physical_devices())
print()
print(device_lib.list_local_devices())

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('{:.1f} gigabytes of available RAM\n'.format(ram_gb))

from google.colab import drive
drive.mount('/content/gdrive')

import matplotlib.pyplot as plt

# set test file path
bird_file_path = '/content/gdrive/MyDrive/Colab Notebooks/datasets/'
bird_file_path = bird_file_path +'bird.jpg'
img = plt.imread(bird_file_path)
# check image
plt.imshow(img)
