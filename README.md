# Capstone_1

### Colab Environment - detection + mongodb

```
!git clone https://github.com/WongKinYiu/yolov7.git
%cd yolov7
!pip install pymongo
!pip install -r requirements.txt
```


### detection + mongodb + deepsort
```
!git clone https://github.com/WongKinYiu/yolov7.git
!pip install pymongo
!pip install -r /content/yolov7/requirements.txt
%cd yolov7
```
```
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from pymongo import MongoClient
from copy import deepcopy
```
```
!pip install -q super-gradients==3.2.1
!pip install filterpy==1.1.0
```
```
from super_gradients.training import models
import math
from IPython.display import HTML
from base64 import b64encode
import os
```
```
!gdown "https://drive.google.com/uc?id=11ZSZcG-bcbueXZC3rN08CM0qqX3eiHxf&confirm=t"
!unzip /content/yolov7/deep_sort_pytorch.zip
```
```
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
```
