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

client = MongoClient("mongodb+srv://OS:MZWl4yS6ylx53ouQ@os.xcm3kqz.mongodb.net/");
db = client.ice_creams
collection = db.ice_creams

classes={0:'Melona', 1:'BBBIG', 2:'PIGBAR', 3:'NUGABAR',4:'JAWSBAR', 5:'OKDONGJA'}
#SOURCE = '/content/test.mp4'
WEIGHTS = '/content/drive/MyDrive/best.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.15
IOU_THRES = 0.25
CLASSES = None
AGNOSTIC_NMS = False

def detect(v):
    source, weights, imgsz = v, WEIGHTS, IMG_SIZE
    n=0

    # Initialize
    device = select_device(DEVICE)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print('device:', device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    video = cv2.VideoCapture(source)
    frame_count=int(video.get(cv2.CAP_PROP_FRAME_COUNT)-1)

    temp=[]
    for i in range(frame_count):
      # Load image
      _, img0 = video.read()
      assert img0 is not None, 'Image Not Found ' + source

      # Padded resize
      img = letterbox(img0, imgsz, stride=stride)[0]

      # Convert
      img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
      #print(img.shape, img0.shape)
      img = np.ascontiguousarray(img)

      img = torch.from_numpy(img).to(device)
      img = img.half() if half else img.float()  # uint8 to fp16/32
      img /= 255.0  # 0 - 255 to 0.0 - 1.0
      if img.ndimension() == 3:
          img = img.unsqueeze(0)

      # Inference
      t0 = time_synchronized()
      pred = model(img, augment=AUGMENT)[0]
      #print('pred shape:', pred.shape)

      # Apply NMS
      pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

      # Process detections
      det = pred[0]
      if i%50==0: print(f'{i}th processing')
      copy_det=det.tolist()
      tmp=[]
      db_data=[]
      for j in range(len(det)):
        t=[]
        st1=0
        st2=0
        if ((copy_det[j][1]+copy_det[j][3])/2)>=((img.shape[3]/6)-40) and ((copy_det[j][1]+copy_det[j][3])/2)<=((img.shape[3]/6)+40):
            print(f'{i}frame ',copy_det[j])
            check=[]
            compare2=copy_det[j]
            for k in range(len(temp)):
              compare1=temp[len(temp)-1-k]
              s=-1
              for q in range(len(compare1)):
                if compare2[5]==compare1[q][0]:
                  if compare2[1]<compare1[q][1] and compare2[3]<compare1[q][2] and (compare1[q][1]-compare2[1])<80 and (compare1[q][2]-compare2[3])<80:
                    if compare1[q][3]!=2:
                      check.append(-1)
                      s=q
                    else:
                      st2=1
                      break
                  elif compare2[1]>compare1[q][1] and compare2[3]>compare1[q][2] and (compare2[1]-compare1[q][1])<80 and (compare2[3]-compare1[q][2])<80:
                    if compare1[q][3]!=1:
                      check.append(1)
                      s=q
                    else:
                      st2=1
                      break
                  elif (compare2[1]>compare1[q][1] and compare2[3]<compare1[q][2]):
                    rr1=abs(compare2[1]-compare1[q][1])
                    rr2=abs(compare2[3]-compare1[q][3])
                    if rr1>rr2 and rr1<80 and rr2<80:
                      if compare1[q][3]!=1:
                        check.append(1)
                        s=q
                      else:
                        st2=1
                        break
                    elif rr2>rr1 and rr1<80 and rr2<80:
                      if compare1[q][3]!=2:
                        check.append(-1)
                        s=q
                      else:
                        st2=1
                        break
                  elif (compare2[1]<compare1[q][1] and compare2[3]>compare1[q][2]):
                    rr1=abs(compare2[1]-compare1[q][1])
                    rr2=abs(compare2[3]-compare1[q][3])
                    if rr1>rr2 and rr1<80 and rr2<80:
                      if compare1[q][3]!=2:
                        check.append(-1)
                        s=q
                      else:
                        st2=1
                        break
                    elif rr2>rr1 and rr1<80 and rr2<80:
                      if compare1[q][3]!=1:
                        check.append(1)
                        s=q
                      else:
                        st2=1
                        break
              if st2==1: break
              if s!=-1:
                cc=[0,compare1[s][1],0,compare1[s][2],0,compare1[s][0]]
                compare2=cc

            if st2!=1 and check.count(1)>=1 and check.count(1)>check.count(-1):
              st1=1
              db_data.append([int(copy_det[j][5]),1])
              print(f'\n{classes[copy_det[j][5]]} select +500won')
            elif st2!=1 and check.count(-1)>=1 and check.count(-1)>check.count(1):
              st1=2
              db_data.append([int(copy_det[j][5]),-1])
              print(f'\n{classes[copy_det[j][5]]} unselect -500won')

        t=[copy_det[j][5],copy_det[j][1],copy_det[j][3],st1]
        tmp.append(t)
      if i<5: temp.append(tmp)
      else:
        a=deepcopy(temp[3])
        b=deepcopy(temp[2])
        c=deepcopy(temp[1])
        temp[3]=deepcopy(temp[4])
        temp[2]=deepcopy(a)
        temp[1]=deepcopy(b)
        temp[0]=deepcopy(c)
        temp[4]=deepcopy(tmp)

      raw_values=[]
      db_read=list(collection.find({},{'_id':False}))
      for k in db_read:
         raw_values.append(list(k.values()))

      key_list = [x[0] for x in raw_values]
      value_list = [x[1] for x in raw_values]
      counts = {key_list[x]: value_list[x] for x in range(len(key_list)-1)}
      for k in db_data:
         if k[1]<0:
          if counts[classes[k[0]]]>0: counts[classes[k[0]]]+=k[1]
          else: counts[classes[k[0]]]+=k[1]

      if len(db_data)>0: print('----------------------')
      for item, count in counts.items():
        collection.update_one({'item': item}, {'$set': {'count': count}}, upsert=True)
        if len(db_data)>0: print(f'{i}frame: Now {item} stock is {count}.')
        #1.5초 정도 쉬는거 넣어야할듯