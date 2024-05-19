client = MongoClient("mongodb://~")
db = client.icecream_store_stock
collection = db.store_stock

classes={0:'Melona', 1:'BBBIG', 2:'PIGBAR', 3:'NUGABAR',4:'JAWSBAR', 5:'OKDONGJA'}
#SOURCE = '/content/test.mp4'
WEIGHTS = '/content/drive/MyDrive/best.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.2
CLASSES = None
AGNOSTIC_NMS = False

weights, imgsz = WEIGHTS, IMG_SIZE
n=0

# Initialize
device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA
print('device:', device)

# Load model
model1 = attempt_load(weights, map_location=device)  # load FP32 model
model1.to(torch.float32)  # 가중치 데이터 타입 변경

# Run inference
if device.type != 'cpu':
    model1(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model1.parameters())).to(torch.float32))  # run once

stride = int(model1.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
if half:
  model1.half()  # to FP16

# Get names and colors
names = model1.module.names if hasattr(model1, 'module') else model1.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


video_path = '/content/drive/MyDrive/KakaoTalk_20231201_014848382.mp4'
cap, frame_width, frame_height = get_video_info(video_path)
frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1)

temp=[]

model_name = 'yolo_nas_l'
model2 = load_model(model_name)
deepsort = initialize_deepsort()

names = cococlassNames()
colors = [[random.randint(0, 255) for _ in range(3)]
          for _ in range(len(names))]
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

classNames = cococlassNames()
for i in range(frame_count):
    xywh_bboxs = []
    confs = []
    oids = []
    outputs = []
    human_xyxy_id_lst=[]
    ret, frame = cap.read()
    if ret:
      result = list(model2.predict(frame, conf=0.5))[0]
      bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
      confidences = result.prediction.confidence
      labels = result.prediction.labels.tolist()
      for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
          bbox = np.array(bbox_xyxy)
          x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          conf = math.ceil((confidence*100))/100
          cx, cy = int((x1+x2)/2), int((y1+y2)/2)
          bbox_width = abs(x1-x2)
          bbox_height = abs(y1-y2)
          xcycwh = [cx, cy, bbox_width, bbox_height]
          xywh_bboxs.append(xcycwh)
          confs.append(conf)
          oids.append(int(cls))
      xywhs = torch.tensor(xywh_bboxs)
      if xywhs.numel() == 0:
        pass
      else:
        confss = torch.tensor(confs)
        outputs = deepsort.update(xywhs, confss, oids, frame)

      if len(outputs)>0:
          bbox_xyxy = outputs[:,:4]
          identities = outputs[:, -2]
          object_id = outputs[:, -1]
          for xyxy, identity in zip(bbox_xyxy, identities):
            xyxy=xyxy.tolist()
            human_xyxy_id_lst.append([xyxy, identity])
          print(human_xyxy_id_lst)

      img = letterbox(frame, imgsz, stride=stride)[0]
      img = img[:, :, ::-1].transpose(2, 0, 1)
      img = np.ascontiguousarray(img)
      img = torch.from_numpy(img).to(device)
      img = img.half() if half else img.float()  # uint8 to fp16/32
      img /= 255.0  # 0 - 255 to 0.0 - 1.0
      if img.ndimension() == 3:
          img = img.unsqueeze(0)

      t0 = time_synchronized()
      pred = model1(img, augment=AUGMENT)[0]

      pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
      det = pred[0]
      print(det)

      if len(human_xyxy_id_lst)>0:
        copy_det=det.tolist()
        tmp=[]
        db_data=[]
        for j in range(len(det)):
          t=[]
          st1=0
          st2=0

          max=0
          max_human_id=-1
          for q in human_xyxy_id_lst:
            if copy_det[j][0] < q[0][2] and copy_det[j][1] < q[0][3]:
              m=((q[0][2]-copy_det[j][0])*(q[0][3]-copy_det[j][1]))/((copy_det[j][2]-copy_det[j][0])*(copy_det[j][3]-copy_det[j][1]))
              if m>max:
                max=m
                max_human_id=q[1]

          if ((copy_det[j][1]+copy_det[j][3])/2)>=((img.shape[3]/6)-20) and ((copy_det[j][1]+copy_det[j][3])/2)<=((img.shape[3]/6)+20):
            check=[]
            for k in range(len(temp)):
              for q in range(len(temp[k])):
                if copy_det[j][5]==temp[k][q][0] and max_human_id==temp[k][q][4]:
                  if temp[k][q][3]==1 or temp[k][q][3]==2:
                    st2=1
                    break
                  elif copy_det[j][1]<temp[k][q][1] and copy_det[j][3]<temp[k][q][2]: check.append(-1)
                  elif copy_det[j][1]>temp[k][q][1] and copy_det[j][3]>temp[k][q][2]: check.append(1)
              if st2==1: break
            if st2!=1 and check.count(1)>=1 and check.count(1)>check.count(-1):
              st1=1
              db_data.append([int(copy_det[j][5]),1,max_human_id])
              print(f'\n{classes[copy_det[j][5]]} select +500won')
            elif st2!=1 and check.count(-1)>=1 and check.count(-1)>check.count(1):
              st1=2
              db_data.append([int(copy_det[j][5]),-1,max_human_id])
              print(f'\n{classes[copy_det[j][5]]} unselect -500won')
            t=[copy_det[j][5],copy_det[j][1],copy_det[j][3],st1,max_human_id]
            tmp.append(t)
        if n<5:
          temp.append(tmp)
          n+=1
        else:
          a=deepcopy(temp[3])
          b=deepcopy(temp[2])
          c=deepcopy(temp[1])
          temp[3]=deepcopy(temp[4])
          temp[2]=deepcopy(a)
          temp[1]=deepcopy(b)
          temp[0]=deepcopy(c)
          temp[4]=deepcopy(tmp)
    else:
        break
cap.release()