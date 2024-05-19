from algorithm.object_detector import YOLOv7
from utils.detections import draw
from tqdm import tqdm
import cv2
from copy import deepcopy
from pymongo import MongoClient

yolov7 = YOLOv7()
yolov7.load('best.pt', classes='coco.yaml', device='cpu')
classes={0:'Melona', 1:'BBBIG', 2:'PIGBAR', 3:'NUGABAR',4:'JAWSBAR', 5:'OKDONGJA'}
classes_values=list(classes.values())

video = cv2.VideoCapture(1)
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if video.isOpened() == False:
	print('[!] error opening the video')

print('[+] tracking video...\n')
pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)
i=0
temp=[]

client = MongoClient("mongodb+srv://OS:MZWl4yS6ylx53ouQ@os.xcm3kqz.mongodb.net/")
db = client.icecream_store_stock
collection = db.store_stock

try:
    while video.isOpened():
      ret, img0 = video.read()
      if ret == True:
        detections = yolov7.detect(img0, track=True)

        det=[]
        for box in detections:
          class_name = box['class']
          conf = box['confidence']
          width = box['width']
          height = box['height']
          top_left_point = {'x':box['x'], 'y':box['y']}
          bottom_right_point = {'x':box['x'] + width, 'y':box['y'] + height}

          det.append([box['x'],box['y'],box['x']+box['width'],box['y']+box['height'],box['confidence'],box['class'],box['id']])
        copy_det=deepcopy(det)
        tmp=[]
        db_data=[]
        for j in range(len(det)):
          for g in range(len(classes_values)):
            if copy_det[j][5]==classes_values[g]:
              copy_det[j][5]=g
              break
          t=[]
          st1=0
          st2=0
          if ((copy_det[j][1]+copy_det[j][3])/2)>=((img0.shape[1]/6)-40) and ((copy_det[j][1]+copy_det[j][3])/2)<=((img0.shape[1]/6)+40):
            print(f'{i}frame:',(copy_det[j]))
            check=[]
            compare2=copy_det[j]
            for k in range(len(temp)):
              compare1=temp[len(temp)-1-k]
              s=-1
              for q in range(len(compare1)):
                if compare2[5]==compare1[q][0] and compare2[6]==compare1[q][4]:
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
                cc=[0,compare1[s][1],0,compare1[s][2],0,compare1[s][0],compare1[s][4]]
                compare2=cc

            print(check)
            if st2!=1 and check.count(1)>=1 and check.count(1)>check.count(-1):
              st1=1
              db_data.append([int(copy_det[j][5]),1])
              print(f'\n{classes[copy_det[j][5]]} select +500won')
            elif st2!=1 and check.count(-1)>=1 and check.count(-1)>check.count(1):
              st1=2
              db_data.append([int(copy_det[j][5]),-1])
              print(f'\n{classes[copy_det[j][5]]} unselect -500won')

          t=[copy_det[j][5],copy_det[j][1],copy_det[j][3],st1,copy_det[j][6]]
          tmp.append(t)
        if i<5:
          temp.append(tmp)
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
        counts = {key_list[x]: value_list[x] for x in range(len(key_list))}
        for k in db_data:
          if k[1]<0:
            if counts[classes[k[0]]]>0: counts[classes[k[0]]]+=k[1]
          else: counts[classes[k[0]]]+=k[1]

        if len(db_data)>0: print('----------------------')
        for item, count in counts.items():
          collection.update_one({'item': item}, {'$set': {'count': count}}, upsert=True)
          if len(db_data)>0: print(f'{i}frame: Now {item} stock is {count}.')

        pbar.update(1)
        i+=1
        cv2.imshow('Detected', img0)
        if cv2.waitKey(1) == ord('q'):
          break
except KeyboardInterrupt:
    pass

pbar.close()
video.release()
yolov7.unload()