import pandas as pd
import glob
import json
import cv2

data=pd.read_csv('/fast/zcb/data/pleatau/data/obj_sample/pairs/label/sample_total.csv')
path=data['id'].values
idx=data['type'].values

path=path[idx==8]
img_path=['/fast/zcb/data/pleatau/data/obj_sample/pairs/mask/'+i+'.png' for i in path]
label_path=['/fast/zcb/data/pleatau/data/obj_sample/pairs/out/'+i+'.png' for i in path]

res=[]
for i in range(len(img_path)):
    res.append({'source':img_path[i],'target':label_path[i],'prompt':'digital surface model, depth map'})

with open('/fast/zcb/data/pleatau/data/obj_sample/pairs/prompt_type8_train.json','w') as f:
    json.dump(res[:int(len(res)*0.8)],f)

with open('/fast/zcb/data/pleatau/data/obj_sample/pairs/prompt_type8_test.json','w') as f:
    json.dump(res[int(len(res)*0.8):],f)