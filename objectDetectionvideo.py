import cv2
import numpy as np
cam = cv2.VideoCapture(0)
# video dimensions

#length
# cam.set(3,740)
# width
# cam.set(4,580)


classNames=[]
classFile='coco.names'

with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')


configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath='frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320 , 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
# clssIds: the returned ids , confs: the confidence , bbox: the drawn box
while True:
    ret, img = cam.read()
   
    classIds,confs,bbox=net.detect(img,confThreshold=0.5)
    #confThreshold
    print(classIds,bbox)
    if len(classIds) !=0:
        for classid, confidence, box in zip(classIds.flatten() ,confs.flatten(), bbox):
            cv2.rectangle(img , box ,color=(0,255,0),thickness=1)
            cv2.putText(img , classNames[classid-1] , (box[0]+10 ,box[1]+20 ),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),thickness=1)

    cv2.imshow('frame',img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
    
   

cam.release()
cv2.destroyAllWindows()
