import cv2

img_path='catDog.jpg'
img=cv2.imread(img_path)

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
classIds,confs,bbox=net.detect(img,confThreshold=0.5)
#confThreshold
print(classIds,bbox)



   









cv2.imshow('Detector',img)

cv2.waitKey(0)