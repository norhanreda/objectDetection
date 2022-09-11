import cv2

img_path='catDog.jpg'
img=cv2.imread(img_path)

classNames=[]
classFile='coco.names'

with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')


configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath='frozen_inference_graph.pb'












cv2.imshow('Detector',img)

cv2.waitKey(0)