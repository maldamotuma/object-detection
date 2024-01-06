import cv2
import matplotlib.pyplot as plt

config_file = 'data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'data/frozen_inference_graph.pb'

model = cv2.dnn.DetectionModel(frozen_model, config_file)

classLabels = []
file_name='data/labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)
# size of new frame
model.setInputSize(320, 320)
# scale factor value for the frame
model.setInputScale(1.0/127.5)
# set mean value for the frame
model.setInputMean((127.5, 127, 5, 127.5))
# set SwapRB flag for every frame
model.setInputSwapRB(True)

img = cv2.imread('data/boy.jpg')
plt.imshow(img)
plt.show()

classIndex, connfidence, bbox = model.detect(img, confThreshold=0.5)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for classInd, conf,  boxes in zip(classIndex.flatten(), connfidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255,0,0), 2)
    cv2.putText(img, classLabels[classInd-1], (boxes[0]+10, boxes[1]+40), font, font_scale, color=(0,255,0), thickness=3)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

cap = cv2.VideoCapture('data/video.mp4')
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('CAN\'T OPEN THE VIDEO')

while True:
    ret, frame = cap.read()
    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    print(classIndex)
    if(len(classIndex) != 0):
        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if(classInd <= 80):
                 cv2.rectangle(frame, boxes, (255,0,0), 2)
                 cv2.putText(frame, classLabels[classInd-1], (boxes[0]+10, boxes[1]+40), font, font_scale, color=(0,255,0), thickness=3)
    cv2.imshow("object detection by malda", frame)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()