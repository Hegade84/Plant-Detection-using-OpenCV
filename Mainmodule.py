from Plant_detection import *

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
# cap.set(10,70)

while True:
    success, img = cap.read()
    result, objectInfo = getObjects(img, thres, nms, objects=['mouse'])
    print(objectInfo)
    cv2.imshow("Output", img)
    cv2.waitKey(1)