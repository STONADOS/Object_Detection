print("[INFO]   Initializing ObjectDetector....")

filee = str(input("What is it Image or Video: "))
filetype = filee.lower() 

from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--prototxt",default="MobileNetSSD_deploy.prototxt.txt")
ap.add_argument("--model",default="MobileNetSSD_deploy.caffemodel")
ap.add_argument("--confidence", type = float,default=0.2)
agrs = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "table",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
    
COLOR =np.random.uniform(0,195,(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe(agrs["prototxt"],agrs["model"])

if filetype == "image":

    col = input("Please provide the File path: ")
    try:
        image = cv2.imread(col)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
    
            confidence = detections[0, 0, i, 2]

            if confidence > agrs["confidence"]:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    COLOR[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR[idx], 2)
        cool = cv2.resize(image,(1250, 750))
        cv2.imshow("Image Analytics", cool)
        cv2.waitKey(0)
    except:
        print("[INFO]  Something got wrong !!")

if filetype == "video":
    what1 = input("Do want to access 'camera' or 'file': ")
    what = what1.lower()
    if what == "file":
        col = input("Please provide the File path: ")
        print('Starting object detection')
        vs = FileVideoStream(col).start()
    if what == "camera":
        vs = VideoStream(0).start()
        print('Starting object detection')
    try:
        fps = FPS().start()


        while True:
            frame = vs.read()
            frame =cv2.resize(frame, (1300, 750))

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0,0,i,2]

                if confidence > agrs["confidence"]:
                    idx = int(detections[0,0,i,1])
                    box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                    (startX , startY , endX, endY)  = box.astype("int")

                    label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
                    cv2.rectangle(frame, (startX,startY), (endX,endY),COLOR[idx],2)
                    y= startY - 15 if startY - 15>15 else startY+15 
                    cv2.putText(frame,label, (startX,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,COLOR[idx],2)

            cv2.imshow("Video Analytics",frame)
            key = cv2.waitKey(1)  & 0xff

            if key == ord('q'):
                break

            fps.update()

        fps.stop()

        print("Elapsed time: {:2f}".format(fps.elapsed()))
        print("Approx FPS: {:2f}".format(fps.fps()))

        cv2.destroyAllWindows()
        vs.stop()
    except:
        print("[INFO]  Something got wrong !!")
