import cv2
import math
import numpy as np



def getFrame(vid_path):
    # Divide the video into frames, 1 frame each second.
    #videoFile = "Video.MOV"
    cap = cv2.VideoCapture(vid_path)
    frameRate = cap.get(5)  # frame rate
    x = 1
    count = 0
    while(cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0 and frameId != 0):
            count = count+1
            # cv2.imwrite("frame%d.jpg" % count, frame)
            # img = cv2.imread("frame%d.jpg" % count)  # RGB
            #img = rgb2gray(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (gray * 255).astype(np.uint8)
            
    cap.release()
    print("DONE")
