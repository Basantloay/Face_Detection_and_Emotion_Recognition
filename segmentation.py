import cv2
import math
import numpy as np


def getHist(img):
    Histogram = np.zeros(256)
    imgCopy = np.copy(img)
    for i in range(imgCopy.shape[0]):
        for j in range(imgCopy.shape[1]):
            Histogram[(imgCopy[i][j]).astype(int)
                      ] = Histogram[(imgCopy[i][j]).astype(int)]+1
    return Histogram


def getThreshold(img):
    histogram = getHist(img)
    levelsCount = len(histogram)
    Tinit = 0
    for k in range(levelsCount):
        segma = k * histogram[k]
        Tinit += segma
    pCount = (img.shape[0]*img.shape[1])
    Tinit = np.around(Tinit/pCount)
    Tcurr = int(Tinit)
    Told = Tcurr+1
    while (abs(Told-Tcurr) > 0.1):
        lower = histogram[0:int(Tcurr)]
        upper = histogram[int(Tcurr):]
        cLower = np.sum(lower)
        cUpper = np.sum(upper)
        sLower = 0
        sUpper = 0
        for i in range(len(lower)):
            sLower += i*lower[i]
        for j in range(len(upper)):
            sUpper += (j+len(lower))*upper[j]
        avgLower = sLower/cLower
        avgUpper = sUpper/cUpper
        Told = Tcurr
        Tcurr = (avgLower+avgUpper)/2
    return Told


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
            cv2.imwrite("frame%d.jpg" % count, frame)
            img = cv2.imread("frame%d.jpg" % count)  # RGB
            #img = rgb2gray(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (gray * 255).astype(np.uint8)
            thr = getThreshold(img)
            #img_thr = img>thr
            img_thr = img.copy()
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j] > thr:
                        img_thr[i][j] = img[i][j]
                    else:
                        img_thr[i][j] = 0

            cv2.imwrite("after_segmentation%d.jpg" % count, img_thr)
            # show_images((img,img_thr),("img","threshold"))
            print(count)
    cap.release()
    print("DONE")
