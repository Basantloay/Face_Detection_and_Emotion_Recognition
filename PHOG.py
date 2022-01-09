from commonfunctions import *
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def PHOG_Algorithm(image, numberOfBins, numberOfLevels):  # Reading Image as gray scale
    dummy_value = 1e-5
    degree = 360/numberOfBins
    x, y = image.shape
    pyramidarr = np.array([])
    # Intializing binary histogram array and binary gradient values array
    binaryHist = np.zeros((x, y))
    binaryGrad = np.zeros((x, y))

    if np.sum(image) > 100:
        medianImg = np.median(image)

        canny_image = cv2.Canny(image, int(
            max(0, (1.0 - 0.33) * medianImg)), int(min(255, (1.0 + 0.33) * medianImg)))
        comps, labels = cv2.connectedComponents(canny_image, connectivity=8)
        double_image = np.array(image, dtype=np.float64)
        # Gradient is defined as (change in y)/(change in x)
        [gx, gy] = np.gradient(double_image)
        # print(gy)
        # print(gx)
        values = np.sqrt(np.square(gy)+np.square(gx))
        # print(value)

        #i=np.array(gx == 0,dtype=np.int32)
        gx[gx == 0] = dummy_value
        #gy2 = np.gradient(gy)[1]

        # consider angle range is always 360 degrees
        aarray = np.divide((np.arctan2(gy, gx) + np.pi) * 180., np.pi)
        # print(labels)
        for k in range(comps):
            xcoordinate, ycoordinate = np.where(labels == k)
            # print(ycoordinate)
            # print(xcoordinate)
            for j in range(xcoordinate.shape[0]):
                ypoint = ycoordinate[j]
                xpoint = xcoordinate[j]

                z = np.ceil(aarray[xpoint, ypoint]/degree)
                if z == 0:
                    numberOfBins = 1
                if values[xpoint, ypoint] > 0:
                    binaryHist[xpoint, ypoint] = z
                    binaryGrad[xpoint, ypoint] = values[xpoint, ypoint]
        # Looping on each level in pyramid
        # histb=binaryHist[0:490,0:490]
        # gradb=binaryGrad[0:490,0:490]
        # print(len(binaryHist))
        # print(len(binaryGrad))
        histb = binaryHist
        gradb = binaryGrad
        for k in range(numberOfBins):
            ind = histb == k
            pyramidarr = np.append(pyramidarr, np.sum(gradb[ind]))

        # higher levels
        for level in range(1, numberOfLevels+1):
            y = int(np.trunc(histb.shape[0]/(2**level)))
            x = int(np.trunc(histb.shape[1]/(2**level)))
            for yy in range(0, histb.shape[0]-y+1, y):
                for xx in range(0, histb.shape[1]-x+1, x):
                    # print(pyramidarr)
                    binaryHist2 = histb[yy:yy+y, xx:xx+x]
                    binaryGrad2 = gradb[yy:yy+y, xx:xx+x]

                    for binofhist in range(numberOfBins):
                        ind = binaryHist2 == binofhist
                        pyramidarr = np.append(
                            pyramidarr, np.sum(binaryGrad2[ind], axis=0))

        if np.sum(pyramidarr) == 0:
            return pyramidarr
        else:
            return pyramidarr/np.sum(pyramidarr)
