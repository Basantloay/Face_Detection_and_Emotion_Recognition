import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2


def apply_LPQ(frame: np.ndarray, window_size: int) -> list:
    # 1- get the window slices
    if(not (window_size % 2)):
        print("Error Window Size can't be Even")
        exit(1)

    a = window_size // 2    # radius of the window
    LPQ_hist = []
    # For all Pixels
    for row in range(a, frame.shape[0]-a):
        for col in range(a, frame.shape[1]-a):
            # 1- get the Window centered at the pixel
            window = frame[row-a:row+a+1, col-a:col+a+1]

            # 2- Find the STFT of this window
            img_fft = np.fft.fft2(window)
            img_fft = np.fft.fftshift(img_fft)

            # 3- Extract the 4 Frequencies of Interest
            freqs = img_fft[a-1:a+2, a:a+2]
            f1 = img_fft[a-1, a]
            f2 = img_fft[a, 0]
            f3 = img_fft[a, a-1]
            f4 = img_fft[a-2, a-1]

            # 4- Build the V Vector -> Concatination of Frequencies
            V = [f1, f2, f3, f4]

            # 5- Build the W Vector concatination of imaginary and real parts
            W = np.concatenate((np.imag(V), np.real(V)))

            # 6- get the decimal Representatoin of the binary mapping
            b = (W > 0).astype('uint8')
            b = int(("".join(map(str, b))), 2)
            LPQ_hist.append(b)

    return LPQ_hist
