from __future__ import division
from PHOG_V2 import PHOG_Algorithm
from LPQ import apply_LPQ
from segmentation import getFrame
import pickle
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import numpy as np
import skimage.io as io
import cv2
import os
import glob
