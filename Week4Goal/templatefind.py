import cv2
import multiprocessing as mp
import math
import numpy as np
import datamani
import drMatches
from drMatches import Position, getXY
import time
import polygons_overlapping
import sys

kp1, kp2, des1, des2 = [], [], [], []
MIN_MATCH_COUNT = 20
img_comp = None
FRAME = None
poly_template = []


def checkRect(array):
    x1 = array[0][0]
    y1 = array[0][1]
    x2 = array[1][0]
    y2 = array[1][1]
    x3 = array[2][0]
    y3 = array[2][1]
    x4 = array[3][0]
    y4 = array[3][1]

    cx=(x1+x2+x3+x4)/4
    cy=(y1+y2+y3+y4)/4

    dd1=math.sqrt(abs(cx-x1))+math.sqrt(abs(cy-y1))
    dd2=math.sqrt(abs(cx-x2))+math.sqrt(abs(cy-y2))
    dd3=math.sqrt(abs(cx-x3))+math.sqrt(abs(cy-y3))
    dd4=math.sqrt(abs(cx-x4))+math.sqrt(abs(cy-y4))
    a = abs(dd1-dd2)/((dd1+dd2)/2)
    b = abs(dd1-dd3)/((dd1+dd3)/2)
    c = abs(dd1-dd4)/((dd1+dd4)/2)

    if a > 0.2 or b>0.2 or c>0.2:
        return False
    else:
        return True

def t_Start(img, frame):
    global img_comp, FRAME, poly_template
    poly_template = []
    img_comp = img.copy()
    FRAME = frame.copy()
    s=np.zeros((4,4))
    pos = Position(0,0,0,0)
    flag = True
    first_run_flag = True
    cmatch = 0
    template_flag = False
    while True:
        cmatch +=1
        good_matches= t_Match()
        break_flag= t_Borders(good_matches)
        if break_flag or cmatch>20:
            return poly_template


def t_Match():
    global kp1, kp2, img_comp, FRAME, poly_template
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_comp,None)
    # print imgt
    kp2, des2 = sift.detectAndCompute(FRAME,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    return good


def t_Borders(good):
    global img_comp, FRAME, kp1, kp2, poly_template
    break_flag = False

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        if mask == None:
            return True
        h,w = img_comp.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        #Extract from dst#
        poly_currentarr = []
        for i in range(len(dst)):
            sub_dst = dst[i]
            sub2_dst = sub_dst[0]
            x1 = sub2_dst[0]
            y1 = sub2_dst[1]
            arr = [x1, y1]
            poly_currentarr.append(arr)
        poly_currentarr.append(poly_currentarr[0])

        current_rec = checkRect(poly_currentarr) ### CHECK IF THIS IS A RECTANGLE
        if current_rec:    
            xb = poly_currentarr[0][0]
            yb = poly_currentarr[0][1]
            for i in range(len(poly_currentarr)):
                arr = [poly_currentarr[i][0]-xb, poly_currentarr[i][1]-yb]
                poly_template.append(arr)
            return True
        FRAME = cv2.fillPoly(FRAME,[np.int32(dst)],(0,0,0))
    else:
        break_flag = True
    return break_flag