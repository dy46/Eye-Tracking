import cv2
import multiprocessing as mp
import math
import numpy as np
import datamani
import drMatches
from drMatches import Position, getXY
import time
import polygons_overlapping
import FrameBFS
import sys
import templatefind

'''Creating instance variables'''
MIN_MATCH_COUNT = 20
font = cv2.FONT_HERSHEY_SIMPLEX
s, pos = None, None
img1, img2, img3, imgt = None, None, None, None
kp1, kp2, des1, des2 = [], [], [], []
first_run_flag = True
framecount, videoData, idx, tail, fps = None, None, None, None, None
flag = True
poly_arr = []
poly_template = []
object_number = None
first = 0
cmatch = 0
template_flag = False


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

def startProcess(img, currentFrame):
    global img1, img2, img3, img4, imgt, s, pos, first_run_flag, poly_arr, poly_template, first, flag, template_flag, polygon_looked
    global idx, tail, framecount, cmatch

    # Initiate SIFT detector
    # find the keypoints and descriptors with SIFT
    s=np.zeros((4,4))
    #print s.dtype
    pos = Position(0,0,0,0)
    flag = True

    first_run_flag = True
    poly_arrays = []
    for i, img1 in enumerate(img):
        if first_run_flag == False:
            img2 = img3
        poly_arr, poly_template= [], []
        imgt = img2.copy()
        first = 0
        cmatch = 0
        template_flag = False
        # print 'New image'
        # print poly_template
        while True:
            cmatch +=1
            good_matches= featureMatch(currentFrame)
            matchesMask, ignore, dst, break_flag = drawBorders(good_matches, currentFrame)
            if break_flag or cmatch>20:
                # print "break"
                break
            x, y = getXY(img2, framecount, videoData, idx, tail, fps, ignore)
            placeText(ignore, i, dst, x, y)
            first += 1
        poly_arrays.append(poly_arr)
    x, y = drawCircleAndMatches(ignore, good_matches)
    # print x, y, framecount
    # print flag
    if not flag:
        order = FrameBFS.determineOrder(poly_arrays[object_number - 1])
        poly_number = 10
        for i in range(len(order)):
            if order[i] in polygon_looked:
                poly_number = i + 1
                break
    if flag:
        cv2.putText(img3,'Gazing at none of the objects',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)
    else:
        cv2.putText(img3,'Gazing at the '+str(poly_number)+' of the '+str(object_number)+' object',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow("hi", img3)
    cv2.waitKey(10)

def featureMatch(currentFrame):
    global kp1, kp2
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    # print imgt
    kp2, des2 = sift.detectAndCompute(imgt,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # if currentFrame == 35:
    #     print 'this is des1: '+ str(des1)
    #     print 'this is des2: '+str(des2)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    # if currentFrame == 14:
    #     print good
    return good

def drawBorders(good, currentFrame):
    global img2, imgt, template_flag, poly_template, poly_arr
    ignore = False
    break_flag = False

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # for x in dst_pts:
        #     cv2.circle(imgs,(int(x[0][0]),int(x[0][1])),2,(255,0,0),2)
        # if currentFrame == 14:
        #     print "I have more good than min match count"
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        if mask == None:
            break_flag = True
            return None, None, None, break_flag
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # print 'pts'
        # print pts
        # print 'dst'
        dst = cv2.perspectiveTransform(pts,M)
        if currentFrame in [557, 558, 559]:
            print dst
            print 'This is dst '+str(dst.size)
        # print dst
        if len(dst) == 0:
            print "SO i am working to break"
            break_flag = True
            return None, None, None, break_flag

        if dst.size ==0:
            print "Size is the best method"
            break_flag = True
            return None, None, None, break_flag
        poly_currentarr = []
        for i in range(len(dst)):
            sub_dst = dst[i]
            sub2_dst = sub_dst[0]
            x1 = sub2_dst[0]
            y1 = sub2_dst[1]
            arr = [x1, y1]
            poly_currentarr.append(arr)
        poly_currentarr.append(poly_currentarr[0])
        # print poly_currentarr
        # print poly_current
        # print 'The current frame is a rectangle: '+str(checkRect(poly_currentarr))
        current_rec = checkRect(poly_currentarr)
        # print template_flag
        if not template_flag:
            poly_template = templatefind.t_Start(img1, img2)
            template_flag = True
        ######## Checking to see if the current mask is a rectangle
        poly_current = np.asarray(poly_currentarr)
        # if currentFrame == 40:
        #     print 'Is template a rectangle: '
        #     print poly_template
        if len(poly_arr)>0:
            # if currentFrame==40:
            #     print "I am close"
            for p in poly_arr:
                if polygons_overlapping.pair_overlapping(p, poly_current) ==2 or not current_rec:
                    xnot=dst[0][0][0]
                    ynot = dst[0][0][1]
                    t2_a = []
                    for i in range(len(poly_template)):
                        # print "Im here"
                        t2_b = []
                        t_a =np.int32([poly_template[i][0]+xnot, poly_template[i][1]+ynot])
                        t2_b.append(t_a)
                        t2_a.append(t2_b)
                    t3_a = np.int32(t2_a)
                    dst = t3_a
        if len(poly_arr)==0 and not current_rec:
            xnot=dst[0][0][0]
            ynot = dst[0][0][1]
            t2_a = []
            for i in range(len(poly_template)):
                t2_b = []
                t_a =np.int32([poly_template[i][0]+xnot, poly_template[i][1]+ynot])
                t2_b.append(t_a)
                t2_a.append(t2_b)
            t3_a = np.int32(t2_a)
            dst = t3_a
        poly_arr.append(poly_current)

        if currentFrame in [557, 558, 559]:
            if currentFrame == 558:
                print 'FUCK i suck'
            print 'dst'
            print dst
            print 'int32'
            print [np.int32(dst)]
            # cv2.imshow('img2', img2)
            # cv2.waitKey(0)
            # cv2.imshow('imgt', imgt)
            # cv2.waitKey(0)
        try: 
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            imgt = cv2.fillPoly(imgt,[np.int32(dst)],(0,0,0))
        except:
            print 'EXCEPT BREAK'
            break_flag = True
            return None, None, None, break_flag
    else:
        # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
        ignore = True
        dst = None
        break_flag = True
    # if currentFrame == 40:
    #     cv2.imshow('img',imgt)
    #     cv2.waitKey(0)
    # cv2.imshow('img2', img2)
    # cv2.waitKey(5)
    return matchesMask, ignore, dst, break_flag

def drawCircleAndMatches(ignore, good):
    global img2, img3, pos, idx, tail, framecount, first_run_flag
    x, y = 0, 0
    if first_run_flag == True:
        img2, x, y, idx, tail ,ignore= datamani.drawCircle(img2, framecount, videoData, idx, tail, fps, ignore)
        framecount = framecount + (1.0/fps)*1000.0
    img3, pos = drMatches.drawMatches(img1,kp1,img2,kp2,good, pos)
    # img3, pos = drMatches.drawMatches(img1,kp1,img2,kp2,good, pos) ## line must not execute
    first_run_flag = False
    return x, y

def placeText(ignore, i, dst, x, y):
    global s, flag, img3, object_number, polygon_looked
    if not ignore:
        s[i][0]=dst[0][0][0];
        s[i][1]=dst[3][0][0];
        s[i][2]=dst[0][0][1];
        s[i][3]=dst[2][0][1];
        if x>s[i][0] and x<s[i][1] and y>s[i][2] and y<s[i][3]:
            polygon_looked = dst
            object_number = i + 1
            flag = False

def processImage(f, currentFrame, index, frames_per_second, FRAMECOUNT, IDX, TAIL, img, data):
    global fps, framecount, idx, tail, img2, videoData
    fps = frames_per_second
    framecount = currentFrame * 1000.0/fps
    idx = IDX[index]
    tail = TAIL[index]
    img2 = f
    videoData = data

    startProcess(img, currentFrame)

    IDX[index] = idx
    TAIL[index] = tail
    FRAMECOUNT[index] = framecount
    return img3