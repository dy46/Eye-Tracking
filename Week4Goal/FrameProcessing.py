import cv2
import multiprocessing as mp
import math
import numpy as np
import datamani
import drMatches
from drMatches import Position, getXY
import time
import sys
import polygons_overlapping

'''Creating instance variables'''
MIN_MATCH_COUNT = 30
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

def startProcess(img):
    global img1, img2, img3, imgt, s, pos, first_run_flag, poly_arr, poly_template, first, flag
    global idx, tail, framecount

    # Initiate SIFT detector
    # find the keypoints and descriptors with SIFT
    s=np.zeros((4,4))
    #print s.dtype
    pos = Position(0,0,0,0)
    flag = True

    first_run_flag = True

    for i, img1 in enumerate(img):
        if first_run_flag == False:
            img2 = img3
        poly_arr, poly_template = [], []
        imgt = img2.copy()
        first = 0
        while True:
            good_matches= featureMatch()
            matchesMask, ignore, dst, break_flag = drawBorders(good_matches)
            if break_flag:
                print "break"
                break
            x, y = getXY(img2, framecount, videoData, idx, tail, fps, ignore)
            placeText(ignore, i, dst, x, y)
            first += 1
    x, y = drawCircleAndMatches(ignore, good_matches)
    print x, y, framecount
    print flag
    if flag:
        cv2.putText(img3,'Gazing at none of the object',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)
    else:
        cv2.putText(img3,'Gazing at the '+str(object_number)+' object',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow("hi", img3)
    cv2.waitKey(10)

def featureMatch():
    global kp1, kp2
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(imgt,None)

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

def drawBorders(good):
    global img2, imgt
    ignore = False
    break_flag = False
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # for x in dst_pts:
        #     cv2.circle(imgs,(int(x[0][0]),int(x[0][1])),2,(255,0,0),2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        if mask == None:
            break_flag = True
            return None, None, None, break_flag
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        poly_currentarr = []
        for i in range(len(dst)):
            sub_dst = dst[i]
            sub2_dst = sub_dst[0]
            x1 = sub2_dst[0]
            y1 = sub2_dst[1]
            arr = [x1, y1]
            poly_currentarr.append(arr)
        poly_currentarr.append(poly_currentarr[0])
        # print poly_current
        if first == 0:
            xb = poly_currentarr[0][0]
            yb = poly_currentarr[0][1]
            for i in range(len(poly_currentarr)):
                arr = [poly_currentarr[i][0]-xb, poly_currentarr[i][1]-yb]
                poly_template.append(arr)

        poly_current = np.asarray(poly_currentarr)
        if len(poly_arr)>0:
            for p in poly_arr:
                if polygons_overlapping.pair_overlapping(p, poly_current) ==2:
                    # print 2
                    # print poly_template[0]
                    xnot=dst[0][0][0]
                    ynot = dst[0][0][1]
                    t2_a = []
                    for i in range(len(poly_template)):
                        # print "Im here"
                        t2_b = []
                        t_a =np.array([poly_template[i][0]+xnot, poly_template[i][1]+ynot])
                        t2_b.append(t_a)
                        t2_a.append(t2_b)
                    t3_a = np.array(t2_a)
                    dst = t3_a

        poly_arr.append(poly_current)
        # print poly_template
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        imgt = cv2.fillPoly(imgt,[np.int32(dst)],(0,0,0))
        index=[]
        # img2 = cv2.polylines(img2,[np.int32(dst)],True,pos.getColor(),3, cv2.LINE_AA)
        # cv2.circle(img2, (int(float(PoRBX[idx])),int(float(PoRBY[idx]))), 10, (255, 0, 0), 20)
    else:
        # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
        ignore = True
        dst = None
        break_flag = True

    # cv2.imshow('img',imgt)
    # cv2.waitKey(10)
    # cv2.imshow('img2', img2)
    # cv2.waitKey(10)

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
    global s, flag, img3, object_number
    if not ignore:
        s[i][0]=dst[0][0][0];
        s[i][1]=dst[3][0][0];
        s[i][2]=dst[0][0][1];
        s[i][3]=dst[2][0][1];
        if x>s[i][0] and x<s[i][1] and y>s[i][2] and y<s[i][3]:
            print "Hi"
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

    startProcess(img)

    IDX[index] = idx
    TAIL[index] = tail
    FRAMECOUNT[index] = framecount
    return img3




# def edit(f, currentFrame, index, fps, FRAMECOUNT, IDX, TAIL, img, videoData):
#     framecount = currentFrame * 1000.0/fps
#     idx = IDX[index]
#     tail = TAIL[index]
#     img3 = 0
#     img2 = f ## get frame from the queue
#     for i,img1 in enumerate(img):
#         ignore = False;
#         if First == False:
#             img2 = img3
#         kp1, des1 = sift.detectAndCompute(img1,None)
#         kp2, des2 = sift.detectAndCompute(img2,None)

#         FLANN_INDEX_KDTREE = 0
#         index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#         search_params = dict(checks = 50)

#         flann = cv2.FlannBasedMatcher(index_params, search_params)

#         matches = flann.knnMatch(des1,des2,k=2)

#         # store all the good matches as per Lowe's ratio test.
#         good = []
#         for m,n in matches:
#             if m.distance < 0.7*n.distance:
#                 good.append(m)

#         if len(good)>MIN_MATCH_COUNT:
#             src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#             dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

#             M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#             matchesMask = mask.ravel().tolist()

#             h,w = img1.shape
#             pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#             dst = cv2.perspectiveTransform(pts,M)
#             img2 = cv2.polylines(img2,[np.int32(dst)],True,pos.getColor(),3, cv2.LINE_AA)
#             # cv2.circle(img2, (int(float(PoRBX[idx])),int(float(PoRBY[idx]))), 10, (255, 0, 0), 20)
#         else:
#             # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#             matchesMask = None
#             ignore = True

#         if First == True:
#             img2, x, y, idx, tail ,ignore= datamani.drawCircle(img2, framecount, videoData, idx, tail, fps, ignore)
#             framecount = framecount + (1.0/fps)*1000.0
#         img3, pos = drMatches.drawMatches(img1,kp1,img2,kp2,good, pos) ## line must not execute
#         First = False
#         # print str(mp.current_process()) + "Ignore: " + str(ignore)
#         if not ignore:
#             s[i][0]=dst[0][0][0];
#             s[i][1]=dst[3][0][0];
#             s[i][2]=dst[0][0][1];
#             s[i][3]=dst[2][0][1];
#             if x>s[i][0] and x<s[i][1] and y>s[i][2] and y<s[i][3]:
#                 cv2.putText(img3,'Gazing at the '+str(i+1)+' object',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)
#                 flag = False
#         ### HERE WHAT I WANT TO DO IS PROCESS THIS WITH A DIFFERENT IMAGE USING IMG3 as my new framw
#     if flag:
#         cv2.putText(img3,'Gazing at none of the object',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)
#     IDX[index] = idx
#     TAIL[index] = tail
#     FRAMECOUNT[index] = framecount
#     return img3