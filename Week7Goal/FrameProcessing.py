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
ref_image, processed_image, output_image, blacked_out_image = None, None, None, None
kp1, kp2, des1, des2 = [], [], [], []
first_run_flag = True
framecount, videoData, idx, tail, fps = None, None, None, None, None
flag = True
poly_arr = []
poly_template = []
poly_arrays = []
object_number = None
first = 0
cmatch = 0
template_flag = False
resized_ref_image, resized_blacked_out_image, resized_processed_image = None, None, None


def checkRect(array):
    x1 = array[0][0]
    y1 = array[0][1]
    x2 = array[1][0]
    y2 = array[1][1]
    x3 = array[2][0]
    y3 = array[2][1]
    x4 = array[3][0]
    y4 = array[3][1]

    cx = (x1+x2+x3+x4)/4
    cy = (y1+y2+y3+y4)/4

    dd1 = math.sqrt(abs(cx-x1))+math.sqrt(abs(cy-y1))
    dd2 = math.sqrt(abs(cx-x2))+math.sqrt(abs(cy-y2))
    dd3 = math.sqrt(abs(cx-x3))+math.sqrt(abs(cy-y3))
    dd4 = math.sqrt(abs(cx-x4))+math.sqrt(abs(cy-y4))
    a = abs(dd1-dd2)/((dd1+dd2)/2)
    b = abs(dd1-dd3)/((dd1+dd3)/2)
    c = abs(dd1-dd4)/((dd1+dd4)/2)

    return a > 0.2 or b > 0.2 or c > 0.2

def startProcess(ref_image_array, current_frame):
    global output_image, s, pos, first_run_flag, poly_arrays, flag, polygon_looked

    # Initiate SIFT detector
    # find the keypoints and descriptors with SIFT
    s=np.zeros((4,4))
    #print s.dtype
    pos = Position(0,0,0,0)
    flag = True

    first_run_flag = True
    poly_arrays = []
    ignore, good_matches = compareImages(ref_image_array, current_frame)
    x, y = drawCircleAndMatches(ignore, good_matches)

    if not flag:
        order = FrameBFS.determineOrder(poly_arrays[object_number - 1])
        poly_number = 10
        for i in range(len(order)):
            if order[i] in polygon_looked:
                poly_number = i + 1
                break
    if flag:
        cv2.putText(output_image,'Gazing at none of the objects',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)
    else:
        cv2.putText(output_image,'Gazing at the '+str(poly_number)+' of the '+str(object_number)+' object',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow("hi", output_image)
    cv2.waitKey(10)

def compareImages(ref_image_array, current_frame):
    global ref_image, processed_image, output_image, blacked_out_image, first_run_flag
    global first, cmatch, template_flag, poly_arr, poly_template, poly_arrays
    for i, ref_image in enumerate(ref_image_array):
        if first_run_flag == False:
            processed_image = output_image
        poly_arr, poly_template= [], []
        blacked_out_image = processed_image.copy()
        # first = 0
        cmatch = 0
        template_flag = False
        while True:
            cmatch +=1
            good_matches = featureMatch(current_frame)
            matchesMask, ignore, dst, break_flag = drawBorders(good_matches, current_frame)
            if break_flag or cmatch>20:
                break
            x, y = getXY(processed_image, framecount, videoData, idx, tail, fps, ignore)
            placeText(ignore, i, dst, x, y)
            # first += 1
        poly_arrays.append(poly_arr)
    return ignore, good_matches

def featureMatch(currentFrame):
    global kp1, kp2, resized_ref_image, resized_processed_image, resized_blacked_out_image
    sift = cv2.xfeatures2d.SIFT_create()

    resized_ref_image = cv2.resize(ref_image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    resized_processed_image = cv2.resize(processed_image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    resized_blacked_out_image = cv2.resize(blacked_out_image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)

    kp1, des1 = sift.detectAndCompute(resized_ref_image,None)
    kp2, des2 = sift.detectAndCompute(resized_blacked_out_image,None)

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

def drawBorders(good, currentFrame):
    global processed_image, blacked_out_image, template_flag, poly_template, poly_arr
    ignore = False
    break_flag = False

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask == None:
            return None, None, None, True
        matchesMask = mask.ravel().tolist()

        h,w = resized_ref_image.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        dst = cv2.perspectiveTransform(pts,M)
        if len(dst) == 0 or dst.size == 0:
            return None, None, None, True

        poly_currentarr = []
        for i in range(len(dst)):
            point_array = [dst[i][0][0], dst[i][0][1]]
            poly_currentarr.append(point_array)
        # poly_currentarr.append(poly_currentarr[0])
        
        current_rec = checkRect(poly_currentarr)
        if not template_flag:
            poly_template = templatefind.t_Start(resized_ref_image, resized_processed_image)
            template_flag = True
        ######## Checking to see if the current mask is a rectangle
        poly_current = np.asarray(poly_currentarr)
        if len(poly_arr)>0:
            for p in poly_arr:
                if polygons_overlapping.pair_overlapping(p, poly_current) ==2 or not current_rec:
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
        formatted_array = []
        for p in dst:
            formatted_array.append(p[0])
        poly_arr.append(formatted_array)

        expanded_dst = []
        for p in dst:
            point = []
            for c in p[0]:
                point.append(c * 2)
            expanded_dst.append([point])

        try: 
            processed_image = cv2.polylines(processed_image,[np.int32(expanded_dst)],True,(0, 255, 0),3, cv2.LINE_AA)
            blacked_out_image = cv2.fillPoly(blacked_out_image,[np.int32(expanded_dst)],(0,0,0))
        except:
            print 'EXCEPT BREAK'
            return None, None, None, True
    else:
        # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        return None, True, None, True
    # cv2.imshow('processed_image', processed_image)
    # cv2.waitKey(5)
    return matchesMask, ignore, dst, break_flag

def drawCircleAndMatches(ignore, good):
    global processed_image, output_image, pos, idx, tail, framecount, first_run_flag
    x, y = 0, 0
    if first_run_flag == True:
        processed_image, x, y, idx, tail ,ignore= datamani.drawCircle(processed_image, framecount, videoData, idx, tail, fps, ignore)
        framecount = framecount + (1.0/fps)*1000.0
    output_image, pos = drMatches.drawMatches(ref_image,kp1,processed_image,kp2,good, pos)
    first_run_flag = False
    return x, y

def placeText(ignore, i, dst, x, y):
    global s, flag, output_image, object_number, polygon_looked
    '''Probability functions here'''
    x, y = x / 2, y / 2
    x_margin = (dst[3][0][0] - dst[0][0][0]) / 20
    y_margin = (dst[2][0][1] - dst[0][0][1]) / 20
    if not ignore:
        s[i][0]=dst[0][0][0] - x_margin;
        s[i][1]=dst[3][0][0] + x_margin;
        s[i][2]=dst[0][0][1] - y_margin;
        s[i][3]=dst[2][0][1] + y_margin;
        if x>s[i][0] and x<s[i][1] and y>s[i][2] and y<s[i][3]:
            polygon_looked = dst
            object_number = i + 1
            flag = False

def processImage(f, currentFrame, index, frames_per_second, FRAMECOUNT, IDX, TAIL, img, data):
    global fps, framecount, idx, tail, processed_image, videoData
    fps = frames_per_second
    framecount = currentFrame * 1000.0/fps
    idx = IDX[index]
    tail = TAIL[index]
    processed_image = f
    videoData = data

    startProcess(img, currentFrame)

    IDX[index] = idx
    TAIL[index] = tail
    FRAMECOUNT[index] = framecount
    return output_image