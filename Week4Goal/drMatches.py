import numpy as np
import cv2
from PIL import Image
import sys

class Position:
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
    def __init__(self, x, y, ymax, f):
        self.x = x
        self.y = y
        self.ymax = ymax
        self.f = f

    def getCoordinates(self):
        return self.__x, self.__y
    def setCoordinates(self, x, y):
        self.__x, self.__y = x, y
    def getColor(self):
        # print self.f
        return (255, 0, 0)

def binarySearch(data, val):
    val = val * 1000
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) / 2
        if data[mid].getTMili() < val:
            lo = mid + 1
        elif data[mid].getTMili() > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind] 
        if abs(data[mid].getTMili() - val) < abs(data[best_ind].getTMili() - val):
            best_ind = mid
    return best_ind

def getXY(frame, frameCount, videoData, Idx, Tail,fps, ignore):
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    alpha = 10
    delayFactor = 6 - int(frameCount/((1.0/fps)*1000.0) /  alpha)#delay factor
    # data = videoData[Idx:Tail]
    # lowt = videoData[Idx].getTMili()
    # hight = videoData[Tail].getTMili()
    idx_sub = binarySearch(videoData, frameCount)
    # Idx = Idx+idx_sub
    # Tail = Idx+10
    point = idx_sub
    # point = idx_sub
    if point > len(videoData) - 1:
        point = len(videoData) - 1
    x, y = videoData[point].getCoordinates()

    tpoint = videoData[point].getTMili()
    # print str(frameCount)+'           '+str(lowt)+'             ' +str(tpoint)+ '            ' +str(hight)
    if x == None or y == None:
        return frame, x, y, Idx, Tail, True

    # cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), 20)
    return x, y

def drawMatches(img1, kp1, img2, kp2, matches, pos):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    if height1>pos.ymax: #yl holds the maximum height so far
        pos.ymax = height1
    foreground = Image.fromarray(img1)
    background = Image.fromarray(img2)
    if pos.x+width1>width2:
        pos.setCoordinates(0, pos.ymax)
    # background.paste(foreground, (pos.x, pos.y), foreground)
    out = np.array(background)
    #print pos.getColor
    #cv2.circle(out, (int(100),int(100)), 1, pos.getColor(), 50)
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        # cv2.circle(out, (int(x1+pos.x),int(y1+pos.y)), 2, pos.getColor(), 2)   
        # cv2.circle(out, (int(x2),int(y2)), 2, pos.getColor(), 2)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        # cv2.line(out, (int(x1+pos.x),int(y1+pos.y)), (int(x2),int(y2)), pos.getColor(), 1)

    pos.x = pos.x+width1
    pos.f = pos.f+1
    # Show the image
    return out, pos
