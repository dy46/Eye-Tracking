import numpy as np
import cv2
from PIL import Image
import sys

class Position:
    def __init__(self, x, y, ymax, f, color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        self.x = x
        self.y = y
        self.ymax = ymax
        self.f = f
        self.color = color

    def getCoordinates(self):
        return self.__x, self.__y
    def setCoordinates(self):
        self.__x, self.__y = x, y
    def getColor(self):
        return self.color[self.f]



def drawMatches(img1, kp1, img2, kp2, matches, pos):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    if height1>pos.ymax: #yl holds the maximum height so far
        pos.ymax = height1
    foreground = Image.fromarray(img1)
    background = Image.fromarray(img2)
    if pos.x+width1>width2:
        pos.setCoordinates(0, pos.ymax)
    background.paste(foreground, (pos.x, pos.y), foreground)
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
        cv2.circle(out, (int(x1+pos.x),int(y1+pos.y)), 2, pos.getColor(), 2)   
        cv2.circle(out, (int(x2),int(y2)), 2, pos.getColor(), 2)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1+pos.x),int(y1+pos.y)), (int(x2),int(y2)), pos.getColor(), 1)

    pos.x = pos.x+width1
    pos.f = pos.f+1
    # Show the image
    return out, pos
