import math
import numpy as np
import cv2
import DataManipulation
import drMatches
from drMatches import Position

MIN_MATCH_COUNT = 5

img = [cv2.imread('1.png',0), cv2.imread('feature1.png', 0)]

font = cv2.FONT_HERSHEY_SIMPLEX

videoData = DataManipulation.createVideoData(open('1.txt', 'r'))

cap = cv2.VideoCapture('cuttwo.mp4')
framecount = 0.0;
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# # print length
fps = cap.get(cv2.CAP_PROP_FPS)

ret,temp=cap.read();
height, width = temp.shape[:2]

capSize = (width,height) # this is the size of my source video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
success = cv2.VideoWriter('TwoShort.mp4',fourcc,fps,capSize)

# Define binary search parameters
idx = 0 
tail = 10
i = 0
while (True):
    img3 = 0
    i+=1
    #print i
    ret,img2=cap.read()
    if ret==False:
        print "Break"
        break
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    s=np.zeros((4,4))
    #print s.dtype
    First = True
    pos = Position(0,0,0,0)
    for i,img1 in enumerate(img):
        if First == False:
            img2 = img3
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            img2 = cv2.polylines(img2,[np.int32(dst)],True,pos.getColor(),3, cv2.LINE_AA)
            # cv2.circle(img2, (int(float(PoRBX[idx])),int(float(PoRBY[idx]))), 10, (255, 0, 0), 20)

        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None
        if First == True:
            img2, x, y, idx, tail = DataManipulation.drawCircle(img2, framecount, videoData, idx, tail, fps)
            framecount = framecount + (1.0/fps)*1000.0
        img3, pos = drMatches.drawMatches(img1,kp1,img2,kp2,good, pos)
        First = False
        s[i][0]=dst[0][0][0];
        s[i][1]=dst[3][0][0];
        s[i][2]=dst[0][0][1];
        s[i][3]=dst[2][0][1];

        ### HERE WHAT I WANT TO DO IS PROCESS THIS WITH A DIFFERENT IMAGE USING IMG3 as my new framw
    flag=True;
    for i in range(2):
        if x>s[i][0] and x<s[i][1] and y>s[i][2] and y<s[i][3]:
            cv2.putText(img3,'Gazing at the '+str(i+1)+' object',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)
            flag=False;
    if flag:
        cv2.putText(img3,'Gazing at none of the object',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)
    #print np.size(img3,1);
    success.write(img3)
    cv2.imshow('img',img3)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

cap.release()
success.release()
cv2.waitKey()
cv2.destroyAllWindows()