import math
import numpy as np
import cv2
import datamani

MIN_MATCH_COUNT = 10

img1 = cv2.imread('1.png',0)          # queryImage


font = cv2.FONT_HERSHEY_SIMPLEX

videoData = datamani.createVideoData(open('1.txt', 'r'))

cap = cv2.VideoCapture('JonyMove.mp4')
framecount = 0.0;
fps = 25.0
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# # print length
width  = 1376
# # print width
height = 960
# # print height
capSize = (width,height) # this is the size of my source video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
success = cv2.VideoWriter('Week2Goal.mp4',fourcc,fps,capSize)

idx = 0 
tail = 10

i = 0

while (True):
    i+=1
    print i
    ret,img2=cap.read()
    if ret==False:
        print "Break"
        break
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
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
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        # cv2.circle(img2, (int(float(PoRBX[idx])),int(float(PoRBY[idx]))), 10, (255, 0, 0), 20)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    img2, x, y, idx, tail = datamani.drawCircle(img2, framecount, videoData, idx, tail)
    framecount = framecount + (1.0/25.0)*1000.0

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    #height = np.size(img3, 0)
    #width = np.size(img3, 1)
    #print width
    #print height
    if x>dst[0][0][0] and x<dst[3][0][0] and y>dst[0][0][1] and y<dst[1][0][1]:
        cv2.putText(img3,'Gazing at the object',(150,30), font, 1,(255,255,255),2,cv2.LINE_AA)
    else:
        cv2.putText(img3,'Not Gazing at the object',(150,30), font, 1,(255,255,255),2,cv2.LINE_AA)
    #height = np.size(img3, 0)
    #width = np.size(img3, 1)
    #print height, width
    #cv2.imwrite('img',img3)
    success.write(img3)
    cv2.imshow('img',img3)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

cap.release()
success.release()
cv2.waitKey()
cv2.destroyAllWindows()