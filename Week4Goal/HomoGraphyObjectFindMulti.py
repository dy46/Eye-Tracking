import numpy as np
import cv2

MIN_MATCH_COUNT = 15

img1 = cv2.imread('current.png',0)          # queryImage
img2 = cv2.imread('feature3.png',0) # trainImage
imgt=img2.copy()
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
for y in range(4):
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	#print des1
	#print des2
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)
	imgs=img2.copy()
	# store all the good matches as per Lowe's ratio test.
	#img4=img2
	for r in range(1):
		good = []
		for m,n in matches:
		    if m.distance < 0.7*n.distance:
		    	x=kp1[m.queryIdx].pt
		    	(x2,y2)=kp2[m.trainIdx].pt
		    	#cv2.circle(img4,(int(x2),int(y2)),2,(255,0,0),2)
		        good.append(m)

		if len(good)>MIN_MATCH_COUNT:
		    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		    for x in dst_pts:
		    	cv2.circle(imgs,(int(x[0][0]),int(x[0][1])),2,(255,0,0),2)
		    cv2.imshow("img",imgs)
		    cv2.waitKey();
		    for p in range(1):
		    	print "=="
		    	print len(src_pts)
		    	print len(dst_pts)
		    	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)	
		    	#print good 
		    	#print mask
		    	matchesMask = mask.ravel().tolist()

		    	h,w = img1.shape
		    	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		    	dst = cv2.perspectiveTransform(pts,M)
		    	imgt = cv2.polylines(imgt,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		    	img2 = cv2.fillPoly(img2,[np.int32(dst)],(0,0,0))
		    	#print len(matchesMask)
		    	#print len(dst_pts)
		    	#print dst_pts
		    	index=[]
		    	#for i in range(len(matchesMask)):
		    		#print matchesMask[i]
		    	#	if matchesMask[i]==1:
		    	#		index.append(i)
		    	#dst_pts=np.delete(dst_pts,index,0)
		    	#src_pts=np.delete(src_pts,index,0)
		    	#print dst_pts
		else:
		    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
		    matchesMask = None


		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
		                   singlePointColor = None,
		                   flags = 2)

		img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
		#cv2.imshow("lll",img3)
		#cv2.waitKey()

cv2.imshow('img',imgt)
#cv2.imshow('ppp',img3)
cv2.waitKey()
