import numpy as np
import cv2
from shapely.geometry import Polygon
import sys
import polygons_overlapping 

MIN_MATCH_COUNT = 15

img1 = cv2.imread('current2.png',0)          # queryImage
img2 = cv2.imread('feature4.png',0) # trainImage
imgt=img2.copy()
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
poly_arr = []
poly_template = []
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
				good.append(m)

		if len(good)>MIN_MATCH_COUNT:
			src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
			for x in dst_pts:
				cv2.circle(imgs,(int(x[0][0]),int(x[0][1])),2,(255,0,0),2)
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
				if y == 0:
					xb = poly_currentarr[0][0]
					yb = poly_currentarr[0][1]
					for i in range(len(poly_currentarr)):
						arr = [poly_currentarr[i][0]-xb, poly_currentarr[i][1]-yb]
						poly_template.append(arr)

				poly_current = np.asarray(poly_currentarr)
				if len(poly_arr)>0:
					for p in poly_arr:
						if polygons_overlapping.pair_overlapping(p, poly_current) ==2:
							print 2
							print poly_template[0]
							xnot=dst[0][0][0]
							ynot = dst[0][0][1]
							t2_a = []
							for i in range(len(poly_template)):
								print "Im here"
								t2_b = []
								t_a =np.array([poly_template[i][0]+xnot, poly_template[i][1]+ynot])
								t2_b.append(t_a)
								t2_a.append(t2_b)
							t3_a = np.array(t2_a)
							dst = t3_a


				print dst

				poly_arr.append(poly_current)
				print poly_template
				imgt = cv2.polylines(imgt,[np.int32(dst)],True,255,3, cv2.LINE_AA)
				img2 = cv2.fillPoly(img2,[np.int32(dst)],(0,0,0))
				index=[]

		else:
			print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
			matchesMask = None


		draw_params = dict(matchColor = (255,0,0), # draw matches in green color
						   singlePointColor = None,
						   flags = 2)

		img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
		#cv2.imshow("lll",img3)
		#cv2.waitKey()

cv2.imshow('img',imgt)
#cv2.imshow('ppp',img3)
cv2.waitKey()