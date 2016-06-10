import math
import numpy as np
import cv2

'''Class for the DataPoint object which stores gaze location at each tick of the raw data.'''
class DataPoint(object):
	def __init__(self, vidname, tmili, x, y):
		self.__vidname = vidname
		self.__tmili = tmili
		self.__x = x
		self.__y = y
	# vidname = ""
	# tmili = 0
	# x = 0
	# y = 0
	def getCoordinates(self):
		return self.__x, self.__y
	def setCoordinates(self, x, y):
		self.__x, self.__y = x, y
	def getTMili(self):
		return self.__tmili
	def setTMili(self, tmili):
		self.__tmili = tmili
	def setVidName(self, name):
		self.__vidname = name

'''Creates a DataPoint object with properties passed in.'''
def make_dpoint(vidname, tmili, x, y):
    dpoint = DataPoint(vidname, tmili, x, y)
    return dpoint

'''Binary search implementation to search for the point in the data that 
is the closest to the passed in value.'''
def binarySearch(data, val):
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

'''Creates the video data based on the raw data passed in.
The parsing within this method would only work with data from the SMI glasses.'''
def createVideoData(textFile): 
	videoData = list(list())
	set_up = True;

	# Skip top line of text file because no data is contained there.
	textFile.next()
	points = list()
	for line in textFile:
		dp = make_dpoint('', 0, 0, 0)
		dataline = line.split()
		cvt = dataline[3] #current video title
		dp.setVidName(cvt)
		if set_up:
			tnot = float(dataline[0])
			pvt = cvt
			set_up = False
		dp.setTMili(float(dataline[0])-tnot)
		if pvt != cvt:
			videoData.append(points)
			points = list()
		if dataline[14]=='Saccade':
			dp.setCoordinates(float(dataline[19]), float(dataline[20]))
		else:
			dp.setCoordinates(float(dataline[20]), float(dataline[21]))
		pvt = cvt
		points.append(dp)

	videoData.append(points)

	vid = videoData[0]
	# print 'Starting'
	# for x in vid:
	# 	print x.getTMili()
	# print 'Ending'
	return vid

'''Draws a circle on the passed in frame that correlates with the gaze location
at the passed in framecount.'''
def drawCircle(frame, frameCount, videoData, idx, tail):
	# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	alpha = 7 #delay factor
	data = videoData[idx:tail]
	idx_sub = binarySearch(data, frameCount)
	idx = idx+idx_sub
	tail = idx+10
	x, y = videoData[idx-alpha].getCoordinates()
	cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), 20)
	return frame, x, y, idx, tail
