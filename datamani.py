#file = open('Jonathan_5-27_raw_data.txt', 'r')
#line = file.readline()
#print(line)
#line = file.readline()
#print(line)
#file.close()
import math
import numpy as np
import cv2

Videoname = list()
Videotime = list()
militime = list()
PoRBY = list()
PoRBX = list()

i = False;
with open('Jonathan_5-27_raw_data.txt', 'r') as f:
	for line in f:
		if i is True:
			#content = f.readlines()
			dataline = line.split()
			Videoname.append(dataline[3])
			militime.append(dataline[0])
			if dataline[14]=='Saccade':
				PoRBY.append(dataline[20]) # MOST IMPORTANT
				PoRBX.append(dataline[19]) # MOST IMPORTANT
				Videotime.append(dataline[33])
			else:
				Videotime.append(dataline[34])
				PoRBY.append(dataline[21]) # MOST IMPORTANT
				PoRBX.append(dataline[20]) # MOST IMPORTANT
		i = True
militime2 = [float(i) for i in militime]
militime2 = [i/1000.0 for i in militime2]
first = Videoname[0]
count = -1
for string in Videoname:
	if first != string:
		break
	count +=1
cap = cv2.VideoCapture('jony2.mp4')
framecount = 0.0;
redtime = militime2[:count]
norm = redtime[0]
redtime = [i-norm for i in redtime]
fps = 24.0
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print length
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print height
capSize = (width,height) # this is the size of my source video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
success = cv2.VideoWriter('adios8.mp4',fourcc,fps,capSize) 
i = 0




while(True):
	print i
	i+=1
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == False:
		break
	# Our operations on the frame come here
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_pink = np.array([104, 103, 120])
	high_pink = np.array([170, 160, 190])

	idx = min(range(len(redtime)), key=lambda x: abs(redtime[x]-framecount))
	idx = idx-25
	cv2.circle(frame, (int(float(PoRBX[idx])),int(float(PoRBY[idx]))), 10, (255, 0, 0), 20)
	framecount = framecount + 1.0/24.0
	# Display the resulting frame
	#out3.write(gray)
	success.write(frame)
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
#out3.release()
success.release()
cv2.destroyAllWindows()
#while True:
 #   flag, frame = cap.read()
  #  print flag
   # if flag ==0:
	#	break
	#cv2.imshow("Video", frame)
	#key_pressed = cv2.waitKey(10)
	#if key_pressed ==27:
	#	break
