import cv2
import numpy as np

drawing=False
ix,iy=-1,-1
dx,dy=-1,-1


def draw(event,x,y,flags,param):
	global ix,iy,drawing,imgc,img,dx,dy

	if event==cv2.EVENT_LBUTTONDOWN:
		drawing=True
		ix,iy=x,y
	elif event==cv2.EVENT_MOUSEMOVE:
		if drawing==True:
			img=imgc.copy()
			cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
	elif event==cv2.EVENT_LBUTTONUP:
		drawing=False
		cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
		dx,dy=x,y
def crop(imgg):
	global ix,iy,drawing,imgc,dx,dy,img
	img=imgg.copy()
	imgc=img.copy()
	cv2.namedWindow('image')
	cv2.setMouseCallback('image',draw)
	cnt=0

	name=[]

	while(1):
		cv2.imshow('image',img)
		k=cv2.waitKey(1) & 0xFF
		if k==ord('c'):
			img=imgc.copy()
		elif k==ord('s'):
			print ix,iy,dx,dy
			crop=imgc[iy:dy,ix:dx]
			cnt+=1
			cv2.imwrite(str(cnt)+'.png',crop)
			name.append(str(cnt)+'.png')
			cv2.imshow('img'+str(cnt),crop)
			img=imgc.copy()
		elif k==27:
			break

	cv2.destroyAllWindows()
	return name

if __name__=='__main__':
	print crop()