import math

def checkRect(array):
	x1 = array[0][0]
	y1 = array[0][1]
	x2 = array[1][0]
	y2 = array[1][1]
	x3 = array[2][0]
	y3 = array[2][1]
	x4 = array[3][0]
	y4 = array[3][1]

	cx=(x1+x2+x3+x4)/4
	cy=(y1+y2+y3+y4)/4

	dd1=math.sqrt(abs(cx-x1))+math.sqrt(abs(cy-y1))
	dd2=math.sqrt(abs(cx-x2))+math.sqrt(abs(cy-y2))
	dd3=math.sqrt(abs(cx-x3))+math.sqrt(abs(cy-y3))
	dd4=math.sqrt(abs(cx-x4))+math.sqrt(abs(cy-y4))
	a = abs(dd1-dd2)/((dd1+dd2)/2)
	b = abs(dd1-dd3)/((dd1+dd3)/2)
	c = abs(dd1-dd4)/((dd1+dd4)/2)

	if a > 0.1 or b>0.1 or c>0.1:
		return False
	else:
		return True

xa = 100
ya = 100



xb = 200
yb = 100


xc = 100
yc = 50


xd = 200
yd = 50

array = []

a = [xc, yc]
b = [xa, ya]
c = [xb, yb]
d = [xd, yd]

array.append(a)
array.append(b)
array.append(c)
array.append(d)
array.append(a)
# print array
print checkRect(array)


	# return dd1==dd2 && dd1==dd3 && dd1==dd4;

	# return res





# [[700, 400], [700, 500], [900, 500], [900, 400], [700, 400]]
# [[100, 100], [200, 100], [100, 50], [200, 50], [100, 100]]
