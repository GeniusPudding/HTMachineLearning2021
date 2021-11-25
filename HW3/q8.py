import numpy as np



def fi(x1,x2):
	return np.array([1,x1,x2,x1*x1,x1*x2,x2*x2])

train = np.array([[0,1,-1],[0,-1,-1],[-1,0,1],[1,0,1]])
w = np.array([[0,-1,0,0,0,0],[0,0,-1,0,0,0], \
[0,0,0,-1,0,0],[0,0,0,0,-1,0],[0,0,0,0,0,-1]])

for vec in train:
	x1, x2, y_label = vec
	fix = fi(x1, x2)
	print("vec:",vec)
	for i,w_ in enumerate(w):
		sign = 1 
		y = np.dot(fix,w_)
		if y != 0:
			sign = np.sign(y)
		if sign == y_label:	 
			print("w"+str(i) )


