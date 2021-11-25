import numpy as np

a = np.linspace(0,1,65537)
w = np.array([[0,1],[0.5,0.5],[-1/6,1],[-1/4,1/4],[1/3,0]])
for ww in w:
	w0,w1 = ww[0],ww[1]
	f = np.power(a, 2)
	h = w1*a+w0
	err = np.dot(f-h,f-h)
	print("sqr err:",err/a.size)
