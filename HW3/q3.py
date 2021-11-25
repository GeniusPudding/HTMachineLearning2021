import numpy as np

a = np.linspace(0,1,65537)

def line(x,x1,x2):
	y1, y2 = x1*x1, x2*x2		
	y = y2 + (y1-y2)*(x-x2)/(x1-x2)
	return y
samples_nb = 10000
err_t = 0
for i in range(samples_nb):
	x1,x2 = np.random.random_sample((2,))
	g = line(a,x1,x2)
	f = np.power(a, 2)
	err = np.dot(f-g,f-g)
	err_t += err/a.size

print("ed:",err_t/samples_nb)
