import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeScale(It, It1, rect, p0 = np.zeros(4)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
	# Put your implementation here
	p = p0
	th = 0.1
	x1, y1, x2, y2 = rect
	h, w = It1.shape
	rbs = RectBivariateSpline(range(h), range(w), It)
	rbs1 = RectBivariateSpline(range(h), range(w), It1)
	all_x, all_y = np.meshgrid(np.arange(x1,x2+1), np.arange(y1,y2+1), sparse=True)
	all_x_dense, all_y_dense = np.meshgrid(np.arange(x1,x2+1), np.arange(y1,y2+1))
	all_x_dense = all_x_dense.reshape((-1, 1))
	all_y_dense = all_y_dense.reshape((-1, 1))
	delta_p = np.ones(4)
	while np.linalg.norm(delta_p) > th:
		all_x_prime = (1 + p[2]) * all_x + p[0]
		all_y_prime = (1 + p[3]) * all_y + p[1]
		Ix_prime = rbs1.ev(all_y_prime, all_x_prime, 0, 1).reshape((-1,1))
		Iy_prime = rbs1.ev(all_y_prime, all_x_prime, 1, 0).reshape((-1,1))
		A = np.hstack((Ix_prime, Iy_prime, all_x_dense * Ix_prime, all_y_dense * Iy_prime))
		b = (rbs.ev(all_y, all_x) - rbs1.ev(all_y_prime, all_x_prime)).flatten()
		delta_p, _, _, _ = np.linalg.lstsq(A, b)
		p += delta_p
	return p
