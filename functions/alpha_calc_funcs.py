import numpy as np
from gaussian_smoothing import gau_smooth

def zprof(arr,L=1.024,dx=0.004,Nz=544):
	import numpy as np
	mlt = (dx/L)**2
	return np.array([np.sum(arr[i])*mlt for i in range(Nz)])


def plottable_hist(data,bins,density=False):
	data = data.ravel()
	pbins = 0.5*(bins[1:]+bins[:-1])
	freqs = np.histogram(data,bins=bins,density=density)[0]
	return [pbins,freqs]

def pdf_moment(bins,pdf,db,m=1):
	return np.sum((bins**m)*pdf)*db

def pdf_var(bins,pdf,db):
	mn = pdf_moment(bins,pdf,db,m=1)
	m2 = pdf_moment(bins,pdf,db,m=2)
	return m2-(mn**2)

def compare_curves(arr1,arr2):
	N1 = arr1.size
	N2 = arr2.size
	if N1==N2:
		cdf1 = np.cumsum(arr1)
		cdf2 = np.cumsum(arr2)
		return np.max(np.fabs(cdf1-cdf2))

def gau_tau(arr1,arr2):
	e12 = gau_smooth(arr1*arr2,typ='piecewise',verbose=True)
	e1 = gau_smooth(arr1,typ='piecewise',verbose=True)
	e2 = gau_smooth(arr2,typ='piecewise',verbose=True)
	return e12-e1*e2

def standardise_var(arr):
	import numpy as np
	return (arr-np.mean(arr))/np.std(arr)

def vec_cross(a,b):
    import numpy as np
    ax,ay,az = a
    bx,by,bz = b
    return np.array([ay*bz-az*by,az*bx-ax*bz,ax*by-ay*bx])

def mod_arr(arr):
	import numpy as np
	if arr.ndim<4:
		return np.sqrt(arr*arr)
	elif arr.ndim==4:
		return np.sqrt(np.sum(arr*arr,axis=0))



