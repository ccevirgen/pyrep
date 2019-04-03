import numpy as np
import scipy as sp

#Gaussian smoothing function
def smooth_md(dt_arr1,typ,krn):
    from scipy.ndimage.filters import gaussian_filter1d as gauss1d
    from scipy.ndimage.filters import gaussian_filter as gaussmd
    if typ=='piecewise':
        if dt_arr1.ndim == 4:
            a1=gauss1d(dt_arr1,axis=1,sigma=krn,mode='reflect')
            a2=gauss1d(a1,axis=2,sigma=krn,mode='wrap')
            sm_dt=gauss1d(a2,axis=3,sigma=krn,mode='wrap')
        elif dt_arr1.ndim == 3:
            a1=gauss1d(dt_arr1,axis=0,sigma=krn,mode='reflect')
            a2=gauss1d(a1,axis=1,sigma=krn,mode='wrap')
            sm_dt=gauss1d(a2,axis=2,sigma=krn,mode='wrap')
    elif typ=='wrap':
        sm_dt=gaussmd(dt_arr1,sigma=[krn,krn,krn],mode='wrap')
    return sm_dt

def gau_smooth(arr0,typ='single',krn=75.0/4.0,verbose=False):
	import numpy as np
	from scipy.ndimage.filters import gaussian_filter1d as gf1d
	from scipy.ndimage.filters import gaussian_filter as gf
	N = arr0.ndim
	ktyp = type(krn)
	modes = ['reflect','wrap','wrap']
	if ktyp==float:
		krn = [krn,krn,krn]
	if typ=='single':
		ret = np.zeros(arr0.shape)
		for i in range(3):
			if verbose:
				print('Smoothing type: wrap, component: {}'.format(i))
			ret[i] = gf(arr0[i],sigma=krn,mode='wrap')
		return ret
	elif typ=='piecewise':
		for i in range(3):
			if N==3:
				cax = i
			if N==4:
				cax = i+1
			if verbose:
				print('Axis = {}; k = {}, mode = {}'.format(cax,krn[i],modes[i]))
			clab = 'arr{}'.format(i)
			nlab = 'arr{}'.format(i+1)
			if verbose:
				print(clab,nlab)
			locals()[nlab] = gf1d(locals()[clab],axis=cax,sigma=krn[i],mode=modes[i])
		if verbose:
			print('Returning {}'.format(nlab))
		return	locals()[nlab]
