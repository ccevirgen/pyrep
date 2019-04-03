def havgz(arr,dx=0.004,dy=0.004,Lx=1.024,Ly=1.024):
	import numpy as np
	N = arr.ndim
	rets = np.zeros(arr.shape)
	D = (dx/Lx)*(dy/Ly)
	if N==3:
		for ii in range(544):
			rets[ii] = np.sum(arr[ii])*D
		#rets = np.array([np.sum(arr[ii]) for ii in np.arange(544)])*D
	elif N==4:
		for i in range(3):
			for ii in range(544):
				rets[i,ii] = np.sum(arr[i,ii])*D
			#rets[i] = np.array([np.sum(arr[i,ii]) for ii in np.arange(544)])*D
	return rets
def random_field(fld,mn):
	import numpy as np
	ret = np.zeros(fld.shape)
	if fld.ndim==mn.ndim:
		return fld-mn
	else:
		raise ValueError('Arrays (fields) must have same shape.')

