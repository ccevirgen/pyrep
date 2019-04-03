#Numerical differentiation for Python
#Cetin Can Evirgen
#02.06.2015

#Forward differencing uses linear combination of f(x+jh)
#Sliding forward scheme only uses first N points and can use backward differencing
#If differencing scheme needs N points
#In first N points, use sliding forward scheme
#ie 1D grid dims [N1] and N points used in scheme with N1>N
#i in {0,1,2,...,N-1} use sliding forward scheme
#i = 0 uses full forward scheme j in {0,1,2,...,N-1}
#Take i: uses j in {-i,-i+1,...,0,1,...,N-(i+1)}
#i=2: j in {-2,-1,0,1,...,N-3}
#i = N-1: j in {-(N-1),-(N-1)+1,...,0}
#Forward differencing scheme
import numpy as np
##############################################################################################
#Backward differencing upto Nth order
def bf_mat(N,h=4,err=1e-8,retm=False):
    import numpy as np
    from scipy.linalg import inv
    from scipy.linalg import det 
    import numpy as np
    def btexp_pi(N,i,h):
        from scipy.misc import factorial as fct
        ds = np.mgrid[0:N+1:1]
        #return (-i*h)**ds
        return ((-i)**ds)/fct(ds)
    ds = np.mgrid[0:N+1:1]
    i_n = len(ds)
    a = np.zeros(i_n)
    m = np.zeros([i_n,i_n])
    for i in np.arange(i_n):
        m[:,i] = btexp_pi(N,i,h)
    detm = np.fabs(det(m))
    if detm>err:
        m_inv = inv(m)
        a[1] = 1.
        sig1 = np.mat(m_inv)*np.mat(a).T
        sig = np.array([sig1[i,0] for i in np.arange(i_n)])
        if retm==True:
            return m,sig
        else:
            return sig
    else:
        raise ValueError("m is singular")
        return detm 
def bt_dx(arr,ind,N):
    import numpy as np
    def dltx_i(arr,i,k,cf):
        return cf*arr[:,:,i-k]
    cfs = bf_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dltx_i(arr,ind,k,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
def bt_dy(arr,ind,N):
    import numpy as np
    def dlty_i(arr,i,k,cf):
        return cf*arr[:,i-k]
    cfs = bf_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dlty_i(arr,ind,k,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
def bt_dz(arr,ind,N):
    import numpy as np
    def dltz_i(arr,i,k,cf):
        return cf*arr[i-k]
    cfs = bf_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dltz_i(arr,ind,k,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
##############################################################################################
#Forward differencing upto Nth order
def ff_mat(N,h=4,err=1e-8,retm=False):
    from scipy.linalg import inv
    from scipy.linalg import det 
    import numpy as np
    def ftexp_pi(N,i,h):
        from scipy.misc import factorial as fct
        ds1 = np.mgrid[0:N+1:1]
        #return (i*h)**ds1
        return ((i)**ds1)/fct(ds1)
    ds = np.mgrid[0:N+1:1]
    i_n = len(ds)
    a = np.zeros(i_n)
    m = np.zeros([i_n,i_n])
    for i in np.arange(i_n):
        m[:,i] = ftexp_pi(N,i,h)
    detm = np.fabs(det(m))
    if detm>err:
        m_inv = inv(m)
        a[1] = 1.
        sig1 = np.mat(m_inv)*np.mat(a).T
        sig = np.array([sig1[i,0] for i in np.arange(i_n)])
        if retm==True:
            return m,sig
        else:
            return sig
    else:
        raise ValueError("m is singular")
        return detm

def ft_dx(arr,ind,N):
    import numpy as np
    def dltx_i(arr,i,k,cf):
        return cf*arr[:,:,i+k]
    cfs = ff_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dltx_i(arr,ind,k,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
def ft_dy(arr,ind,N):
    import numpy as np
    def dlty_i(arr,i,k,cf):
        return cf*arr[:,i+k]
    cfs = ff_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dlty_i(arr,ind,k,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
def ft_dz(arr,ind,N):
    import numpy as np
    def dltz_i(arr,i,k,cf):
        return cf*arr[i+k]
    cfs = ff_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dltz_i(arr,ind,k,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
##############################################################################################
#Nth order central differencing
def cf_mat(N,err=1e-8,retm=False):
    """Calculate weighting of central diffs, sig, using a = inv(M)*c\n
    Nth order scheme. Only use even N.\n
    Grid spacing, h, is not required here.\n
    err specifies min. value of det(M); default 1e-8\n
    ie det(M)!=0; otherwise singular matrix.\n
    retm decides whether to return M matrix as well as sig.\n
        """
    #N = N1+1
    from scipy.linalg import inv
    from scipy.linalg import det 
    import numpy as np
    #Calculate ith column of M
    def ctexp_pi(N,i):
        #import numpy as np
        from scipy.misc import factorial as fct
        ds = np.mgrid[1:N:2]
        #return (2/fct(ds))*(i*h)**ds
        return (2/fct(ds))*(i)**ds
    #All points required for Nth order scheme
    #N=6 --> [1,3,5]
    ds = np.mgrid[1:N:2]
    #Number of points used in diff. routine
    i_n = len(ds)
    #Initialise a vector, M matrix
    a = np.zeros(i_n).T
    m = np.zeros([i_n,i_n])
    #Calculate M
    for i in np.arange(i_n):
        m[:,i] = ctexp_pi(N,i+1)
    #Determinant of M
    detm = np.fabs(det(m))
    #If det>err --> non-singular matrix
    if detm>err:
        #Calculate inverse of m
        m_inv = inv(m)
        #Only recover f' term
        a[0] = 1.
        sig1 = np.mat(m_inv)*np.mat(a).T
        sig = np.array([sig1[i,0] for i in np.arange(i_n)])
        if retm==True:
            return m,sig
        else:
            return sig
    else:
        print(detm)
        raise ValueError("m is singular")
#Central differencing at step i for 3D array
#In x direction
def ct_dx(arr,ind,N):
    import numpy as np
    #dltx_i takes f(x+k,y,z)-f(x-k,y,z)
    #weighted by a coefficient cf
    def dltx_i(arr,i,k,cf):
        return cf*(arr[:,:,i+k]-arr[:,:,i-k])
    #Coefficient matrix calculated using cf_mat
    cfs = cf_mat(N)
    i_n = cfs.shape[0]
    #Calculate df/dx at current point for all yz plane
    term = np.sum(np.array([dltx_i(arr,ind,k+1,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
#Comments for ct_dy and ct_dz similar to ct_dx
def ct_dy(arr,ind,N):
    import numpy as np
    def dlty_i(arr,i,k,cf):
        return cf*(arr[:,i+k]-arr[:,i-k])
    cfs = cf_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dlty_i(arr,ind,k+1,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
def ct_dz(arr,ind,N):
    import numpy as np
    def dltz_i(arr,i,k,cf):
        return cf*(arr[i+k]-arr[i-k])
    cfs = cf_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dltz_i(arr,ind,k+1,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
##############################################################################################
def diffx(arr,N,h):
    import numpy as np
    d1,d2,d3 = arr.shape
    a_dx = np.zeros([d1,d2,d3])
    for ind in np.arange(d3):
        if ind<N:
            a_dx[:,:,ind] = ft_dx(arr,ind,N)
        elif ind<(d3-N):
            a_dx[:,:,ind] = ct_dx(arr,ind,N)
        else:
            a_dx[:,:,ind] = bt_dx(arr,ind,N)
    return a_dx/h
def diffy(arr,N,h):
    import numpy as np
    d1,d2,d3 = arr.shape
    a_dy = np.zeros([d1,d2,d3])
    for ind in np.arange(d2):
        if ind<N:
            a_dy[:,ind] = ft_dy(arr,ind,N)
        elif ind<(d2-N):
            a_dy[:,ind] = ct_dy(arr,ind,N)
        else:
            a_dy[:,ind] = bt_dy(arr,ind,N)
    return a_dy/h
def diffz(arr,N,h):
    import numpy as np
    d1,d2,d3 = arr.shape
    a_dz = np.zeros([d1,d2,d3])
    for ind in np.arange(d1):
        if ind<N:
            a_dz[ind] = ft_dz(arr,ind,N)
        elif ind<(d1-N):
            a_dz[ind] = ct_dz(arr,ind,N)
        else:
            a_dz[ind] = bt_dz(arr,ind,N)
    return a_dz/h
def diff(arr,N,h):
    import numpy as np
    d1 = len(arr)
    a_dz = np.zeros(d1)
    for ind in np.arange(d1):
        if ind<N:
            a_dz[ind] = ft_dz(arr,ind,N)
        elif ind<(d1-N):
            a_dz[ind] = ct_dz(arr,ind,N)
        else:
            a_dz[ind] = bt_dz(arr,ind,N)
    return a_dz/h
###############################################################################################
def sc_grad(arr,N,h):
    import numpy as np
    tz = diffz(arr,N,h)
    ty = diffy(arr,N,h)
    tx = diffx(arr,N,h)
    return np.array([tx,ty,tz])
def grad(arr,N,h):
    import numpy as np
    if arr.ndim == 3:
        return np.array([diffx(arr,N,h),diffy(arr,N,h),diffz(arr,N,h)])
    elif arr.ndim == 4:
        ax,ay,az = arr
        return np.array([diffx(ax,N,h),diffy(ay,N,h),diffz(az,N,h)])
    else:
        raise ValueError('Array needs to be 3 or 4 dim')
def div(arr,N,h):
    import numpy as np
    ax,ay,az = arr
    tz = diffz(az,N,h)
    ty = diffy(ay,N,h)
    tx = diffx(ax,N,h)
    return tx+ty+tz
def vec_div(arr,N,h):
    import numpy as np
    ax,ay,az = arr
    tz = div(az,N,h)
    ty = div(ay,N,h)
    tx = div(ax,N,h)
    return np.array([tx,ty,tz])
def curl(arr,N,h):
    import numpy as np
    ax,ay,az = arr
    tx = diffy(az,N,h)-diffz(ay,N,h)
    ty = diffz(ax,N,h)-diffx(az,N,h)
    tz = diffx(ay,N,h)-diffy(ax,N,h)
    return np.array([tx,ty,tz])
def sc_lap(arr,N,h):
    grd = sc_grad(arr,N,h) 
    return div(grd,N,h)
def vec_lap(arr,N,h):
    grd = grad(arr,N,h)
    div = vec_div(grd,N,h)
    return div
def adv_drv(arr,uu,N,h):
    gr = sc_grad(arr,N,h)
    #tz = np.sum(uu*gr[0],axis=0)
    #ty = np.sum(uu*gr[1],axis=0)
    #tx = np.sum(uu*gr[2],axis=0)
    return np.sum(uu*gr,axis=0)
    #return np.array([tz,ty,tx])
###############################################################################################
#Second order derivative
###############################################################################################
##############################################################################################
#Backward differencing upto Nth order
def bf2_mat(N,h=4,err=1e-8,retm=False):
    import numpy as np
    from scipy.linalg import inv
    from scipy.linalg import det 
    import numpy as np
    def btexp_pi(N,i,h):
        from scipy.misc import factorial as fct
        ds = np.mgrid[0:N+2:1]
        #return (-i*h)**ds
        return ((-i)**ds)/fct(ds1)
    ds = np.mgrid[0:N+2:1]
    i_n = len(ds)
    a = np.zeros(i_n)
    m = np.zeros([i_n,i_n])
    for i in np.arange(i_n):
        m[:,i] = btexp_pi(N,i+1,h)
    detm = np.fabs(det(m))
    if detm>err:
        m_inv = inv(m)
        a[2] = 1.
        sig1 = np.mat(m_inv)*np.mat(a).T
        sig = np.array([sig1[i,0] for i in np.arange(i_n)])
        if retm==True:
            return m,sig
        else:
            return sig
    else:
        raise ValueError("m is singular")
        return detm 
def bt_d2x(arr,ind,N):
    import numpy as np
    def dltx_i(arr,i,k,cf):
        return cf*arr[:,:,i-k]
    cfs = bf_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dltx_i(arr,ind,k+1,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
def bt_d2y(arr,ind,N):
    import numpy as np
    def dlty_i(arr,i,k,cf):
        return cf*arr[:,i-k]
    cfs = bf_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dlty_i(arr,ind,k+1,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
def bt_d2z(arr,ind,N):
    import numpy as np
    def dltz_i(arr,i,k,cf):
        return cf*arr[i-k]
    cfs = bf_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dltz_i(arr,ind,k+1,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
##############################################################################################
#Forward differencing upto Nth order
def ff2_mat(N,h=4,err=1e-8,retm=False):
    from scipy.linalg import inv
    from scipy.linalg import det 
    import numpy as np
    def ftexp_pi(N,i,h):
        from scipy.misc import factorial as fct
        ds1 = np.mgrid[1:N+3:1]
        #return (i*h)**ds1
        return ((i+1)**ds1)/fct(ds1)
    ds = np.mgrid[0:N+2:1]
    i_n = len(ds)
    a = np.zeros(i_n)
    m = np.zeros([i_n,i_n])
    for i in np.arange(i_n):
        m[:,i] = ftexp_pi(N,i,h)
    detm = np.fabs(det(m))
    if detm>err:
        m_inv = inv(m)
        a[2] = 1.
        sig1 = np.mat(m_inv)*np.mat(a).T
        sig = np.array([sig1[i,0] for i in np.arange(i_n)])
        if retm==True:
            return m,sig
        else:
            return sig
    else:
        raise ValueError("m is singular")
        return detm

def ft_d2x(arr,ind,N):
    import numpy as np
    def dltx_i(arr,i,k,cf):
        return cf*arr[:,:,i+k]
    cfs = ff_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dltx_i(arr,ind,k+1,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
def ft_d2y(arr,ind,N):
    import numpy as np
    def dlty_i(arr,i,k,cf):
        return cf*arr[:,i+k]
    cfs = ff_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dlty_i(arr,ind,k+1,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
def ft_d2z(arr,ind,N):
    import numpy as np
    def dltz_i(arr,i,k,cf):
        return cf*arr[i+k]
    cfs = ff_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dltz_i(arr,ind,k+1,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
##############################################################################################
#Nth order central differencing
def cf2_mat(N,err=1e-8,retm=False):
    """Calculate weighting of central diffs, sig, using a = inv(M)*c\n
    Nth order scheme. Only use even N.\n
    Grid spacing, h, is not required here.\n
    err specifies min. value of det(M); default 1e-8\n
    ie det(M)!=0; otherwise singular matrix.\n
    retm decides whether to return M matrix as well as sig.\n
        """
    #N = N1+1
    from scipy.linalg import inv
    from scipy.linalg import det 
    import numpy as np
    #Calculate ith column of M
    def ctexp_pi(N,i):
        #import numpy as np
        from scipy.misc import factorial as fct
        ds = np.mgrid[0:N+1:2]
        #return (2/fct(ds))*(i*h)**ds
        return (2/fct(ds))*(i)**ds
    #All points required for Nth order scheme
    #N=6 --> [1,3,5]
    ds = np.mgrid[0:N+1:2]
    #Number of points used in diff. routine
    i_n = len(ds)
    #Initialise a vector, M matrix
    a = np.zeros(i_n).T
    m = np.zeros([i_n,i_n])
    #Calculate M
    for i in np.arange(i_n):
        m[:,i] = ctexp_pi(N,i)
    #Determinant of M
    detm = np.fabs(det(m))
    #If det>err --> non-singular matrix
    if detm>err:
        #Calculate inverse of m
        m_inv = inv(m)
        #Only recover f' term
        a[1] = 1.
        sig1 = np.mat(m_inv)*np.mat(a).T
        sig = np.array([sig1[i,0] for i in np.arange(i_n)])
        if retm==True:
            return m,sig
        else:
            return sig
    else:
        print(detm)
        raise ValueError("m is singular")
#Central differencing at step i for 3D array
#In x direction
def ct_d2x(arr,ind,N):
    import numpy as np
    #dltx_i takes f(x+k,y,z)-f(x-k,y,z)
    #weighted by a coefficient cf
    def dltx_i(arr,i,k,cf):
        return cf*(arr[:,:,i+k]+arr[:,:,i-k])
    #Coefficient matrix calculated using cf_mat
    cfs = cf2_mat(N)
    i_n = cfs.shape[0]
    #Calculate df/dx at current point for all yz plane
    term = np.sum(np.array([dltx_i(arr,ind,k,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
#Comments for ct_dy and ct_dz similar to ct_dx
def ct_d2y(arr,ind,N):
    import numpy as np
    def dlty_i(arr,i,k,cf):
        return cf*(arr[:,i+k]+arr[:,i-k])
    cfs = cf2_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dlty_i(arr,ind,k,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
def ct_d2z(arr,ind,N):
    import numpy as np
    def dltz_i(arr,i,k,cf):
        return cf*(arr[i+k]+arr[i-k])
    cfs = cf2_mat(N)
    i_n = cfs.shape[0]
    term = np.sum(np.array([dltz_i(arr,ind,k,cfs[k]) for k in np.arange(i_n)]),axis=0)
    return term
##############################################################################################
##############################################################################################
def diff2x(arr,N,h):
    import numpy as np
    d1,d2,d3 = arr.shape
    a_dx = np.zeros([d1,d2,d3])
    for ind in np.arange(d3):
        if ind<N+1:
            a_dx[:,:,ind] = ft_d2x(arr,ind,N)
        elif ind<(d3-N-1):
            a_dx[:,:,ind] = ct_d2x(arr,ind,N)
        else:
            a_dx[:,:,ind] = bt_d2x(arr,ind,N)
    return a_dx/(h**2)
def diff2y(arr,N,h):
    import numpy as np
    d1,d2,d3 = arr.shape
    a_dy = np.zeros([d1,d2,d3])
    for ind in np.arange(d2):
        if ind<N+1:
            a_dy[:,ind] = ft_d2y(arr,ind,N)
        elif ind<(d2-N-1):
            a_dy[:,ind] = ct_d2y(arr,ind,N)
        else:
            a_dy[:,ind] = bt_d2y(arr,ind,N)
    return a_dy/(h**2)
def diff2z(arr,N,h):
    import numpy as np
    d1,d2,d3 = arr.shape
    a_dz = np.zeros([d1,d2,d3])
    for ind in np.arange(d1):
        if ind<N+1:
            a_dz[ind] = ft_d2z(arr,ind,N)
        elif ind<(d1-N-1):
            a_dz[ind] = ct_d2z(arr,ind,N)
        else:
            a_dz[ind] = bt_d2z(arr,ind,N)
    return a_dz/(h**2)
def diff2(arr,N,h):
    import numpy as np
    d1 = len(arr)
    a_dz = np.zeros(d1)
    for ind in np.arange(d1):
        if ind<N+1:
            a_dz[ind] = ft_d2z(arr,ind,N)
        elif ind<(d1-N-1):
            a_dz[ind] = ct_d2z(arr,ind,N)
        else:
            a_dz[ind] = bt_d2z(arr,ind,N)
    return a_dz/(h**2)

