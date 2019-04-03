#Plotting functions for Python
#Cetin Can Evirgen
#13.02.16

#Preamble
import numpy as np
import matplotlib.pyplot as plt

def plot_arr(freqs):
    N = len(freqs)
    sz = freqs[0].shape[0]
    #rets = np.zeros([N,sz])
    rets = np.array([freqs[i] for i in np.arange(N)])        
    return rets
#1D line plot
def line_1d(x,y,xl,yl,xrn='Default',yrn='Default',l_col = 'b',l_sty = '-',fg_height = 6,l_width=1.5,sv_fig=False,sv_lab='None',sv_format='pdf',sv_dir='Current',plt_title = 'None',plt_show=True):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    golden = (1.+5.**0.5)/2.
    fg_width = fg_height*golden
    fig = plt.figure(figsize=[fg_width,fg_height])
    ax = fig.add_subplot(111)
    ax.plot(x,y,lw=l_width,ls=l_sty,color=l_col)
    ax.set_xlabel(xl,fontsize=18)
    ax.set_ylabel(yl,fontsize=18)
    ax.tick_params(labelsize=16,length=5,width=1.5)
    if xrn=='Default':
        ax.set_xlim([np.amin(x),np.amax(x)])
    else:
        ax.set_xlim(xrn)
    if yrn=='Default':
        ax.set_ylim([np.amin(y),1.1*np.amax(y)])
    else:
        ax.set_ylim(yrn)
    if plt_title!='None':
        ax.set_title(plt_title,fontsize=22)
    if sv_fig==True:
        if sv_dir=='Current':
            sv_dir = os.getcwd()
        os.chdir(sv_dir)
        plt.savefig(sv_lab+'.'+sv_format)
    if plt_show==True:
        plt.show()
#Partition sample space by one variable and find mean of other variable for each partition
def part_mean(x,y,bns,ret_freqs=False,nrm=True):
    import numpy as np
    if x.ndim>1:
        x = x.ravel()
    if y.ndim>1:
        y = y.ravel()
    if type(bns)==int:
        x_b = np.mgrid[np.amin(x):np.amax(x):eval(str(bns)+'j')]
        y_b = np.mgrid[np.amin(y):np.amax(y):eval(str(bns)+'j')]
    else:
        x_b,y_b = bns
    freqs = np.histogram2d(x,y,bins=[x_b,y_b])[0].T
    if nrm:
        dx = np.ediff1d(x_b).mean(); dy = np.ediff1d(y_b).mean()
        freqs /= float(np.sum(freqs*dx*dy))
    xb = 0.5*(x_b[1:]+x_b[:-1]); yb = 0.5*(y_b[1:]+y_b[:-1])
    Nx = len(xb); Ny = len(yb)
    rets = np.zeros(Nx)
    for i in np.arange(Nx):
        t = np.sum(yb*freqs[:,i])
        b = np.sum(freqs[:,i])
        if b==0:
            rets[i] = np.nan
        else:
            rets[i] = t/b
    if ret_freqs:
        return x_b,y_b,rets,freqs
    else:
        return [xb,rets]
#Partition sample space and compute error
#def part_error(x,y):
#    
#2D histogram
def hist2d_data(x,y,bns,ret_bns=False,nrm=False):
    if x.ndim>1:
        x = x.ravel()
    if y.ndim>1:
        y = y.ravel()
    if type(bns)==int:
        x_b = np.mgrid[np.amin(x):np.amax(x):eval(str(bns)+'j')]
        y_b = np.mgrid[np.amin(y):np.amax(y):eval(str(bns)+'j')]
    elif type(bns)==np.ndarray:
        if bns.shape[0]!=2:
            raise ValueError('Incorrect shape')
        x_b,y_b = bns
        if len(x_b)!=len(y_b):
            raise ValueError('x and y do not have same dimensions.')
    elif type(bns)==list:
        if len(x)!=2:
            raise ValueError('Need tuple containing two arrays; one for x and one for y')
        x_b,y_b = bns
    dx = np.ediff1d(x_b).mean(); dy = np.ediff1d(y_b).mean()
    freqs = np.histogram2d(x,y,bins=[x_b,y_b])[0].T
    if nrm:
        freqs /= float(np.sum(freqs)*dx*dy)
    if ret_bns:
        return [x_b,y_b,freqs]
    else:
        return freqs
#1D multi-plot
def mline_1d(x,ys,xl,yl,leg_labs,xrn='Default',yrn='Default',fg_height = 6,l_width=1.5):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    def ax_subplot(x,y,ax_obj,l_width):
        import matplotlib.pyplot as plt
        ax_obj.plot(x,y,lw=l_width)
    #Initialise figure object
    golden = (1.+5.**0.5)/2.
    fg_width = fg_height*golden
    #Number of plots decide plot grid shape
    N_plt = ys.shape[0]
    grid_dim = int(np.sqrt(N_plt))
    fig = plt.figure(figsize=[fg_width,fg_height])
    ax = fig.add_subplot(111)
    for i in np.arange(N_plt):
        ax.plot(x,ys[i],lw=l_width)
    ax.set_xlabel(xl,fontsize=18)
    ax.set_ylabel(yl,fontsize=18)
    ax.tick_params(labelsize=16,length=5,width=1.5)
    if xrn=='Default':
        ax.set_xlim([np.amin(x),np.amax(x)])
    else:
        ax.set_xlim(xrn)
    if yrn=='Default':
        ax.set_ylim([np.amin(ys),1.1*np.amax(ys)])
    else:
        ax.set_ylim(yrn)
    ax.legend(leg_labs,fontsize=16,loc=2)
#1D histogram as line plot
def hist_pdf(arr1,no_bins,xl,yl,rng='Default',dns=True,fg_height = 6,l_width=1.5,sv_fig=False,sv_lab='hist1d',sv_format='pdf',sv_dir='Current',plt_title = 'None',plt_show=True):
    import os
    golden = (1 + 5 ** 0.5) / 2
    #Processing data
    dims = arr1.ndim
    if dims>1:
        arr = arr1.ravel()
        print(str(dims)+'D array flattened')
    else:
        arr=arr1
    #Calculating histogram
    if rng=='Default':
        rng = [np.amin(arr),np.amax(arr)]
    freqs,bns = np.histogram(arr,range=rng,bins=no_bins,density=dns)
    bins = 0.5*(bns[1:]+bns[:-1])
    fg_width = fg_height*golden
    fig = plt.figure(figsize=[fg_width,fg_height])
    ax = fig.add_subplot(111)
    ax.plot(bins,freqs,linewidth=l_width)
    ax.set_xlabel(xl,fontsize=18)
    ax.set_ylabel(yl,fontsize=18)
    ax.tick_params(labelsize=16,length=5,width=1.5)
    if plt_title !='None':
        ax.set_title(plt_title,fontsize=20)
    if sv_fig==True:
        if sv_dir == 'Current':
            sv_dir = os.getcwd()+'/'
        else:
            print('Save directory is '+sv_dir)
        os.chdir(sv_dir)
        plt.savefig(sv_lab+'.'+sv_format)
    if plt_show==True:
        plt.show()
    return bins,freqs
#Multiple PDF plots
def input_array(dset1,dset2,dset3):
    if dset1.ndim>1:
        arr1 = dset1.ravel()
    else:
        arr1 = dset1
    N = len(arr1)
    rets = np.zeros([3,N])
    rets[0] = arr1
    if dset2.ndim>1:
        arr2 = dset2.ravel()
    else:
        arr2 = dset2
    rets[1] = arr2
    if dset3.ndim>1:
        arr3 = dset3.ravel()
    else:
        arr3 = dset3
    rets[0] = arr3
    return rets
def phase_hist_pdf(arr,no_bins,xl,yl,fltr,rng='Default',fg_height = 6,l_width=1.5,sv_fig=False,sv_lab='None',sv_format='pdf',sv_dir='Current',plt_title = 'None',plt_show=False):
    import os
    golden = (1 + 5 ** 0.5) / 2
    #Processing data
    c = arr[fltr[0]]
    w = arr[fltr[1]]
    h = arr[fltr[2]]
    cols = np.array(['b-','g','m-'])
    #Calculating histogram
    if rng=='Default':
        rng = [np.amin(arr),np.amax(arr)]
    fg_width = fg_height*golden
    fig = plt.figure(figsize=[fg_width,fg_height])
    ax = fig.add_subplot(111)
    freq_arr = np.zeros([3,no_bins])
    for i in np.arange(3):
        if i==0:
            freqs,bns = np.histogram(c,range=rng,bins=no_bins)
            freq_arr[0] = freqs
        elif i==1:
            freqs,bns = np.histogram(w,range=rng,bins=no_bins)
            freq_arr[1] = freqs
        elif i==2:
            freqs,bns = np.histogram(h,range=rng,bins=no_bins)
            freq_arr[2] = freqs
        bins = 0.5*(bns[1:]+bns[:-1])
        ax.plot(bins,freqs,cols[i],linewidth=l_width)
    ax.set_xlabel(xl,fontsize=18)
    ax.set_ylabel(yl,fontsize=18)
    ax.tick_params(labelsize=16,length=5,width=1.5)
    ax.legend(['Cold phase','Warm phase','Hot phase'],fontsize=16)
    if plt_title !='None':
        ax.set_title(plt_title,fontsize=20)
    if sv_fig==True:
        if sv_dir == 'Current':
            sv_dir = os.getcwd()+'/'
        else:
            print('Save directory is '+sv_dir)
        os.chdir(sv_dir)
        plt.savefig(sv_lab+'.'+sv_format)
    if plt_show==True:
        plt.show()
    else:
        plt.close()
    return bins,freq_arr
