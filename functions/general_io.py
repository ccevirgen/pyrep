def var_check(vname):
	ret = True
	if vname not in globals():
		ret = False
	return ret

def file_check(fname,cdir='.',make=False,del_exist=False,verbose=False):
	import os
	path = '{}/{}'.format(cdir,fname)
	file_exist = os.path.isfile(path)
	message = ''
	if not file_exist:
		message += '{} does not exist in {}.\n'.format(fname,cdir)
		if make:
			os.system("touch {}".format(path))
			file_exist = os.path.isfile(path)
			if file_exist:
				message += '{} created in {}.'.format(fname,cdir)
	else:
		message += '{} already exists in {}.\n'.format(fname,cdir)
		if del_exist:
			os.system("rm {}".format(path))
	if verbose:
		print(message)
	return file_exist

def dir_check(fname,cdir='.',make=False,del_exist=False,verbose=False):
	import os
	if fname[0]=='/':
		path = '{}{}'.format(cdir,fname)
	else:
		path = '{}/{}'.format(cdir,fname)
	dir_exist = os.path.isdir(path)
	message = ''
	if dir_exist:
		message += '{} already exists in {}.\n'.format(fname,cdir)		
	else:
		message += '{} does not exist in {}.\n'.format(fname,cdir)
		if make:
			os.mkdir(path)
			dir_exist = os.path.isdir(path)
			if dir_exist:
				message += '{} created in {}.'.format(fname,cdir)
	if verbose==True:
		print(message)
	return dir_exist

