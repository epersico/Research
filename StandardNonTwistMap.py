import sys
import numpy as np
import cProfile
import matplotlib.pyplot as plt
import scipy.optimize
#import pyqtgraph as pg
import functools
from bokeh.plotting import figure, show, output_file, output_server, curdoc
import bokeh
from fractions import gcd
"""
import datashader as ds
from datashader import transfer_functions as tf
import pandas as pd
from datashader.bokeh_ext import InteractiveImage
"""

def xcoord(z):
	"""
	returns the x component of the orbit
	"""
	return z[0]

def ycoord(z):
	"""
	returns the y component of the orbit
	"""
	return z[1]

def zeroY(y):
	"""
	Used in function composition to plug in an array with zero intial x 
	and arbitrary y.
	"""
	return np.array([[0,y]])

def compose(functions):
	"""
	Defined recursively.  Provide a tuple of function names and this returns 
	a function which is their composition. For more information look here:
	https://mathieularose.com/function-composition-in-python/
	"""
	return functools.reduce(lambda f,g: lambda x: f(g(x)),functions, lambda x: x)

def subtraction(a,b):
	return a-b

"""
These are initialization functions.
"""

def seta(avalue):
	"""
	In order for the rest of these functions to use the a and b values 
	that I want they must be set inside here.  This function sets the 
	value of a.  Similarly with setb
	"""
	global a
	a = float(avalue)
	return

def setb(bvalue):
	"""
	See seta
	"""
	global b 
	b = float(bvalue)
	return

def Map(z):
	"""
	Takes in an array points and iterates them with the SNTM.  
	Must be numpy array. This does not mod.
	NOTE: This changes what is passed. 
	The points should be in shape [[x0,y0],[x1,y1],[x2,y2]]
	init.shape = (n,2) where n is the number of points.
	This will work with only one point as well, however,
	currently it must still be a numpy array.
	"""
	z[:,1] = z[:,1] - b*np.sin(2*np.pi*z[:,0])
	z[:,0] = (z[:,0] + a*(1-z[:,1]**2))
	return z

def tangentMap(z):
	x = z[0]
	y = z[1]
	return np.array([[1+4*a*b*np.pi*np.cos(2*np.pi*x)*(y-b*np.sin(2*np.pi*x)),-2*a*(y-b*np.sin(2*np.pi*x))],[-2*b*np.pi*np.cos(2*np.pi*x),1]])

def R(z,n):
	z[0] -= n
	return z

def MapSinglePoint(z):
	"""
	Takes in an array points and iterates them with the SNTM.  
	Must be numpy array. This does not mod.
	NOTE: This changes what is passed. 
	The points should be in shape [[x0,y0],[x1,y1],[x2,y2]]
	init.shape = (n,2) where n is the number of points.
	This will work with only one point as well, however,
	currently it must still be a numpy array.
	"""
	z[1] = z[1] - b*np.sin(2*np.pi*z[0])
	z[0] = (z[0] + a*(1-z[1]**2))
	return z

def nonMonotoneCurvey(x):
	return b*np.sin(2*np.pi*x)


"""
Functions for computing and plotting large number of orbts.
"""

def orbit(initialConditions, orbitLength):
	"""
	Takes array of shape (n,2) of n initial conditions and returns an
	array of shape (n,2,orbitLength) such that the orbit goes from 0 
	to orbitLength.  This means if it is a periodic orbit of length equal
	to the orbitLength the first and last entry should be the same

	NOTE: For index purposes: 
	The first index corresponds to the index of the initial condition.  
	The second index chooses x or y 
	The third index is the position on the orbit
	"""
	initCondLength= len(initialConditions[:,0])
	orbit = np.zeros((initCondLength,2,orbitLength+1),dtype='float64')
	orbit[:,:,0]=initialConditions
	print('Calculating orbits using SNTM....')
	print('There are',initCondLength,'orbits of length', orbitLength)
	for i in range(orbitLength):
		if i%1000 == 0:
			print('finished with:',i)
		orbit[:,:,i+1]=orbit[:,:,i]
		Map(orbit[:,:,i+1])
	print('Orbits calculated.')
	return orbit

def orbitSingle(initialConditions, orbitLength):
	"""
	Takes array of shape (n,2) of n initial conditions and returns an
	array of shape (n,2,orbitLength) such that the orbit goes from 0 
	to orbitLength.  This means if it is a periodic orbit of length equal
	to the orbitLength the first and last entry should be the same

	NOTE: For index purposes: 
	The first index corresponds to the index of the initial condition.  
	The second index chooses x or y 
	The third index is the position on the orbit
	"""
	initCondLength= len(initialConditions[:,0])
	orbit = np.zeros((initCondLength,2,orbitLength+1),dtype='float64')
	orbit[:,:,0]=initialConditions
	print('Calculating orbits using SNTM....')
	print('There are',initCondLength,'orbits of length', orbitLength)
	for i in range(orbitLength):
		orbit[:,:,i+1]=orbit[:,:,i]
		Map(orbit[:,:,i+1])
	print('Orbits calculated.')
	return orbit

def Mapn(z,n):
	for i in range(n):
		MapSinglePoint(z)
	return z

def nIterationFunc(n):
	"""
	Returns a function which is the composition of n copies
	of the SNTM.  This is done iteratively to avoid recursion depth issues.
	NOTE: This returns a function, so it can be used for root finding 
	compositions later.  This can also be used to calculate a single 
	point along an orbit without storing all the information.  
	"""
	maptuple = (MapSinglePoint for i in range(n-1))
	mapComp = MapSinglePoint
	for SNTM in maptuple:
		mapComp = compose((mapComp,MapSinglePoint))
	return mapComp

def windingNumber(y):
	"""
	Winding Number function goes here
	"""
	nMax = int(3e6)
	epsilon = 1e-6
	m=int(1e5)
	n=0
	omegas = [0]

	z = [0,y]
	
	for i in range(nMax):
		MapSinglePoint(z)
		omega = z[0]/(i+1)
		omegas.append(omega)
		#omega2 = orbitarray[-2,0]/(i+1)
		
		if abs(omegas[-1]-omegas[-2]) <= epsilon:
			if n == 0:
				n=i
				
			elif i>n+m:
				print('Has been close to itself for',m,'iterations.')	
				sup = max(omegas[:n])
				inf = min(omegas[:n])
				if sup >= omega and inf <= omega:
					print('/n -----/n converged!/n ----/n ')
					break
		else:
			n=0
		if i == nMax-1:
			print('failed to converge')
			omega =.6025
	return omega



"""
Here is some code for the creation of the root function.  f2 in Fuchss dissertation.  
See pg. 105 (pdf 120)
"""
def xsubi(i,y):
	if i == 1:
		return 0
	elif i ==2:
		return 1/2
	elif i == 3:
		return .5*a*(1-y**2)
	elif i == 4:
		return .5*a*(1-y**2)+.5

def xofyi(y,i):
	func = functools.partial(xsubi,i)
	return func(y)

def mapSelectorFunction(j,m2):
	if j == 0 or j== 3:
		return .5*(m2+1)
	else:
		return .5*(m2-1)

def xSelectorFunction(j,m1):
	if j%2==1:
		return .5*(m1+1)
	else:
		return .5*(m1-1)

def totalSelectorFunction(i,m1,m2):
	if m2%2 == 0:
		j = (3-i)%4
		Riteration = xSelectorFunction(j,m1)
		mapIteration = m2/2
	else:
		if m1%2==1:
			j = (5-i)%4
			Riteration = xSelectorFunction(j,m1)
		else: 
			j = (i+2)%4
			Riteration = m1/2
		mapIteration = mapSelectorFunction(j,m2)
	if j == 0: 
		j=4
	print('n,m before being turned into ints',Riteration,mapIteration)
	return [int(j),int(Riteration),int(mapIteration)]

def finalFunc(x):
	return x/(1+abs(x))

def rootFinding(i,q,p,updown='up',plot=False,resetBounds=False,lowerBound = -1,upperBound = 1):
	"""
	For winding number q/p.  In p iterations this will go around the torus q times. 
	"""
	root =0
	stepsize = 1e-3
	stepCounter = 0
	endpoint = 0
	error = False



	j,n,m = totalSelectorFunction(i,q,p)
	xofy = functools.partial(xsubi,i)
	xofyFin = functools.partial(xsubi,j)

	print('Starting SLN:',i,'\nEnd Symmetry Line Number:', j,'\nq (numerator) is',q,'\np (denominator)',p)

	def rootfunction(y):
		z = [xofy(y),y]
		Mapn(z,m)
		xiPrime = z[0] - n 
		yiPrime = z[1]
		
		xjPrime = xofyFin(yiPrime) 
		return xiPrime - xjPrime

	func = compose((finalFunc,rootfunction))

	if plot:
		title = 'root function for winding number '+str(q)+'/'+str(p)+' '+updown+' sln:'+str(i)
		ys = np.linspace(lowerBound,upperBound,10000)
		fs = [func(y) for y in ys]
		f = figure(title = title,x_range=(lowerBound,upperBound),y_range=(-1,1))
		f.circle(ys,fs)
		show(f)

	if resetBounds:
		lowerBound = float(input('Please specify a new Lower Bound for the root search:'))
		upperBound = float(input('Please specify a new Upper Bound for the root search:'))
		
	else:
		print('Tryint to intelligently set bounds...')
		print('first have to calculate the root function...')
		ys = np.linspace(lowerBound,upperBound,10000)
		fs = [func(y) for y in ys]
		fsReverse = fs[:]
		fsReverse.reverse()
		threshold = .99 
		print('done with that...  Now to find bounds...')
		try:
			value = next(f for f in fs if f > threshold)
			index = fs.index(value)
			lowerBound = ys[index]
			value = next(f for f in fsReverse if f > threshold)
			index = fs.index(value)
			upperBound = ys[index]
		except StopIteration:
			threshold -= .01
	print('Root search will now take place between',lowerBound,'and',upperBound)

	if updown == 'up':
		r = upperBound
		plusminus = -1
	elif updown == 'down':
		r = lowerBound
		plusminus = 1
	else:
		print('Please specify "up" or "down" for the keyword "updown".')

	l = abs(upperBound-lowerBound)
	stepsize=l/10

	while not root or abs(root)>1:
	#while not root or abs(root)>.6 or abs(endpoint - q) > 1e-6:
		try:
			root = scipy.optimize.brentq(func,r-stepsize/2,r+stepsize/2,maxiter=500,xtol=1e-16)
			print('root is',root)
			z=[xofy(root),root]
			Mapn(z,p)
			endpoint = z[0]
			print('The r value that worked was:',r,'with stepsize:',stepsize)
			print('endpoint is',endpoint)
		except ValueError:
			pass
			#print('Values did not bracket roots')
		stepCounter += 1
		r=(r+plusminus*stepsize)

		if r>upperBound:
			r=r-l
		if r<lowerBound:
			r = r+l
		if stepCounter>l/stepsize:
			stepsize=stepsize/10
			stepCounter = 0
			print('Went through specified range, didn\'t find a root. Moving on to stepsize:',stepsize)
		if stepsize < l/(1e4) or stepsize < 1e-7:
			print('No root found')
			error = True 
			break
	print('for winding number',q,'/',p,updown)
	print('Final root was',root)

	if not error:
		res = residue([xofy(root),root],p)
	else:
		res = 0

	return [root,res,error]

def residue(z,n):
	print('Calculating the rest of the orbit for residue calculation...')
	orbitarray = orbit(np.array([z]),n)
	print('final point on orbit was',orbitarray[0,:,-1])
	orbitarray[0,0,:] = orbitarray[0,0,:]%1 
	print('the last point in the res calc is',orbitarray[0,:,n-1])
	tangentMatrix = np.array([[1,0],[0,1]])
	for i in range(n):
		tangentMatrix = np.dot(tangentMap(orbitarray[0,:,i]),tangentMatrix)
	trace = np.trace(tangentMatrix)
	res = .25*(2-trace)
	print('residue is:',res)
	return res


"""
Bifurcation curves.
"""
def makeBifurcationCurve(i,q,p):
	"""
	For winding number q/p.  In p iterations this will go around the torus q times. 
	"""
	root =0
	r = q/p
	stepsize = .1
	stepCounter = 0
	endpoint = 0

	j,n,m = totalSelectorFunction(i,q,p)
	xofy = functools.partial(xsubi,i)
	xofyFin = functools.partial(xsubi,j)

	print('Starting SLN:',i,'\nEnd Symmetry Line Number:', j,'\nq (numerator) is',q,'\np (denominator)',p)

	def fofb(b):
		setb(b)
		z = [xofy(y),y]
		Mapn(z,m)
		xiPrime = z[0] - n 
		yiPrime = z[1]
		
		xjPrime = xofyFin(yiPrime) 
		return xiPrime - xjPrime

	func = compose((finalFunc,fofb))	
	return func



def main():
	args = sys.argv[1:]
	output_file("test.html", title="test")

	if not args:
		print('usage description to come later!')

	if '--test' in args:
		output_file("test.html", title="test")
		
		seta(0.94)
		global y
		y=-.3
		func = rootFinding(1,7,8)

		bs = np.linspace(0,.3,100)
		fs = [func(b) for b in bs]
		f = figure(title = 'winding number test',x_range=(-1,1),y_range=(0,.3))
		f.circle(fs,bs)
		show(f)
	
		sys.exit()


	if '--density' in args:
		"""
		The goal
		"""
		def density(b):
			n = 840
			seta(.615)
			setb(b)
			rootsFound = []
			for i in range(n-1):
				q=i+1
				a = gcd(q,n)
				p = int(n/a)
				q = int(q/a)
				if all(rootsFound[-4:]) and len(rootsFound) > 4:
					rootsFound.append(True)
				else:
					result = rootFinding(1,q,p,updown='up',lowerBound=-.2,upperBound=.2)
					rootsFound.append(result[2])
					result = rootFinding(1,q,p,updown='down',lowerBound=-.2,upperBound=.2)
					rootsFound.append(result[2])
				


			print(rootsFound)
			return rootsFound.count(False)

		bs = np.linspace(0,1,20)
		fs = [density(b) for b in bs]
		f = figure(title='number density test',x_range=(0,1))
		f.circle(bs,fs)
		show(f)
		sys.exit()
		

	if '--manyroots1' in args:
		seta(0.686048)
		setb(.742489259544)


		ilist = [1,3,1,3,1,3,3,1,3,1]
		l = len(ilist)
		qs = [1,21,3,55,8,144,1,21,8,144]
		ps = [2,34,5,89,13,233,2,34,13,233]
		updowns = ['down','up','down','up','down','up','up','down','up','down']
		residues = []
		for i in range(l):
			result = rootFinding(ilist[i],qs[i],ps[i],updown=updowns[i],plot=True)
			residues.append(result[1])

		for i in range(l):
			print('Orbit from symmetry line',ilist[i],'winding number',qs[i],'/',ps[i]
				,'had residue',residues[i])

	if '--manyroots2' in args:
		seta(0.686048)
		setb(.742489259544)


		ilist = [1,3,1,3,1,3,3,1,3,1]
		l = len(ilist)
		qs = [377,6765,987,17711,2584,46368,377,6765,2584,46368]
		ps = [610,10946,1597,28657,4181,75025,610,10946,4181,75025]
		updowns = ['down','up','down','up','down','up','up','down','up','down']
		residues = []


		for i in range(l):
			result = rootFinding(ilist[i],qs[i],ps[i],updown=updowns[i])
			residues.append(result[1])

		for i in range(l):
			print('Orbit from symmetry line',ilist[i],'winding number',qs[i],'/',ps[i]
				,'had residue',residues[i])

	if '--rftest' in args:
		output_file("test.html", title="test")
		
		i = 3
		m1 = 17711
		m2 = 28657

		seta(0.686048)
		setb(.742489259544)
		rootFinding(i,m1,m2,updown='up')


	if '--windingnumbertest' in args:
		print('Running only the test section of code!')

		output_file("test.html", title="test")

		seta(.923)
		setb(0.22465)
		
		ys = np.linspace(-.3,.2,400)

		print('starting calculation of winding numbers')
		omegas = [windingNumber(y) for y in ys]
		print('finished with that! Now to plot!')

		p = figure(title = 'winding number test',x_range=(-.3,.2),y_range=(.57,.605))
		p.circle(ys,omegas)
		show(p)


		sys.exit()

	"""
	Below is the code where I was working on datashader stuff.  Where I left it it
	it was saving the image very quickly.  However, I was not getting the zooming 
	functionality of InteractiveImage.  This is because InteractiveImage is set up 
	to work in a jupyter notebook and not in bokeh server mode.  There were promising 
	things I found on google that might resolve this.  
	"""
	if '--dstest' in args:

		seta(.615)
		setb(.4)

		#output_file("datashadertest.html", title="DStest")
		output_server('hover')


		initconds = np.random.rand(5e2,2)-.5
		orbitarray = orbit(initconds,int(1e4))

		x_range, y_range = ((-.5,.5),(-.5,.5))

		plot_width = int(750)
		plot_height = plot_width//1.3

		background = 'black'
		export = functools.partial(ds.utils.export_image, export_path="export", background=background)

		points = np.array([[],[]])
		print('putting the orbits into arrays for plotting...')
		for i in range(len(orbitarray[:,0,0])):
			points = np.append(points,orbitarray[i,:,:],axis=1)
		points[0,:]=points[0,:]%1-.5


		df = pd.DataFrame(points.T,columns=['x','y'])
		print('finished. Here are the last three rows:', df.tail(3))

		def create_image(x_range,y_range,w=plot_width,h=plot_height):
			canvas = ds.Canvas(plot_width=plot_width,plot_height=plot_height,
			x_range=x_range, y_range=y_range)
			agg = canvas.points(df,'x','y')
			img = tf.interpolate(agg,cmap=ds.colors.Hot,how='log')
			return tf.dynspread(img,threshold=0.5,max_px=4)

		def base_plot(tools='pan,wheel_zoom,reset',plot_width=plot_width,plot_height=plot_height, **plot_args):
			p = figure(tools=tools, plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range, outline_line_color=None,min_border=0, min_border_left=0, min_border_right=0,min_border_top=0, min_border_bottom=0, **plot_args)
			p.axis.visible = True
			p.xgrid.grid_line_color = None
			p.ygrid.grid_line_color = None
			return p

		print('making the image now...')
		session = bokeh.client.push_session(curdoc())

		p = base_plot(background_fill_color=background)
		export(create_image(x_range,y_range),'testexport')
		print('image saved.')
		
		#curdoc().add_periodic_callback(create_image, 50)
		session.show()
		InteractiveImage(p,create_image,throttle=200)
		session.loop_until_closed()

	

	
	if '--a' in args:
		index = args.index('--a')
		del args[index]
		a = args.pop(index)
		seta(a)
		print('a has been set to:',a)
	

	if '--b' in args:
		index = args.index('--b')
		del args[index]
		b = args.pop(index)
		setb(b)
		print('b has been set to:',b)
	
	if '--orbitLength' in args:
		index = args.index('--orbitLength')
		del args[index]
		orbitLength = args.pop(index)
		orbitLength = int(orbitLength)
		initconds = np.random.rand(200,2)
		orbitarray = orbit(initconds, orbitLength)

	if '--plot' in args:
		if not orbitLength:
			print('please specify orbit length using command line flag --orbitLength')
			sys.exit()
		#output_file("test.html", title="SNTM plots!!!")
		output_server('hover')
		p = figure(title ='testplot!',x_range=(0,1),y_range=(-.5,.5),webgl=True)
		for i in range(len(orbitarray[:,0,0])):
			p.circle(orbitarray[i,0,:],orbitarray[i,1,:],size=.1)
		print('showing plot...')
		show(p)
		print('plot should be showing')
		
	return

if __name__ == '__main__':
	cProfile.run('main()')
#	main()