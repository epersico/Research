import sys
import numpy as np
import cProfile
import matplotlib.pyplot as plt
import scipy.optimize
#import pyqtgraph as pg
import functools
from bokeh.plotting import figure, show, output_file, output_server, curdoc
import bokeh
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
	print('a has been set to:',a)
	return

def setb(bvalue):
	"""
	See seta
	"""
	global b 
	b = float(bvalue)
	print('b has been set to:',b)
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
	return

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
		return .5*a*(1-y**2)+1/2

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

def rootFunction(i,m1,m2):
	j, n, m  = totalSelectorFunction(i,m1,m2)
	print('j,n,m:', j,n,m)
	xofy = functools.partial(xsubi,i)
	xofyFin = functools.partial(xsubi,j)
	"""
	The step that follows is very confusing.  functools.partial lets one specidfy some 
	of the parameters.  In this case I use it on itself and specify subtraction.  
	now func = partialFunc(y) = y - ? and the ? still remains to be specified
	now func(x) = y - x.  I use this basic function combined with further compositions
	to make the requisite root function.
	"""
	partialFunc = functools.partial(functools.partial,subtraction)

	mapIterationsFunc = nIterationFunc(m)
	RIterationsFunc = functools.partial(R,n=n)
	

	def zofy(y):
		return[xofy(y),y]

	zPrimeFunc = compose((RIterationsFunc,mapIterationsFunc,zofy))
	xPrimeFunc = compose((xcoord,zPrimeFunc))
	yPrimeFunc = compose((ycoord,zPrimeFunc))

	def f2(y):
		func = compose((partialFunc,xPrimeFunc))
		func = func(y)
		func = compose((func,xofyFin,yPrimeFunc))
		return func(y)

	func = compose((finalFunc,f2))
	return func


def residue(i,m1,m2,upordown):
	
	"""
	This is the beginning of my residue function.  m1 and m2 define
	the winding number of the orbit such that r = m1/m2.
	"""

	r = m1/m2
	r = (1-r/a)**.5
	if upordown == 1:
		r=-1*r
	endpoint = 0
	counter = 0
	stepCounter =0
	stepsize = .001
	root = 0
	func = rootFunction(i,m1,m2)	
	m2Function = nIterationFunc(m2)
	xofy = functools.partial(xsubi,i)
	while not root or abs(root)>.5:
		try:
			root = scipy.optimize.brentq(func,r-stepsize/2,r+stepsize/2,maxiter=200,xtol=1e-14,rtol=1e-14)
			print('root is',root)
			z=[xofy(root),root]
			Mapn(z,m2)
			endpoint = z[0]
			print('endpoint is',endpoint)
		except ValueError:
			pass
			#print('Values did not bracket roots')
		stepCounter += 1
		r=(r+stepsize)%1-.5
		if stepCounter>1/stepsize:
			stepsize=stepsize/10
			stepCounter = 0
			print('stepsize',stepsize)
		if stepsize < 1e-6:
			print('No root found')
			sys.exit()
	print('for winding number',m1/m2)
	print('Final root was',root)
	

	
	print('Calculating the rest of the orbit for residue calculation...')
	orbitarray = orbit(np.array([[xofy(root),root]]),m2)
	print('final point on orbit was',orbitarray[0,:,-1])
	orbitarray[0,0,:] = orbitarray[0,0,:]%1 
	print('the last point in the res calc is',orbitarray[0,:,m2-1])
	tangentMatrix = np.array([[1,0],[0,1]])
	for i in range(m2):
		tangentMatrix = np.dot(tangentMap(orbitarray[0,:,i]),tangentMatrix)
	trace = np.trace(tangentMatrix)
	res = .25*(2-trace)
	print('residue is:',res)
	return res

def main():
	args = sys.argv[1:]
	if not args:
		print('usage description to come later!')

	if '--rftest' in args:
		output_file("test.html", title="test")

		i = 3
		m1 = 46368
		m2 = 75025
		ud = -1
	
		seta(0.686048)
		setb(.742489259544)
		
		"""
		func = rootFunction(i,m1,m2)
		ys = np.linspace(-.6,.6,500)
		fs = [func(y) for y in ys]
		p = figure(title = 'winding number test',x_range=(-.6,.6),y_range=(-1,1))
		p.circle(ys,fs)
		show(p)
		"""
		residue(i,m1,m2,ud)

		
		
		sys.exit()

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
	

	if '--b' in args:
		index = args.index('--b')
		del args[index]
		b = args.pop(index)
		setb(b)
	
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
#	cProfile.run('main()')
	main()