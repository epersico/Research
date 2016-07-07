import sys
import numpy as np
import cProfile
import matplotlib.pyplot as plt
import scipy.optimize
import pyqtgraph as pg
import functools
"""
IMPORTANT: Conventions I changed in the SNTM file are as follows
7/6/16: orbit iterates orbitLength times instead of orbitLength -1 
		No longer use singleOrbitPoint, instead define iteratively a 
			function to produce the single point.  
"""
def setk(kvalue):
	"""
	In order for the k value to be passed to the different functions
	in particular the standard map itself we must define k globally.
	For some reason doing it in the environment where you are working 
	doesn't work.  Must do it within MapFunction namespace.
	"""
	global k
	k=kvalue
	print('k=',k)
	return

def Map(z):
	"""
	Takes in an array points and iterates them with the Standard Map.  
	Must be numpy array. NOTE: This changes what is passed. 
	The points should be in shape [[x0,y0],[x1,y1],[x2,y2]]
	init.shape = (n,2) where n is the number of points.
	This will work with only one point as well, however,
	currently it must still be a numpy array.
	"""
	z[:,1] = (z[:,1] - k/(2*np.pi)* np.sin(2*np.pi*z[:,0]))
	z[:,0] = (z[:,0] + z[:,1])
	return z

def orbit(initialConditions,orbitLength):
	"""
	Takes array of shape (n,2) of n initial conditions and returns an
	array of shape (n,2,orbitLength) such that the orbit goes from 0 
	to orbitLength - 1
	"""
	initCondLength= len(initialConditions[:,0])
	orbit = np.zeros((initCondLength,2,orbitLength),dtype='float64')
	orbit[:,:,0]=initialConditions
	print('Calculating orbits....')
	print('There are',initCondLength,'orbits of length', orbitLength)
	for i in range(orbitLength-1):
		orbit[:,:,i+1]=orbit[:,:,i]
		Map(orbit[:,:,i+1])
	print('orbits calculated.')
	return orbit

def singleOrbitPoint(n):
	"""
	Simply iterates the point z n times using the Map
	"""
	tupleofSMs = (Map for i in range(n))
	SMcomposition = compose(tupleofSMs)
	return SMcomposition

def reduceOrbit(orbitLength):
	"""
	takes orbitLength and returns the appropraite m/2 or (m+1)/2
	depending on parity. 
	"""
	m = orbitLength;
	if m%2 ==0:
		m = m/2
	else:
		m = (m+1)/2
	return int(m)

def xcoord(z):
	"""
	returns the x component of the orbit
	"""
	return z[:,0]

def ycoord(z):
	"""
	returns the y component of the orbit
	"""
	return z[:,1]

def evenRootFunction(z):
	"""
	If the orbit length is even halfway through the orbit
	it is guarenteed to lie on the s1 or s3 symmetry lines.  
	The sin function removes the ambiguity between these two 
	Wrote for the purpose of function composition
	"""
	x = xcoord(z)
	y = ycoord(z)
	return np.sin(2*np.pi*x)

def oddRootFunction(z):
	"""
	If the orbit length is odd halfway (rounded up) through the orbit
	it is guarenteed to lie on the s2 or s4 symmetry lines.  
	The sin function removes the ambiguity between these two 
	Wrote for the purpose of function composition
	"""
	x = xcoord(z)
	y = ycoord(z)
	return np.sin(2*np.pi*(x - .5*(y-1)))

def rootFunction(orbitLength):
	"""
	Given the length of the orbit returns a function of the initial y
	coordinate.  The meat of this is plugging in the correct iterations
	of the standard map into the even or odd root function.
	"""
	m = reduceOrbit(orbitLength)
	tupleofSMs = (Map for i in range(m-1))
	#SMcomposition = compose(tupleofSMs)
	SMcomposition = Map
	for SM in tupleofSMs:
		SMcomposition = compose((SMcomposition,SM))

	if orbitLength%2:
		func = compose((oddRootFunction,SMcomposition,zeroY))
	else:
		func = compose((evenRootFunction,SMcomposition,zeroY))
	return func

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

def windingNumber(z):
	x = xcoord(z)
	y = ycoord(z)
	return z

def tangentMap(z):
	x = z[0]
	return np.array([[1 - k*np.cos(2*np.pi*x),1],[-k*np.cos(2*np.pi*x),1]])

def residue(m1,m2):
	"""
	This is the beginning of my residue function.  m1 and m2 define
	the winding number of the orbit such that r = m1/m2.
	"""
	r = m1/m2
	endpoint = 0
	counter = 0
	stepCounter =0
	stepsize = .1
	func = rootFunction(m2)		
	orbitpoint = singleOrbitPoint(m2)
	while abs(endpoint-m1) >= 1e-6:
		if counter > 0:
			r =(r-stepsize)%1
		counter += 1
		stepCounter += 1
		try:
			root = scipy.optimize.brentq(func,r-stepsize/2,r+stepsize/2,maxiter=200)
			#root = scipy.optimize.newton(func,r,tol=1e-4,maxiter=50)
			endpoint = xcoord(orbitpoint(np.array([[0,root]])))[0]
		except ValueError:
			print('Values did not bracket roots')
		except RuntimeError:
			print('oops the values got away from us')
		if not counter%10:
			print('Attempt:',counter)
		if stepCounter>1/stepsize:
			stepsize=stepsize/10
			stepCounter = 0
			print('stepsize',stepsize)
			print('moving on to smaller step size')

	print('Went through',counter,'iterations to find correct root')
	print('for winding number',m1/m2)
	print('Final root was',root)
	print('final point on orbit was',orbitpoint(np.array([[0,root]])))

	print('Calculating the rest of the orbit for residue calculation...')
	orbitarray = orbit(np.array([[0,root]]),m2)
	tangentMatrix = np.array([[1,0],[0,1]])
	for i in range(m2):
		tangentMatrix = np.dot(tangentMap(orbitarray[0,:,i]),tangentMatrix)
	trace = np.trace(tangentMatrix)
	res = .25*(2-trace)
	print('residue was:',res)
	meanRes = (abs(res)*4)**(1/m2)
	print('mean residue was:',meanRes)
	return res



def pyplot(orbitarray):
	"""
	Plots the orbits that are passed in the array.  Assumed to be of shape
	(number of orbits, 2, length of orbits).  
	"""
	print('Beginning plots...')
	orbitarray = orbitarray%1
	#colors = abs(1-2*orbitarray[0,1,:])
	colors = orbitarray[0,1,:]
	print(colors.shape)
	for i in range(len(orbitarray[:,0,0])):
		plt.scatter(orbitarray[i,0,:],orbitarray[i,1,:],marker='o',lw=0.0,s=1,c=colors,rasterized=True)
		print('plotting orbit',i+1,'of',len(orbitarray[:,0,0]))
	print('Plots are done.  Now saving...')
	plt.ylim([0,1])
	plt.xlim([0,1])
	plt.savefig('foo.png',bbox_inches='tight')
	print('Plot is saved! Yay!')
	return



def main():
	args = sys.argv[1:]
	if not args:
		print('usage: --k kvalue --orbitLength orbitlength --plot')


	##Test section of code here.  
	if '--test' in args:
		print('test')





	##Defining the k value
	global k 
	if '--k' in args:
		index = args.index('--k')
		del args[index]
		k = args.pop(index)
		k= float(k)
	else:
		k = 0
	print('k=',k)

	if '--residue' in args:
		index = args.index('--residue')
		del args[index]
		m1 = int(args.pop(index))
		m2 = int(args.pop(index))
		print('Finding residue for winding number:',m1,'/',m2)
		res = residue(m1,m2)
		print('Calculated reside is:',res)

	if '--orbitLength' in args:
		index = args.index('--orbitLength')
		del args[index]
		orbitLength=int(args.pop(index))
	else:
		orbitLength=100
	print('Orbit Length is:',orbitLength)

	

	if '--plot' in args:
		init = np.random.rand(200,2)
		print('Initial conditions set.')
		orbitarray = orbit(init,orbitLength)
		pyplot(orbitarray)
	#	pyqtPlot(orbitarray)

	
	
def pyqtPlot(orbitarray):
	print('Beginning plots...')
	plotwidget = pg.plot(title='Standard Map test plot')
	for i in range(len(orbitarray[:,0,0])):
		plotwidget.plot(orbitarray[i,0,:],orbitarray[i,1,:],pen=None,symbol='o',size=.01)
		print('plotting orbit',i,'of',len(orbitarray[:,0,0]))
	print('Plots are done.  Now saving...')
	print('Plot is saved! Yay!')
	input()
	return


if __name__ == '__main__':
#	cProfile.run('main()')
	main()