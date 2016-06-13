import sys
import numpy as np
import cProfile
import matplotlib.pyplot as plt
import scipy.optimize
import pyqtgraph as pg
import functools

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

def standardMap(z):
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
		standardMap(orbit[:,:,i+1])
	print('orbits calculated.')
	return orbit

def singleOrbitPoint(n,z):
	tupleofSMs = (standardMap for i in range(n))
	SMcomposition = compose(tupleofSMs)
	return SMcomposition(z)

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
	return z[:,0]
def ycoord(z):
	return z[:,1]

def evenRootFunction(z):
	x = xcoord(z)
	y = ycoord(z)
	return np.sin(2*np.pi*x)

def oddRootFunction(z):
	x = xcoord(z)
	y = ycoord(z)
	return np.sin(2*np.pi*(x - .5*(y-1)))

def rootFunction(orbitLength):
	"""
	Takes orbit length and returns the proper number of compositions
	of the standard map for rootfinding purposes. 
	"""
	m = reduceOrbit(orbitLength)
	tupleofSMs = (standardMap for i in range(m))
	SMcomposition = compose(tupleofSMs)
	if orbitLength%2:
		func = compose((oddRootFunction,SMcomposition,zeroY))
	else:
		func = compose((evenRootFunction,SMcomposition,zeroY))
	return func
def zeroY(y):
	return np.array([[0,y]])

def compose(functions):
	return functools.reduce(lambda f,g: lambda x: f(g(x)),functions, lambda x: x)

def residue(m1,m2):
	r = m1/m2
	r=.58
	endpoint = xcoord(singleOrbitPoint(m2,np.array([[0,0]])))[0]
	counter = 0
	func = rootFunction(m2)
	while abs(endpoint-m1) >= .001:
		if counter > 0:
			r =(r+.0001)%1
		counter += 1
		try:
			#root = scipy.optimize.brentq(func,0+r,1)
			root = scipy.optimize.newton(func,r)
			orbitpoint = singleOrbitPoint(m2,np.array([[0,root]]))
			endpoint = xcoord(orbitpoint)[0]
		except RuntimeError:
			print('oops the values got away from us')

	print('Went through',counter,'iterations to find correct root')
	print('for winding number',m1/m2)
	print('Final root was',root)
	print('final point on orbit was',orbitpoint)
	return



def pyplot(orbitarray):
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
		sys.exit()


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
		residue(m1,m2)

	if '--orbitLength' in args:
		index = args.index('--orbitLength')
		del args[index]
		orbitLength=int(args.pop(index))
	else:
		orbitLength=100
	print('Orbit Length is:',orbitLength)

	init = np.random.rand(200,2)
	print('Initial conditions set.')


	orbitarray = orbit(init,orbitLength)

	if '--plot' in args:
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