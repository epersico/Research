import sys
import numpy as np
import cProfile
import matplotlib.pyplot as plt
import scipy.optimize
import pyqtgraph as pg
import functools
from bokeh.plotting import figure, show, output_file, output_server

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
		orbit[:,:,i+1]=orbit[:,:,i]
		Map(orbit[:,:,i+1])
	print('Orbits calculated.')
	return orbit

def nIterationFunc(n):
	"""
	Returns a function which is the composition of n copies
	of the SNTM.  This is done iteratively to avoid recursion depth issues.
	NOTE: This returns a function, so it can be used for root finding 
	compositions later.  This can also be used to calculate a single 
	point along an orbit without storing all the information.  
	"""
	maptuple = (Map for i in range(n-1))
	mapComp = Map
	for SNTM in maptuple:
		mapComp = compose((mapComp,SNTM))
	return mapComp

def windingNumber(y):
	"""
	Winding Number function goes here
	"""
	n = 100000
	initconds = np.array([[0,y]])
	orbitarray = orbit(initconds,n)
	'''
	func = nIterationFunc(n)
	func = compose((func,zeroY))
	func = compose((xcoord,func))
	'''
	omega = orbitarray[0,0,-1]
	omega = omega/n
	return omega
	

def main():
	args = sys.argv[1:]
	if not args:
		print('usage description to come later!')

	if '--test' in args:
		print('Running only the test section of code!')

		output_file("test.html", title="test")

		seta(.615)
		setb(.4)
		
		ys = np.random.rand(1000)-.5
		omegas = np.random.rand(1000)
		for i in range(len(omegas)):
			omegas[i] = windingNumber(ys[i])

		p = figure(title = 'winding number test')
		p.circle(ys,omegas)
		show(p)


		sys.exit()

	if '--orbitLength' in args:
		index = args.index('orbitLength')
		del args[index]
		orbitLength = args.pop(index)
		initconds = np.random.rand(200,2)-.5
		orbitarray = orbit(initconds, orbitLength)
	
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
	
	if '--plot' in args:
		if not orbitLength:
			print('please specify orbit length using command line flag --orbitLength')
			sys.exit()
		#output_file("test.html", title="SNTM plots!!!")
		output_server('hover')
		p = figure(title ='testplot!',x_range=(0,1),y_range=(-.5,.5),webgl=True)
		for i in range(len(orbitarray[:,0,0])):
			p.circle(orbitarray[i,0,:],orbitarray[i,1,:],size=.1)
		print('showing plot')
		show(p)
		print('plot should be showing')
		
	return

if __name__ == '__main__':
	main()