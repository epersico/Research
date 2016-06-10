import sys
import numpy as np
import cProfile
import matplotlib.pyplot as plt
import scipy
import pyqtgraph as pg
import functools

def standardMap(z):
	if len(z) >= 1:
		z[:,1] = (z[:,1] - k/(2*scipy.pi)* np.sin(2*scipy.pi*z[:,0]))
		z[:,0] = (z[:,0] + z[:,1])
	
	else:
		z[1] = (z[1] - k/(2*scipy.pi)* np.sin(2*scipy.pi*z[0,:]))
		z[0] = (z[0] + z[1])%1
	return z

def orbit(initialConditions,orbitLength):
	initCondLength= len(initialConditions[:,0])
	orbit = np.zeros((initCondLength,2,orbitLength),dtype='float64')
	orbit[:,:,0]=initialConditions
	print(initialConditions)
	print(orbit[:,:,0])
	print(orbit[:,:,0].shape)
	print('Calculating orbits....')
	print(initialConditions,'0')
	for i in range(orbitLength-1):
		orbit[:,:,i+1]=standardMap(orbit[:,:,i])
		print(orbit[:,:,i+1],i+1)
	print('orbits calculated.')
	return orbit

def compose(functions):
	return functools.reduce(lambda f,g: lambda x: f(g(x)),functions, lambda x: x)

def pyplot(orbitarray):
	print('Beginning plots...')
	colors = abs(1-2*orbitarray[1,:,0])
	for i in range(len(orbitarray[0,0,:])):
		plt.scatter(orbitarray[0,:,i],orbitarray[1,:,i],c=colors,s=1,rasterized=True)
		print('plotting orbit',i,'of',len(orbitarray[0,0,:]))
	print('Plots are done.  Now showing...')
	plt.show()
	print('Plot is shown! Yay!')
	return

def pyqtPlot(orbitarray):
	print('Beginning plots...')
	plotwidget = pg.plot(title='Standard Map test plot')
	for i in range(len(orbitarray[0,:,0])):
		plotwidget.plot(orbitarray[0,i,:],orbitarray[1,i,:],pen=None,symbol='o',size=.01)
		print('plotting orbit',i,'of',len(orbitarray[0,:,0]))
	print('Plots are done.  Now showing...')
	print('Plot is shown! Yay!')
	input()
	return
def main():

	args = sys.argv[1:]
	if not args:
		print('usage: --k kvalue --orbitLength orbitlength --plot')

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


	if '--orbitLength' in args:
		index = args.index('--orbitLength')
		del args[index]
		orbitLength=int(args[index])
	else:
		orbitLength=100
	print('Orbit Length is:',orbitLength)

	init = np.random.rand(1000,2)
	print('Initial conditions set.')


	orbitarray = orbit(init,orbitLength)

	if '--plot' in args:
		pyplot(orbitarray)
	#	pyqtPlot(orbitarray)

	
	

if __name__ == '__main__':
#	cProfile.run('main()')
	main()