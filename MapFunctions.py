import sys
import numpy as np
import cProfile
import matplotlib.pyplot as plt
import scipy

def standardMap(z):
##	if len(z) > 2:
	z[1,:] = (z[1,:] - k/(2*scipy.pi)* np.sin(2*scipy.pi*z[0,:]))%1
	z[0,:] = (z[0,:] + z[1,:])%1
	
##	z[:,1] = z[:,1] + k * np.sin(z[:,1])
##	z[:,1] = z[:,1] + z[:,1]
##	else:
##		z[1] = z[1] + k * np.sin(z[0])
##		z[0] = z[0] + z[1]
	return z
def orbit(initialConditions,orbitLength):
	initCondLength= len(initialConditions[:,0])
	orbit = np.zeros((2,initCondLength,orbitLength),dtype='float64')
	orbit[:,:,0]=initialConditions.T
	print(orbit[:,:,0].shape)
	print('Calculating orbits....')
	for i in range(orbitLength-1):
		orbit[:,:,i+1]=standardMap(orbit[:,:,i])
	print('orbits calculated.')
	return orbit

def pyplot(orbitarray):
	print('Beginning plots...')
	for i in range(len(orbitarray[0,0,:])):
		plt.scatter(orbitarray[0,:,i],orbitarray[1,:,i],c=orbitarray[1,:,0],rasterized=True)
	print('Plots are done.  Now showing...')
	#plt.show()
	print('Plot is shown! Yay!')
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

	init = np.random.rand(100,2)
	print('Initial conditions set.')

	orbitarray = orbit(init,orbitLength)

	if '--plot' in args:
		pyplot(orbitarray)

	
	

if __name__ == '__main__':
	cProfile.run('main()')
##	main()