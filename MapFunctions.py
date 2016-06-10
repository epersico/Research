import sys
import numpy as np
import cProfile
import matplotlib.pyplot as plt
import scipy

def standardMap(z):
##	if len(z) > 2:
	z[1,:] = (z[1,:] - k * np.sin(2*scipy.pi*z[0,:]))%1
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
	for i in range(orbitLength-1):
		orbit[:,:,i+1]=standardMap(orbit[:,:,i])
	return orbit

def main():
	global k 
	k = .87
	init = np.random.rand(1000,2)
	##init = np.array([[0,.5],[0,.5],[0,.5]])

	orbitLength=10000
	orbitarray = orbit(init,orbitLength)
	for i in range(orbitLength):
		plt.scatter(orbitarray[0,:,i],orbitarray[1,:,i],c=orbitarray[1,:,0],s=.1)
	plt.show()


if __name__ == '__main__':
	cProfile.run('main()')
##	main()