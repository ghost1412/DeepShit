	
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.misc
import matplotlib.cm as cm
from PIL import Image
import matplotlib.image as mpimg

class PCA:


	def doPca(self, X):
		mean = np.mean(X, axis = 0)
		X_norm = X - mean
		std  = np.std(X_norm, axis = 0)
		X_norm = X_norm / std
		cov = X_norm.T.dot(X_norm) / X_norm.shape[0]
		U, S, V = np.linalg.svd(cov, full_matrices = True, compute_uv = True)

		return mean, std, X_norm, U, S, V

	def displayPics(self, X, rows, columns, width, height):
		print(X.shape)
		pic = np.zeros((rows * height, columns * width))
		row = 0;  col = 0
		for i in np.arange(100, 100+rows * columns):
			if col == columns:
				row += 1
				col = 0
			image = (X[i].reshape(width, height)).T
			pic[height * row : height * row + image.shape[0], width * col : width * col + image.shape[1]] = image
			col += 1
		plt.figure(figsize = (12, 12))
		#pics = scipy.misc.toimage(pic)
		pics = Image.fromarray(pic)
		plt.imshow(pics, cmap = cm.Greys_r)
		plt.show()


	def projectData(self, X, U, k):
		print(U.shape)  
		U_reduce = U[:, :k]
		z = np.dot(X, U_reduce)

		return z


	def recoverX(self, z, U, k):  
		U_reduce = U[:, :k]
		X_rec = np.dot(z, U_reduce.T)

		return X_rec

if __name__ == '__main__':
	losses = []
	p = PCA()
	faces = loadmat('data/ex7faces.mat')  
	X4 = faces['X'] 
	print('X4: {0}'.format(X4.shape))
	p.displayPics(X4, 3, 3, 32, 32)
	X4_mean, X4_std, X4_norm, X4_U, X4_S, X4_V = p.doPca(X4)
	print("X4_U: {0}".format(X4_U.shape))
	p.displayPics(X4_U.T, 3, 3, 32, 32)
	val = [8, 32, 64, 128, 256, 512, 1048]
	for i in val:
		z4 = p.projectData(X4, X4_U, i)
		X4_recov = p.recoverX(z4, X4_U, i)
		print("X4_recov: {0}".format(X4_recov.shape))
		loss = ((X4 - X4_recov) ** 2).mean()
		losses.append(loss)
		p.displayPics(X4_recov, 3, 3, 32, 32)
		
	plt.plot(val, losses)
	plt.xlabel('Dimenstion')
	plt.ylabel('Loss')
	plt.title('Reconstruction loss vs dimentions')
	plt.show()


