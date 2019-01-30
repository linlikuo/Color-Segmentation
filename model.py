#
from scipy.stats import multivariate_normal as mvn
import numpy as np
# single gaussian
def mvnpdf(x, mu, sigma):
	#print('x', np.shape(x))
	#print('mu', np.shape(mu))
	#print(mu)
	#print(np.shape(sigma))
	#k = len(sigma)
	#exp_term = -0.5*np.dot((x-mu),np.linalg.inv(sigma))*(x-mu).T
	#print(np.dot((x-mu),np.linalg.inv(sigma))*(x-mu).T)
	#ans = np.exp(exp_term)/(np.sqrt(((2*np.pi)**k)*np.linalg.det(sigma)))
	#return ans

	L = np.linalg.cholesky(np.linalg.inv(sigma))
	e_term = np.exp(-0.5*np.sum(np.square(np.dot(x-mu,L)), axis = 1))
	le = e_term/(np.sqrt(((2*np.pi)**3)*(np.linalg.det(sigma))))
	return le
	
def gau_model(data):
	mean = np.mean(data, axis = 0)
	cov = np.cov(data.T)
	return mean, cov
def single_gaussian(data, model_data):
	mean, sig = gau_model(model_data)
	return mvnpdf(data, mean, sig)


class Single_Gaussian(object):
	def __init__(self, train):
		self.train = train
		self.mean, self.sigma = self._model(self.train)
		
	def _model(self, x):
		mean = np.mean(x, axis = 0)
		cov = np.cov(x.T)
		return mean, cov
	
	def _phi(self, x):
		L = np.linalg.cholesky(np.linalg.inv(self.sigma))
		print(L.shape)
		e_term = np.exp(-0.5*np.sum(np.square(np.dot(x-self.mean,L)), axis = 1))
		le = e_term/(np.sqrt(((2*np.pi)**3)*(np.linalg.det(self.sigma))))
		return le
	
	def show_mean_cov(self):
		return [self.mean, self.sigma]		
				
	def predict(self, x):
		return self._phi(x)




	
