import numpy as np

def _linInterp(x, xp, fp, j):
    d = ((x-xp[j])/(xp[j+1]-xp[j])).reshape(j.shape+(1,)*(fp.ndim-1))
    return (1-d)*fp[j] + d*fp[j+1]

class customLinInterp:
	""" xp = 1d array, fp = ndarray. 
		Linear interpolation with support extrapolate ∈ {'linear', 'Nearest'}"""
	def __init__(self, xp, fp, extrapolate = 'linear', **kwargs):
		self.extrapolate = extrapolate
		self.f = getattr(self, f'extrapolate_{self.extrapolate}')(xp, fp, **kwargs)

	def __call__(self, x, **kwargs):
		return self.f(x, **kwargs)

	def extrapolate_linear(self, xp, fp, **kwargs):
		def interpolator(x, **kwargs):
			xb = np.clip(x, min(xp)+np.finfo(float).eps, max(xp))
			j = np.searchsorted(xp, xb, side = 'left') - 1
			return _linInterp(x, xp, fp, j)
		return interpolator

	def extrapolate_nearest(self, xp, fp, **kwargs):
		def interpolator(x, **kwargs):
			xb = np.clip(x, min(xp)+np.finfo(float).eps, max(xp))
			j = np.searchsorted(xp, xb, side = 'left') - 1
			return _linInterp(xb, xp, fp, j)
		return interpolator
