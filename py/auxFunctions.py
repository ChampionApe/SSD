import numpy as np
from scipy import interpolate

def _linInterp(x, xp, fp, j):
    d = ((x-xp[j])/(xp[j+1]-xp[j])).reshape(j.shape+(1,)*(fp.ndim-1))
    return (1-d)*fp[j] + d*fp[j+1]

class CustomLinInterp:
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

## A few methods used for calibration purposes:
class Calibrate:
	def __init__(self, m, **kwargs):
		self.m = m
		self.kwargs = self.defaultKwargs
		[self.kwargs.__setitem__(k,v) for k,v in kwargs.items()];

	@property
	def defaultKwargs(self):
		return {'method': 'hybr', 'tol': 1e-5, 'options': {}}

	def onGrid(self, grid, parameter, kt = 'x_unbounded', kT = 'τ_unbounded', maxIter = 10):
		""" Start with a simple iteration that breaks if it does not calibrate. 
			Then, extrapolate from existing solutions and re-try """
		cals, paths, sols = self.onGrid_simpleIte(grid, parameter)
		for i in range(maxIter):
			if all((isinstance(j, np.ndarray) for j in cals.values())):
				break
			else:
				x, cals_x0, sols_x0 = self.approxInitials(cals, sols, kt = kt, kT = kT)
				cals_i, paths_i, sols_i = self.onGrid_simpleIte(x, parameter, cals_x0 = cals_x0, sols_x0 = sols_x0)
				cals.update(cals_i), paths.update(paths_i), sols.update(sols_i)
				print(i)
		return cals, paths, sols

	def approxInitials(self, cals, sols, kt = 'x_unbounded', kT = 'τ_unbounded'):
		cals_x0 = self.extrapolateParametersFromSols(cals, sols)
		x, sols_x0 = self.extrapolateInitialsFromSols(cals, sols, kt = kt, kT = kT)
		return x, cals_x0, sols_x0
	
	def extrapolateParametersFromSols(self, cals, sols):
		""" Return dictionary with extrapolated initial guesses for calibration parameters for 
			entries in cals/sols that are not yet inhabited by solutions (represented by np.ndarrays)"""
		fp = np.vstack([v for k,v in cals.items() if isinstance(v, np.ndarray)])
		xp = np.array([k for k,v in cals.items() if isinstance(v, np.ndarray)])
		x = np.array([k for k,v in cals.items() if not isinstance(v, np.ndarray)])
		cals2d = interpolate.PchipInterpolator(np.sort(xp), fp[np.argsort(xp)], extrapolate = True)(x)
		return {x[i]: cals2d[i,:] for i in range(len(x))}
	
	def extrapolateInitialsFromSols(self, cals, sols, kt = 'x_unbounded', kT = 'τ_unbounded'):
		""" Return dictionary with extrapolated initial guesses for policy functions
			entries in cals/sols that are not yet inhabited by solutions (represented by np.ndarrays)"""
		xp = np.array([k for k,v in cals.items() if isinstance(v, np.ndarray)])	
		x = np.array([k for k,v in cals.items() if not isinstance(v, np.ndarray)])
		sols_x0 = dict.fromkeys(x)
		[sols_x0.__setitem__(xi, {}) for xi in x]; # dictionary with initial guesses 
		for t in self.m.db['txE']:
			fp = np.vstack([sols[xi][t][kt] for xi in xp])
			x0 = interpolate.PchipInterpolator(np.sort(xp), fp[np.argsort(xp)], extrapolate = True)(x)
			[sols_x0[x[i]].__setitem__(t, x0[i,:]) for i in range(len(x))];
		t = self.m.db['t'][-1]
		fp = np.vstack([sols[xi][t][kT] for xi in xp])
		x0 = interpolate.PchipInterpolator(np.sort(xp), fp[np.argsort(xp)], extrapolate = True)(x)
		[sols_x0[x[i]].__setitem__(t, x0[i,:]) for i in range(len(x))];
		return x, sols_x0
	
	def onGrid_simpleIte(self, grid, parameter, cals_x0 = None, sols_x0 = None):
		""" Calibrate model instance for grid of parameters with simple looping. """
		cals, sols, paths = dict.fromkeys(grid), dict.fromkeys(grid), dict.fromkeys(grid)
		for v in grid:
			cals[v], paths[v], sols[v] = self.basicIte(parameter, v, cals_x0 = cals_x0, sols_x0 = sols_x0)
			print(v)
		return cals, paths, sols
	
	def onGrid_simpleBreak(self, grid, parameter, cals_x0 = None, sols_x0 = None):
		""" Calibrate model instance for grid of parameters with simple looping. """
		cals, sols, paths = dict.fromkeys(grid), dict.fromkeys(grid), dict.fromkeys(grid)
		for v in grid:
			cals[v], paths[v], sols[v] = self.basicIte(parameter, v, cals_x0 = cals_x0, sols_x0 = sols_x0)
			if not isinstance(cals[v], np.ndarray):
				break
			print(v)
		return cals, paths, sols
	
	def basicIte(self, k, v, cals_x0 = None, sols_x0 = None):
		self.m.db.update(self.m.adjPar(k,v))
		try:
			if cals_x0:
				self.m.calibUpdateParameters(cals_x0[v])
			if sols_x0:
				self.m.PEE.x0 = sols_x0[v]
			cal = self.m.calibPEE(**self.kwargs)
			path, sol = self.m.solvePEE()
		except AssertionError:
			cal = """Failed to calibrate"""
			path, sol = None, None
		return cal, path, sol
	
