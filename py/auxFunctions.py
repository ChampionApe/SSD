import numpy as np
from scipy import interpolate

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


## A few methods used for calibration purposes:
def calibrateOnGrid(m, grid, parameter, kt = 'x_unbounded', kT = 'τ_unbounded'):
	""" Start with a simple iteration that breaks if it does not calibrate. 
		Then, extrapolate from existing solutions and re-try """
	cals, paths, sols = calibrateOnGrid_simpleBreak(m, grid, parameter)
	for i in range(3):
		x, cals_x0, sols_x0 = calibrateApproxInitials(m, cals, sols, kt = kt, kT = kT)
		cals_i, paths_i, sols_i = calibrateOnGrid_simpleBreak(m, x, parameter, cals_x0 = cals_x0, sols_x0 = sols_x0)
		cals.update(cals_i), paths.update(paths_i), sols.update(sols_i)
	return cals, paths, sols

def calibrateApproxInitials(m, cals, sols, kt = 'x_unbounded', kT = 'τ_unbounded'):
	cals_x0 = calibrateExtrapolateParametersFromSols(cals, sols)
	x, sols_x0 = calibrateExtrapolateInitialsFromSols(m, cals, sols, kt = kt, kT = kT)
	return x, cals_x0, sols_x0

def calibrateExtrapolateParametersFromSols(cals, sols):
	""" Return dictionary with extrapolated initial guesses for calibration parameters for 
		entries in cals/sols that are not yet inhabited by solutions (represented by np.ndarrays)"""
	fp = np.vstack([v for k,v in cals.items() if isinstance(v, np.ndarray)])
	xp = np.array([k for k,v in cals.items() if isinstance(v, np.ndarray)])
	x  = np.array([k for k,v in cals.items() if not isinstance(v, np.ndarray)])
	cals2d = interpolate.PchipInterpolator(xp, fp, extrapolate = True)(x)
	return {x[i]: cals2d[i,:] for i in range(len(x))}

def calibrateExtrapolateInitialsFromSols(m, cals, sols, kt = 'x_unbounded', kT = 'τ_unbounded'):
	""" Return dictionary with extrapolated initial guesses for policy functions
		entries in cals/sols that are not yet inhabited by solutions (represented by np.ndarrays)"""
	fp = np.vstack([v for k,v in cals.items() if isinstance(v, np.ndarray)])
	xp = np.array([k for k,v in cals.items() if isinstance(v, np.ndarray)])	
	x  = np.array([k for k,v in cals.items() if not isinstance(v, np.ndarray)])
	sols_x0 = dict.fromkeys(x) 
	[sols_x0.__setitem__(xi, {}) for xi in x]; # dictionary with initial guesses 
	for t in m.db['txE']:
		fp = np.vstack([sols[xi][t][kt] for xi in xp])
		x0 = interpolate.PchipInterpolator(xp, fp, extrapolate = True)(x)
		[sols_x0[x[i]].__setitem__(t, x0[i,:]) for i in range(len(x))];
	t = m.db['t'][-1]
	fp = np.vstack([sols[xi][t][kT] for xi in xp])
	x0 = interpolate.PchipInterpolator(xp, fp, extrapolate = True)(x)
	[sols_x0[x[i]].__setitem__(t, x0[i,:]) for i in range(len(x))];
	return x, sols_x0

def calibrateOnGrid_simpleIte(m, grid, parameter, cals_x0 = None, sols_x0 = None):
	""" Calibrate model instance for grid of parameters with simple looping. """
	cals, sols, paths = dict.fromkeys(grid), dict.fromkeys(grid), dict.fromkeys(grid)
	for v in grid:
		cals[v], paths[v], sols[v] = calibrateBasicIte(m, parameter, v, cals_x0 = cals_x0, sols_x0 = sols_x0)
	return cals, paths, sols

def calibrateOnGrid_simpleBreak(m, grid, parameter, cals_x0 = None, sols_x0 = None):
	""" Calibrate model instance for grid of parameters with simple looping. """
	cals, sols, paths = dict.fromkeys(grid), dict.fromkeys(grid), dict.fromkeys(grid)
	for v in grid:
		cals[v], paths[v], sols[v] = calibrateBasicIte(m, parameter, v, cals_x0 = cals_x0, sols_x0 = sols_x0)
		if not isinstance(cals[v], np.ndarray):
			break
		print v
	return cals, paths, sols

def calibrateBasicIte(m, k, v, cals_x0 = None, sols_x0 = None):
	m.db.update(m.adjPar(k,v))
	try:
		cal = m.calibPEE()
		if cals_x0:
			m.calibUpdateParameters(cals_x0[v])
		if sols_x0:
			m.PEE.x0 = sols_x0[v]
		path, sol = m.solvePEE()
	except AssertionError:
		cal = """ Failed to calibrate"""
		path, sol = None, None
	return cal, path, sol
