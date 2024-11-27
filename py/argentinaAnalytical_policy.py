import numpy as np, pandas as pd
from scipy import optimize, interpolate
from pyDbs import is_iterable, adj
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
# warnings.filterwarnings("ignore", message = "FutureWarning: The previous implementation of stack is deprecated and will be removed in a future version of pandas. See the What's New notes for pandas 2.1.0 for details. Specify future_stack=True to adopt the new implementation and silence this warning.")
_numtypes = (int,float,np.generic)

def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

def aux_soli(sol, i):
	sol_i = {k: sol[k][i] if (k != 'Bi') and is_iterable(sol[k]) else sol[k] for k in sol}
	if 'Bi' in sol:
		sol_i['Bi'] = sol['Bi'][i:i+1,:]
	return sol_i

def x0_solp(x0, solp, x0_from_solp):
	return solp if x0_from_solp else x0

def inverseInterp1d(x1, x2, y1, y2):
	return x1+y1*(x2-x1)/(y1-y2)

def customInterp2d(x, xp, fp):
	xb = np.clip(x, min(xp)+np.finfo(float).eps, max(xp))
	j = np.searchsorted(xp, xb, side = 'left') - 1
	d = ((x-xp[j])/(xp[j+1]-xp[j]))[:,None]
	return (1-d)*fp[j,:] + d * fp[j+1,:]

def customLinIntp(x, y, **kwargs):
	return lambda z: np.interp(z,x,y, **kwargs)

def interpSol(x, xp, fp):
	""" linear interpolation where x, xp are 1d vectors and fp may be 1d or 2d (simply repeats interpolation over the 2d)"""
	if fp.ndim == 1:
		return np.interp(x,xp,fp)
	else:
		return np.vstack([np.interp(x,xp,fp[:,i]) for i in range(fp.shape[1])]).T

def cartesianGrids(grids1d):
	""" grids1d = dict of grids. """
	idx1ds = {k: pd.Index(range(grids1d[k].size), name = f'{k}Idx') for k in grids1d}
	idxnd = pd.MultiIndex.from_product(list(idx1ds.values()))
	gridsnd = {k: pd.Series(0, index = idxnd).add(pd.Series(grids1d[k], index = idx1ds[k])) for k in grids1d}
	return idx1ds, idxnd, gridsnd

def defaultGrid_(n, l, u, kl, ku):
	return np.insert(np.linspace(l,u,n-2), [0, n-2], [l-1/kl-1e-4, u+1/ku+1e-4])

def defaultGrid(k, db):
	return defaultGrid_(db[f'{k}_n'], db[f'{k}_l'], db[f'{k}_u'], db[f'k{k}_l'], db[f'k{k}_u'])

class LOG_A:
	def __init__(self, m):
		self.m = m
		self.BG = m.BG # requires gridded solutions
		self.BT = m.BT # 
		self.db = m.db
		self.x0 = self.defaultInitials
		self.fInterp = customLinIntp
		self.kwargsInterp = {}
		# self.fInterp = interpolate.PchipInterpolator
		# self.kwargsInterp = {'extrapolate': True}

	@property
	def defaultInitials(self):
		return np.full(self.m.T, .2)



class PEE_A:
	def __init__(self, m):
		self.m = m
		self.BG = m.BG # requires gridded solutions
		self.db = m.db
		self.x0 = self.defaultInitials
		self.kwargs_T = {'style': 'Vector', 'method': 'krylov'}
		self.kwargs_T_ = {'style': 'Vector', 'method': 'krylov', 'x0_from_solp': False}
		self.kwargs_t = {'style': 'Vector', 'method': 'krylov', 'x0_from_solp': True}
		self.kwargsInterp = {'method': 'linear', 'bounds_error': False, 'fill_value': None}
