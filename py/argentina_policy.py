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

class LOG:
	def __init__(self, m):
		self.m = m
		self.BG = m.BG # requires gridded solutions
		self.db = m.db
		self.x0 = self.defaultInitials
		self.kwargs_T = {'style': 'Vector'}
		self.kwargs_T_ = {'style': 'Vector', 'x0_from_solp': False}
		self.kwargs_t = {'style': 'Vector', 'x0_from_solp': True}
		self.fInterp = customLinIntp
		self.kwargsInterp = {}
		# self.fInterp = interpolate.PchipInterpolator
		# self.kwargsInterp = {'extrapolate': True}

	@property
	def defaultInitials(self):
		return dict.fromkeys(self.db['txE'], np.full(self.m.ns['LOGpol'].len, .1)) | {self.m.T-1: np.full(self.m.ns0, .2)}

	@property
	def FH_kwargs(self):
		return dict.fromkeys(self.db['t'][:-2], self.kwargs_t) | {self.m.T-2: self.kwargs_T_, self.m.T-1: self.kwargs_T}

	def FH(self, pars = None, update = True):
		""" Return dict of policy functions. If kwargs are passed, this should be a dict of kwargs for each t. """
		kwargs = noneInit(pars, self.FH_kwargs)
		sols = dict.fromkeys(self.db['t'])
		t = self.m.T-1
		self.BG.t = self.m.T-1
		sols[t] = self.solve(t = 'T', **({'x0': self.x0[t]} | kwargs[t]))
		if update:
			self.x0[t] = sols[t]['τ_unbounded']
		for t in range(self.m.T-2, -1, -1):
			self.BG.t = t
			sols[t] = self.solve(solp = sols[t+1], **({'x0': self.x0[t]} | kwargs[t]))
			if update:
				self.x0[t] = sols[t]['x_unbounded']
		return sols

	def vectorPolicy(self, sols, y = 'τ', grids = None, kwargs = None):
		""" Return vector of predicted policies from vector of states; the "0" index is used to make sure that it returns a 1d object, but it requires that we query a single point at a time. """
		d = {t: sols[t][y] for t in self.db['t']}
		kwargs = self.kwargsInterp | noneInit(kwargs, {})
		grids = noneInit(grids, self.db['s0Grid'])
		return lambda x: np.array([self.fInterp(grids, d[t], **kwargs)(x[t]) for t in self.db['t']])

	def gridPolicy(self, v, grids = None, kwargs = None):
		return self.gridPolicy1d(v, noneInit(grids, self.db['s0Grid']), kwargs = kwargs) if v.ndim == 1 else self.gridPolicy2d(v, noneInit(grids, self.db['s0Grid']), kwargs = kwargs)

	def gridPolicy1d(self, v, grids, kwargs = None):
		kwargs = self.kwargsInterp | noneInit(kwargs, {})
		return self.fInterp(grids, v, **kwargs)

	def gridPolicy2d(self, v, grids, kwargs = None):
		ite = tuple(v[:,i] for i in range(v.shape[1])) 
		kwargs = self.kwargsInterp | noneInit(kwargs, {})
		return lambda x: np.vstack([self.fInterp(grids,vi, **kwargs)(x) for vi in ite]).T

	def solve(self, style = 'Vector', t = 't', **kwargs):
		return getattr(self, f'solve{style}_{t}')(**kwargs)

	def solveVector_t(self, solp = None, x0_from_solp = False, x0 = None, **kwargs):
		sol = self.precomputations_t(solp, self.db['s0Grid'])
		fp = {k: self.gridPolicy(solp[k]) for k in ('τ','dτ/d(s0/s)')}
		x = optimize.root(lambda x: self.objective_t(x, sol, fp), x0_solp(x0, solp['x_unbounded'], x0_from_solp), **kwargs)
		assert x['success'], f""" Coulnd't identify policy function for year t = {self.BG.t}. Previous solution x = {solp['x_unbounded']}"""
		return self.report_t(x['x'], sol, fp)

	def objective_t(self, x, sol, fp):
		sol, solp = self.funcOfτ_t(x, sol, fp)
		PEEobj = self.BG.LOG_PEE_t(τBound = sol['τ'], τ  = sol['τ_unbounded'], τp = solp['τ'], Γs = solp['Γs'], Bip = sol['Bi'], B0p = sol['B0'], si_s = sol['si/s'], s0_s = sol['s0/s[t-1]'], Θs = sol['Θs'],
							dlnh_Dτ = sol['dln(h)/dτ'], dlns_Dτ = sol['dln(s)/dτ'], dlnΓs_Dτ = sol['dln(Γs)/dτ'], dlnhp_Dlns = solp['dln(h)/dln(s[t-1])'], dτp_dτ = sol['dτ[t+1]/dτ'])
		return np.hstack([PEEobj, sol['s0/s']-self.BG.s0_s(B0 = sol['B0'], Θs = sol['Θs'], τp = solp['τ'])])

	def precomputations_t(self, solp, s0idx):
		sol = {'s0/s[t-1]': s0idx, 'Bi': solp['Bi'], 'B0': solp['B0']}
		return sol

	def funcOfτ_t(self, x, sol, fp):
		sol['τ_unbounded'] = self.m.ns['LOGpol'](x,'τ')
		sol['s0/s'] = self.m.ns['LOGpol'](x,'s0/s')
		sol['τ'] = np.clip(sol['τ_unbounded'], self.db['τ_l'], self.db['τ_u'])
		solp = {k: v(sol['s0/s']) for k,v in fp.items()} # interpolated solution
		solp['Γs'] = self.BG.Γs(Bi = sol['Bi'], τp = solp['τ'])
		solp['dln(h)/dln(s[t-1])'] = self.BG.power_h()
		sol['Θh'] = self.BG.Θh_t(τ = sol['τ'], τp = solp['τ'], Γs = solp['Γs'])
		sol['Θs'] = self.BG.Θs_t(Θh = sol['Θh'], Γs = solp['Γs'])
		sol['Γs']	= self.BG.Γs(Bi = sol['Bi'], τp = sol['τ'])
		sol['si/s']	= self.BG.si_s(Bi = sol['Bi'], Γs = sol['Γs'], τp = sol['τ'])
		sol['Ω'] = self.BG.Ω(Γs = solp['Γs'], τp = solp['τ'])
		sol.update(self.BG.LOG_EEDerivatives(τ = sol['τ'], B0p = sol['B0'], Θs = sol['Θs']))
		sol.update(self.BG.LOG_LaggedEEDerivatives(Ω = sol['Ω'], Bip = sol['Bi'], τp = solp['τ'], B0p = sol['B0'], Θs = sol['Θs']))
		sol.update(self.auxStrategicEffects(sol, solp))
		return sol, solp

	def report_t(self, x, sol, fp):
		sol, _ = self.funcOfτ_t(x, sol, fp)
		sol.update(self.getGriddedGradients(sol))
		sol['x_unbounded'] = np.hstack([sol['s0/s'], sol['τ_unbounded']])
		return sol

	def auxStrategicEffects(self, sol, solp):
		ds0_s_dτ = sol['∂(s0/s)/∂τ'] /(1-sol['∂(s0/s)/∂τ[t+1]']*solp['dτ/d(s0/s)'])
		dτp_dτ = ds0_s_dτ * solp['dτ/d(s0/s)']
		return {'dτ[t+1]/dτ': dτp_dτ, 'd(s0/s)/dτ': ds0_s_dτ} |  {f'dln({k})/dτ': self.auxStrategy(k, dτp_dτ, sol) for k in ('s','h','Γs')}

	def auxStrategy(self, k, dτp_dτ, sol):
		return sol[f'∂ln({k})/∂τ']+dτp_dτ * sol[f'∂ln({k})/∂τ[t+1]']

	### TERMINAL STATE FUNCTIONS
	def solveVector_T(self, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['s0Grid'])
		x = optimize.root(lambda τ: self.objective_T(τ, sol), x0, **kwargs)
		assert x['success'], f""" Could not identify PEE solution (self.solveVector_T)"""
		return self.report_T(x['x'], sol)

	def precomputations_T(self, s0idx):
		return {'s' : np.zeros(len(s0idx)), 's0/s': np.zeros(len(s0idx)), 's0/s[t-1]': s0idx, 'Bi': np.tile(self.BG.get('βi')[None,:], (len(s0idx),1)), 'B0': np.full(len(s0idx), self.BG.get('β0'))}

	def objective_T(self, τ, sol):
		τBound = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		funcOfτ = self.funcOfτ_T(τBound, sol)
		return self.BG.LOG_PEE_T(τBound = τBound, τ = τ, dlnh_Dτ = funcOfτ['∂ln(h)/∂τ'], si_s = funcOfτ['si/s'], s0_s = sol['s0/s[t-1]'])

	def funcOfτ_T(self, τ, sol):
		funcOfτ = {'∂ln(h)/∂τ': self.BG.dlnh_Dτ_T(τ), 'Γs': self.BG.Γs(Bi = sol['Bi'], τp = τ)}
		funcOfτ['si/s'] = self.BG.si_s(Bi = sol['Bi'], Γs = funcOfτ['Γs'], τp = τ)
		return funcOfτ

	def report_T(self, τ, sol):
		sol['τ'] = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		sol['τ_unbounded'] = τ
		sol.update(self.funcOfτ_T(sol['τ'], sol))
		sol.update(self.getGriddedGradients(sol))
		sol['x_unbounded'] = np.hstack([sol['s0/s'], sol['τ_unbounded']])
		return sol

	def getGriddedGradients(self, sol):
		return {'dτ/d(s0/s)': np.gradient(sol['τ'], sol['s0/s[t-1]'])}

class PEE:
	def __init__(self, m):
		self.m = m
		self.BG = m.BG # requires gridded solutions
		self.db = m.db
		self.x0 = self.defaultInitials
		self.kwargs_T = {'style': 'Vector', 'method': 'krylov'}
		self.kwargs_T_ = {'style': 'Vector', 'method': 'krylov', 'x0_from_solp': False}
		self.kwargs_t = {'style': 'Vector', 'method': 'krylov', 'x0_from_solp': True}
		self.kwargsInterp = {'method': 'linear', 'bounds_error': False, 'fill_value': None}

	def interpInitialsFromSols(self, sols, idx, grids):
		queryCurrentGrid = np.vstack([self.db['sGrid_ss0'].values, self.db['s0Grid_ss0'].values]).T
		self.x0[self.m.T-1] = self.gridPolicy(sols[self.m.T-1]['τ_unbounded'], idx = idx, grids = grids)(queryCurrentGrid)
		for t in range(self.m.T-2,-1,-1):
			self.x0[t] = np.hstack([self.gridPolicy(sols[t][k], idx = idx, grids = grids)(queryCurrentGrid) for k in ('s','s0/s','τ_unbounded')])

	def interpInitialsFromLOG(self, sols, path):
		""" Get self.x0 dictionary from log solution """
		t = self.m.T-1
		self.x0[t] = pd.Series(0, index = self.db['ss0Idx']).add(pd.Series(sols[t]['τ'], index = self.db['s0Idx'])).values
		[self.x0.__setitem__(t, np.hstack([pd.Series(0, index = self.db['ss0Idx']).add(pd.Series(sols[t][k] if k != 's' else path[k][t], index = self.db['s0Idx'])).values for k in ('s','s0/s','τ')])) for t in self.db['txE']];

	@property
	def defaultInitials(self):
		return dict.fromkeys(self.db['txE'], np.full(self.m.ns['PEEpol'].len, .1)) | {self.m.T-1: np.full(self.m.nss0grid, .2)}

	@property
	def FH_kwargs(self):
		return dict.fromkeys(self.db['t'][:-2], self.kwargs_t) | {self.m.T-2: self.kwargs_T_, self.m.T-1: self.kwargs_T}

	def FH(self, pars = None, update = True):
		""" Return dict of policy functions. If kwargs are passed, this should be a dict of kwargs for each t. """
		kwargs = noneInit(pars, self.FH_kwargs)
		sols = dict.fromkeys(self.db['t'])
		t = self.m.T-1
		self.BG.t = self.m.T-1
		sols[t] = self.solve(t = 'T', **({'x0': self.x0[t]} | kwargs[t]))
		if update:
			self.x0[t] = sols[t]['τ_unbounded']
		for t in range(self.m.T-2, -1, -1):
			self.BG.t = t
			sols[t] = self.solve(solp = sols[t+1], **({'x0': self.x0[t]} | kwargs[t]))
			if update:
				self.x0[t] = sols[t]['x_unbounded']
		return sols

	def gridPolicy(self, v, idx = None, grids = None, kwargs = None):
		return self.gridPolicy1d(v, noneInit(idx, self.db['ss0Idx']), noneInit(grids, (self.db['sGrid'], self.db['s0Grid'])), kwargs = kwargs) if v.ndim == 1 else self.gridPolicy2d(v, noneInit(idx, self.db['ss0Idx']), noneInit(grids, (self.db['sGrid'], self.db['s0Grid'])), kwargs = kwargs)

	def gridPolicy1d(self, v, idx, grids, kwargs = None):
		vals = pd.Series(v, index = idx).unstack('s0Idx').values
		kwargs = self.kwargsInterp | noneInit(kwargs, {})
		return lambda x: interpolate.interpn(grids, vals, x, **kwargs)

	def gridPolicy2d(self, v, idx, grids, kwargs = None):
		""" Repeat and stack columns """
		ite = tuple(pd.Series(v[:,i], index = idx).unstack('s0Idx').values for i in range(v.shape[1])) 
		kwargs = self.kwargsInterp | noneInit(kwargs, {})
		return lambda x: np.vstack([interpolate.interpn(grids, vi, x, **kwargs) for vi in ite]).T

	def vectorPolicy(self, sols, y = 'τ', idx = None, grids = None, kwargs = None):
		""" Return vector of predicted policies from vector of states; the "0" index is used to make sure that it returns a 1d object, but it requires that we query a single point at a time. """
		d = {t: pd.Series(sols[t][y], index = noneInit(idx, self.db['ss0Idx'])).unstack('s0Idx').values for t in self.db['t']}
		kwargs = self.kwargsInterp | noneInit(kwargs, {})
		grids = noneInit(grids, (self.db['sGrid'], self.db['s0Grid']))
		return lambda x: np.array([interpolate.interpn(grids, d[t], x[t], **kwargs)[0] for t in self.db['t']])

	def solve(self, style = 'Vector', t = 't', **kwargs):
		return getattr(self, f'solve{style}_{t}')(**kwargs)

	def solveVector_t(self, solp = None, x0_from_solp = False, x0 = None, **kwargs):
		""" Solve problem as a sequence of root-finding problems. """
		fp = {k: self.gridPolicy(solp[k]) for k in ('τ','Bi','B0', '∂ln(h)/∂τ', '∂ln(h)/∂ln(s[t-1])', 'dτ/ds[t-1]','dτ/d(s0/s)','dln(h)/dln(s[t-1])')} # dict interpolants
		x = optimize.root(lambda x: self.objective_t(x, fp), x0_solp(x0, solp['x_unbounded'], x0_from_solp), **kwargs)
		assert x['success'], f""" Couldn't identify policy function for year t={self.BG.t}. Previous solution x = {solp['x_unbounded']}. """
		return self.reportVector_t(x['x'], fp)

	def objective_t(self, x, fp):
		sol, solp = self.funcOfτ_t(x ,fp)
		PEEobj = self.BG.PEE_t(τBound = sol['τ'], τ  = sol['τ_unbounded'], τp = solp['τ'], s_ = sol['s[t-1]'], h = sol['h'], Γs = solp['Γs'], Bip = solp['Bi'], B0p = solp['B0'], si_s = sol['si/s'], s0_s = sol['s0/s[t-1]'], Θs = sol['Θs'],
						dlnh_Dτ = sol['dln(h)/dτ'], dlns_Dτ = sol['dln(s)/dτ'], dlnΓs_Dτ = sol['dln(Γs)/dτ'], dlnhp_Dlns = solp['dln(h)/dln(s[t-1])'], dτp_dτ = sol['dτ[t+1]/dτ'])
		return np.hstack([PEEobj, 
						sol['s']-self.BG.s_t(h=sol['h'], Γs = sol['Γs']),
						sol['s0/s']-self.BG.s0_s(B0 = solp['B0'], Θs = sol['Θs'], τp = solp['τ'])])

	def funcOfτ_t(self, x, fp):
		sol = {	'τ_unbounded': self.m.ns['PEEpol'](x, 'τ'), 's': self.m.ns['PEEpol'](x, 's'), 's0/s': self.m.ns['PEEpol'](x, 's0/s'),
				's[t-1]': self.db['sGrid_ss0'].values, 's0/s[t-1]': self.db['s0Grid_ss0'].values}
		sol['s'] = np.clip(sol['s'], self.db['s_l'], None)
		sol['τ'] = np.clip(sol['τ_unbounded'], self.db['τ_l'], self.db['τ_u']) 
		z0 = np.vstack([sol['s'], sol['s0/s']]).T # vector of states to pass to interpolators 
		solp = {k: v(z0) for k,v in fp.items()} # query t+1 solution
		solp['Γs'] = self.BG.Γs(Bi = solp['Bi'], τp = solp['τ'])
		sol['Θh']	= self.BG.Θh_t(τ = sol['τ'], τp = solp['τ'], Γs = solp['Γs'])
		sol['h']	= self.BG.hFromΘh_t(s_ = sol['s[t-1]'], Θh = sol['Θh'])
		sol['Θs'] = self.BG.Θs_t(Θh = sol['Θh'], Γs = solp['Γs'])
		# sol['Θs']	= self.BG.backOutΘs(s_ = sol['s[t-1]'], s = sol['s'])
		return self.getAuxVars_t(sol, solp), solp

	def getAuxVars_t(self, sol, solp):
		sol['Bi']	= self.BG.Bi(s_ = sol['s[t-1]'], h = sol['h'])
		sol['B0']	= self.BG.B0(s_ = sol['s[t-1]'], h = sol['h'])
		sol['Γs']	= self.BG.Γs(Bi = sol['Bi'], τp = sol['τ'])
		sol['si/s']	= self.BG.si_s(Bi = sol['Bi'], Γs = sol['Γs'], τp = sol['τ'])
		sol['Ω']	= self.BG.Ω(Γs = solp['Γs'], τp = solp['τ'])
		sol['Ψ']	= self.BG.Ψ(Bip = solp['Bi'], τp = solp['τ'])
		sol['σ']	= self.BG.σ(Ω = sol['Ω'], Ψ = sol['Ψ'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'], τp = solp['τ'])
		sol.update(self.BG.EELaggedDerivatives(Ω = sol['Ω'], Ψ = sol['Ψ'], Bip = solp['Bi'], dlnhp_dτp = solp['∂ln(h)/∂τ'], τp = solp['τ'], B0p = solp['B0'], Θs = sol['Θs']))
		sol.update(self.BG.EEDerivatives(Ψ = sol['Ψ'], σ = sol['σ'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'], τ = sol['τ'], τp = solp['τ'], B0p = solp['B0'], Θs = sol['Θs']))
		sol.update(self.auxStrategicEffects(sol, solp))
		return sol

	def reportVector_t(self, x, fp):
		sol, solp = self.funcOfτ_t(x ,fp)
		sol['x_unbounded'] = x
		sol.update(self.getGriddedGradients(sol))
		sol['∂ln(h)/∂ln(s[t-1])'] = self.BG.recursive_dlnh_dlns_(Ψ = sol['Ψ'], σ = sol['σ'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'])
		return sol

	def auxStrategicEffects(self, sol, solp):
		a11 = sol['∂ln(s)/∂τ[t+1]']  * solp['dτ/ds[t-1]'] * sol['s']
		a12 = sol['∂ln(s)/∂τ[t+1]']  * solp['dτ/d(s0/s)']
		a21 = sol['∂(s0/s)/∂τ[t+1]'] * solp['dτ/ds[t-1]'] * sol['s']
		a22 = sol['∂(s0/s)/∂τ[t+1]'] * solp['dτ/d(s0/s)']
		b1,b2  = sol['∂ln(s)/∂τ'], sol['∂(s0/s)/∂τ']
		det = (1-a22)*(1-a11)-a12*a21
		dlns_dτ, ds0_s_dτ = (b1*(1-a22)+a12*b2)/det, (b2*(1-a11)+a21*b1)/det
		dτp_dτ = dlns_dτ*solp['dτ/ds[t-1]'] * sol['s'] + ds0_s_dτ* solp['dτ/d(s0/s)']
		return {'dτ[t+1]/dτ': dτp_dτ, 'dln(s)/dτ': dlns_dτ, 'd(s0/s)/dτ': ds0_s_dτ} | {f'dln({k})/dτ': self.auxStrategy(k, dτp_dτ, sol) for k in ('h','Γs')}

	def auxStrategy(self, k, dτp_dτ, sol):
		""" Works for ln(h) and ln(Γs) - they are not states in the pee problem. """
		return sol[f'∂ln({k})/∂τ']+dτp_dτ * sol[f'∂ln({k})/∂τ[t+1]']

	### TERMINAL STATE FUNCTIONS
	def solveVector_T(self, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid_ss0'].values, self.db['s0Grid_ss0'].values)
		x = optimize.root(lambda τ: self.objective_T(τ, sol), x0, **kwargs)
		assert x['success'], f""" Could not identify PEE solution (self.solveVector_T)"""
		return self.report_T(x['x'], sol)

	def precomputations_T(self, sidx, s0idx):
		return {'s' : np.zeros(len(sidx)), 's0/s': np.zeros(len(sidx)), 's[t-1]': sidx, 's0/s[t-1]': s0idx}

	def objective_T(self, τ, sol):
		τBound = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		funcOfτ = self.funcOfτ_T(τBound, sol)
		return self.BG.PEE_T(τBound = τBound, τ = τ, s_ = sol['s[t-1]'], h = funcOfτ['h'], dlnh_Dτ = funcOfτ['∂ln(h)/∂τ'], si_s = funcOfτ['si/s'], s0_s = sol['s0/s[t-1]'])

	def funcOfτ_T(self, τ, sol):
		funcOfτ = {'∂ln(h)/∂τ': self.BG.dlnh_Dτ_T(τ), 'h': self.BG.h_T(s_ = sol['s[t-1]'], τ = τ)}
		funcOfτ['Bi'] = self.BG.Bi(s_ = sol['s[t-1]'], h = funcOfτ['h'])
		funcOfτ['Γs'] = self.BG.Γs(Bi = funcOfτ['Bi'], τp = τ)
		funcOfτ['si/s'] = self.BG.si_s(Bi = funcOfτ['Bi'], Γs = funcOfτ['Γs'], τp = τ)
		return funcOfτ

	def report_T(self, τ, sol):
		sol['τ'] = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		sol['τ_unbounded'] = τ
		sol.update(self.funcOfτ_T(sol['τ'], sol))
		sol.update(self.getGriddedGradients(sol))
		sol['∂ln(h)/∂ln(s[t-1])'] = np.full(τ.shape, self.BG.power_h())
		sol['B0'] = self.BG.B0(s_ = sol['s[t-1]'], h = sol['h'])
		sol['s0[t-1]'] = sol['s[t-1]'] * sol['s0/s[t-1]']
		sol['x_unbounded'] = np.hstack([sol['s'], sol['s0/s'], sol['τ_unbounded']])
		return sol

	def getGriddedGradients(self, sol):
		s = pd.Series(sol['τ'], index = self.db['ss0Idx']).unstack('sIdx')
		h = pd.Series(sol['h'], index = self.db['ss0Idx']).unstack('sIdx')
		grdnt = np.gradient(s.values, self.db['s0Grid'], self.db['sGrid'])
		grdnt_h = np.gradient(h.values, self.db['s0Grid'], self.db['sGrid'])
		return {'dτ/ds[t-1]': pd.DataFrame(grdnt[1], index = s.index, columns = s.columns).stack().values, 
			    'dτ/d(s0/s)': pd.DataFrame(grdnt[0], index = s.index, columns = s.columns).stack().values,
			    'dln(h)/dln(s[t-1])': pd.DataFrame(grdnt_h[1], index = h.index, columns = h.columns).stack().values * sol['s[t-1]'] /sol['h']}


