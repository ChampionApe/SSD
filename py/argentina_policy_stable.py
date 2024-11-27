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

class PEE:
	def __init__(self, m):
		self.m = m
		self.BG = m.BG # requires gridded solutions
		self.db = m.db
		self.x0 = dict.fromkeys(self.db['t'], np.full(self.m.nss0grid, .2))
		self.x0pol = dict.fromkeys(self.db['txE'], np.full(self.m.ns['PEEpol'].len, .2)) | {self.m.T-1: np.full(self.m.nss0grid, .2)}
		self.kwargs_T = {'style': '', 'method': 'krylov'}
		self.kwargs_t = {'style': 'Vector', 'method': 'krylov'}
		# self.fInterp = customLinIntp
		self.fInterp = interpolate.PchipInterpolator
		self.kwargsPolInterp = {'method': 'linear', 'bounds_error': False, 'fill_value': None}

	@property
	def FH_kwargs(self):
		return dict.fromkeys(self.db['txE'], self.kwargs_t) | {self.m.T-1: self.kwargs_T}

	def FH(self, pars = None):
		""" Return dict of policy functions. If kwargs are passed, this should be a dict of kwargs for each t. """
		kwargs = noneInit(pars, self.FH_kwargs)
		sols = dict.fromkeys(self.db['t'])
		t = self.m.T-1
		self.BG.t = self.m.T-1
		sols[t] = self.solve(t = 'T', **({'x0': self.x0[t]} | kwargs[t]))
		for t in range(self.m.T-2, -1, -1):
			self.BG.t = t
			sols[t] = self.solve(solp = sols[t+1], **({'x0': self.x0[t]} | kwargs[t]))
		return sols

	def gridPolicy(self, v, kwargs = None):
		return self.gridPolicy1d(v, kwargs = kwargs) if v.ndim == 1 else self.gridPolicy2d(v, kwargs = kwargs)

	def gridPolicy1d(self, v, kwargs = None):
		vals = pd.Series(v, index = self.db['ss0Idx']).unstack('s0Idx').values
		kwargs = self.kwargsPolInterp | noneInit(kwargs, {})
		return lambda x: interpolate.interpn((self.db['sGrid'], self.db['s0Grid']), vals, x, **kwargs)

	def gridPolicy2d(self, v, kwargs = None):
		""" Repeat and stack columns """
		ite = tuple(pd.Series(v[:,i], index = self.db['ss0Idx']).unstack('s0Idx').values for i in range(v.shape[1])) 
		kwargs = self.kwargsPolInterp | noneInit(kwargs, {})
		return lambda x: np.vstack([interpolate.interpn((self.db['sGrid'], self.db['s0Grid']), vi, x, **kwargs) for vi in ite]).T

	def vectorPolicy(self, sols, y = 'τ', kwargs = None):
		""" Return vector of predicted policies from vector of states; the "0" index is used to make sure that it returns a 1d object, but it requires that we query a single point at a time. """
		d = {t: pd.Series(sols[t][y], index = self.db['ss0Idx']).unstack('s0Idx').values for t in self.db['t']}
		kwargs = self.kwargsPolInterp | noneInit(kwargs, {})
		return lambda x: np.array([interpolate.interpn((self.db['sGrid'], self.db['s0Grid']), d[t], x[t], **kwargs)[0] for t in self.db['t']])

	def solve(self, style = 'Vector', t = 't', **kwargs):
		return getattr(self, f'solve{style}_{t}')(**kwargs)

	def solveVector_t(self, solp, x0_from_solp = False, x0 = None, **kwargs):
		""" Solve problem as a sequence of root-finding problems. """
		fp = {k: self.gridPolicy(solp[k]) for k in ('τ','Bi','B0','Γs', '∂ln(h)/∂τ', '∂ln(h)/∂ln(s[t-1])', 'dτ/ds[t-1]','dτ/d(s0/s)','dln(h)/dln(s[t-1])')} # dict interpolants
		x = optimize.root(lambda x: self.objectiveVector_t(x, fp), x0_solp(x0, solp['x_unbounded'], x0_from_solp), **kwargs)
		assert x['success'], f""" Couldn't identify policy function for year t={self.t}. Previous solution x = {solp['x_unbounded']}. """
		return self.reportVector_t(x['x'], fp)

	def objectiveVector_t(self, x, fp):
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
		sol['Θh']	= self.BG.Θh_t(τ = sol['τ'], τp = solp['τ'], Γs = solp['Γs'])
		sol['h']	= self.BG.hFromΘh_t(s_ = sol['s[t-1]'], Θh = sol['Θh'])
		sol['Θs']	= self.BG.backOutΘs(s_ = sol['s[t-1]'], s = sol['s'])
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



	# GRIDSEARCH SOLUTION - REQUIRES RELATIVELY LARGE GRID TO SEARCH OVER
	def solveGridSearch_t(self, solp, **kwargs):
		sol = self.precomputationsGrid_t(solp)
		fsol, fsolp, idx = self.keepFeasibleDicts(sol, solp) # only keep values within certain bounds
		objective = self.objectiveOnGrid_t(fsol, fsolp, self.db['s0Grid']) # compute objective value on 2d array. Rows = solution grid from t, col = solution grid t-1 for s0/s
		# Identify sign changes in PEE objective on grids:
		objDF = pd.DataFrame(objective, index = self.db['ss0Idx'][idx], columns = self.db['s0Idx'].rename('s0alias')).stack().unstack('s0Idx').sort_index()
		obj = objDF.values
		changeSign = np.diff(np.sign(obj), axis=1) <0
		csIdx1, csIdx2 = np.hstack([changeSign, np.full((changeSign.shape[0],1), False)]), np.hstack([np.full((changeSign.shape[0],1), False), changeSign])
		obj1, obj2 = obj[csIdx1], obj[csIdx2]
		solIdx = objDF.where(csIdx1).stack().index # Get boolean array of what columns (in the 6460 x 91 array) includes sign changes (the rest we can drop):

		# Extract more sparse arrays: Only include rows/columns that include solutions. 
		signChangeCols = np.vstack([csIdx1.any(axis=0), csIdx2.any(axis=0)]).T.any(axis=1) # check any columns in csIdx matrices that include the sign of objective changing from + to -
		sparseIdx = ~np.isnan(objDF.loc[:,signChangeCols].values).all(axis=1) # what rows of the array to include
		csIdx1Sparse = csIdx1[sparseIdx,:][:,signChangeCols]
		csIdx2Sparse = csIdx2[sparseIdx,:][:,signChangeCols]

		# get boolen array that indicates the relevant rows in the sol[k] variables
		ss0Idx = self.db['ss0Idx'][idx]
		ss0IdxBool = ss0Idx.get_level_values('s0Idx').isin(objDF.columns[signChangeCols])

		# Get solution grids:
		solIntrpd = {k: self.interpSol(v, ss0Idx, csIdx1Sparse, csIdx2Sparse, ss0IdxBool, solIdx, obj1, obj2, len(self.db['s0Grid'])) for k,v in fsol.items()}
		return self.reportGridSearch_t(self.resampleSol(solIntrpd))

	def interpSol(self, v, idx, csIdx1, csIdx2, idxBool, solIdx, obj1, obj2, n, secondIdx = None):
		return self.interpSol1d(v, idx, csIdx1, csIdx2, idxBool, solIdx, obj1, obj2, n) if v.ndim == 1 else self.interpSol2d(v, idx, csIdx1, csIdx2, idxBool, solIdx, obj1, obj2, n, secondIdx = noneInit(secondIdx, self.db['i']))

	def interpSol1d(self, v, idx, csIdx1, csIdx2, idxBool, solIdx, obj1, obj2, n):
		kdf = pd.DataFrame(np.tile(v[:,None], (1,n)), index = idx, columns = self.db['s0Idx'].rename('s0alias'))
		kdf = kdf.loc[idxBool,:].stack().unstack('s0Idx').sort_index()
		vsol = inverseInterp1d(kdf.values[csIdx1], kdf.values[csIdx2] ,obj1,obj2)
		return pd.Series(vsol, index = solIdx)

	def interpSol2d(self, v, idx, csIdx1, csIdx2, idxBool, solIdx, obj1, obj2, n, secondIdx):
		kdf = pd.DataFrame(np.tile(v, (1,n)), index = idx, columns = pd.MultiIndex.from_product([secondIdx, self.db['s0Idx'].rename('s0alias')]))
		kdf = kdf.loc[idxBool,:].stack('s0alias').unstack('s0Idx').sort_index()
		v1  = kdf.where(np.tile(csIdx1, (1,len(secondIdx)))).stack('s0Idx').values
		v2  = kdf.where(np.tile(csIdx2, (1,len(secondIdx)))).stack('s0Idx').values
		return pd.DataFrame(inverseInterp1d(v1, v2, obj1[:,None], obj2[:,None]), index = solIdx, columns = secondIdx)

	def reportGridSearch_t(self, sol):
		sol.update(self.getGriddedGradients(sol))
		sol['∂ln(h)/∂ln(s[t-1])'] = self.BG.recursive_dlnh_dlns_(Ψ = sol['Ψ'], σ = sol['σ'], dlnhp_dlns = sol['∂ln(h[t+1])/∂ln(s)'])
		sol['B0'] = self.BG.B0(s_ = sol['s[t-1]'], h = sol['h'])
		sol['s0[t-1]'] = sol['s[t-1]'] * sol['s0/s[t-1]']
		return sol

	def resampleSol(self, sol, **kwargs):
		if len(sol['s[t-1]'].index.get_level_values('s0alias').unique()) == len(self.db['s0Grid']):
			resampleF = self.resampleInterp
		else:
			resampleF = self.resampleInterp_NoneType
		d = {k: resampleF(sol['s[t-1]'], v, self.db['sGrid'], len(self.db['s0Grid'])) for k,v in sol.items() if k not in ('s[t-1]','s0/s[t-1]')}
		d['s[t-1]'], d['s0/s[t-1]'] = self.db['sGrid_ss0'].values, self.db['s0Grid_ss0'].values
		return d
	def resampleInterp(self, s_, v, samplePoints, n, fInterp = None, **kwargs):
		f = self.resampleInterp1d if v.ndim == 1 else self.resampleInterp2d
		return f(s_, v, samplePoints, n, fInterp = fInterp, **kwargs)

	def resampleInterp1d(self, s_, v, samplePoints, n, fInterp = None, **kwargs):
		fInterp = noneInit(fInterp, self.fInterp)
		return np.hstack([fInterp(s_.xs(i,level='s0alias'), v.xs(i,level='s0alias'), **kwargs)(samplePoints) for i in range(n)])
	def resampleInterp2d(self, s_, v, samplePoints, n, fInterp = None, **kwargs):
		return np.vstack([self.resampleInterp1d(s_, v[j], samplePoints, n, fInterp = fInterp, **kwargs) for j in v.columns]).T

	def resampleInterp_NoneType(self, s_, v, samplePoints, n, fInterp = None, **kwargs):
		f = self.resampleInterp1d_NoneType if v.ndim == 1 else self.resampleInterp2d_NoneType
		return f(s_, v, samplePoints, n, fInterp = fInterp, **kwargs)

	def resampleInterp1d_NoneType(self, s_, v, samplePoints, n, fInterp = None, **kwargs):
		fInterp = noneInit(fInterp, self.fInterp)
		d = {i: self.noneIfErr(i, s_,v, samplePoints, fInterp, **kwargs) for i in range(n)}
		fp = np.vstack([v for k,v in d.items() if v is not None])
		return customInterp2d(self.db['s0Grid'], 
							  self.db['s0Grid'][np.array([False if v is None else True for k,v in d.items()])], 
							  					np.vstack([v for k,v in d.items() if v is not None])).reshape(-1)
	def resampleInterp2d_NoneType(self, s_, v, samplePoints, n, fInterp = None, **kwargs):
		return np.vstack([self.resampleInterp1d_NoneType(s_, v[j], samplePoints, n, fInterp = fInterp, **kwargs) for j in v.columns]).T

	def noneIfErr(self, i, s_, v, samplePoints, fInterp, **kwargs):
		try:
			return fInterp(s_.xs(i,level='s0alias'), v.xs(i,level='s0alias'),**kwargs)(samplePoints)
		except KeyError:
			return None

	def precomputationsGrid_t(self, solp):
		sol = { 's': solp['s[t-1]'], 's0/s': solp['s0/s[t-1]'],
				'h': self.BG.backOutH(s = solp['s[t-1]'], Γs = solp['Γs']),
				's[t-1]': self.BG.backOutS_(s = solp['s[t-1]'], s0 = solp['s0[t-1]'], B0 = solp['B0'], τp = solp['τ'])}
		sol['Θs'] = self.BG.backOutΘs(s_ = sol['s[t-1]'], s = sol['s'])
		sol['τ'] = self.BG.backOutτ(Γs = solp['Γs'], τp = solp['τ'], Θs = sol['Θs'])
		sol['∂ln(h[t+1])/∂ln(s)'] = solp['∂ln(h)/∂ln(s[t-1])']
		return self.getAuxVars_t(sol, solp)

	def keepFeasibleDicts(self, sol, solp):
		idx = np.where((sol['s[t-1]']>=self.db['s_l']) & (sol['s[t-1]']<=self.db['s_u']) & (sol['τ']>=0) & (sol['τ']<=1))[0]
		# idx = np.where(np.logical_and(sol['τ']>self.db['τ_l'], sol['τ']<self.db['τ_u']))[0]
		return {k: sol[k][idx] for k in sol}, {k: solp[k][idx] for k in solp}, idx

	def objectiveOnGrid_t(self, sol, solp, s0grid):
		return self.BG.PEE_t_s0grid(τBound = sol['τ'], τ  = sol['τ'], τp = solp['τ'], s_ = sol['s[t-1]'], h = sol['h'], Γs = solp['Γs'], Bip = solp['Bi'], B0p = solp['B0'], si_s = sol['si/s'], s0_s = s0grid, Θs = sol['Θs'],
									dlnh_Dτ = sol['dln(h)/dτ'], dlns_Dτ = sol['dln(s)/dτ'], dlnΓs_Dτ = sol['dln(Γs)/dτ'], dlnhp_Dlns = solp['dln(h)/dln(s[t-1])'], dτp_dτ = sol['dτ[t+1]/dτ'])

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
	def solve_T(self, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid_ss0'].values, self.db['s0Grid_ss0'].values)
		x = optimize.root(lambda τ: self.objective_T(τ, sol), x0, **kwargs)
		assert x['success'], f""" Could not identify PEE solution (PEE.solveVector_T)"""
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
		s = pd.Series(sol['τ'], index = self.db['ss0Idx']).unstack('s0Idx')
		h = pd.Series(sol['h'], index = self.db['ss0Idx']).unstack('s0Idx')
		grdnt = np.gradient(s.values)
		grdnt_h = np.gradient(h.values)
		return {'dτ/ds[t-1]': pd.DataFrame(grdnt[0], index = s.index, columns = s.columns).stack().values, 
			    'dτ/d(s0/s)': pd.DataFrame(grdnt[1], index = s.index, columns = s.columns).stack().values,
			    'dln(h)/dln(s[t-1])': pd.DataFrame(grdnt_h[0], index = h.index, columns = h.columns).stack().values * sol['s[t-1]'] /sol['h']}


class PEE_Analytical:
	def __init__(self, m):
		self.m = m
		self.BG = m.BG # battle gööse?
		self.db = m.db
		self.x0 = dict.fromkeys(self.db['t'], np.full(self.m.ngrid, .2))
		self.kwargs_T = {'style': 'Vector'}
		self.kwargs_t = {'style': 'Vector', 'x0_from_solp': True}
