import numpy as np, pandas as pd
from scipy import optimize
from pyDbs import is_iterable
_numtypes = (int,float,np.generic)

def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

def aux_soli(sol, i):
	sol_i = {k: sol[k][i] if (k != 'B') and is_iterable(sol[k]) else sol[k] for k in sol}
	if 'B' in sol:
		sol_i['B'] = sol['B'][:,i:i+1]
	return sol_i

def x0_solp(x0, solp, x0_from_solp, var = 'τ_unbounded'):
	return solp[var] if x0_from_solp else x0

def inverseInterp1d(x1, x2, y1, y2):
	return x1+y1*(x2-x1)/(y1-y2)

def boolOperator_i(obj, i):
    return obj.le if i == 0 else obj.gt

def boolOperator(obj, d):
    return pd.DataFrame({k: boolOperator_i(obj[k], v)(0) for k, v in d.items()}).all(axis=1)

def interpSol(x, xp, fp):
	""" linear interpolation where x, xp are 1d vectors and fp may be 1d or 2d (simply repeats interpolation over the 2d)"""
	if isinstance(fp, _numtypes):
		return fp
	elif fp.ndim == 1:
		return np.interp(x,xp,fp)
	else:
		return np.vstack([np.interp(x,xp,fp[i,:]) for i in range(fp.shape[0])])

def defaultGrid_(n, l, u, kl, ku):
	return np.insert(np.linspace(l,u,n-2), [0, n-2], [l-1/kl-1e-4, u+1/ku+1e-4])

def defaultGrid(k, db):
	return defaultGrid_(db[f'{k}_n'], db[f'{k}_l'], db[f'{k}_u'], db[f'k{k}_l'], db[f'k{k}_u'])


class LOG:
	def __init__(self, m):
		self.m = m
		self.C = m.C
		self.db = m.db
		self.x0 = {'PEE': np.full(self.m.T, .2), 'ESC': np.full(self.m.ns['ESC'].len, .2)}
		self.grids = {k: defaultGrid(k, self.db) for k in ('τ','κ')}

	def nGrids(self, k, n = None, l = None, u = None, kl = None, ku = None):
		return defaultGrid_(noneInit(n, self.db[f'{k}_n']), noneInit(l, self.db[f'{k}_l']), noneInit(u, self.db[f'{k}_u']), noneInit(kl, self.db[f'k{k}_l']), noneInit(ku, self.db[f'k{k}_u']))

	def createGrids(self, grids1d):
		idx1ds = {k: pd.Index(range(grids1d[k].size), name = k) for k in grids1d}
		idxnd = pd.MultiIndex.from_product(idx1ds.values())
		gridsnd = {k: pd.Series(0, index = idxnd).add(pd.Series(grids1d[k], index = idx1ds[k])) for k in grids1d}
		grids2d = {k: pd.DataFrame(np.tile(gridsnd[k].values, (self.m.T,1)).T, index = idxnd, columns = self.db['t']) for k in grids1d}
		return idx1ds, idxnd, gridsnd, grids2d

	def solve(self, c, style = 'VeryRobust', **kwargs):
		return getattr(self, f'solve{style}_{c}')(**kwargs)

	def solveRobust_PEE(self, **kwargs):
		return self.solveRobust('PEE', **kwargs)

	def solveVeryRobust_PEE(self, **kwargs):
		return self.solveVeryRobust('PEE', **kwargs)

	def solveRobust_ESC(self, **kwargs):
		return self.solveRobust('ESC', **kwargs)

	def solveVeryRobust_ESC(self, **kwargs):
		return self.solveVeryRobust('ESC', **kwargs)

	def solveVeryRobust(self, c, **kwargs):
		""" Use grid-search to get x0 """
		try:
			return getattr(self, f'solveVector_{c}')(**kwargs)
		except AssertionError:
			gridSol = getattr(self, f'solveGridSC_{c}')(**kwargs)
			try:
				return getattr(self, f'solveVector_{c}')(x0 = gridSol)
			except AssertionError:
				# print(f"""Warning: Could only solve model with grid search""")
				return gridSol

	def solveRobust(self, c, **kwargs):
		""" Use grid-search to get x0 """
		try:
			return getattr(self, f'solveVector_{c}')(**kwargs)
		except AssertionError:
			gridSol = getattr(self, f'solveGridSC_{c}')(**kwargs)
			return getattr(self, f'solveVector_{c}')(x0 = gridSol)

	def solveGridSC_PEE(self, τGrid = None, **kwargs):
		τ = np.tile(noneInit(τGrid, self.grids['τ']), (self.m.T,1))
		κ, ν = self.db['κ'].values.reshape(self.m.T,1), self.db['ν'].reshape(self.m.T,1)
		o = self.C.PEE_log_FH(τ = τ, κ = κ, ν = ν)
		changeSign = np.diff(np.sign(o), axis = 1) <0
		return inverseInterp1d(τ[:,:-1][changeSign], τ[:,1:][changeSign], o[:,:-1][changeSign], o[:,1:][changeSign])

	def solveGridSC_ESC(self, grids  = None, **kwargs):
		grids1d = noneInit(grids, self.grids)
		# Create nd grids over policy choices - identify optimal κ on a grid of (t, τ):
		idx1ds, idxnd, gridsnd, grids2d = self.createGrids(grids1d)
		obj = self.C.ESC_log_FH(τ = grids2d['τ'].values.T, κ = grids2d['κ'].values.T, ν = self.db['ν'].reshape(self.m.T,1))
		obj2d = pd.DataFrame(obj[:,len(idxnd):].T, columns = self.db['t'], index = idxnd).unstack('τ')
		κ2d = grids2d['κ'].unstack('τ').values
		changeSign = np.diff(np.sign(obj2d), axis = 0) <0
		κ = pd.Series(inverseInterp1d(κ2d[:-1,:][changeSign],κ2d[1:,:][changeSign],obj2d.values[:-1,:][changeSign],obj2d.values[1:,:][changeSign]), index = obj2d.columns).unstack('τ').values
		# Solve PEE grid choice:
		τ = np.tile(grids1d['τ'], (self.m.T,1))
		o = self.C.ESC_log_FH(τ = τ, κ = κ, ν = self.db['ν'].reshape(self.m.T,1))[:,:grids1d['τ'].shape[0]]
		changeSign = np.diff(np.sign(o), axis=1)<0
		τ = inverseInterp1d(τ[:,:-1][changeSign], τ[:,1:][changeSign], o[:,:-1][changeSign], o[:,1:][changeSign])
		κ = inverseInterp1d(κ[:,:-1][changeSign], κ[:,1:][changeSign], o[:,:-1][changeSign], o[:,1:][changeSign])
		return np.hstack([τ, κ])

	def solveVector_PEE(self, x0 = None, **kwargs):
		sol = optimize.root(lambda x: self.C.PEE_log_FH(x, κ = self.db['κ'].values, ν = self.db['ν']), noneInit(x0, self.x0['PEE']))
		assert sol['success'], f""" Couldn't identify PEE in LOG.solveVector_PEE"""
		return sol['x']

	def solveVector_ESC(self, x0 = None, **kwargs):
		sol = optimize.root(lambda x: self.C.ESC_log_FH(self.m(x,'τ', ns = 'ESC'), κ = self.m(x,'κ',ns = 'ESC'), ν = self.db['ν']), noneInit(x0, self.x0['ESC']))
		assert sol['success'], f"""Couldn't identify ESC in LOG.solveVector_ESC)"""
		return sol['x']

class ESC:
	def __init__(self, m):
		self.m = m
		self.C = m.C
		self.db = m.db
		self.x0 = dict.fromkeys(self.db['t'], np.full(self.m.ns['ESCpol'].len, .2))
		self.grids = {k: defaultGrid(k, self.db) for k in ('τ','κ')}
		self.kwargs_T = {'style': 'VeryRobust'}
		self.kwargs_t = {'style': 'VeryRobust', 'x0_from_solp': True}

	@property
	def FH_kwargs(self):
		return dict.fromkeys(self.db['txE'], self.kwargs_t) | {self.m.T-1: self.kwargs_T}

	def FH(self, pars = None):
		""" Return dict of policy functions. If kwargs are passed, this should be a dict of kwargs for each t. """
		kwargs = noneInit(pars, self.FH_kwargs)
		sols = dict.fromkeys(self.db['t'])
		sols[self.m.T-1] = self.solve(self.db['ν'][-1], t = 'T', **({'x0': self.x0[self.m.T-1]} | kwargs[self.m.T-1]))
		for t in range(self.m.T-2, -1, -1):
			sols[t] = self.resampleSolution(self.solve(self.db['ν'][t], νp = self.db['ν'][t+1], solp = sols[t+1], **({'x0': self.x0[t]} | kwargs[t])))
		return sols

	def resampleSolution(self, sol):
		d = {k: interpSol(self.db['sGrid'], sol['s[t-1]'], v) for k,v in sol.items() if k not in ('s[t-1]','x_unbounded')}
		d['s[t-1]'] = self.db['sGrid']
		d['x_unbounded'] = interpSol(self.db['sGrid'], sol['s[t-1]'], sol['x_unbounded'].reshape((2, self.m.ngrid))).reshape(sol['x_unbounded'].shape)
		return d

	def nGrids(self, k, n = None, l = None, u = None, kl = None, ku = None):
		return defaultGrid_(noneInit(n, self.db[f'{k}_n']), noneInit(l, self.db[f'{k}_l']), noneInit(u, self.db[f'{k}_u']), noneInit(kl, self.db[f'k{k}_l']), noneInit(ku, self.db[f'k{k}_u']))

	def createGrids(self, grids1d):
		idx1ds = {k: pd.Index(range(grids1d[k].size), name = k) for k in grids1d}
		idxnd = pd.MultiIndex.from_product(list(idx1ds.values())+[self.db['sIdx']])
		gridsnd = {k: pd.Series(0, index = idxnd).add(pd.Series(grids1d[k], index = idx1ds[k])) for k in grids1d}
		gridsnd['s'] = pd.Series(0, index = idxnd).add(pd.Series(self.db['sGrid'], index = self.db['sIdx']))
		return idx1ds, idxnd, gridsnd

	def mapToIdxnd(self, k, idxnd, solp):
		if solp[k].ndim == 1:
			return pd.Series(0, index = idxnd).add(pd.Series(solp[k], index = self.db['sIdx'])).values
		else:
			return pd.DataFrame(0, index = idxnd, columns = self.db['i']).add(pd.DataFrame(solp[k].T, index = self.db['sIdx'], columns = self.db['i'])).values.T

	def interp2dSol_(self, f0, f1, x0, x1):
		return (f1 * x0 + abs(f0) * x1)/(abs(f0)+f1)

	def interp2dSol(self, f0, f1, x0, x1):
		return np.hstack([self.interp2dSol_(f0[k].values, f1[k].values, x0[k].values, x1[k].values) for k in ('τ','κ')])

	def aux_strategy(self, sol, solp, k):
		return (sol[f'∂ln({k})/∂τ[t+1]'] * solp['dτ/ds[t-1]'] + sol[f'∂ln({k})/∂κ[t+1]']*solp['dκ/ds[t-1]']) * sol['s']

	def aux_logDev(self, sol, funcOfτ, k):
		if k == 's':
			return funcOfτ[f'∂ln({k})/∂τ']/(1-sol[f'{k}_strategy'])
		else:
			return funcOfτ[f'∂ln({k})/∂τ']+sol[f'{k}_strategy'] * funcOfτ['dln(s)/dτ']

	######## Std. period t:
	def solve(self, ν, style = 'VeryRobust', t = 't', **kwargs):
		return getattr(self, f'solve{style}_{t}')(ν, **kwargs)

	def solveVeryRobust_t(self, ν, νp = None, solp = None, grids = None, weights = None, **kwargs):
		try:
			return self.solveVector_t(ν, νp = νp, solp = solp, **kwargs)
		except AssertionError:
			gridSol = self.solveGridSC_t(ν, νp = νp, solp = solp, grids = grids, weights = weights)
			try:
				return self.solveVector_t(ν, νp = νp, solp = solp, x0_from_solp = False, x0 = gridSol['x_unbounded'])
			except AssertionError:
				print(f"""Warning: Could only solve model with grid search - for ν = {ν}, ρ = {self.db['ρ']}, ξ: {self.db['ξ']}""")
				return gridSol

	def solveRobust_t(self, ν, νp = None, solp = None, grids = None, weights = None, **kwargs):
		""" Use grid-search to get x0 """
		try:
			return self.solveVector_t(ν, νp = νp ,solp = solp, **kwargs)
		except AssertionError:
			gridSol = self.solveGridSC_t(ν, νp = νp, solp = solp, grids = grids, weights = weights)
			return self.solveVector_t(ν, νp = νp, solp = solp, x0_from_solp = False, x0 = gridSol['x_unbounded'])

	def solveVector_t(self, ν, νp = None, solp = None, x0_from_solp = True, x0 = None, **kwargs):
		sol = self.precomputations_t(ν, solp)
		x = optimize.root(lambda x: self.objective_t(self.m(x,'τ'), self.m(x, 'κ'), ν, νp, sol, solp), x0_solp(x0, solp, x0_from_solp, var = 'x_unbounded'))
		assert x['success'], f""" Couldn't identify ESC solution (ESC.solveVector_t) with ν = {ν} and previous solution:
		τ: {solp['τ']}
		κ: {solp['κ']}"""
		return self.report_t(x['x'], ν, sol, solp)

	def solveScalarLoop_t(self, ν, νp = None, solp = None, x0_from_loop = True, x0_from_solp = True, x0 = None, **kwargs):
		sol = self.precomputations_t(ν, solp)
		esc = np.empty((self.m.ngrid, 2))
		x = optimize.root(lambda x: self.objective_t(x[0], x[1], ν, νp, aux_soli(sol, 0), aux_soli(solp,0)), x0_solp(x0, solp, x0_from_solp, var = 'x_unbounded').reshape((2, self.m.ngrid)).T[0])
		esc[0] = x['x']
		assert x['success'], f""" Couldn't identify ESC with ν = {ν}, loop i=0."""
		for i in range(1, self.m.ngrid):
			x = optimize.root(lambda x: self.objective_t(x[0], x[1], ν, νp, aux_soli(sol, i), aux_soli(solp,i)), esc[i-1] if x0_from_loop else x0_solp(x0, solp, x0_from_solp, var = 'x_unbounded').reshape((2, self.m.ngrid)).T[i])
			esc[i] = x['x']
			assert x['success'], f""" Couldn't identify ESC iteration i = {i} """
		return self.report_t(np.hstack([esc[:,0], esc[:,1]]), ν, sol, solp)

	def solveGridSearch_t(self, ν, νp = None, solp = None, grids = None, **kwargs):
		grids1d = noneInit(grids, self.grids)
		idx1ds, idxnd, gridsnd = self.createGrids(grids1d)
		solp_nd = {k: self.mapToIdxnd(k, idxnd, solp) for k in solp if k != 'x_unbounded'}
		sol_nd  = self.precomputations_t(ν, solp_nd)
		objective = self.objective_t(gridsnd['τ'].values, gridsnd['κ'].values, ν, νp, sol_nd, solp_nd)
		idxMin = pd.Series(abs(objective).reshape((2,len(idxnd))).sum(axis=0), index = idxnd).groupby(['sIdx']).idxmin()
		x = np.hstack([gridsnd[k][idxMin].values for k in ('τ','κ')])
		sol = self.precomputations_t(ν, solp)
		return self.report_t(x, ν, sol, solp)

	def solveGridSC_t(self, ν, νp = None, solp = None, grids = None, **kwargs):
		""" Solve by sequentially looking for sign changes"""
		grids1d = noneInit(grids, self.grids).copy()
		idx1ds, idxnd, gridsnd = self.createGrids(grids1d)
		solp_nd = {k: self.mapToIdxnd(k, idxnd, solp) for k in solp if k != 'x_unbounded'}
		sol_nd  = self.precomputations_t(ν, solp_nd)
		# Identify κ:
		objective = pd.DataFrame(self.objective_t(gridsnd['τ'].values, gridsnd['κ'].values, ν, νp, sol_nd, solp_nd).reshape((2,len(idxnd))).T, 
								index = idxnd, columns = ['τ','κ'])
		obj2d = objective['κ'].unstack('κ').values.T
		κ2d = gridsnd['κ'].unstack('κ').values.T
		changeSign = np.diff(np.sign(obj2d), axis = 0) <0
		κ = inverseInterp1d(κ2d[:-1,:][changeSign],κ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		grids1d.pop('κ'); # remove 'κ' from grids
		idx1ds, idxnd, gridsnd = self.createGrids(grids1d)
		solp_nd = {k: self.mapToIdxnd(k, idxnd, solp) for k in solp if k != 'x_unbounded'}
		sol_nd  = self.precomputations_t(ν, solp_nd)
		objective = pd.DataFrame(self.objective_t(gridsnd['τ'].values, κ, ν, νp, sol_nd, solp_nd).reshape((2,len(idxnd))).T, 
								index = idxnd, columns = ['τ','κ'])
		obj2d = objective['τ'].unstack('τ').values.T
		τ2d = gridsnd['τ'].unstack('τ').values.T
		κ2d = pd.Series(κ, index = idxnd).unstack('τ').values.T
		changeSign = np.diff(np.sign(obj2d), axis = 0) <0
		τ = inverseInterp1d(τ2d[:-1,:][changeSign],τ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		κ = inverseInterp1d(κ2d[:-1,:][changeSign],κ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		sol = self.precomputations_t(ν, solp)
		return self.report_t(np.hstack([τ,κ]), ν, sol, solp)

	def solveGrid_t(self, ν, νp = None, solp = None, grids = None, weights = None, **kwargs):
		""" Neutral weights is a dict {'τ': 1, 'κ': 1}"""
		grids1d = noneInit(grids, self.grids)
		idx1ds, idxnd, gridsnd = self.createGrids(grids1d)
		solp_nd = {k: self.mapToIdxnd(k, idxnd, solp) for k in solp if k != 'x_unbounded'}
		sol_nd  = self.precomputations_t(ν, solp_nd)
		objective = pd.DataFrame(self.objective_t(gridsnd['τ'].values, gridsnd['κ'].values, ν, νp, sol_nd, solp_nd).reshape((2,len(idxnd))).T,
								index = idxnd, columns = ['τ','κ'])
		weightedObj = abs(objective.mul(weights) if weights else objective).sum(axis=1)
		subsetGrid = {k: boolOperator(objective, {'τ': k[0], 'κ': k[1]}) for k in [(0,0), (1,1)]} 
		# subsetGrid = {k: boolOperator(objective, {'τ': k[0], 'κ': k[1], 'eps': k [2]}) for k in itertools.product(*[[0,1]]*3)} # do all 8 combinations
		nodes = {k: weightedObj[subsetGrid[k]].groupby('sIdx').idxmin() for k in subsetGrid}
		x = self.interp2dSol(objective.loc[nodes[(0,0)]], objective.loc[nodes[(1,1)]],
							 {k: gridsnd[k].loc[nodes[(0,0)]] for k in ('τ','κ')}, {k: gridsnd[k].loc[nodes[(1,1)]] for k in ('τ','κ')})
		sol = self.precomputations_t(ν, solp)
		return self.report_t(x,ν,sol, solp)

	def precomputations_t(self, ν, solp):
		sol = { 's': solp['s[t-1]'], 
				'h': (solp['s[t-1]']/solp['Γs'])**(self.db['ξ']/(1+self.db['ξ'])), 
				'Ω': self.C.Ω(Γs = solp['Γs'], τp = solp['τ'], κp = solp['κ']), 
				'Ψ': self.C.Ψ(Bp = solp['B'] , τp = solp['τ'], κp = solp['κ'])}		
		sol['s_τ0'] = self.C.aux_sτ0(h = sol['h'], ν = ν, τp = solp['τ'], κp= solp['κ'], Γs = solp['Γs'])
		sol.update(self.C.EELaggedDerivatives_τ(Ω = sol['Ω'], Ψ = sol['Ψ'], Bp = solp['B'], dlnhp_dτp = solp['∂ln(h)/∂τ'], τp = solp['τ'], κp = solp['κ']))
		sol['σ'] = self.C.σ(Ω = sol['Ω'], Ψ = sol['Ψ'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'], τp = solp['τ'], κp = solp['κ'])
		sol.update(self.C.EELaggedDerivatives_κ(Ω = sol['Ω'], Ψ = sol['Ψ'], σ = sol['σ'], Bp = solp['B'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'], τp = solp['τ'], κp = solp['κ']))
		sol.update({f'{k}_strategy': self.aux_strategy(sol, solp, k) for k in ('s','Γs','h')})
		return sol

	def objective_t(self, τ, κ, ν, νp, sol, solp):
		τBound, κBound = np.clip(τ, self.db['τ_l'], self.db['τ_u']), np.clip(κ, self.db['κ_l'], self.db['κ_u'])
		funcOfτ = self.func_t(τBound, κBound, ν, sol, solp)
		return self.C.ESC_t(τBound = τBound, κBound = κBound, τ = τ, κ = κ, ν = ν, τp= solp['τ'], κp= solp['κ'], νp= νp,
							s_ = funcOfτ['s[t-1]'], s = sol['s'], h = sol['h'], hp= solp['h'], Γs =solp['Γs'], Bp = solp['B'], sSpread = funcOfτ['si/s'],
							dlnh_Dτ = funcOfτ['dln(h)/dτ'], dlns_Dτ = funcOfτ['dln(s)/dτ'], dlnΓs_Dτ = funcOfτ['dln(Γs)/dτ'], dlnhp_Dlns = solp['dln(h)/dln(s[t-1])'], 
							dτp_dτ = funcOfτ['dτ[t+1]/dτ'], dκp_dτ = funcOfτ['dκ[t+1]/dτ'])
 
	def func_t(self, τ, κ, ν, sol, solp):
		""" Return functions of τ on the grid of s"""
		funcOfτ = {'s[t-1]': sol['s_τ0']*(1-τ)**(-1/self.db['α'])}
		funcOfτ['B'] = self.C.B(funcOfτ['s[t-1]'], sol['h'], ν)
		funcOfτ['Γs'] = self.C.Γs(funcOfτ['B'], τ, κ)
		funcOfτ['si/s'] = self.C.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, κ)
		funcOfτ.update(self.C.EEDerivatives(sol['Ψ'], sol['σ'], solp['∂ln(h)/∂ln(s[t-1])'], τ))
		funcOfτ['dln(s)/dτ'] = self.aux_logDev(sol, funcOfτ, 's')
		funcOfτ.update({f'dln({k})/dτ': self.aux_logDev(sol, funcOfτ, k) for k in ('Γs','h')})
		funcOfτ['dτ[t+1]/dτ']  = solp['dτ/ds[t-1]'] * funcOfτ['dln(s)/dτ'] * solp['s[t-1]']
		funcOfτ['dκ[t+1]/dτ']  = solp['dκ/ds[t-1]'] * funcOfτ['dln(s)/dτ'] * solp['s[t-1]']
		return funcOfτ

	def report_t(self, x, ν, sol, solp):
		sol['τ'], sol['κ'] = np.clip(self.m(x,'τ'), self.db['τ_l'], self.db['τ_u']), np.clip(self.m(x,'κ'), self.db['κ_l'], self.db['κ_u'])
		sol['x_unbounded'] = x
		sol['s[t-1]'] = sol['s_τ0']*(1-sol['τ'])**(-1/self.db['α'])
		sol['B']  = self.C.B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.C.Γs(sol['B'], sol['τ'], sol['κ'])
		sol.update({f'd{k}/ds[t-1]': np.gradient(sol[k], sol['s[t-1]']) for k in ('τ','κ')})
		sol['dln(h)/dln(s[t-1])'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s[t-1])'] = self.db['α']*self.db['ξ']*(1+sol['Ψ']*(1-solp['∂ln(h)/∂ln(s[t-1])']))/((1+self.db['α']*self.db['ξ'])*sol['σ'])
		sol.update(self.C.EEDerivatives(sol['Ψ'], sol['σ'], solp['∂ln(h)/∂ln(s[t-1])'], sol['τ']))
		return sol

	### FH, terminal state
	def solveVeryRobust_T(self, ν, grids = None, weights = None, **kwargs):
		try:
			return self.solveVector_T(ν, **kwargs)
		except AssertionError:
			gridSol = self.solveGridSC_T(ν, grids = grids, weights = weights)
			try:
				return self.solveVector_T(ν, x0 = gridSol['x_unbounded'])
			except AssertionError:
				# print(f"""Warning: Could only solve model with grid search in terminal state T, ρ = {self.db['ρ']}, ξ: {self.db['ξ']}""")
				return gridSol

	def solveRobust_T(self, ν, grids = None, weights = None, **kwargs):
		""" Use grid-search to get x0 """
		try:
			return self.solveVector_T(ν, **kwargs)
		except AssertionError:
			gridSol = self.solveGridSC_T(ν, grids = grids, weights = weights)
			return self.solveVector_T(ν, x0 = gridSol['x_unbounded'])

	def solveVector_T(self, ν, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid'])
		x = optimize.root(lambda x: self.objective_T(self.m(x,'τ'), self.m(x,'κ'), ν, sol), x0)
		assert x['success'], f""" Could not identify ESC solution (ESC.solveVector_T) with parameters:
		ν: {ν}"""
		return self.report_T(x['x'], ν, sol)

	def solveAdHoc_T(self, ν, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(x0, ν, sol)

	def solveScalarLoop_T(self, ν, x0_from_loop = True, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid'])
		esc = np.empty((self.m.ngrid, 2))
		x = optimize.root(lambda x: self.objective_T(x[0], x[1], ν, aux_soli(sol, 0)), x0.reshape((2, self.m.ngrid)).T[0])
		esc[0] = x['x']
		assert x['success'], f""" Couldn't identify ESC with ν = {ν}, loop i=0."""
		for i in range(1, self.m.ngrid):
			x = optimize.root(lambda x: self.objective_T(x[0], x[1], ν, aux_soli(sol, i)), esc[i-1] if x0_from_loop else x0.reshape((2, self.m.ngrid)).T[i])
			esc[i] = x['x']
			assert x['success'], f""" Couldn't identify ESC iteration i = {i} """
		return self.report_T(np.hstack([esc[:,0], esc[:,1]]), ν, sol)

	def solveGridSearch_T(self, ν, grids = None, **kwargs):
		grids1d = noneInit(grids, self.grids)
		idx1ds, idxnd, gridsnd = self.createGrids(grids1d)
		sol = self.precomputations_T(gridsnd['s'].values)
		objective = self.objective_T(gridsnd['τ'].values, gridsnd['κ'].values, ν, sol)
		idxMin = pd.Series(abs(objective).reshape((2,len(idxnd))).sum(axis=0), index = idxnd).groupby(['sIdx']).idxmin()
		x = np.hstack([gridsnd[k][idxMin].values for k in ('τ','κ')])
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(x, ν, sol)

	def solveGridSC_T(self, ν, grids = None, **kwargs):
		""" Solve by sequentially looking for sign changes"""
		grids1d = noneInit(grids, self.grids).copy()
		idx1ds, idxnd, gridsnd = self.createGrids(grids1d)
		sol = self.precomputations_T(gridsnd['s'].values)
		objective = pd.DataFrame(self.objective_T(gridsnd['τ'].values, gridsnd['κ'].values, ν, sol).reshape((2,len(idxnd))).T, 
								index = idxnd, columns = ['τ','κ'])
		obj2d = objective['κ'].unstack('κ').values.T
		κ2d = gridsnd['κ'].unstack('κ').values.T
		changeSign = np.diff(np.sign(obj2d), axis = 0) <0
		κ = inverseInterp1d(κ2d[:-1,:][changeSign],κ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		# identify τ:
		grids1d.pop('κ'); # remove 'κ' from grids
		idx1ds, idxnd, gridsnd = self.createGrids(grids1d)
		sol = self.precomputations_T(gridsnd['s'].values)
		objective = pd.DataFrame(self.objective_T(gridsnd['τ'].values, κ, ν, sol).reshape((2,len(idxnd))).T, 
								index = idxnd, columns = ['τ','κ'])
		obj2d = objective['τ'].unstack('τ').values.T
		τ2d = gridsnd['τ'].unstack('τ').values.T
		κ2d = pd.Series(κ, index = idxnd).unstack('τ').values.T
		changeSign = np.diff(np.sign(obj2d), axis = 0) <0
		τ = inverseInterp1d(τ2d[:-1,:][changeSign],τ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		κ = inverseInterp1d(κ2d[:-1,:][changeSign],κ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(np.hstack([τ,κ]), ν, sol)

	def solveGrid_T(self, ν, grids = None, weights = None, **kwargs):
		""" Neutral weights is a dict {'τ': 1, 'κ': 1, 'eps': 1}"""
		grids1d = noneInit(grids, self.grids)
		idx1ds, idxnd, gridsnd = self.createGrids(grids1d)
		sol = self.precomputations_T(gridsnd['s'].values)
		objective = pd.DataFrame(self.objective_T(gridsnd['τ'].values, gridsnd['κ'].values, ν, sol).reshape((2,len(idxnd))).T, 
								index = idxnd, columns = ['τ','κ'])
		weightedObj = abs(objective.mul(weights) if weights else objective).sum(axis=1)
		subsetGrid = {k: boolOperator(objective, {'τ': k[0], 'κ': k[1]}) for k in [(0,0), (1,1)]} 
		nodes = {k: weightedObj[subsetGrid[k]].groupby('sIdx').idxmin() for k in subsetGrid}
		x = self.interp2dSol(objective.loc[nodes[(0,0)]], objective.loc[nodes[(1,1)]],
							 {k: gridsnd[k].loc[nodes[(0,0)]] for k in ('τ','κ')}, {k: gridsnd[k].loc[nodes[(1,1)]] for k in ('τ','κ')})
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(x,ν,sol)

	def precomputations_T(self, sidx):
		return {'s' : np.zeros(len(sidx)), 's[t-1]': sidx}

	def objective_T(self, τ, κ, ν, sol):
		τBound, κBound = np.clip(τ, self.db['τ_l'], self.db['τ_u']), np.clip(κ, self.db['κ_l'], self.db['κ_u'])
		funcOfτ = self.func_T(τBound, κBound, ν, sol)
		return self.C.ESC_T(τBound = τBound, κBound = κBound, τ = τ, κ = κ, ν = ν, s_ = sol['s[t-1]'], h = funcOfτ['h'], dlnh_Dτ = funcOfτ['dln(h)/dτ'], sSpread = funcOfτ['si/s'])

	def func_T(self, τ, κ, ν, sol):
		funcOfτ = {'h': self.C.h_T(τ, sol['s[t-1]'], ν), 'dln(h)/dτ': -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-τ))}
		funcOfτ['B'] = self.C.B(sol['s[t-1]'], funcOfτ['h'], ν)
		funcOfτ['Γs'] = self.C.Γs(funcOfτ['B'], τ, κ)
		funcOfτ['si/s'] = self.C.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, κ)
		return funcOfτ

	def report_T(self, x, ν, sol):
		""" Return solution dictionary given vector of taxes"""
		sol['τ'], sol['κ'] = np.clip(self.m(x,'τ'), self.db['τ_l'], self.db['τ_u']), np.clip(self.m(x,'κ'), self.db['κ_l'], self.db['κ_u'])
		sol['x_unbounded'] = x
		sol['h'] = self.C.h_T(sol['τ'], sol['s[t-1]'], ν)
		sol['B']  = self.C.B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.C.Γs(sol['B'], sol['τ'], sol['κ'])
		sol.update({f'd{k}/ds[t-1]': np.gradient(sol[f'{k}'], sol['s[t-1]']) for k in ('τ','κ')})
		sol['dln(h)/dln(s[t-1])'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s[t-1])'] = np.full(self.m.ngrid, self.C.power_h)
		sol['∂ln(h)/∂τ'] = -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-sol['τ']))
		return sol


class PEE:
	def __init__(self, m):
		self.m = m
		self.C = m.C
		self.db = m.db
		self.x0 = dict.fromkeys(self.db['t'], np.full(self.m.ngrid, .2))
		self.grids = defaultGrid('τ', self.db)
		self.kwargs_T = {'style': 'Vector'}
		self.kwargs_t = {'style': 'Vector', 'x0_from_solp': True}

	def nGrids(self, n = None, l = None, u = None, kl = None, ku = None):
		return defaultGrid_(noneInit(n, self.db['τ_n']), noneInit(l, self.db[f'τ_l']), noneInit(u, self.db[f'τ_u']), noneInit(kl, self.db[f'kτ_l']), noneInit(ku, self.db[f'kτ_u']))

	@property
	def FH_kwargs(self):
		return dict.fromkeys(self.db['txE'], self.kwargs_t) | {self.m.T-1: self.kwargs_T}

	def FH(self, pars = None):
		""" Return dict of policy functions. If kwargs are passed, this should be a dict of kwargs for each t. """
		kwargs = noneInit(pars, self.FH_kwargs)
		sols = dict.fromkeys(self.db['t'])
		sols[self.m.T-1] = self.solve(self.db['κ'].iloc[-1], self.db['ν'][-1], t = 'T', **({'x0': self.x0[self.m.T-1]} | kwargs[self.m.T-1]))
		for t in range(self.m.T-2, -1, -1):
			sols[t] = self.resampleSolution(self.solve(self.db['κ'].values[t], self.db['ν'][t],
											 κp = self.db['κ'].values[t+1], νp = self.db['ν'][t+1], solp = sols[t+1], **({'x0': self.x0[t]} | kwargs[t])))
		return sols

	def resampleSolution(self, sol):
		d = {k: interpSol(self.db['sGrid'], sol['s[t-1]'], v) for k,v in sol.items() if k != 's[t-1]'}
		d['s[t-1]'] = self.db['sGrid']
		return d

	def createGrids(self, grid1d):
		idx1d = pd.Index(range(grid1d.size), name = 'τ')
		idxnd = pd.MultiIndex.from_product([idx1d, self.db['sIdx']])
		gridnd_τ = pd.Series(0, index = idxnd).add(pd.Series(grid1d, index = idx1d))
		gridnd_s = pd.Series(0, index = idxnd).add(pd.Series(self.db['sGrid'], index = self.db['sIdx']))
		return {'idx1d': idx1d, 'grid1d': grid1d, 'idxnd': idxnd, 'gridnd_τ': gridnd_τ, 'gridnd_s': gridnd_s}

	def interpGridSolFromSC(self, grids, objective, n):
		""" Assumes a single crossing """
		o = objective.reshape((n, self.m.ngrid))
		τ = grids['gridnd_τ'].values.reshape((n, self.m.ngrid))
		changeSign = np.diff(np.sign(o), axis = 0) < 0
		return inverseInterp1d(τ[:-1].T[changeSign.T], τ[1:].T[changeSign.T], o[:-1].T[changeSign.T], o[1:].T[changeSign.T])

	def mapToIdxnd(self, k, idxnd, solp):
		if solp[k].ndim == 1:
			return pd.Series(0, index = idxnd).add(pd.Series(solp[k], index = self.db['sIdx'])).values
		else:
			return pd.DataFrame(0, index = idxnd, columns = self.db['i']).add(pd.DataFrame(solp[k].T, index = self.db['sIdx'], columns = self.db['i'])).values.T

	def interpGridSol(self, y, x):
		""" Interpolate tax level that solves for y = 0"""
		id1, id2 = y[y>0].groupby('sIdx').idxmin(), y[y<0].groupby('sIdx').idxmax()
		y1, x1 = y[id1].values, x[id1].values
		y2, x2 = y[id2].values, x[id2].values
		return x1+y1*(x2-x1)/(y1-y2)

	def aux_strategy(self, sol, solp, k):
		return sol[f'∂ln({k})/∂τ[t+1]'] * solp['dτ/ds[t-1]'] * sol['s']

	def aux_logDev(self, sol, funcOfτ, k):
		if k == 's':
			return funcOfτ[f'∂ln({k})/∂τ']/(1-sol[f'{k}_strategy'])
		else:
			return funcOfτ[f'∂ln({k})/∂τ']+sol[f'{k}_strategy'] * funcOfτ['dln(s)/dτ']

	######## Std. period t:
	def solve(self, κ, ν, style = 'Vector', t = 't', **kwargs):
		return getattr(self, f'solve{style}_{t}')(κ, ν, **kwargs)

	def solveRobust_t(self, κ, ν, κp = None, νp = None, solp = None, grids = None, **kwargs):
		""" Use grid-search to get x0 """
		gridSol = self.solveGridSC_t(κ, ν, κp = κp, νp = νp, solp = solp, grids = grids)
		return self.solveVector_t(κ, ν, κp = κp, νp = νp, solp = solp, x0_from_solp = False, x0 = gridSol['τ_unbounded'])

	def solveVector_t(self, κ, ν, κp = None, νp = None, solp = None, x0_from_solp = True, x0 = None, **kwargs):
		sol = self.precomputations_t(κp, ν, solp)
		x = optimize.root(lambda τ: self.objective_t(τ, κ, ν, κp, νp, sol, solp), x0_solp(x0, solp, x0_from_solp))
		assert x['success'], f""" Could not identify PEE solution (PEE.solveVector_t) with parameters:
		κ: {κ}, ν: {ν}, κ[t+1]: {κp}, ν[t+1]: {νp}, and solution from t+1 with taxes
		τ[t+1]: {solp['τ']}"""
		return self.report_t(x['x'], κ, ν, sol, solp)

	def solveScalarLoop_t(self, κ, ν, κp = None, νp = None, solp = None, x0_from_loop = True, x0_from_solp = True, x0 = None, **kwargs):
		sol = self.precomputations_t(κp, ν, solp)
		τ = np.empty(self.m.ngrid)
		x = optimize.root(lambda τ: self.objective_t(τ, κ, ν, κp, νp, aux_soli(sol, 0), aux_soli(solp,0)), x0_solp(x0, solp, x0_from_solp)[0])
		τ[0] = x['x']
		assert x['success'], f""" Couldn't identify PEE with ν = {ν}, loop i=0."""
		for i in range(1, self.m.ngrid):
			x = optimize.root(lambda τ: self.objective_t(τ, κ, ν, κp, νp, aux_soli(sol, i), aux_soli(solp,i)), τ[i-1] if x0_from_loop else x0_solp(x0, solp, x0_from_solp)[i])
			τ[i] = x['x']
			assert x['success'], f""" Couldn't identify PEE iteration i = {i} out of {self.m.ngrid} """
		return self.report_t(τ, κ, ν, sol, solp)

	def solveGridSearch_t(self, κ, ν, κp = None, νp = None, solp = None, grids = None, **kwargs):
		grids = self.createGrids(noneInit(grids, self.grids))
		solp_nd = {k: self.mapToIdxnd(k, grids['idxnd'], solp) for k in solp} # map solp to the full grid
		sol_nd = self.precomputations_t(κp, ν, solp_nd)
		objective = self.objective_t(grids['gridnd_τ'].values, κ, ν, κp, νp, sol_nd, solp_nd)
		idxMin = pd.Series(abs(objective), index = grids['idxnd']).groupby(['sIdx']).idxmin()
		τ = grids['gridnd_τ'][grids['gridnd_τ'].index.isin(idxMin.values)].values
		sol = self.precomputations_t(κp, ν, solp)
		return self.report_t(τ, κ, ν, sol, solp)

	def solveGrid_t(self, κ, ν, κp = None, νp = None, solp = None, grids = None, **kwargs):
		grids = self.createGrids(noneInit(grids, self.grids))
		solp_nd = {k: self.mapToIdxnd(k, grids['idxnd'], solp) for k in solp} # map solp to the full grid
		sol_nd = self.precomputations_t(κp, ν, solp_nd)
		objective = self.objective_t(grids['gridnd_τ'].values, κ, ν, κp, νp, sol_nd, solp_nd)
		τ = self.interpGridSol(pd.Series(objective, index = grids['idxnd']), pd.Series(grids['gridnd_τ'], index = grids['idxnd']))
		sol = self.precomputations_t(κp, ν, solp)
		return self.report_t(τ, κ, ν, sol, solp)

	def solveGridSC_t(self, κ, ν, κp = None, νp = None, solp = None, grids = None, **kwargs):
		grids = self.createGrids(noneInit(grids, self.grids))
		solp_nd = {k: self.mapToIdxnd(k, grids['idxnd'], solp) for k in solp} # map solp to the full grid
		sol_nd = self.precomputations_t(κp, ν, solp_nd)
		objective = self.objective_t(grids['gridnd_τ'].values, κ, ν, κp, νp, sol_nd, solp_nd)
		τ = self.interpGridSolFromSC(grids, objective, grids['grid1d'].size)
		sol = self.precomputations_t(κp, ν, solp)
		return self.report_t(τ, κ, ν, sol, solp)

	def precomputations_t(self, κp, ν, solp):
		sol = { 's': solp['s[t-1]'], 
				'h': (solp['s[t-1]']/solp['Γs'])**(self.db['ξ']/(1+self.db['ξ'])), 
				'Ω': self.C.Ω(Γs = solp['Γs'], τp = solp['τ'], κp = κp), 
				'Ψ': self.C.Ψ(Bp = solp['B'] , τp = solp['τ'], κp = κp)}
		sol['s_τ0'] = self.C.aux_sτ0(h = sol['h'], ν = ν, κp= κp, τp = solp['τ'], Γs = solp['Γs'])
		sol.update(self.C.EELaggedDerivatives_τ(Ω = sol['Ω'], Ψ = sol['Ψ'], Bp = solp['B'], dlnhp_dτp = solp['∂ln(h)/∂τ'], τp = solp['τ'], κp = κp))
		sol['σ'] = self.C.σ(Ω = sol['Ω'], Ψ = sol['Ψ'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'], τp = solp['τ'], κp = κp)
		sol.update({f'{k}_strategy': self.aux_strategy(sol, solp, k) for k in ('s','Γs','h')})
		return sol

	def objective_t(self, τ, κ, ν, κp, νp, sol, solp):
		τBound = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		funcOfτ = self.func_t(τBound, κ, ν, sol, solp)
		return self.C.PEE_t(τBound = τBound, τ = τ, κ = κ, ν = ν, τp = solp['τ'], κp = κp, νp = νp, 
							s_ = funcOfτ['s[t-1]'], s = sol['s'], h = sol['h'], hp = solp['h'], Γs = solp['Γs'], Bp = solp['B'], 
							dlnh_Dτ = funcOfτ['dln(h)/dτ'], dlns_Dτ = funcOfτ['dln(s)/dτ'], dlnΓs_Dτ = funcOfτ['dln(Γs)/dτ'], dlnhp_Dlns = solp['dln(h)/dln(s[t-1])'], dτp_dτ = funcOfτ['dτ[t+1]/dτ'], sSpread = funcOfτ['si/s'])

	def func_t(self, τ, κ, ν, sol, solp):
		""" Return functions of τ on the grid of s"""
		funcOfτ = {'s[t-1]': sol['s_τ0']*(1-τ)**(-1/self.db['α'])}
		funcOfτ['B'] = self.C.B(funcOfτ['s[t-1]'], sol['h'], ν)
		funcOfτ['Γs'] = self.C.Γs(funcOfτ['B'], τ, κ)
		funcOfτ['si/s'] = self.C.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, κ)
		funcOfτ.update(self.C.EEDerivatives(sol['Ψ'], sol['σ'], solp['∂ln(h)/∂ln(s[t-1])'], τ))
		funcOfτ['dln(s)/dτ'] = self.aux_logDev(sol, funcOfτ, 's')
		funcOfτ.update({f'dln({k})/dτ': self.aux_logDev(sol, funcOfτ, k) for k in ('Γs','h')})
		funcOfτ['dτ[t+1]/dτ'] = solp['dτ/ds[t-1]'] * funcOfτ['dln(s)/dτ'] * solp['s[t-1]']
		return funcOfτ

	def report_t(self, τ, κ, ν, sol, solp):
		sol['τ'] = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		sol['τ_unbounded'] = τ
		sol['s[t-1]'] = sol['s_τ0']*(1-sol['τ'])**(-1/self.db['α'])
		sol['B']  = self.C.B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.C.Γs(sol['B'], sol['τ'], κ)
		sol['dτ/ds[t-1]'] = np.gradient(sol['τ'], sol['s[t-1]'])
		sol['dln(h)/dln(s[t-1])'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s[t-1])'] = self.db['α']*self.db['ξ']*(1+sol['Ψ']*(1-solp['∂ln(h)/∂ln(s[t-1])']))/((1+self.db['α']*self.db['ξ'])*sol['σ'])
		sol.update(self.C.EEDerivatives(sol['Ψ'], sol['σ'], solp['∂ln(h)/∂ln(s[t-1])'], sol['τ']))
		return sol

	######## FH terminal state:
	def solveRobust_T(self, κ, ν, grids = None, **kwargs):
		""" Use grid-search to get x0 """
		gridSol = self.solveGridSC_T(κ, ν, grids)
		return self.solveVector_T(κ, ν, x0 = gridSol['τ_unbounded'])

	def solveVector_T(self, κ, ν, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid'])
		x = optimize.root(lambda τ: self.objective_T(τ, κ, ν, sol), x0)
		assert x['success'], f""" Could not identify PEE solution (PEE.solveVector_T) with parameters:
		κ: {κ}, ν: {ν}"""
		return self.report_T(x['x'], κ, ν, sol)

	def solveScalarLoop_T(self, κ, ν, x0_from_loop = True, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid'])
		τ = np.empty(self.m.ngrid)
		x = optimize.root(lambda τ: self.objective_T(τ, κ, ν, aux_soli(sol, 0)), x0[0])
		τ[0] = x['x']
		assert x['success'], f""" Couldn't identify PEE with ν = {ν}, loop i=0."""
		for i in range(1, self.m.ngrid):
			x = optimize.root(lambda τ: self.objective_T(τ, κ, ν, aux_soli(sol, i)), τ[i-1] if x0_from_loop else x0[i])
			τ[i] = x['x']
			assert x['success'], f""" Couldn't identify PEE iteration i = {i} out of {self.m.ngrid} """
		return self.report_T(τ, κ, ν, sol)

	def solveGridSearch_T(self, κ, ν, grids = None, **kwargs):
		grids = self.createGrids(noneInit(grids, self.grids))
		sol = self.precomputations_T(grids['gridnd_s'].values)
		objective = self.objective_T(grids['gridnd_τ'].values, κ, ν, sol)
		idxMin = pd.Series(abs(objective), index = grids['idxnd']).groupby(['sIdx']).idxmin()
		τ = grids['gridnd_τ'][grids['gridnd_τ'].index.isin(idxMin.values)].values
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(τ, κ, ν, sol)

	def solveGrid_T(self, κ, ν, grids = None, **kwargs):
		grids = self.createGrids(noneInit(grids, self.grids))
		sol = self.precomputations_T(grids['gridnd_s'].values)
		objective = self.objective_T(grids['gridnd_τ'].values, κ, ν, sol)
		τ = self.interpGridSol(pd.Series(objective, index = grids['idxnd']), pd.Series(grids['gridnd_τ'], index = grids['idxnd']))
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(τ, κ, ν, sol)

	def solveGridSC_T(self, κ, ν, grids = None, **kwargs):
		grids = self.createGrids(noneInit(grids, self.grids))
		sol = self.precomputations_T(grids['gridnd_s'].values)
		objective = self.objective_T(grids['gridnd_τ'].values, κ, ν, sol)
		τ = self.interpGridSolFromSC(grids, objective, grids['grid1d'].size)
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(τ, κ, ν, sol)

	def precomputations_T(self, sidx):
		return {'s' : np.zeros(len(sidx)), 's[t-1]': sidx}

	def objective_T(self, τ, κ, ν, sol):
		τBound = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		funcOfτ = self.func_T(τBound, κ, ν, sol)
		return self.C.PEE_T(τBound = τBound, τ = τ, κ = κ, ν = ν, s_ = sol['s[t-1]'], h =funcOfτ['h'], dlnh_Dτ = funcOfτ['dln(h)/dτ'], sSpread = funcOfτ['si/s'])

	def func_T(self, τ, κ, ν, sol):
		funcOfτ = {'h': self.C.h_T(τ, sol['s[t-1]'], ν), 'dln(h)/dτ': -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-τ))}
		funcOfτ['B'] = self.C.B(sol['s[t-1]'], funcOfτ['h'], ν)
		funcOfτ['Γs'] = self.C.Γs(funcOfτ['B'], τ, κ)
		funcOfτ['si/s'] = self.C.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, κ)
		return funcOfτ

	def report_T(self, τ, κ, ν, sol):
		sol['τ'] = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		sol['τ_unbounded'] = τ
		sol['h'] = self.C.h_T(sol['τ'], sol['s[t-1]'], ν)
		sol['B']  = self.C.B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.C.Γs(sol['B'], sol['τ'], κ)
		sol['dτ/ds[t-1]'] = np.gradient(sol['τ'], sol['s[t-1]'])
		sol['dln(h)/dln(s[t-1])'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s[t-1])'] = np.full(self.m.ngrid, self.C.power_h)
		sol['∂ln(h)/∂τ'] = -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-sol['τ']))
		return sol