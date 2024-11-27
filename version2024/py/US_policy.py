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

class ESC:
	def __init__(self, m):
		self.m = m
		self.C = m.C
		self.db = m.db
		self.x0 = dict.fromkeys(self.db['t'], np.full(self.m.ns['ESCpol'].len, .2))
		self.kwargs_T = {'style': 'ScalarLoop'}
		self.kwargs_t = {'style': 'Vector', 'x0_from_solp': True}

	@property
	def grids(self):
		v = ('τ','θ','eps')
		return {'n': dict.fromkeys(v, 11), 'l': dict.fromkeys(v, -.1), 'u': dict.fromkeys(v, 1.1)}
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
		d['x_unbounded'] = interpSol(self.db['sGrid'], sol['s[t-1]'], sol['x_unbounded'].reshape((3, self.db['glob_ns']))).reshape(sol['x_unbounded'].shape)
		return d

	####### Auxiliary methods
	def createGrids(self, n = None, l = None, u = None):
		idx1ds = {k: pd.Index(range(n[k]), name = k) for k in n}
		grids1d = {k: np.linspace(l[k], u[k], n[k]) for k in n}
		idxnd = pd.MultiIndex.from_product(list(idx1ds.values())+[self.db['sIdx']])
		gridsnd = {k: pd.Series(0, index = idxnd).add(pd.Series(grids1d[k], index = idx1ds[k])) for k in n}
		gridsnd['s'] = pd.Series(0, index = idxnd).add(pd.Series(self.db['sGrid'], index = self.db['sIdx']))
		return idx1ds, grids1d, idxnd, gridsnd

	def mapToIdxnd(self, k, idxnd, solp):
		if solp[k].ndim == 1:
			return pd.Series(0, index = idxnd).add(pd.Series(solp[k], index = self.db['sIdx'])).values
		else:
			return pd.DataFrame(0, index = idxnd, columns = self.db['i']).add(pd.DataFrame(solp[k].T, index = self.db['sIdx'], columns = self.db['i'])).values.T

	def interp3dSol_(self, f0, f1, x0, x1):
		return (f1 * x0 + abs(f0) * x1)/(abs(f0)+f1)

	def interp3dSol(self, f0, f1, x0, x1):
		return np.hstack([self.interp3dSol_(f0[k].values, f1[k].values, x0[k].values, x1[k].values) for k in ('τ','θ','eps')])

	def aux_strategy(self, sol, solp, k):
		return (sol[f'∂ln({k})/∂τ[t+1]'] * solp['dτ/ds[t-1]'] + sol[f'∂ln({k})/∂θ[t+1]']*solp['dθ/ds[t-1]']+sol[f'∂ln({k})/∂eps[t+1]']*solp['deps/ds[t-1]']) * sol['s']

	def aux_logDev(self, sol, funcOfτ, k):
		if k == 's':
			return funcOfτ[f'∂ln({k})/∂τ']/(1-sol[f'{k}_strategy'])
		else:
			return funcOfτ[f'∂ln({k})/∂τ']+sol[f'{k}_strategy'] * funcOfτ['dln(s)/dτ']

	######## Std. period t:
	def solve(self, ν, style = 'Vector', t = 't', **kwargs):
		return getattr(self, f'solve{style}_{t}')(ν, **kwargs)

	def solveRobust_t(self, ν, νp = None, solp = None, n = None, l = None, u = None, weights = None, **kwargs):
		""" Use grid-search to get x0 """
		gridSol = self.solveGridSC_t(ν, νp = νp, solp = solp, n = n, l = l, u = u, weights = weights)
		return self.solveVector_t(ν, νp = νp, solp = solp, x0_from_solp = False, x0 = gridSol['x_unbounded'])

	def solveVector_t(self, ν, νp = None, solp = None, x0_from_solp = True, x0 = None, **kwargs):
		sol = self.precomputations_t(ν, solp)
		x = optimize.root(lambda x: self.objective_t(self.m(x,'τ'), self.m(x, 'θ'), self.m(x,'eps'), ν, νp, sol, solp), x0_solp(x0, solp, x0_from_solp, var = 'x_unbounded'))
		assert x['success'], f""" Couldn't identify ESC solution (ESC.solveVector_t) with ν = {ν} and previous solution:
		τ: {solp['τ']}
		θ: {solp['θ']}
		ε: {solp['eps']}"""
		return self.report_t(x['x'], ν, sol, solp)

	def solveScalarLoop_t(self, ν, νp = None, solp = None, x0_from_loop = True, x0_from_solp = True, x0 = None, **kwargs):
		sol = self.precomputations_t(ν, solp)
		esc = np.empty((self.db['glob_ns'], 3))
		x = optimize.root(lambda x: self.objective_t(x[0], x[1], x[2], ν, νp, aux_soli(sol, 0), aux_soli(solp,0)), x0_solp(x0, solp, x0_from_solp, var = 'x_unbounded').reshape((3, self.db['glob_ns'])).T[0])
		esc[0] = x['x']
		assert x['success'], f""" Couldn't identify ESC with ν = {ν}, loop i=0."""
		for i in range(1, self.db['glob_ns']):
			x = optimize.root(lambda x: self.objective_t(x[0], x[1], x[2], ν, νp, aux_soli(sol, i), aux_soli(solp,i)), esc[i-1] if x0_from_loop else x0_solp(x0, solp, x0_from_solp, var = 'x_unbounded').reshape((3, self.db['glob_ns'])).T[i])
			esc[i] = x['x']
			assert x['success'], f""" Couldn't identify ESC iteration i = {i} """
		return self.report_t(np.hstack([esc[:,0], esc[:,1], esc[:,2]]), ν, sol, solp)

	def solveGridSearch_t(self, ν, νp = None, solp = None, n = None, l = None, u = None, **kwargs):
		idx1ds, grids1d, idxnd, gridsnd = self.createGrids(n = n, l = l, u = u)
		solp_nd = {k: self.mapToIdxnd(k, idxnd, solp) for k in solp if k != 'x_unbounded'}
		sol_nd  = self.precomputations_t(ν, solp_nd)
		objective = self.objective_t(gridsnd['τ'].values, gridsnd['θ'].values, gridsnd['eps'].values, ν, νp, sol_nd, solp_nd)
		idxMin = pd.Series(abs(objective).reshape((3,len(idxnd))).sum(axis=0), index = idxnd).groupby(['sIdx']).idxmin()
		x = np.hstack([gridsnd[k][idxMin].values for k in ('τ','θ','eps')])
		sol = self.precomputations_t(ν, solp)
		return self.report_t(x, ν, sol, solp)

	def solveGridSC_t(self, ν, νp = None, solp = None, n = None, l = None, u = None, **kwargs):
		""" Solve by sequentially looking for sign changes"""
		n, l, u = n.copy(), l.copy(), u.copy()
		idx1ds, grids1d, idxnd, gridsnd = self.createGrids(n = n, l = l, u = u)
		solp_nd = {k: self.mapToIdxnd(k, idxnd, solp) for k in solp if k != 'x_unbounded'}
		sol_nd  = self.precomputations_t(ν, solp_nd)
		objective = pd.DataFrame(self.objective_t(gridsnd['τ'].values, gridsnd['θ'].values, gridsnd['eps'].values, ν, νp, sol_nd, solp_nd).reshape((3,len(idxnd))).T, 
								index = idxnd, columns = ['τ','θ','eps'])
		obj2d = objective['eps'].unstack('eps').values.T
		eps2d = gridsnd['eps'].unstack('eps').values.T
		changeSign = np.diff(np.sign(obj2d), axis = 0) <0
		eps = inverseInterp1d(eps2d[:-1,:][changeSign],eps2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		# Identify theta:
		[d.pop('eps') for d in (n,l,u)]; # remove epsilon grid
		idx1ds, grids1d, idxnd, gridsnd = self.createGrids(n = n, l = l, u = u)
		solp_nd = {k: self.mapToIdxnd(k, idxnd, solp) for k in solp if k != 'x_unbounded'}
		sol_nd  = self.precomputations_t(ν, solp_nd)
		objective = pd.DataFrame(self.objective_t(gridsnd['τ'].values, gridsnd['θ'].values, eps, ν, νp, sol_nd, solp_nd).reshape((3,len(idxnd))).T, 
								index = idxnd, columns = ['τ','θ','eps'])
		obj2d = objective['θ'].unstack('θ').values.T
		θ2d = gridsnd['θ'].unstack('θ').values.T
		eps2d = pd.Series(eps, index = idxnd).unstack('θ').values.T
		changeSign = np.diff(np.sign(obj2d), axis = 0) <0
		θ = inverseInterp1d(θ2d[:-1,:][changeSign],θ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		eps = inverseInterp1d(eps2d[:-1,:][changeSign],eps2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		# identify τ:
		[d.pop('θ') for d in (n,l,u)];
		idx1ds, grids1d, idxnd, gridsnd = self.createGrids(n = n, l = l, u = u)
		solp_nd = {k: self.mapToIdxnd(k, idxnd, solp) for k in solp if k != 'x_unbounded'}
		sol_nd  = self.precomputations_t(ν, solp_nd)
		objective = pd.DataFrame(self.objective_t(gridsnd['τ'].values, θ, eps, ν, νp, sol_nd, solp_nd).reshape((3,len(idxnd))).T, 
								index = idxnd, columns = ['τ','θ','eps'])
		obj2d = objective['τ'].unstack('τ').values.T
		τ2d = gridsnd['τ'].unstack('τ').values.T
		eps2d = pd.Series(eps, index = idxnd).unstack('τ').values.T
		θ2d = pd.Series(θ, index = idxnd).unstack('τ').values.T
		changeSign = np.diff(np.sign(obj2d), axis = 0) <0
		τ = inverseInterp1d(τ2d[:-1,:][changeSign],τ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		θ = inverseInterp1d(θ2d[:-1,:][changeSign],θ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		eps = inverseInterp1d(eps2d[:-1,:][changeSign],eps2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		sol = self.precomputations_t(ν, solp)
		return self.report_t(np.hstack([τ,θ,eps]), ν, sol, solp)

	def solveGrid_t(self, ν, νp = None, solp = None, n = None, l = None, u = None, weights = None, **kwargs):
		""" Neutral weights is a dict {'τ': 1, 'θ': 1, 'eps': 1}"""
		idx1ds, grids1d, idxnd, gridsnd = self.createGrids(n = n, l = l, u = u)
		solp_nd = {k: self.mapToIdxnd(k, idxnd, solp) for k in solp if k != 'x_unbounded'}
		sol_nd  = self.precomputations_t(ν, solp_nd)
		objective = pd.DataFrame(self.objective_t(gridsnd['τ'].values, gridsnd['θ'].values, gridsnd['eps'].values, ν, νp, sol_nd, solp_nd).reshape((3,len(idxnd))).T,
								index = idxnd, columns = ['τ','θ','eps'])
		weightedObj = abs(objective.mul(weights) if weights else objective).sum(axis=1)
		subsetGrid = {k: boolOperator(objective, {'τ': k[0], 'θ': k[1], 'eps': k [2]}) for k in [(0,0,0), (1,1,1)]} 
		# subsetGrid = {k: boolOperator(objective, {'τ': k[0], 'θ': k[1], 'eps': k [2]}) for k in itertools.product(*[[0,1]]*3)} # do all 8 combinations
		nodes = {k: weightedObj[subsetGrid[k]].groupby('sIdx').idxmin() for k in subsetGrid}
		x = self.interp3dSol(objective.loc[nodes[(0,0,0)]], objective.loc[nodes[(1,1,1)]],
							 {k: gridsnd[k].loc[nodes[(0,0,0)]] for k in ('τ','θ','eps')}, {k: gridsnd[k].loc[nodes[(1,1,1)]] for k in ('τ','θ','eps')})
		sol = self.precomputations_t(ν, solp)
		return self.report_t(x,ν,sol, solp)

	def precomputations_t(self, ν, solp):
		sol = { 's': solp['s[t-1]'], 
				'h': (solp['s[t-1]']/solp['Γs'])**(self.db['ξ']/(1+self.db['ξ'])), 
				'Ω': self.C.Ω(Γs = solp['Γs'], τp = solp['τ'], θp = solp['θ'], epsp = solp['eps']), 
				'Ψ': self.C.Ψ(Bp = solp['B'] , τp = solp['τ'], θp = solp['θ'], epsp = solp['eps'])}		
		sol['s_τ0'] = self.C.aux_sτ0(h = sol['h'], ν = ν, τp = solp['τ'], θp= solp['θ'], epsp = solp['eps'], Γs = solp['Γs'])
		sol.update(self.C.EELaggedDerivatives_τ(Ω = sol['Ω'], Ψ = sol['Ψ'], Bp = solp['B'], dlnhp_dτp = solp['∂ln(h)/∂τ'], τp = solp['τ'], θp = solp['θ'], epsp = solp['eps']))
		sol['σ'] = self.C.σ(Ω = sol['Ω'], Ψ = sol['Ψ'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'], τp = solp['τ'], θp = solp['θ'])
		sol.update(self.C.EELaggedDerivatives_θ(Ω = sol['Ω'], Ψ = sol['Ψ'], σ = sol['σ'], Bp = solp['B'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'], τp = solp['τ'], θp = solp['θ'], epsp = solp['eps']))
		sol.update(self.C.EELaggedDerivatives_eps(Ω = sol['Ω'], Ψ = sol['Ψ'], σ = sol['σ'], Bp = solp['B'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'], τp = solp['τ'], θp = solp['θ'], epsp = solp['eps']))
		sol.update({f'{k}_strategy': self.aux_strategy(sol, solp, k) for k in ('s','Γs','h')})
		return sol

	def objective_t(self, τ, θ, eps, ν, νp, sol, solp):
		τBound, θBound, epsBound = np.clip(τ, self.db['τ_l'], self.db['τ_u']), np.clip(θ, self.db['θ_l'], self.db['θ_u']), np.clip(eps, self.db['eps_l'], self.db['eps_u'])
		funcOfτ = self.func_t(τBound, θBound, epsBound, ν, sol, solp)
		return self.C.ESC_t(τBound = τBound, θBound = θBound, epsBound = epsBound, τ = τ, θ = θ, eps = eps, ν = ν,
																   τp= solp['τ'], θp= solp['θ'], epsp= solp['eps'], νp= νp,
							s_ = funcOfτ['s[t-1]'], s = sol['s'], h = sol['h'], hp= solp['h'], Γs =solp['Γs'], Bp = solp['B'], sSpread = funcOfτ['si/s'],
							dlnh_Dτ = funcOfτ['dln(h)/dτ'], dlns_Dτ = funcOfτ['dln(s)/dτ'], dlnΓs_Dτ = funcOfτ['dln(Γs)/dτ'], dlnhp_Dlns = solp['dln(h)/dln(s[t-1])'], 
							dτp_dτ = funcOfτ['dτ[t+1]/dτ'], dθp_dτ = funcOfτ['dθ[t+1]/dτ'], depsp_dτ = funcOfτ['deps[t+1]/dτ'])

	def func_t(self, τ, θ, eps, ν, sol, solp):
		""" Return functions of τ on the grid of s"""
		funcOfτ = {'s[t-1]': sol['s_τ0']*(1-τ)**(-1/self.db['α'])}
		funcOfτ['B'] = self.C.B(funcOfτ['s[t-1]'], sol['h'], ν)
		funcOfτ['Γs'] = self.C.Γs(funcOfτ['B'], τ, θ, eps)
		funcOfτ['si/s'] = self.C.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, θ, eps)
		funcOfτ.update(self.C.EEDerivatives(sol['Ψ'], sol['σ'], solp['∂ln(h)/∂ln(s[t-1])'], τ))
		funcOfτ['dln(s)/dτ'] = self.aux_logDev(sol, funcOfτ, 's')
		funcOfτ.update({f'dln({k})/dτ': self.aux_logDev(sol, funcOfτ, k) for k in ('Γs','h')})
		funcOfτ['dτ[t+1]/dτ']  = solp['dτ/ds[t-1]'] * funcOfτ['dln(s)/dτ'] * solp['s[t-1]']
		funcOfτ['dθ[t+1]/dτ']  = solp['dθ/ds[t-1]'] * funcOfτ['dln(s)/dτ'] * solp['s[t-1]']
		funcOfτ['deps[t+1]/dτ']= solp['deps/ds[t-1]'] * funcOfτ['dln(s)/dτ'] * solp['s[t-1]']
		return funcOfτ

	def report_t(self, x, ν, sol, solp):
		sol['τ'], sol['θ'], sol['eps'] = np.clip(self.m(x,'τ'), self.db['τ_l'], self.db['τ_u']), np.clip(self.m(x,'θ'), self.db['θ_l'], self.db['θ_u']), np.clip(self.m(x,'eps'), self.db['eps_l'], self.db['eps_u'])
		sol['x_unbounded'] = x
		sol['s[t-1]'] = sol['s_τ0']*(1-sol['τ'])**(-1/self.db['α'])
		sol['B']  = self.C.B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.C.Γs(sol['B'], sol['τ'], sol['θ'], sol['eps'])
		sol.update({f'd{k}/ds[t-1]': np.gradient(sol[k], sol['s[t-1]']) for k in ('τ','θ','eps')})
		sol['dln(h)/dln(s[t-1])'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s[t-1])'] = self.db['α']*self.db['ξ']*(1+sol['Ψ']*(1-solp['∂ln(h)/∂ln(s[t-1])']))/((1+self.db['α']*self.db['ξ'])*sol['σ'])
		sol.update(self.C.EEDerivatives(sol['Ψ'], sol['σ'], solp['∂ln(h)/∂ln(s[t-1])'], sol['τ']))
		return sol

	### FH, terminal state
	def solveRobust_T(self, ν, n = None, l = None, u = None, weights = None, **kwargs):
		""" Use grid-search to get x0 """
		gridSol = self.solveGridSC_T(ν, n = n, l = l, u = u, weights = weights)
		return self.solveVector_T(ν, x0 = gridSol['x_unbounded'])

	def solveVector_T(self, ν, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid'])
		x = optimize.root(lambda x: self.objective_T(self.m(x,'τ'), self.m(x,'θ'), self.m(x,'eps'), ν, sol), x0)
		assert x['success'], f""" Could not identify ESC solution (ESC.solveVector_T) with parameters:
		ν: {ν}"""
		return self.report_T(x['x'], ν, sol)

	def solveAdHoc_T(self, ν, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(x0, ν, sol)

	def solveScalarLoop_T(self, ν, x0_from_loop = True, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid'])
		esc = np.empty((self.db['glob_ns'], 3))
		x = optimize.root(lambda x: self.objective_T(x[0], x[1], x[2], ν, aux_soli(sol, 0)), x0.reshape((3, self.db['glob_ns'])).T[0])
		esc[0] = x['x']
		assert x['success'], f""" Couldn't identify ESC with ν = {ν}, loop i=0."""
		for i in range(1, self.db['glob_ns']):
			x = optimize.root(lambda x: self.objective_T(x[0], x[1], x[2], ν, aux_soli(sol, i)), esc[i-1] if x0_from_loop else x0.reshape((3, self.db['glob_ns'])).T[i])
			esc[i] = x['x']
			assert x['success'], f""" Couldn't identify ESC iteration i = {i} """
		return self.report_T(np.hstack([esc[:,0], esc[:,1], esc[:,2]]), ν, sol)

	def solveGridSearch_T(self, ν, n = None, l = None, u = None, **kwargs):
		idx1ds, grids1d, idxnd, gridsnd = self.createGrids(n = n, l = l, u = u)
		sol = self.precomputations_T(gridsnd['s'].values)
		objective = self.objective_T(gridsnd['τ'].values, gridsnd['θ'].values, gridsnd['eps'].values, ν, sol)
		idxMin = pd.Series(abs(objective).reshape((3,len(idxnd))).sum(axis=0), index = idxnd).groupby(['sIdx']).idxmin()
		x = np.hstack([gridsnd[k][idxMin].values for k in ('τ','θ','eps')])
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(x, ν, sol)

	def solveGridSC_T(self, ν, n = None, l = None, u = None, **kwargs):
		""" Solve by sequentially looking for sign changes"""
		n, l, u = n.copy(), l.copy(), u.copy()
		idx1ds, grids1d, idxnd, gridsnd = self.createGrids(n = n, l = l, u = u)
		sol = self.precomputations_T(gridsnd['s'].values)
		objective = pd.DataFrame(self.objective_T(gridsnd['τ'].values, gridsnd['θ'].values, gridsnd['eps'].values, ν, sol).reshape((3,len(idxnd))).T, 
								index = idxnd, columns = ['τ','θ','eps'])
		obj2d = objective['eps'].unstack('eps').values.T
		eps2d = gridsnd['eps'].unstack('eps').values.T
		changeSign = np.diff(np.sign(obj2d), axis = 0) <0
		eps = inverseInterp1d(eps2d[:-1,:][changeSign],eps2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		# Identify theta:
		[d.pop('eps') for d in (n,l,u)]; # remove epsilon grid
		idx1ds, grids1d, idxnd, gridsnd = self.createGrids(n = n, l = l, u = u)
		sol = self.precomputations_T(gridsnd['s'].values)
		objective = pd.DataFrame(self.objective_T(gridsnd['τ'].values, gridsnd['θ'].values, eps, ν, sol).reshape((3,len(idxnd))).T, 
								index = idxnd, columns = ['τ','θ','eps'])
		obj2d = objective['θ'].unstack('θ').values.T
		θ2d = gridsnd['θ'].unstack('θ').values.T
		eps2d = pd.Series(eps, index = idxnd).unstack('θ').values.T
		changeSign = np.diff(np.sign(obj2d), axis = 0) <0
		θ = inverseInterp1d(θ2d[:-1,:][changeSign],θ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		eps = inverseInterp1d(eps2d[:-1,:][changeSign],eps2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		# identify τ:
		[d.pop('θ') for d in (n,l,u)];
		idx1ds, grids1d, idxnd, gridsnd = self.createGrids(n = n, l = l, u = u)
		sol = self.precomputations_T(gridsnd['s'].values)
		objective = pd.DataFrame(self.objective_T(gridsnd['τ'].values, θ, eps, ν, sol).reshape((3,len(idxnd))).T, 
								index = idxnd, columns = ['τ','θ','eps'])
		obj2d = objective['τ'].unstack('τ').values.T
		τ2d = gridsnd['τ'].unstack('τ').values.T
		eps2d = pd.Series(eps, index = idxnd).unstack('τ').values.T
		θ2d = pd.Series(θ, index = idxnd).unstack('τ').values.T
		changeSign = np.diff(np.sign(obj2d), axis = 0) <0
		τ = inverseInterp1d(τ2d[:-1,:][changeSign],τ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		θ = inverseInterp1d(θ2d[:-1,:][changeSign],θ2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		eps = inverseInterp1d(eps2d[:-1,:][changeSign],eps2d[1:,:][changeSign],obj2d[:-1,:][changeSign],obj2d[1:,:][changeSign])
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(np.hstack([τ,θ,eps]), ν, sol)

	def solveGrid_T(self, ν, n = None, l = None, u = None, weights = None, **kwargs):
		""" Neutral weights is a dict {'τ': 1, 'θ': 1, 'eps': 1}"""
		idx1ds, grids1d, idxnd, gridsnd = self.createGrids(n = n, l = l, u = u)
		sol = self.precomputations_T(gridsnd['s'].values)
		objective = pd.DataFrame(self.objective_T(gridsnd['τ'].values, gridsnd['θ'].values, gridsnd['eps'].values, ν, sol).reshape((3,len(idxnd))).T, 
								index = idxnd, columns = ['τ','θ','eps'])
		weightedObj = abs(objective.mul(weights) if weights else objective).sum(axis=1)
		subsetGrid = {k: boolOperator(objective, {'τ': k[0], 'θ': k[1], 'eps': k [2]}) for k in [(0,0,0), (1,1,1)]} 
		# subsetGrid = {k: boolOperator(objective, {'τ': k[0], 'θ': k[1], 'eps': k [2]}) for k in itertools.product(*[[0,1]]*3)} # do all 8 combinations
		nodes = {k: weightedObj[subsetGrid[k]].groupby('sIdx').idxmin() for k in subsetGrid}
		x = self.interp3dSol(objective.loc[nodes[(0,0,0)]], objective.loc[nodes[(1,1,1)]],
							 {k: gridsnd[k].loc[nodes[(0,0,0)]] for k in ('τ','θ','eps')}, {k: gridsnd[k].loc[nodes[(1,1,1)]] for k in ('τ','θ','eps')})
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(x,ν,sol)

	def precomputations_T(self, sidx):
		return {'s' : np.zeros(len(sidx)), 's[t-1]': sidx}

	def objective_T(self, τ, θ, eps, ν, sol):
		τBound, θBound, epsBound = np.clip(τ, self.db['τ_l'], self.db['τ_u']), np.clip(θ, self.db['θ_l'], self.db['θ_u']), np.clip(eps, self.db['eps_l'], self.db['eps_u'])
		funcOfτ = self.func_T(τBound, θBound, epsBound, ν, sol)
		return self.C.ESC_T(τBound = τBound, θBound = θBound, epsBound = epsBound, τ = τ, θ = θ, eps = eps, ν = ν, s_ = sol['s[t-1]'], h = funcOfτ['h'], dlnh_Dτ = funcOfτ['dln(h)/dτ'], sSpread = funcOfτ['si/s'])

	def func_T(self, τ, θ, eps, ν, sol):
		funcOfτ = {'h': self.C.h_T(τ, sol['s[t-1]'], ν), 'dln(h)/dτ': -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-τ))}
		funcOfτ['B'] = self.C.B(sol['s[t-1]'], funcOfτ['h'], ν)
		funcOfτ['Γs'] = self.C.Γs(funcOfτ['B'], τ, θ, eps)
		funcOfτ['si/s'] = self.C.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, θ, eps)
		return funcOfτ

	def report_T(self, x, ν, sol):
		""" Return solution dictionary given vector of taxes"""
		sol['τ'], sol['θ'], sol['eps'] = np.clip(self.m(x,'τ'), self.db['τ_l'], self.db['τ_u']), np.clip(self.m(x,'θ'), self.db['θ_l'], self.db['θ_u']), np.clip(self.m(x,'eps'), self.db['eps_l'], self.db['eps_u'])
		sol['x_unbounded'] = x
		sol['h'] = self.C.h_T(sol['τ'], sol['s[t-1]'], ν)
		sol['B']  = self.C.B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.C.Γs(sol['B'], sol['τ'], sol['θ'], sol['eps'])
		sol.update({f'd{k}/ds[t-1]': np.gradient(sol[f'{k}'], sol['s[t-1]']) for k in ('τ','θ','eps')})
		sol['dln(h)/dln(s[t-1])'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s[t-1])'] = np.full(self.db['glob_ns'], self.C.power_h)
		sol['∂ln(h)/∂τ'] = -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-sol['τ']))
		return sol


class PEE:
	def __init__(self, m):
		self.m = m
		self.C = m.C
		self.db = m.db
		self.x0 = dict.fromkeys(self.db['t'], np.full(self.db['glob_ns'], .2))
		self.kwargs_T = {'style': 'Vector'}
		self.kwargs_t = {'style': 'Vector', 'x0_from_solp': True}

	@property
	def FH_kwargs(self):
		return dict.fromkeys(self.db['txE'], self.kwargs_t) | {self.m.T-1: self.kwargs_T}

	def FH(self, pars = None):
		""" Return dict of policy functions. If kwargs are passed, this should be a dict of kwargs for each t. """
		kwargs = noneInit(pars, self.FH_kwargs)
		sols = dict.fromkeys(self.db['t'])
		sols[self.m.T-1] = self.solve(self.db['θ'].iloc[-1], self.db['eps'].iloc[-1], self.db['ν'][-1], t = 'T', **({'x0': self.x0[self.m.T-1]} | kwargs[self.m.T-1]))
		for t in range(self.m.T-2, -1, -1):
			sols[t] = self.resampleSolution(self.solve(self.db['θ'].values[t], self.db['eps'].values[t], self.db['ν'][t],
											 θp = self.db['θ'].values[t+1], epsp = self.db['eps'].values[t+1], νp = self.db['ν'][t+1], solp = sols[t+1], **({'x0': self.x0[t]} | kwargs[t])))
		return sols

	def resampleSolution(self, sol):
		d = {k: interpSol(self.db['sGrid'], sol['s[t-1]'], v) for k,v in sol.items() if k != 's[t-1]'}
		d['s[t-1]'] = self.db['sGrid']
		return d

	####### Grid search solutions
	def createGrids(self, n, l = 0, u = 1):
		idx1d = pd.Index(range(n), name = 'τ')
		grid1d = np.linspace(l,u,n)
		idxnd = pd.MultiIndex.from_product([idx1d, self.db['sIdx']])
		gridnd_τ = pd.Series(0, index = idxnd).add(pd.Series(grid1d, index = idx1d))
		gridnd_s = pd.Series(0, index = idxnd).add(pd.Series(self.db['sGrid'], index = self.db['sIdx']))
		return {'idx1d': idx1d, 'grid1d': grid1d, 'idxnd': idxnd, 'gridnd_τ': gridnd_τ, 'gridnd_s': gridnd_s}

	def interpGridSolFromSC(self, grids, objective, n):
		""" Assumes a single crossing """
		o = objective.reshape((n, self.db['glob_ns']))
		τ = grids['gridnd_τ'].values.reshape((n, self.db['glob_ns']))
		changeSign = np.diff(np.sign(o), axis = 0) < 0
		return inverseInterp1d(τ[:-1][changeSign], τ[1:][changeSign], o[:-1][changeSign], o[1:][changeSign])

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
	def solve(self, θ, eps, ν, style = 'Vector', t = 't', **kwargs):
		return getattr(self, f'solve{style}_{t}')(θ, eps, ν, **kwargs)

	def solveRobust_t(self, θ, eps, ν, θp = None, epsp = None, νp = None, solp = None, n = None, l = None, u = None, **kwargs):
		""" Use grid-search to get x0 """
		gridSol = self.solveGridSC_t(θ, eps, ν, θp = θp, epsp = epsp, νp = νp, solp = solp, n = n, l = l, u = u)
		return self.solveVector_t(θ, eps, ν, θp = θp, epsp = epsp, νp = νp, solp = solp, x0_from_solp = False, x0 = gridSol['τ_unbounded'])

	def solveVector_t(self, θ, eps, ν, θp = None, epsp = None, νp = None, solp = None, x0_from_solp = True, x0 = None, **kwargs):
		sol = self.precomputations_t(θp, epsp, ν, solp)
		x = optimize.root(lambda τ: self.objective_t(τ, θ, eps, ν, θp, epsp, νp, sol, solp), x0_solp(x0, solp, x0_from_solp))
		assert x['success'], f""" Could not identify PEE solution (PEE.solveVector_t) with parameters:
		θ: {θ}, ε: {eps}, ν: {ν}, θ[t+1]: {θp}, ε[t+1]: {epsp}, ν[t+1]: {νp}, and solution from t+1 with taxes
		τ[t+1]: {solp['τ']}"""
		return self.report_t(x['x'], θ, eps, ν, sol, solp)

	def solveScalarLoop_t(self, θ, eps, ν, θp = None, epsp = None, νp = None, solp = None, x0_from_loop = True, x0_from_solp = True, x0 = None, **kwargs):
		sol = self.precomputations_t(θp, epsp, ν, solp)
		τ = np.empty(self.db['glob_ns'])
		x = optimize.root(lambda τ: self.objective_t(τ, θ, eps, ν, θp, epsp, νp, aux_soli(sol, 0), aux_soli(solp,0)), x0_solp(x0, solp, x0_from_solp)[0])
		τ[0] = x['x']
		assert x['success'], f""" Couldn't identify PEE with ν = {ν}, loop i=0."""
		for i in range(1, self.db['glob_ns']):
			x = optimize.root(lambda τ: self.objective_t(τ, θ, eps, ν, θp, epsp, νp, aux_soli(sol, i), aux_soli(solp,i)), τ[i-1] if x0_from_loop else x0_solp(x0, solp, x0_from_solp)[i])
			τ[i] = x['x']
			assert x['success'], f""" Couldn't identify PEE iteration i = {i} out of {self.db['glob_ns']} """
		return self.report_t(τ, θ, eps, ν, sol, solp)

	def solveGridSearch_t(self, θ, eps, ν, θp = None, epsp = None, νp = None, solp = None, n = 1000, l = 0, u = 1, **kwargs):
		grids = self.createGrids(n, l = l, u = u)
		solp_nd = {k: self.mapToIdxnd(k, grids['idxnd'], solp) for k in solp} # map solp to the full grid
		sol_nd = self.precomputations_t(θp, epsp, ν, solp_nd)
		objective = self.objective_t(grids['gridnd_τ'].values, θ, eps, ν, θp, epsp, νp, sol_nd, solp_nd)
		idxMin = pd.Series(abs(objective), index = grids['idxnd']).groupby(['sIdx']).idxmin()
		τ = grids['gridnd_τ'][grids['gridnd_τ'].index.isin(idxMin.values)].values
		sol = self.precomputations_t(θp, epsp, ν, solp)
		return self.report_t(τ, θ, eps, ν, sol, solp)

	def solveGrid_t(self, θ, eps, ν, θp = None, epsp = None, νp = None, solp = None, n = 1000, l = 0, u = 1, **kwargs):
		grids = self.createGrids(n, l = l, u = u)
		solp_nd = {k: self.mapToIdxnd(k, grids['idxnd'], solp) for k in solp} # map solp to the full grid
		sol_nd = self.precomputations_t(θp, epsp, ν, solp_nd)
		objective = self.objective_t(grids['gridnd_τ'].values, θ, eps, ν, θp, epsp, νp, sol_nd, solp_nd)
		τ = self.interpGridSol(pd.Series(objective, index = grids['idxnd']), pd.Series(grids['gridnd_τ'], index = grids['idxnd']))
		sol = self.precomputations_t(θp, epsp, ν, solp)
		return self.report_t(τ, θ, eps, ν, sol, solp)

	def solveGridSC_t(self, θ, eps, ν, θp = None, epsp = None, νp = None, solp = None, n = 1000, l = -.1, u = 1.1, **kwargs):
		grids = self.createGrids(n, l = l, u = u)
		solp_nd = {k: self.mapToIdxnd(k, grids['idxnd'], solp) for k in solp} # map solp to the full grid
		sol_nd = self.precomputations_t(θp, epsp, ν, solp_nd)
		objective = self.objective_t(grids['gridnd_τ'].values, θ, eps, ν, θp, epsp, νp, sol_nd, solp_nd)
		τ = self.interpGridSolFromSC(grids, objective, n)
		sol = self.precomputations_t(θp, epsp, ν, solp)
		return self.report_t(τ, θ, eps, ν, sol, solp)

	def precomputations_t(self, θp, epsp, ν, solp):
		sol = { 's': solp['s[t-1]'], 
				'h': (solp['s[t-1]']/solp['Γs'])**(self.db['ξ']/(1+self.db['ξ'])), 
				'Ω': self.C.Ω(Γs = solp['Γs'], τp = solp['τ'], θp = θp, epsp = epsp), 
				'Ψ': self.C.Ψ(Bp = solp['B'] , τp = solp['τ'], θp = θp, epsp = epsp)}
		sol['s_τ0'] = self.C.aux_sτ0(h = sol['h'], ν = ν, θp= θp, τp = solp['τ'], epsp = epsp, Γs = solp['Γs'])
		sol.update(self.C.EELaggedDerivatives_τ(Ω = sol['Ω'], Ψ = sol['Ψ'], Bp = solp['B'], dlnhp_dτp = solp['∂ln(h)/∂τ'], τp = solp['τ'], θp = θp, epsp = epsp))
		sol['σ'] = self.C.σ(Ω = sol['Ω'], Ψ = sol['Ψ'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'], τp = solp['τ'], θp = θp)
		sol.update({f'{k}_strategy': self.aux_strategy(sol, solp, k) for k in ('s','Γs','h')})
		return sol

	def objective_t(self, τ, θ, eps, ν, θp, epsp, νp, sol, solp):
		τBound = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		funcOfτ = self.func_t(τBound, θ, eps, ν, sol, solp)
		return self.C.PEE_t(τBound = τBound, τ = τ, θ = θ, eps = eps, ν = ν, τp = solp['τ'], θp = θp, epsp = epsp, νp = νp, 
							s_ = funcOfτ['s[t-1]'], s = sol['s'], h = sol['h'], hp = solp['h'], Γs = solp['Γs'], Bp = solp['B'], 
							dlnh_Dτ = funcOfτ['dln(h)/dτ'], dlns_Dτ = funcOfτ['dln(s)/dτ'], dlnΓs_Dτ = funcOfτ['dln(Γs)/dτ'], dlnhp_Dlns = solp['dln(h)/dln(s[t-1])'], dτp_dτ = funcOfτ['dτ[t+1]/dτ'], sSpread = funcOfτ['si/s'])

	def func_t(self, τ, θ, eps, ν, sol, solp):
		""" Return functions of τ on the grid of s"""
		funcOfτ = {'s[t-1]': sol['s_τ0']*(1-τ)**(-1/self.db['α'])}
		funcOfτ['B'] = self.C.B(funcOfτ['s[t-1]'], sol['h'], ν)
		funcOfτ['Γs'] = self.C.Γs(funcOfτ['B'], τ, θ, eps)
		funcOfτ['si/s'] = self.C.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, θ, eps)
		funcOfτ.update(self.C.EEDerivatives(sol['Ψ'], sol['σ'], solp['∂ln(h)/∂ln(s[t-1])'], τ))
		funcOfτ['dln(s)/dτ'] = self.aux_logDev(sol, funcOfτ, 's')
		funcOfτ.update({f'dln({k})/dτ': self.aux_logDev(sol, funcOfτ, k) for k in ('Γs','h')})
		funcOfτ['dτ[t+1]/dτ'] = solp['dτ/ds[t-1]'] * funcOfτ['dln(s)/dτ'] * solp['s[t-1]']
		return funcOfτ

	def report_t(self, τ, θ, eps, ν, sol, solp):
		sol['τ'] = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		sol['τ_unbounded'] = τ
		sol['s[t-1]'] = sol['s_τ0']*(1-sol['τ'])**(-1/self.db['α'])
		sol['B']  = self.C.B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.C.Γs(sol['B'], sol['τ'], θ, eps)
		sol['dτ/ds[t-1]'] = np.gradient(sol['τ'], sol['s[t-1]'])
		sol['dln(h)/dln(s[t-1])'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s[t-1])'] = self.db['α']*self.db['ξ']*(1+sol['Ψ']*(1-solp['∂ln(h)/∂ln(s[t-1])']))/((1+self.db['α']*self.db['ξ'])*sol['σ'])
		sol.update(self.C.EEDerivatives(sol['Ψ'], sol['σ'], solp['∂ln(h)/∂ln(s[t-1])'], sol['τ']))
		return sol

	######## FH terminal state:
	def solveRobust_T(self, θ, eps, ν, n = None, l = None, u = None, **kwargs):
		""" Use grid-search to get x0 """
		gridSol = self.solveGridSC_T(θ, eps, ν, n = n, l = l, u = u)
		return self.solveVector_T(θ, eps, ν, x0 = gridSol['τ_unbounded'])

	def solveVector_T(self, θ, eps, ν, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid'])
		x = optimize.root(lambda τ: self.objective_T(τ, θ, eps, ν, sol), x0)
		assert x['success'], f""" Could not identify PEE solution (PEE.solveVector_T) with parameters:
		θ: {θ}, ε: {eps}, ν: {ν}"""
		return self.report_T(x['x'], θ, eps, ν, sol)

	def solveScalarLoop_T(self, θ, eps, ν, x0_from_loop = True, x0 = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid'])
		τ = np.empty(self.db['glob_ns'])
		x = optimize.root(lambda τ: self.objective_T(τ, θ, eps, ν, aux_soli(sol, 0)), x0[0])
		τ[0] = x['x']
		assert x['success'], f""" Couldn't identify PEE with ν = {ν}, loop i=0."""
		for i in range(1, self.db['glob_ns']):
			x = optimize.root(lambda τ: self.objective_T(τ, θ, eps, ν, aux_soli(sol, i)), τ[i-1] if x0_from_loop else x0[i])
			τ[i] = x['x']
			assert x['success'], f""" Couldn't identify PEE iteration i = {i} out of {self.db['glob_ns']} """
		return self.report_T(τ, θ, eps, ν, sol)

	def solveGridSearch_T(self, θ, eps, ν, n = 1000, l = 0, u = 1, **kwargs):
		grids = self.createGrids(n, l = l, u = u)
		sol = self.precomputations_T(grids['gridnd_s'].values)
		objective = self.objective_T(grids['gridnd_τ'].values, θ, eps, ν, sol)
		idxMin = pd.Series(abs(objective), index = grids['idxnd']).groupby(['sIdx']).idxmin()
		τ = grids['gridnd_τ'][grids['gridnd_τ'].index.isin(idxMin.values)].values
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(τ, θ, eps, ν, sol)

	def solveGrid_T(self, θ, eps, ν, n = 1000, l = 0, u = 1, **kwargs):
		grids = self.createGrids(n, l = l, u = u)
		sol = self.precomputations_T(grids['gridnd_s'].values)
		objective = self.objective_T(grids['gridnd_τ'].values, θ, eps, ν, sol)
		τ = self.interpGridSol(pd.Series(objective, index = grids['idxnd']), pd.Series(grids['gridnd_τ'], index = grids['idxnd']))
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(τ, θ, eps, ν, sol)

	def solveGridSC_T(self, θ, eps, ν, n = 1000, l = -.1, u = 1.1, **kwargs):
		grids = self.createGrids(n, l = l, u = u)
		sol = self.precomputations_T(grids['gridnd_s'].values)
		objective = self.objective_T(grids['gridnd_τ'].values, θ, eps, ν, sol)
		τ = self.interpGridSolFromSC(grids, objective, n)
		sol = self.precomputations_T(self.db['sGrid'])
		return self.report_T(τ, θ, eps, ν, sol)

	def precomputations_T(self, sidx):
		return {'s' : np.zeros(len(sidx)), 's[t-1]': sidx}

	def objective_T(self, τ, θ, eps, ν, sol):
		τBound = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		funcOfτ = self.func_T(τBound, θ, eps, ν, sol)
		return self.C.PEE_T(τBound = τBound, τ = τ, θ = θ, eps = eps, ν = ν, s_ = sol['s[t-1]'], h =funcOfτ['h'], dlnh_Dτ = funcOfτ['dln(h)/dτ'], sSpread = funcOfτ['si/s'])		

	def func_T(self, τ, θ, eps, ν, sol):
		funcOfτ = {'h': self.C.h_T(τ, sol['s[t-1]'], ν), 'dln(h)/dτ': -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-τ))}
		funcOfτ['B'] = self.C.B(sol['s[t-1]'], funcOfτ['h'], ν)
		funcOfτ['Γs'] = self.C.Γs(funcOfτ['B'], τ, θ, eps)
		funcOfτ['si/s'] = self.C.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, θ, eps)
		return funcOfτ

	def report_T(self, τ, θ, eps, ν, sol):
		sol['τ'] = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		sol['τ_unbounded'] = τ
		sol['h'] = self.C.h_T(sol['τ'], sol['s[t-1]'], ν)
		sol['B']  = self.C.B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.C.Γs(sol['B'], sol['τ'], θ, eps)
		sol['dτ/ds[t-1]'] = np.gradient(sol['τ'], sol['s[t-1]'])
		sol['dln(h)/dln(s[t-1])'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s[t-1])'] = np.full(self.db['glob_ns'], self.C.power_h)
		sol['∂ln(h)/∂τ'] = -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-sol['τ']))
		return sol
