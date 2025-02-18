import numpy as np, pandas as pd
from scipy import optimize, interpolate
from auxFunctions import CustomLinInterp
from pyDbs import is_iterable
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
	def __init__(self, m, kwargsMain = None):
		self.m = m
		self.BG = m.BG
		self.BTG = m.BTG # requires gridded solutions
		self.BT = m.BT # 
		self.db = m.db
		self.x0 = self.defaultInitials
		self.kwargsMain = self.defaultKwargs | noneInit(kwargsMain, {})
		self.main = m.main # what method is used in __call__ method.

	def __call__(self, main = None):
		method = noneInit(main, self.main)
		return getattr(self, f'f_{method}')()

	def f_FH(self, **kwargs):
		return self.solve(**(self.kwargsMain['FH'] | noneInit(kwargs, {})))

	@property
	def defaultInitials(self):
		return np.full(self.m.T, .2)

	@property
	def defaultKwargs(self):
		return {'FH': {'style': 'Vector'}}

	def solve(self, style = 'Vector', **kwargs):
		return getattr(self, f'solve{style}')(**kwargs)

	def solveRobust(self, grid = None, **kwargs):
		""" Solve with grid-search, then with vector """
		self.solveGridSC(grid = None, update = True)
		return self.solveVector(**kwargs)

	def solveVector(self, x0 = None, update = True, **kwargs):
		sol = optimize.root(lambda τ: self.objectiveVector(τ), noneInit(x0, self.x0), **kwargs)
		assert sol['success'], f""" Couldn't identify PEE in LOG.solveVector"""
		return self.report(sol['x'], update = update)

	def solveApproxGridSC(self, grid = None, update = True, **kwargs):
		"""  NOTE: This is a steady state approximation """
		if grid is None:
			grid = np.tile(self.db['τGrid'][:,None], self.m.T).T
		τ = self.interpGridSolFromSC(grid.T, self.objectiveApproxGridSC(grid).T)
		return self.report(τ, update = update)

	def solveGridSC(self, grid = None, update = True, **kwargs):
		if grid is None:
			grid = self.db['τGrid']
		sols = dict.fromkeys(self.db['t'])
		t = self.db['t'][-1]
		self.BG.t = t
		sols[t] = self.solveGridSC_T(grid)
		for t in self.db['t'][-2::-1]:
			self.BG.t = t
			sols[t] = self.solveGridSC_t(grid, sols[t+1])
		return self.report(np.array([soli['τ_unbounded'][0] for soli in sols.values()]), update = update)

	def solveGridSC_T(self, grid):
		sol = self.funcOfτGridSC_T(grid)
		τ = self.interpGridSolFromSC_1d(grid, self.objectiveGridSC_T(grid))
		sol = {'τ_unbounded': τ, 'τ': np.clip(τ, self.db['τ_l'], self.db['τ_u'])}
		sol['Γs'] = self.BG.Γs(Bi = self.BG.get('βi[t-1]'), τp = sol['τ'])
		return sol

	def solveGridSC_t(self, grid, solp):
		sol = self.funcOfτGridSC_t(grid, solp)
		τ = self.interpGridSolFromSC_1d(grid, self.objectiveGridSC_t(grid, solp))
		sol = {'τ_unbounded': τ, 'τ': np.clip(τ, self.db['τ_l'], self.db['τ_u'])}
		sol['Γs'] = self.BG.Γs(Bi = self.BG.get('βi[t-1]'), τp = sol['τ'])
		return sol

	def interpGridSolFromSC_1d(self, τ, o):
		""" Assumes a single crossing"""
		changeSign = np.diff(np.sign(o), axis = 0) < 0
		return inverseInterp1d(τ[:-1][changeSign], τ[1:][changeSign], o[:-1][changeSign], o[1:][changeSign])

	def interpGridSolFromSC(self, τ, o):
		""" Assumes a single crossing. 
		Note that when numpy slices a 2d array in a way that results in a 1d vector, 
		it "sorts" after matches in the row-direction first. """
		changeSign = np.diff(np.sign(o), axis = 0) < 0
		return inverseInterp1d(τ[:-1].T[changeSign.T], τ[1:].T[changeSign.T], o[:-1].T[changeSign.T], o[1:].T[changeSign.T])

	def report(self, τ, update = True):
		sol = self.funcOfτVector(τ)
		if update:
			self.x0 = sol['τ_unbounded']
		return sol
	def objectiveVector(self, τ):
		sol = self.funcOfτVector(τ)
		return self.BT.FH_LOG_PEE(τBound = sol['τ'], τ = sol['τ_unbounded'], si_s = sol['si/s'], Θh = sol['Θh'], dlnh_Dτ = sol['dlnh_Dτ'])
	def objectiveApproxGridSC(self, τ):
		sol = self.funcOfτApproxGridSC(τ)
		return self.BTG.FH_LOG_PEE(τBound = sol['τ'], τ = sol['τ_unbounded'], si_s = sol['si/s'], Θh = sol['Θh'], dlnh_Dτ = sol['dlnh_Dτ'])
	def objectiveGridSC_t(self, τ, solp):
		sol = self.funcOfτGridSC_t(τ, solp)
		return self.BG.LOG_PEE_t(τBound = sol['τ'], τ = sol['τ_unbounded'], si_s = sol['si/s'], Θh = sol['Θh'], dlnh_Dτ = sol['dlnh_Dτ'])
	def objectiveGridSC_T(self, τ):
		sol = self.funcOfτGridSC_T(τ)
		return self.BG.LOG_PEE_T(τBound = sol['τ'], τ = sol['τ_unbounded'], si_s = sol['si/s'], Θh = sol['Θh'], dlnh_Dτ = sol['dlnh_Dτ'])

	def _funcOfτ(self, τ, Bi, base):
		sol = {'τ_unbounded': τ, 'τ': np.clip(τ, self.db['τ_l'], self.db['τ_u'])}
		sol['τ[t+1]'] = self.m.leadSym(sol['τ'])
		sol['dlnh_Dτ'] = base.LOG_dlnh_Dτ(τ=sol['τ'])
		sol['Γs[t-1]'] = base.FH_LOG_ΓsLagged(τ = sol['τ'])
		sol['Θh'] = base.FH_LOG_Θh(τ = sol['τ'], τp = sol['τ[t+1]'], Γs = sol['Γs[t-1]'][1:])
		sol['si/s'] = base.si_s(Bi = Bi, Γs = sol['Γs[t-1]'], τp = sol['τ'])
		return sol
	def funcOfτVector(self, τ):
		return self._funcOfτ(τ, self.BT.get('βi[t-1]'), self.BT)
	def funcOfτApproxGridSC(self, τ):
		return self._funcOfτ(τ, self.BTG.get('βi[t-1]')[:,None,:], self.BTG)		
	def funcOfτGridSC_t(self, τ, solp):
		""" τp is a scalar - τ is a vector over a grid """
		sol = {'τ_unbounded': τ, 'τ': np.clip(τ, self.db['τ_l'], self.db['τ_u'])}
		sol['dlnh_Dτ'] = self.BG.LOG_dlnh_Dτ(τ = sol['τ'])
		sol['Γs'] = self.BG.Γs(Bi = self.BG.get('βi[t-1]'), τp = sol['τ'])
		sol['Θh'] = self.BG.Θh_t(τ = sol['τ'], τp = solp['τ'], Γs = solp['Γs'])
		sol['si/s'] = self.BG.si_s(Bi = self.BG.get('βi[t-1]')[None,:], Γs = sol['Γs'], τp = sol['τ'])
		return sol
	def funcOfτGridSC_T(self, τ):
		sol = {'τ_unbounded': τ, 'τ': np.clip(τ, self.db['τ_l'], self.db['τ_u'])}
		sol['dlnh_Dτ'] = self.BG.LOG_dlnh_Dτ(τ = sol['τ'])
		sol['Γs'] = self.BG.Γs(Bi = self.BG.get('βi[t-1]'), τp = sol['τ'])
		sol['Θh'] = self.BG.Θh_T(τ = sol['τ'])
		sol['si/s'] = self.BG.si_s(Bi = self.BG.get('βi[t-1]')[None,:], Γs = sol['Γs'], τp = sol['τ'])
		return sol


class PEE_A:
	def __init__(self, m, kwargsMain = None):
		self.m = m
		self.BG = m.BG # requires gridded solutions
		self.db = m.db
		self.x0 = self.defaultInitials
		self.kwargs_T = {'style': 'Vector', 'method': 'krylov', 'options': None}
		self.kwargs_T_ = {'style': 'Vector', 'method': 'krylov', 'options': None, 'x0_from_solp': False}
		self.kwargs_t = {'style': 'Vector', 'method': 'krylov', 'options': None, 'x0_from_solp': True}
		self.fInterp = CustomLinInterp
		self.kwargsInterp = {}
		self.kwargsMain = self.defaultKwargs | noneInit(kwargsMain, {})
		self.main = m.main # what method is used in __call__ method.
		# self.fInterp = interpolate.PchipInterpolator
		# self.kwargsInterp = {'extrapolate': True}

	def __call__(self, main = None):
		method = noneInit(main, self.main)
		return getattr(self, f'f_{method}')()

	@property
	def defaultKwargs(self):
		return {'FH': dict.fromkeys(self.db['t'][:-2], self.kwargs_t) | {self.db['t'][-2]: self.kwargs_T_, self.db['t'][-1]: self.kwargs_T}}

	@property
	def defaultInitials(self):
		return dict.fromkeys(self.db['t'], np.full(self.m.ngrid, .2))

	def interpInitialsFromLOG(self, path):
		""" Get self.x0 dictionary from log solution """
		[self.x0.__setitem__(self.db['t'][t], np.full(self.m.ngrid, path['τ_unbounded'][t])) for t in range(len(self.db['t']))];


	def f_FH(self, update = True, **kwargs):
		""" Return dict of policy functions. If kwargs are passed, this should be a dict of kwargs for each t. """
		kwargs = self.kwargsMain['FH'] | noneInit(kwargs, {})
		sols = dict.fromkeys(self.db['t'])
		t = self.db['t'][-1]
		self.BG.t = t
		sols[t] = self.solve(t = 'T', **({'x0': self.x0[t]} | kwargs[t]))
		if update:
			self.x0[t] = sols[t]['τ_unbounded']
		for t in self.db['t'][-2::-1]:
			self.BG.t = t
			sols[t] = self.resampleSolution(self.solve(solp = sols[t+1], **({'x0': self.x0[t]} | kwargs[t])))
			if update:
				self.x0[t] = sols[t]['τ_unbounded']
		return sols

	def resampleSolution(self, sol):
		d = {k: self.fInterp(sol['s[t-1]'], v)(self.db['sGrid']) for k,v in sol.items() if k != 's[t-1]'}
		d['s[t-1]'] = self.db['sGrid']
		return d

	def interpGridSolFromSC(self, objective):
		""" Assumes a single crossing. 
		Note that when numpy slices a 2d array in a way that results in a 1d vector, 
		it "sorts" after matches in the row-direction first. """
		o = objective.reshape((self.db['τ_n'], self.m.ngrid))
		τ = self.db['τGrid_sτ'].values.reshape((self.db['τ_n'], self.m.ngrid))
		changeSign = np.diff(np.sign(o), axis = 0) < 0
		return inverseInterp1d(τ[:-1].T[changeSign.T], τ[1:].T[changeSign.T], o[:-1].T[changeSign.T], o[1:].T[changeSign.T])

	def mapToIdxnd(self, k, idxnd, solp):
		if solp[k].ndim == 1:
			return pd.Series(0, index = idxnd).add(pd.Series(solp[k], index = self.db['sIdx'])).values
		else:
			return pd.DataFrame(0, index = idxnd, columns = self.db['i']).add(pd.DataFrame(solp[k], index = self.db['sIdx'], columns = self.db['i'])).values

	def gridPolicy(self, v, grids = None, kwargs = None):
		kwargs = self.kwargsInterp | noneInit(kwargs, {})
		return self.fInterp(noneInit(grids, self.db['sGrid']), v, **kwargs)

	def vectorPolicy(self, sols, y = 'τ', grids = None, kwargs = None):
		""" Return vector of predicted policies from vector of states; the "0" index is used to make sure that it returns a 1d object, but it requires that we query a single point at a time. """
		grids = noneInit(grids, self.db['sGrid'])
		kwargs = self.kwargsInterp | noneInit(kwargs, {})
		fps = (soli[y] for soli in sols.values())
		interps = tuple(self.fInterp(grids, fps_i, **kwargs) for fps_i in fps)
		return lambda x: np.array([interps[i](x[i]) for i in range(len(x))])

	def solve(self, style = 'Vector', t = 't', **kwargs):
		return getattr(self, f'solve{style}_{t}')(**kwargs)

	def solveRobust_t(self, solp = None, **kwargs):
		""" Use grid-search to get x0 """
		gridSol = self.solveGridSC_t(solp = solp)
		return self.solveVector_t(solp = solp, x0 = gridSol['τ_unbounded'], **{k:v for k,v in noneInit(kwargs, {}).items() if k != 'x0'})

	def solveVeryRobust_t(self, solp = None, **kwargs):
		""" Use vectorized optimization, then gridsearch, retry vectorized optimization or exit with gridsearch result"""
		try:
			return self.solveVector_t(solp = solp, **kwargs)
		except AssertionError:
			gridSol = self.solveGridSC_t(solp = solp)
			try:
				return self.solveVector_t(solp = solp, x0 = gridSol['τ_unbounded'], **{k:v for k,v in noneInit(kwargs, {}).items() if k != 'x0'})
			except AssertionError:
				print(f"""Warning: Could only solve with gridsearch.""")
				return gridSol

	def solveGridSC_t(self, solp = None, **kwargs):
		solp_nd = {k: self.mapToIdxnd(k, self.db['sτIdx'], solp) for k in solp} # map solp to full grid
		τ = self.interpGridSolFromSC(self.objective_t(self.db['τGrid_sτ'].values, self.precomputations_t(solp_nd), solp_nd))
		return self.report_t(τ, self.precomputations_t(solp), solp)

	def solveVector_t(self, solp = None, x0_from_solp = False, x0 = None, method = 'hybr', options = None, **kwargs):
		sol = self.precomputations_t(solp)
		x = optimize.root(lambda τ: self.objective_t(τ, sol, solp), x0_solp(x0, solp['τ_unbounded'], x0_from_solp), method = method, options = options)
		assert x['success'], f""" Could not identify PEE solution (PEE.solveVector_t) with parameters"""
		return self.report_t(x['x'], sol, solp)

	def objective_t(self, τ, sol, solp):
		sol = self.funcOfτ_t(τ, sol, solp)
		return self.BG.PEE_t(τBound = sol['τ'], τ  = sol['τ_unbounded'], τp = solp['τ'], s_ = sol['s[t-1]'], s = sol['s'], h = sol['h'], Γs = solp['Γs'], Bip = sol['Bi'], si_s = sol['si/s'], Θh = sol['Θh'], Θhp = solp['Θh'],
							dlnh_Dτ = sol['dln(h)/dτ'], dlns_Dτ = sol['dln(s)/dτ'], dlnΓs_Dτ = sol['dln(Γs)/dτ'], dlnhp_Dlns = solp['dln(h)/dln(s[t-1])'], dlnΘhp_Dτ = sol['dln(Θh[t+1])/dτ'], dτp_dτ = sol['dτ[t+1]/dτ'])

	def precomputations_t(self, solp):
		sol = { 's': solp['s[t-1]'], 'h': self.BG.backOutH(s = solp['s[t-1]'], Γs = solp['Γs']), 'Ω': self.BG.Ω(Γs = solp['Γs'], τp = solp['τ']), 'Ψ': self.BG.Ψ(Bip = solp['Bi'], τp= solp['τ'])}
		sol['σ'] = self.BG.σ(Ω = sol['Ω'], Ψ = sol['Ψ'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'], τp = solp['τ'])
		sol.update(self.BG.EELaggedDerivatives(Ω = sol['Ω'], Ψ = sol['Ψ'], Bip = solp['Bi'], dlnhp_dτp = solp['∂ln(h)/∂τ'], τp = solp['τ']))
		return sol

	def funcOfτ_t(self, τ, sol, solp):
		""" Return functions of τ on the grid of s"""
		sol['τ_unbounded'] = τ
		sol['τ'] = np.clip(sol['τ_unbounded'], self.db['τ_l'], self.db['τ_u'])
		sol['s[t-1]'] = self.BG.backOutS_(h = sol['h'], τ = sol['τ'], τp = solp['τ'], Γs = solp['Γs'])
		sol['Θh'] = self.BG.Θh_t(τ = sol['τ'], τp = solp['τ'], Γs = solp['Γs'])
		sol['Bi'] = self.BG.Bi(s_ = sol['s[t-1]'], h = sol['h'])
		sol['Γs'] = self.BG.Γs(Bi = sol['Bi'], τp = sol['τ'])
		sol['si/s']	= self.BG.si_s(Bi = sol['Bi'], Γs = sol['Γs'], τp = sol['τ'])
		sol.update(self.BG.EEDerivatives(Ψ = sol['Ψ'], σ = sol['σ'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'], τ = sol['τ'], τp = solp['τ']))
		sol.update(self.auxStrategicEffects(sol, solp))
		sol['dln(Θh[t+1])/dτ'] = (solp['dln(h)/dln(s[t-1])']-self.BG.power_h())*sol['dln(s)/dτ']
		return sol

	def report_t(self, τ, sol, solp):
		sol = self.funcOfτ_t(τ ,sol, solp)
		sol.update(self.getGriddedGradients(sol))
		sol['∂ln(h)/∂ln(s[t-1])'] = self.BG.recursive_dlnh_dlns_(Ψ = sol['Ψ'], σ = sol['σ'], dlnhp_dlns = solp['∂ln(h)/∂ln(s[t-1])'])
		return sol

	def auxStrategicEffects(self, sol, solp):
		dlns_dτ = sol['∂ln(s)/∂τ']/(1-sol['∂ln(s)/∂τ[t+1]']*solp['dτ/ds[t-1]'] * sol['s'])
		dτp_dτ  = dlns_dτ * solp['dτ/ds[t-1]'] * sol['s']
		return {'dτ[t+1]/dτ': dτp_dτ, 'dln(s)/dτ': dlns_dτ} |  {f'dln({k})/dτ': self.auxStrategy(k, dτp_dτ, sol) for k in ('h','Γs')}

	def auxStrategy(self, k, dτp_dτ, sol):
		return sol[f'∂ln({k})/∂τ']+dτp_dτ * sol[f'∂ln({k})/∂τ[t+1]']

	## TERMINAL STATE FUNCTIONS
	def solveRobust_T(self, **kwargs):
		""" Use grid-search to get x0 """
		gridSol = self.solveGridSC_T()
		return self.solveVector_T(x0 = gridSol['τ_unbounded'], **{k:v for k,v in noneInit(kwargs, {}).items() if k != 'x0'})

	def solveVeryRobust_T(self, **kwargs):
		""" Use vectorized optimization, then gridsearch, retry vectorized optimization or exit with gridsearch result"""
		try:
			return self.solveVector_T(**kwargs)
		except AssertionError:
			gridSol = self.solveGridSC_T()
			try:
				return self.solveVector_T(x0 = gridSol['τ_unbounded'], **{k:v for k,v in noneInit(kwargs, {}).items() if k != 'x0'})
			except AssertionError:
				print(f"""Warning: Could only solve with gridsearch.""")
				return gridSol

	def solveGridSC_T(self, **kwargs):
		τ = self.interpGridSolFromSC(self.objective_T(self.db['τGrid_sτ'].values, self.precomputations_T(self.db['sGrid_sτ'].values)))
		return self.report_T(τ, self.precomputations_T(self.db['sGrid']))

	def solveVector_T(self, x0 = None, method = 'hybr', options = None, **kwargs):
		sol = self.precomputations_T(self.db['sGrid'])
		x = optimize.root(lambda τ: self.objective_T(τ, sol), x0, method = method, options = options)
		assert x['success'], f""" Could not identify PEE solution (self.solveVector_T)"""
		return self.report_T(x['x'], sol)

	def precomputations_T(self, sidx):
		return {'s' : np.zeros(len(sidx)), 's[t-1]': sidx}

	def objective_T(self, τ, sol):
		τBound = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		funcOfτ = self.funcOfτ_T(τBound, sol)
		return self.BG.PEE_T(τBound = τBound, τ = τ, s_ = sol['s[t-1]'], h = funcOfτ['h'], Θh = funcOfτ['Θh'], dlnh_Dτ = funcOfτ['∂ln(h)/∂τ'], si_s = funcOfτ['si/s'])

	def funcOfτ_T(self, τ, sol):
		funcOfτ = {'∂ln(h)/∂τ': self.BG.dlnh_Dτ_T(τ), 'h': self.BG.h_T(s_ = sol['s[t-1]'], τ = τ), 'Θh': self.BG.Θh_T(τ = τ)}
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
		return sol

	def getGriddedGradients(self, sol):
		return {'dτ/ds[t-1]': np.gradient(sol['τ'], sol['s[t-1]'], edge_order = 2),
				'dln(h)/dln(s[t-1])': np.gradient(sol['h'], sol['s[t-1]'], edge_order =2) * sol['s[t-1]']/sol['h']}