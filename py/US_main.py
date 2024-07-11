import numpy as np, pandas as pd, scipy
from pyDbs import is_iterable, SymMaps as sm
from scipy import optimize
from US_c import C
from US_policy import PEE, ESC

def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

def polGrid(v0, vT, n, exp = 1):
	""" Create polynomial grid with exponent 'exp'. 
		If exp>1 there are more gridpoint in the lower end of the grid."""
	return v0+(vT-v0)*((np.arange(1,n+1)-1)/(n-1))**exp

def interpFixedPoint(ŝ, s):
	""" Interpolation of fixed point problem with two grids"""
	Δs = ŝ-s # distance from steady state
	changeSign = np.diff(np.sign(Δs))!=0
	s1,s2 = s[:-1][changeSign], s[1:][changeSign]
	ŝ1,ŝ2 = ŝ[:-1][changeSign], ŝ[1:][changeSign]
	return ŝ1+((ŝ2-ŝ1)*(ŝ1-s1))/(s2-s1-(ŝ2-ŝ1)) # lin. approximation

class Model:
	def __init__(self, ni = 3, T = 10, ngrid = 50, gridkwargs = None, **kwargs):
		""" Fixed namespace """
		self.ni, self.T, self.ngrid = ni, T, ngrid
		self.db = self.defaultParameters | kwargs # default parameters
		self.initIdxs()
		self.inferPars(**kwargs)
		self.db.update(self.initGrids(**noneInit(gridkwargs, {})))
		self.addNamespaces() # define auxiliary class with "namespace" that helps organize move from stacked numpy arrays to sliced vectors and to pd.Series with indices.
		self.initUS() # initialize various settings for the specific US case
		self.C = C(self)
		self.PEE = PEE(self)
		self.ESC = ESC(self)
		self.x0 = self.defaultInitials

	#######################################################################
	##########					1. INIT METHODS				 	###########
	#######################################################################
	def initIdxs(self):
		self.db['t'] = pd.Index(range(self.T), name = 't')
		self.db['txE'] = pd.Index(range(self.T-1), name = 't') # Time index without terminal period
		self.db['t_']= self.db['t'].append(pd.Index([-1], name = 't')).sort_values() # Time index with -1 included
		self.db['i'] = pd.Index(range(self.ni), name = 'i')

	@property
	def defaultParameters(self):
		return {'α': .5, 'A': np.ones(self.T), 'ν': np.ones(self.T), 'ξ' : .35, 'ρ': 1.2, 'ω': 2,  # parameters
				'τ0': .158, 'RR0': 39.4/50.1, 'UShare0': 3.4/15.8, 'R0': 2.443}

	def initGrids(self, sgridExp = 1, **kwargs):
		d = self.defaultGridSettings | kwargs
		[d.__setitem__(f'{x}Idx', pd.Index(range(d[f'glob_n{x}']), name = f'{x}Idx')) for x in ('s','τ','θ','eps')]; # grid index
		d['sGrid'] = polGrid(d['s_l'], d['s_u'], d['glob_ns'], sgridExp) # grid value 1d, nonlinear for state 's' 
		return d

	@property
	def defaultGridSettings(self):
		d = {'glob_ns': self.ngrid, 'glob_nτ': 10, 'glob_nθ': 10, 'glob_neps': 10}
		d.update({f'k{x}_l': 10 for x in ('τ','θ','eps')})
		d.update({f'k{x}_u': 10 for x in ('τ','θ','eps')})
		d.update({f'{x}_l': 1e-4 for x in ('τ','θ','eps')})
		d.update({f'{x}_u': 1-1e-4 for x in ('τ','θ','eps')})
		d['s_l'] = .01 # stuff and stuff
		d['s_u'] = .07 # update later to ss when it has been implemented
		# d['s_u'] = self.solve_ss(0,0,0,self.db['ν'][-1])['s']
		return d

	def inferPars(self, **kwargs):
		""" Split full vectors syntax 'xj' into subsets """
		self.db['αr'] = (1-self.db['α'])/self.db['α'] # aux parameter
		d = self.defaultHeterogeneity | kwargs
		self.db['γ0'], self.db['γi'] = d['γj'][0], d['γj'][1:]
		self.db['X0'], self.db['Xi'] = d['Xj'][0], d['Xj'][1:]
		self.db['p0'], self.db['pi'] = d['pj'][0], d['pj'][1:]
		self.db['p'] = sum(self.db['γi']*self.db['pi'])/sum(self.db['γi'])
		self.db['p̄'] = self.db['γ0']*self.db['p0']+(1-self.db['γ0'])*self.db['p']
		self.db['μ0'], self.db['μi'] = d['μj'][0], d['μj'][1:]
		self.db['zx0'], self.db['zxi'] = d['zxj'][0], d['zxj'][1:]
		self.db['zη0'], self.db['zηi'] = d['zηj'][0], d['zηj'][1:]
		if 'βj' not in kwargs:
			self.db['βj'] = self.simpleβj*2
		self.db['β0'], self.db['βi'] = self.db['βj'][0], self.db['βj'][1:]

	@property
	def defaultHeterogeneity(self):
		return {'γj': np.full(self.ni+1, 1/(self.ni+1)),
				'pj': np.full(self.ni+1, 1),
				'μj': np.full(self.ni+1, 1),
				'Xj': np.ones(self.ni+1),
				'zxj': np.full(self.ni+1, 1),
				'zηj': np.full(self.ni+1, 1)}
	@property
	def simpleβj(self):
		return self.US_β(1/self.db['R0'])

	def addNamespaces(self):
		self.ns = {}		
		self.ns['EE_IH'] = sm(symbols = {x: self.db['t'] for x in ('s','h','Γs')}) # solve EE given policy
		self.ns['EE_FH'] = sm(symbols = {'h': self.db['t'], 's': self.db['txE'], 'Γs': self.db['txE']}) # solve EE given policy, finite horizon
		self.ns['ESC'] = sm(symbols = {k: self.db['t'] for k in ('τ', 'θ', 'eps')}) # Endogenous system characteristics solution
		self.ns['ESCpol'] = sm(symbols = {x: self.db['sIdx'] for x in ('τ', 'θ', 'eps')}) # namespace used in policy function identifications
		[ns.compile() for ns in self.ns.values()];
		# Define auxiliary lagged/leaded symbols
		self.ns['EE_IH'].addShiftedSym('h[t+1]','h',-1, opt = {'useLoc':'nn'})
		self.ns['EE_FH'].addLaggedSym('h[t+1]','h',-1, c= ('not',self.db['t'][-1:])) 
		[self.ns['ESC'].addShiftedSym(f'{k}[t+1]',f'{k}', -1, opt = {'useLoc':'nn'}) for k in ('τ', 'θ','eps')];
		self.ns['EE_IH'].addLaggedSym('s[t-1]', 's', 1, c = ('not', self.db['t'][0:1])) # this will not include the s0 value as it is not endogenous in the optimization

	@property
	def defaultInitials(self):
		return {'EE_FH': np.full(self.ns['EE_FH'].len, .2), 
				'EE_IH': np.full(self.ns['EE_IH'].len, .2)}

	# Some basic methods for navigating symbols:
	def initSC(self, sc, name, ns = 'ESC'):
		""" Define relevant eps or θ parameters"""
		sc = pd.Series(sc, index = self.db['t'], name = name) if not is_iterable(sc) else sc
		return {name: sc, f'{name}[t+1]': self.leadSym(sc, ns = ns)}

	def __call__(self, x, name, ns = 'ESCpol', **kwargs):
		return self.ns[ns](x, name, **kwargs)

	def get(self, x, name, ns = 'ESCpol'):
		return self.ns[ns].get(x, name)

	def leadSym(self, symbol, lead = -1, opt = None, ns = 'ESC'):
		return self.ns[ns].getShift(symbol, lead, opt = noneInit(opt, {'useLoc':'nn'})) if isinstance(symbol, pd.Series) else pd.Series(symbol).shift(lead, fill_value=symbol[-1]).values

	def j(self, x):
		""" return full vector of symbol x by combining 0 and i types"""
		return np.hstack([self.db[f'{x}0'], self.db[f'{x}i']])

	######### 1.1 US parameters
	def initUS(self):
		self.db.update(self.initSC(self.US_eps(), 'eps')) # start at calibration target
		self.US_addEigenVectors()
		self.db['ηi'] = self.US_ηi()
		self.US_Xi()
		self.US_η0()
		self.US_X0init()
		self.db.update(self.initSC(self.US_θ(), 'θ')) # start at calibration target - requires Xi,ηi,ξ
	def US_β(self, β):
		return β*np.hstack([self.db['p0'], np.full(self.ni, self.db['p'])])
	# Simple calibration:
	def US_addEigenVectors(self):
		valx, vecx = scipy.sparse.linalg.eigs(self.db['zxi'].reshape(self.ni,1) * self.db['γi'].reshape(1, self.ni) / (1-self.db['γ0']), k = 1)
		valη, vecη = scipy.sparse.linalg.eigs(self.db['zηi'].reshape(self.ni,1) * self.db['γi'].reshape(1, self.ni) / (1-self.db['γ0']), k = 1)
		self.db['yx'], self.db['yη'] = abs(np.real(vecx)).reshape(self.ni), abs(np.real(vecη).reshape(self.ni))
	def US_ηi(self):
		return self.db['yη']/(self.db['yx']*sum(self.db['γi']*self.db['yη']))
	# Calibration given ξ:
	def US_Xi(self):
		self.db['Xi'] = self.db['ηi']/self.db['yx']**(1/self.db['ξ'])
	def US_η0(self):
		self.db['η0'] = (1-self.db['τ0'])*(self.db['zη0']/self.db['zx0'])/sum(self.db['γi']*(self.db['ηi']/self.db['Xi'])**self.db['ξ'])
	def US_X0init(self):
		self.db['X0'] = self.US_X0(.5)
	def US_X0(self, ht0):
		return ((1-self.db['γ0'])/(self.db['zη0']*(1-self.db['τ0'])*ht0))**(1/self.db['ξ'])*self.db['η0']**((1+self.db['ξ'])/self.db['ξ'])*(1-self.db['α'])*(1-self.db['τ0'])*(self.db['α']/self.db['R0'])**((self.db['α']/(1+self.db['α'])))
	# Targets in calibration:
	def US_eps(self):
		return (self.db['UShare0']/(1-self.db['UShare0']))*(self.db['p']/self.db['p̄'])
	def US_θ(self):
		h1, h2 = self.db['Xi'][0]**(self.db['ξ'])/self.db['ηi'][0]**(1+self.db['ξ']), self.db['Xi'][1]**(self.db['ξ'])/self.db['ηi'][1]**(1+self.db['ξ'])
		return (self.db['RR0']*h1-h2)/(1-self.db['γ0']-h2-self.db['RR0']*(1-self.db['γ0']-h1))

	#######################################################################
	##########				2. SOLVE PEE/ESC PATHS				###########
	#######################################################################

	def PEE_FH(self, s0 = None, returnPols = False, x0_EE = None, pars = None):
		policy = self.PEE.FH(pars = pars)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], self.db['sGrid'])
		sols = self.EE_FH_PEE_solve(policy, self.db['θ'].values, self.db['eps'].values, s0, x0 = noneInit(x0_EE, self.x0['EE_FH']))
		return (self.PEE_FH_report(sols), policy) if returnPols else self.PEE_FH_report(sols)

	def PEE_FH_report(self, sol):
		sol['B'] = pd.DataFrame(self.C.B(s_ = sol['s[t-1]'].values, h = sol['h'].values, ν = self.db['ν']).T, columns  = self.db['i'], index = self.db['t'])
		sol['Γs[t-1]'] = pd.Series(self.C.Γs(sol['B'].values.T, sol['τ'].values, self.db['θ'].values, self.db['eps'].values), index = self.db['t'])
		sol['τ[t+1]'] = self.leadSym(sol['τ'])
		sol['R'] = pd.Series(self.C.R(sol['s[t-1]'].values, sol['h'].values, self.db['ν']), index = self.db['t'])
		return sol

	def ESC_FH(self, s0 = None, returnPols = False, x0_EE = None, pars = None):
		policy = self.ESC.FH(pars = pars)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], self.db['sGrid'])
		sols = self.EE_FH_ESC_solve(policy, s0, x0 = noneInit(x0_EE, self.x0['EE_FH']))
		return (self.ESC_FH_report(sols), policy) if returnPols else self.ESC_FH_report(sols)

	def ESC_FH_report(self, sol):
		sol['B'] = pd.DataFrame(self.C.B(s_ = sol['s[t-1]'].values, h = sol['h'].values, ν = self.db['ν']).T, columns  = self.db['i'], index = self.db['t'])
		sol['Γs[t-1]'] = pd.Series(self.C.Γs(sol['B'].values.T, sol['τ'].values, self.db['θ'].values, self.db['eps'].values), index = self.db['t'])
		sol.update({f'{k}[t+1]': self.leadSym(sol[f'{k}']) for k in ('τ', 'θ','eps')})
		sol['R'] = pd.Series(self.C.R(sol['s[t-1]'].values, sol['h'].values, self.db['ν']), index = self.db['t'])
		return sol	

	#######################################################################
	##########				3. STEADY STATE MEHTHODS			###########
	#######################################################################

	def SS_solveScalarLoop(self, τ, θ, eps, ν, x0_from_loop = True, x0 = None):
		""" Solve with τ being a vector on the grid of states """
		B, Γs = np.empty((self.ni, self.ngrid)), np.empty(self.ngrid)
		sol = optimize.root(lambda x: self.SS_objective(x[1:].reshape(self.ni, 1), x[:1], τ[0], θ, eps, ν), x0[:self.ni+1])
		Γs[0], B[:,0] = sol['x'][0], sol['x'][1:]
		for i in range(1,self.ngrid):
			sol = optimize.root(lambda x: self.SS_objective(x[1:].reshape(self.ni,1), x[:1], τ[i], θ, eps, ν), sol['x'] if x0_from_loop else x0[i*(self.ni+1):(i+1)*(self.ni+1)])
			Γs[i], B[:,i] = sol['x'][0], sol['x'][1:]
		return self.SS_report(B, Γs, τ, θ, eps, ν)

	def SS_solveScalarLoop_ESC(self, τ, θ, eps, ν, x0_from_loop = True, x0 = None):
		""" Solve with τ, θ, eps being vectors on the grid of states """
		B, Γs = np.empty((self.ni, self.ngrid)), np.empty(self.ngrid)
		sol = optimize.root(lambda x: self.SS_objective(x[1:].reshape(self.ni, 1), x[:1], τ[0], θ[0], eps[0], ν), x0[:self.ni+1])
		Γs[0], B[:,0] = sol['x'][0], sol['x'][1:]
		for i in range(1,self.ngrid):
			sol = optimize.root(lambda x: self.SS_objective(x[1:].reshape(self.ni,1), x[:1], τ[i], θ[i], eps[i], ν), sol['x'] if x0_from_loop else x0[i*(self.ni+1):(i+1)*(self.ni+1)])
			Γs[i], B[:,i] = sol['x'][0], sol['x'][1:]
		return self.SS_report(B, Γs, τ, θ, eps, ν)

	def SS_solveVector(self, τ, θ, eps, ν, length = 1, x0 = None):
		sol = optimize.root(lambda x: self.SS_objective(x[length:].reshape(self.ni, length), x[:length], τ, θ, eps, ν), x0)
		assert sol['success'], f""" Couldn't identify SS with parameters: 
		τ: {τ}, θ: {θ}, ε: {eps}, ν: {ν}"""
		return self.SS_report(sol['x'][length:].reshape(self.ni, length), sol['x'][:length], τ, θ, eps, ν)

	def SS_objective(self, B, Γs, τ, θ, eps, ν):
		""" B = Matrix with shape (self.ni, len(Γs)), Γs = vector."""
		return np.hstack([self.C.steadyStateB_eq(B, Γs, τ, θ, eps,ν).reshape(self.ni*len(Γs)), self.C.steadyStateΓs_eq(B, Γs, τ, θ, eps, ν)])

	def SS_report(self, B, Γs, τ, θ, eps, ν):
		return {'B': B, 'Γs': Γs, 's': np.nan_to_num(self.C.s_SS(Γs, τ, θ, eps, ν), nan = 0)}



	#######################################################################
	##########				4. ECONOMIC EQUILIBRIUM				###########
	#######################################################################

	def EE_FH_solve(self, τ, θ, eps, s0, x0 = None):
		τp, θp, epsp = self.leadSym(τ), self.leadSym(θ), self.leadSym(eps)
		sol = optimize.root(lambda x: self.EE_FH_objective(x, τ, θ, eps, τp, θp, epsp, s0), x0)
		assert sol['success'], f""" Could not identify economic equilibrium (self.EE_FH_solve) with parameter inputs: 
		τ: {τ}, θ: {θ}, ε: {eps}, s0: {s0}"""
		return self.EE_FH_report(sol, s0)

	def EE_FH_objective(self, x, τ, θ, eps, τp, θp, epsp, s0):
		Γs, h, s, s_ = self(x, 'Γs',ns = 'EE_FH'), self(x, 'h',ns='EE_FH'), self(x,'s',ns='EE_FH'), self.FH_sLag(x, s0)
		return np.hstack([self.C.h_FH(τ = τ, τp = τp, θp = θp, epsp = epsp, Γs = Γs, s_ = s_, ν = self.db['ν'])-h,
						  self.C.s_t(h[:-1], Γs)-s,
						  self.C.Γs_EE(s, h[1:], self.db['ν'][1:], τ[1:], θ[1:], eps[1:])-Γs])

	def EE_FH_report(self, sol, s0):
		d = self.ns['EE_FH'].unloadSol(sol['x'])
		d['s[t-1]'] = pd.Series(self.FH_sLag(sol['x'], s0), index = self.db['t'])
		return d

	def FH_sLag(self, x, s0):
		return np.insert(self(x,'s', ns = 'EE_FH'), 0, s0)

	def EE_FH_PEE_solve(self, gridSol, θ, eps, s0, x0 = None):
		θp, epsp = self.leadSym(θ), self.leadSym(eps)
		policyFunction = self.aux_vecPolFunction(gridSol)
		sol = optimize.root(lambda x: self.EE_FH_PEE_objective(x, policyFunction, θ, eps, θp, epsp, s0), x0)
		assert sol['success'], f""" Could not identify economic equilibrium (self.EE_FH_PEE_solve) with parameter inputs: 
		θ: {θ}, ε: {eps}, s0: {s0}"""
		return self.EE_FH_PEE_report(sol, policyFunction, s0)

	def EE_FH_PEE_objective(self, x, policyFunction, θ, eps, θp, epsp, s0):
		τ = policyFunction(self.FH_sLag(x, s0))
		τp = self.leadSym(τ)
		return self.EE_FH_objective(x, τ, θ, eps, τp, θp, epsp, s0)

	def EE_FH_PEE_report(self, sol, policyFunction, s0):
		d = self.ns['EE_FH'].unloadSol(sol['x'])
		d['s[t-1]'] = pd.Series(self.FH_sLag(sol['x'], s0), index = self.db['t'])
		d['τ'] = pd.Series(policyFunction(d['s[t-1]']), index = self.db['t'])
		return d

	def EE_FH_ESC_solve(self, gridSol, s0, x0 = None):
		τf = self.aux_vecPolFunction(gridSol, y = 'τ')
		θf = self.aux_vecPolFunction(gridSol, y = 'θ')
		epsf = self.aux_vecPolFunction(gridSol, y = 'eps')
		sol = optimize.root(lambda x: self.EE_FH_ESC_objective(x, τf, θf, epsf, s0), x0)
		assert sol['success'], f""" Could not identify economic equilibrium (self.EE_FH_ESC_solve)."""
		return self.EE_FH_ESC_report(sol, τf, θf, epsf, s0)

	def EE_FH_ESC_objective(self, x, τf, θf, epsf, s0):
		s_ = self.FH_sLag(x, s0)
		τ, θ, eps = τf(s_), θf(s_), epsf(s_)
		τp, θp, epsp = self.leadSym(τ), self.leadSym(θ), self.leadSym(eps)
		return self.EE_FH_objective(x, τ, θ, eps, τp, θp, epsp, s0)

	def EE_FH_ESC_report(self, sol, τf, θf, epsf, s0):
		d = self.ns['EE_FH'].unloadSol(sol['x'])
		d['s[t-1]'] = pd.Series(self.FH_sLag(sol['x'], s0), index = self.db['t'])
		d['τ'] = pd.Series(τf(d['s[t-1]']), index = self.db['t'])
		d['θ'] = pd.Series(θf(d['s[t-1]']), index = self.db['t'])
		d['eps'] = pd.Series(epsf(d['s[t-1]']), index = self.db['t'])
		return d

	def aux_vecPolFunction(self, griddedSolution, x = 's[t-1]', y = 'τ'):
		""" Return vectorized policy function from dict of gridded solutions (keys = t and values = dict of gridded policy)"""
		return lambda k: np.array([np.interp(k[t], griddedSolution[t][x], griddedSolution[t][y]) for t in self.db['t']])


	###########################################################
	##########			5. Calibration US			###########
	###########################################################
	def USCalSimple_PEE_FH(self, t0, update = True):
		sol = optimize.root(lambda x: self.USCalSimple_PEE_FH_objective(x, t0, update = update), [self.db['ω'], self.db['β0']/self.db['p0'], self.db['X0']])
		assert sol['success'], f"""Couldn't calibrate model"""
		return sol['x']
		
	def USCalSimple_PEE_FH_objective(self, x, t0, update = True):
		self.db['ω'] = x[0]
		self.db['βj']= self.US_β(x[1])
		self.db['β0'], self.db['βi'] = self.db['βj'][0], self.db['βj'][1:]
		self.db['X0'] = x[2]
		sol, pol = self.PEE_FH(returnPols = True)
		if update:
			self.PEE.x0 = {t: pol[t]['τ'] for t in self.db['t']}
		return np.hstack([sol['τ'].xs(t0)-self.db['τ0'],
						  sol['R'].xs(t0)-self.db['R0'],
						  x[2]-self.US_X0(sol['h'].xs(t0))])

	def USCalSimple_ESC_FH(self, t0, update = True):
		sol = optimize.root(lambda x: self.USCalSimple_ESC_FH_objective(x, t0, update = update), [self.db['ω'], self.db['β0']/self.db['p0'], self.db['X0']])
		assert sol['success'], f"""Couldn't calibrate model"""
		return sol['x']

	def USCalSimple_ESC_FH_objective(self, x, t0, update = True):
		self.db['ω'] = x[0]
		self.db['βj']= self.US_β(x[1])
		self.db['β0'], self.db['βi'] = self.db['βj'][0], self.db['βj'][1:]
		self.db['X0'] = x[2]
		sol, pol = self.ESC_FH(returnPols = True)
		if update:
			self.ESC.x0 = {t: pol[t]['x_unbounded'] for t in self.db['t']}
		return np.hstack([sol['τ'].xs(t0)-self.db['τ0'],
						  sol['R'].xs(t0)-self.db['R0'],
						  x[2]-self.US_X0(sol['h'].xs(t0))])
