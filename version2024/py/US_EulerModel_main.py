import numpy as np, pandas as pd, scipy
from pyDbs import is_iterable, SymMaps as sm
from scipy import optimize
from US_EulerModel_c import C
from US_EulerModel_policy import PEE, ESC, LOG, inverseInterp1d

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
	def __init__(self, ni = 3, T = 10, ngrid = 50, RRgroups = (1,2), gridkwargs = None, **kwargs):
		""" Fixed namespace """
		self.ni, self.T, self.ngrid = ni, T, ngrid
		self.db = self.defaultParameters | kwargs # default parameters
		self.initIdxs()
		self.inferPars(**kwargs)
		self.db.update(self.initGrids(**noneInit(gridkwargs, {})))
		self.RRgroups = RRgroups
		self.addNamespaces() # define auxiliary class with "namespace" that helps organize move from stacked numpy arrays to sliced vectors and to pd.Series with indices.
		self.initUS() # initialize various settings for the specific US case
		self.C = C(self)
		self.PEE = PEE(self)
		self.ESC = ESC(self)
		self.LOG = LOG(self)
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
		d['sIdx'] = pd.Index(range(self.ngrid), name = 'sIdx')
		d['sGrid'] = polGrid(d['s_l'], d['s_u'], self.ngrid, sgridExp) # grid value 1d, nonlinear for state 's' 
		return d

	@property
	def defaultGridSettings(self):
		d = {f'k{x}_l': 10 for x in ('τ','κ')}
		d.update({f'k{x}_u': 10 for x in ('τ','κ')})
		d.update({f'{x}_l': 1e-4 for x in ('τ','κ')})
		d.update({f'{x}_n': 101 for x in ('τ','κ')})
		d.update({f'{x}_l': 1e-4 for x in ('τ','κ')})
		d.update({f'{x}_u': 1-1e-4 for x in ('τ','κ')})
		d['s_l'] = .01 # stuff and stuff
		d['s_u'] = .07 # update later to ss when it has been implemented
		# d['s_u'] = self.solve_ss(0,0,0,self.db['ν'][-1])['s']
		return d

	def inferPars(self, **kwargs):
		""" Split full vectors syntax 'xj' into subsets """
		self.db['αr'] = (1-self.db['α'])/self.db['α'] # aux parameter
		self.db.update(self.defaultHeterogeneity | kwargs)
		self.db['p'] = sum(self.db['γi']*self.db['pi'])/sum(self.db['γi'])
		if 'βi' not in kwargs:
			self.db['βi'] = self.simpleβi*2

	@property
	def defaultHeterogeneity(self):
		return {'γi': np.full(self.ni, 1/(self.ni)),
				'pi': np.full(self.ni, 1),
				'μi': np.full(self.ni, 1),
				'Xi': np.ones(self.ni),
				'zxi': np.full(self.ni, 1),
				'zηi': np.full(self.ni, 1)}
	@property
	def simpleβi(self):
		return self.US_β(1/self.db['R0'])

	def addNamespaces(self):
		self.ns = {}		
		self.ns['EE_IH'] = sm(symbols = {x: self.db['t'] for x in ('s','h','Γs')}) # solve EE given policy
		self.ns['EE_FH'] = sm(symbols = {'h': self.db['t'], 's': self.db['txE'], 'Γs': self.db['txE']}) # solve EE given policy, finite horizon
		self.ns['ESC'] = sm(symbols = {k: self.db['t'] for k in ('τ', 'κ')}) # Endogenous system characteristics solution
		self.ns['ESCpol'] = sm(symbols = {x: self.db['sIdx'] for x in ('τ', 'κ')}) # namespace used in policy function identifications
		[ns.compile() for ns in self.ns.values()];
		# Define auxiliary lagged/leaded symbols
		self.ns['EE_IH'].addShiftedSym('h[t+1]','h',-1, opt = {'useLoc':'nn'})
		self.ns['EE_FH'].addLaggedSym('h[t+1]','h',-1, c= ('not',self.db['t'][-1:])) 
		[self.ns['ESC'].addShiftedSym(f'{k}[t+1]',f'{k}', -1, opt = {'useLoc':'nn'}) for k in ('τ', 'κ')];
		self.ns['EE_IH'].addLaggedSym('s[t-1]', 's', 1, c = ('not', self.db['t'][0:1])) # this will not include the s0 value as it is not endogenous in the optimization

	@property
	def defaultInitials(self):
		return {'EE_FH': np.full(self.ns['EE_FH'].len, .2), 
				'EE_IH': np.full(self.ns['EE_IH'].len, .2),
				'PEE_log': np.full(self.T, .2),
				'ESC_log': np.full(self.ns['ESC'].len, .2)}

	# Some basic methods for navigating symbols:
	def initSC(self, sc, name, ns = 'ESC'):
		""" Define relevant eps, θ, κ, parameters"""
		sc = pd.Series(sc, index = self.db['t'], name = name) if not is_iterable(sc) else sc
		return {name: sc, f'{name}[t+1]': self.leadSym(sc, ns = ns)}

	def __call__(self, x, name, ns = 'ESCpol', **kwargs):
		return self.ns[ns](x, name, **kwargs)

	def get(self, x, name, ns = 'ESCpol'):
		return self.ns[ns].get(x, name)

	def leadSym(self, symbol, lead = -1, opt = None, ns = 'ESC'):
		return self.ns[ns].getShift(symbol, lead, opt = noneInit(opt, {'useLoc':'nn'})) if isinstance(symbol, pd.Series) else pd.Series(symbol).shift(lead, fill_value=symbol[-1]).values

	######### 1.1 US parameters
	def initUS(self):
		self.US_addEigenVectors()
		self.db['ηi'] = self.US_ηi()
		self.US_Xi()
		self.db.update(self.initSC(self.US_κ(), 'κ'))

	def US_β(self, β):
		return β*np.full(self.ni, self.db['p'])
	def US_βinv(self):
		return self.db['βi'][0]/self.db['p']
	# Simple calibration:
	def US_addEigenVectors(self):
		valx, vecx = scipy.sparse.linalg.eigs(self.db['zxi'].reshape(self.ni,1) * self.db['γi'].reshape(1, self.ni), k = 1)
		valη, vecη = scipy.sparse.linalg.eigs(self.db['zηi'].reshape(self.ni,1) * self.db['γi'].reshape(1, self.ni), k = 1)
		self.db['yx'], self.db['yη'] = abs(np.real(vecx)).reshape(self.ni), abs(np.real(vecη).reshape(self.ni))
	def US_ηi(self):
		return self.db['yη']/(self.db['yx']*sum(self.db['γi']*self.db['yη']))
	# Calibration given ξ:
	def US_Xi(self):
		self.db['Xi'] = self.db['ηi']/self.db['yx']**(1/self.db['ξ'])
	# Targets in calibration:
	def US_κ(self, ξ = None):
		return self.US_θ(ξ = ξ)/(1+self.US_eps())
	def US_eps(self):
		return (self.db['UShare0']/(1-self.db['UShare0']))
	def US_θ(self, ξ = None):
		i,ii = self.RRgroups[0], self.RRgroups[1]
		ξ = noneInit(ξ, self.db['ξ'])
		h1, h2 = self.db['Xi'][i]**(ξ)/self.db['ηi'][i]**(1+ξ), self.db['Xi'][ii]**(ξ)/self.db['ηi'][ii]**(1+ξ)
		return (self.db['RR0']*h1-h2)/(1-h2-self.db['RR0']*(1-h1))

	#######################################################################
	##########				2. SOLVE PEE/ESC PATHS				###########
	#######################################################################

	def PEE_FH(self, s0 = None, returnPols = False, x0_EE = None, pars = None):
		policy = self.PEE.FH(pars = pars)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], self.db['sGrid'])
		sols = self.EE_FH_PEE_solve(policy, self.db['κ'].values, s0, x0 = noneInit(x0_EE, self.x0['EE_FH']))
		return (self.PEE_FH_report(sols), policy) if returnPols else self.PEE_FH_report(sols)

	def PEE_FH_report(self, sol):
		sol['B'] = pd.DataFrame(self.C.B(s_ = sol['s[t-1]'].values, h = sol['h'].values, ν = self.db['ν']).T, columns  = self.db['i'], index = self.db['t'])
		sol['Γs[t-1]'] = pd.Series(self.C.Γs(sol['B'].values.T, sol['τ'].values, self.db['κ'].values), index = self.db['t'])
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
		sol['Γs[t-1]'] = pd.Series(self.C.Γs(sol['B'].values.T, sol['τ'].values, self.db['κ'].values), index = self.db['t'])
		sol.update({f'{k}[t+1]': self.leadSym(sol[f'{k}']) for k in ('τ', 'κ')})
		sol['R'] = pd.Series(self.C.R(sol['s[t-1]'].values, sol['h'].values, self.db['ν']), index = self.db['t'])
		return sol

	def PEE_log_FH(self, s0 = None, **kwargs):
		τ = self.LOG.solve('PEE',**kwargs)
		sol = self.EE_log_FH(τ, self.db['κ'].values, s0 = s0)
		return self.log_FH_report(sol)

	def ESC_log_FH(self, s0 = None, **kwargs):
		sol = self.ns['ESC'].unloadSol(self.LOG.solve('ESC',**kwargs))
		sol = self.EE_log_FH(sol['τ'].values, sol['κ'].values, s0 = s0)
		return self.log_FH_report(sol)

	def log_FH_report(self, sol):
		sol['B'] = pd.DataFrame(np.repeat(self.db['βi'].reshape(1, self.ni), self.T, axis =0), columns = self.db['i'], index = self.db['t'])
		sol['Γs[t-1]'] = pd.Series(self.C.Γs(sol['B'].values.T, sol['τ'].values, sol['κ'].values), index = self.db['t'])
		sol['R'] = pd.Series(self.C.R(sol['s[t-1]'].values, sol['h'].values, self.db['ν']), index = self.db['t'])
		return sol

	#######################################################################
	##########				3. STEADY STATE MEHTHODS			###########
	#######################################################################

	def SS_log(self, τ, κ, ν):
		""" Steady state savings with log"""
		return np.nan_to_num(self.C.s_SS(self.C.log_Γs(τ, κ), τ, κ, ν), nan = 0)

	def SS_solveScalarLoop(self, τ, κ, ν, x0_from_loop = True, x0 = None):
		""" Solve with τ being a vector on the grid of states """
		B, Γs = np.empty((self.ni, self.ngrid)), np.empty(self.ngrid)
		sol = optimize.root(lambda x: self.SS_objective(x[1:].reshape(self.ni, 1), x[:1], τ[0], κ, ν), x0[:self.ni+1])
		Γs[0], B[:,0] = sol['x'][0], sol['x'][1:]
		for i in range(1,self.ngrid):
			sol = optimize.root(lambda x: self.SS_objective(x[1:].reshape(self.ni,1), x[:1], τ[i], κ, ν), sol['x'] if x0_from_loop else x0[i*(self.ni+1):(i+1)*(self.ni+1)])
			Γs[i], B[:,i] = sol['x'][0], sol['x'][1:]
		return self.SS_report(B, Γs, τ, κ, ν)

	def SS_solveScalarLoop_ESC(self, τ, κ, ν, x0_from_loop = True, x0 = None):
		""" Solve with τ, κ, eps being vectors on the grid of states """
		B, Γs = np.empty((self.ni, self.ngrid)), np.empty(self.ngrid)
		sol = optimize.root(lambda x: self.SS_objective(x[1:].reshape(self.ni, 1), x[:1], τ[0], κ[0], ν), x0[:self.ni+1])
		Γs[0], B[:,0] = sol['x'][0], sol['x'][1:]
		for i in range(1,self.ngrid):
			sol = optimize.root(lambda x: self.SS_objective(x[1:].reshape(self.ni,1), x[:1], τ[i], κ[i], ν), sol['x'] if x0_from_loop else x0[i*(self.ni+1):(i+1)*(self.ni+1)])
			Γs[i], B[:,i] = sol['x'][0], sol['x'][1:]
		return self.SS_report(B, Γs, τ, κ, ν)

	def SS_solveVector(self, τ, κ, ν, length = 1, x0 = None):
		sol = optimize.root(lambda x: self.SS_objective(x[length:].reshape(self.ni, length), x[:length], τ, κ, ν), x0)
		assert sol['success'], f""" Couldn't identify SS with parameters: 
		τ: {τ}, κ: {κ}, ν: {ν}"""
		return self.SS_report(sol['x'][length:].reshape(self.ni, length), sol['x'][:length], τ, κ, ν)

	def SS_objective(self, B, Γs, τ, κ, ν):
		""" B = Matrix with shape (self.ni, len(Γs)), Γs = vector."""
		return np.hstack([self.C.steadyStateB_eq(B, Γs, τ, κ, ν).reshape(self.ni*len(Γs)), self.C.steadyStateΓs_eq(B, Γs, τ, κ, ν)])

	def SS_report(self, B, Γs, τ, κ, ν):
		return {'B': B, 'Γs': Γs, 's': np.nan_to_num(self.C.s_SS(Γs, τ, κ, ν), nan = 0)}

	#######################################################################
	##########				4. ECONOMIC EQUILIBRIUM				###########
	#######################################################################

	def EE_FH_solve(self, τ, κ, s0, x0 = None):
		τp, κp = self.leadSym(τ), self.leadSym(κ)
		sol = optimize.root(lambda x: self.EE_FH_objective(x, τ, κ, τp, κp, s0), x0)
		assert sol['success'], f""" Could not identify economic equilibrium (self.EE_FH_solve) with parameter inputs: 
		τ: {τ}, κ: {κ}, s0: {s0}"""
		return self.EE_FH_report(sol, s0)

	def EE_FH_objective(self, x, τ, κ, τp, κp, s0):
		Γs, h, s, s_ = self(x, 'Γs',ns = 'EE_FH'), self(x, 'h',ns='EE_FH'), self(x,'s',ns='EE_FH'), self.FH_sLag(x, s0)
		return np.hstack([self.C.h_FH(τ = τ, τp = τp, κp = κp, Γs = Γs, s_ = s_, ν = self.db['ν'])-h,
						  self.C.s_t(h[:-1], Γs)-s,
						  self.C.Γs_EE(s, h[1:], self.db['ν'][1:], τ[1:], κ[1:])-Γs])

	def EE_log_FH(self, τ, κ, s0 = None):
		sol = self.initSC(pd.Series(τ, index = self.db['t']), 'τ') | self.initSC(pd.Series(κ, index = self.db['t']), 'κ')
		if s0 is None:
			s0 = self.SS_log(τ[0], κ[0], self.db['ν'][0])
		τp, κp = sol['τ[t+1]'].values, sol['κ[t+1]'].values
		Γs = self.C.log_Γs(τp = τp[:-1], κp = κp[:-1])
		sol['Θh'] = pd.Series(self.C.Θh_FH(τ = τ, τp = τp, κp = κp, Γs = Γs), index = self.db['t'])
		sol['Θs'] = sol['Θh'].iloc[:-1]**((1+self.db['ξ'])/self.db['ξ'])*Γs
		sol['s'] = self.C.log_s_FH(sol, s0 = s0)
		sol['s[t-1]'] = pd.Series(np.insert(sol['s'].values, 0, s0), index = self.db['t'])
		sol['h'] = self.C.log_h_FH(sol)
		return sol

	def EE_FH_report(self, sol, s0):
		d = self.ns['EE_FH'].unloadSol(sol['x'])
		d['s[t-1]'] = pd.Series(self.FH_sLag(sol['x'], s0), index = self.db['t'])
		return d

	def FH_sLag(self, x, s0):
		return np.insert(self(x,'s', ns = 'EE_FH'), 0, s0)

	def EE_FH_PEE_solve(self, gridSol, κ, s0, x0 = None):
		κp = self.leadSym(κ)
		policyFunction = self.aux_vecPolFunction(gridSol)
		sol = optimize.root(lambda x: self.EE_FH_PEE_objective(x, policyFunction, κ, κp, s0), x0)
		assert sol['success'], f""" Could not identify economic equilibrium (self.EE_FH_PEE_solve) with parameter inputs: 
		κ: {κ}, s0: {s0}"""
		return self.EE_FH_PEE_report(sol, policyFunction, s0)

	def EE_FH_PEE_objective(self, x, policyFunction, κ, κp, s0):
		τ = policyFunction(self.FH_sLag(x, s0))
		τp = self.leadSym(τ)
		return self.EE_FH_objective(x, τ, κ, τp, κp, s0)

	def EE_FH_PEE_report(self, sol, policyFunction, s0):
		d = self.ns['EE_FH'].unloadSol(sol['x'])
		d['x'] = sol['x']
		d['s[t-1]'] = pd.Series(self.FH_sLag(sol['x'], s0), index = self.db['t'])
		d['τ'] = pd.Series(policyFunction(d['s[t-1]']), index = self.db['t'])
		return d

	def EE_FH_ESC_solve(self, gridSol, s0, x0 = None):
		τf = self.aux_vecPolFunction(gridSol, y = 'τ')
		κf = self.aux_vecPolFunction(gridSol, y = 'κ')
		sol = optimize.root(lambda x: self.EE_FH_ESC_objective(x, τf, κf, s0), x0)
		if not sol['success']:
			sol = optimize.root(lambda x: self.EE_FH_ESC_objective(x, τf, κf, s0), self.defaultInitials['EE_FH'])
		assert sol['success'], f""" Could not identify economic equilibrium (self.EE_FH_ESC_solve)."""
		return self.EE_FH_ESC_report(sol, τf, κf, s0)

	def EE_FH_ESC_objective(self, x, τf, κf, s0):
		s_ = self.FH_sLag(x, s0)
		τ, κ = τf(s_), κf(s_)
		τp, κp = self.leadSym(τ), self.leadSym(κ)
		return self.EE_FH_objective(x, τ, κ, τp, κp, s0)

	def EE_FH_ESC_report(self, sol, τf, κf, s0):
		d = self.ns['EE_FH'].unloadSol(sol['x'])
		d['x'] = sol['x']
		d['s[t-1]'] = pd.Series(self.FH_sLag(sol['x'], s0), index = self.db['t'])
		d['τ'] = pd.Series(τf(d['s[t-1]']), index = self.db['t'])
		d['κ'] = pd.Series(κf(d['s[t-1]']), index = self.db['t'])
		return d

	def aux_vecPolFunction(self, griddedSolution, x = 's[t-1]', y = 'τ'):
		""" Return vectorized policy function from dict of gridded solutions (keys = t and values = dict of gridded policy)"""
		return lambda k: np.array([np.interp(k[t], griddedSolution[t][x], griddedSolution[t][y]) for t in self.db['t']])


	###########################################################
	##########			5. Calibration US			###########
	###########################################################
	#### 5.1. CALIBRATION GIVEN ρ, ξ

	def USCalSimple_PEE_FH(self, t0, update = True):
		sol = optimize.root(lambda x: self.USCalSimple_PEE_FH_objective(x, t0, update = update), [self.db['ω'], self.US_βinv()])
		assert sol['success'], f"""Couldn't calibrate model"""
		return sol['x']
		
	def USCalSimple_PEE_FH_objective(self, x, t0, update = True):
		self.db['ω'] = x[0]
		self.db['βi']= self.US_β(x[1])
		sol, pol = self.PEE_FH(returnPols = True)
		if update:
			self.PEE.x0 = {t: pol[t]['τ'] for t in self.db['t']}
			self.x0['EE_FH'] = sol['x']
		return np.hstack([sol['τ'].xs(t0)-self.db['τ0'],
						  sol['R'].xs(t0)-self.db['R0']])

	def USCalSimple_ESC_FH(self, t0, update = True, **kwargs):
		sol = optimize.root(lambda x: self.USCalSimple_ESC_FH_objective(x, t0, update = update, **kwargs), [self.db['ω'], self.US_βinv()])
		assert sol['success'], f"""Couldn't calibrate model"""
		return sol['x']

	def USCalSimple_ESC_FH_objective(self, x, t0, update = True, **kwargs):
		self.db['ω'] = max(x[0], .1)
		self.db['βi']= self.US_β(x[1])
		sol, pol = self.ESC_FH(returnPols = True, **kwargs)
		if update:
			self.ESC.x0 = {t: pol[t]['x_unbounded'] for t in self.db['t']}
			self.x0['EE_FH'] = sol['x']
		return np.hstack([sol['τ'].xs(t0)-self.db['τ0'],
						  sol['R'].xs(t0)-self.db['R0']])


	def USCalSimple_ESC_log_FH(self, t0, update = True):
		sol = optimize.root(lambda x: self.USCalSimple_ESC_log_FH_objective(x, t0, update = update), [self.db['ω'], self.US_βinv()])
		assert sol['success'], f"""Couldn't calibrate model"""
		return sol['x']
		
	def USCalSimple_ESC_log_FH_objective(self, x, t0, update = True):
		self.db['ω'] = x[0]
		self.db['βi']= self.US_β(x[1])
		sol = self.ESC_log_FH()
		if update:
			self.LOG.x0['ESC'] = np.hstack([sol['τ'].values, sol['κ'].values])
		return np.hstack([sol['τ'].xs(t0)-self.db['τ0'],
						  sol['R'].xs(t0)-self.db['R0']])

	def USCalSimple_PEE_log_FH(self, t0, update = True):
		sol = optimize.root(lambda x: self.USCalSimple_PEE_log_FH_objective(x, t0, update = update), [self.db['ω'], self.US_βinv()])
		assert sol['success'], f"""Couldn't calibrate model"""
		return sol['x']
		
	def USCalSimple_PEE_log_FH_objective(self, x, t0, update = True):
		self.db['ω'] = x[0]
		self.db['βi']= self.US_β(x[1])
		sol = self.PEE_log_FH()
		if update:
			self.LOG.x0['PEE'] = sol['τ'].values
		return np.hstack([sol['τ'].xs(t0)-self.db['τ0'],
						  sol['R'].xs(t0)-self.db['R0']])

	#### 5.2. CALIBRATION OF ξ GIVEN ρ (so far only implemented with log - but simple to extend to general ρ)
	###### Golden section search for ξ:
	def _getCalF(self, log = None):
		log = noneInit(log, True if self.db['ρ'] == 1 else False)
		return self.USCalSimple_ESC_log_FH if log else self.USCalSimple_ESC_FH
	def _getSolF(self, log = None):
		log = noneInit(log, True if self.db['ρ'] == 1 else False)
		return self.ESC_log_FH if log else self.ESC_FH

	def USCal_GoldenSection(self, grid, t0, n = 0, tol = 1e-5, iterMax = 5, log = None, var = 'ξ'):
		""" Golden rule search-like for ξ; n>0 includes a linearly spaced grid on top of linear-interpolated-guided search"""
		sols = self.USCal_OnGrid(grid, t0, full_output = False, log = log, var = var)
		out = self.USCal_checkTol(sols, tol)
		i = 0
		if out:
			return out
		else:
			while i<iterMax:
				sols = self.USCal_GR_iteration(t0, sols, n, log = log, var = var)
				out = self.USCal_checkTol(sols, tol)
				if out:
					return out
				i += 1
		print(f"""Model did not reach tolerance with iterations {iterMax}.""")
		return sols

	def USCal_OnGrid(self, grid, t0, full_output = False, log = None, var = 'ξ'):
		sols = {var: grid, 'obj': np.empty(grid.size)}
		if full_output:
			sols['sol'] = dict.fromkeys(grid)
		sols.update({k: sols['obj'].copy() for k in ('ω','β')})
		for i in range(grid.size):
			sols = self.USCal_OnGrid_i(t0, sols, i, full_output = full_output, log = log, var = var)
		return sols

	def USCal_OnGrid_i(self, t0, sol, i, ωi = None, βi = None, full_output = False, log = None, var = 'ξ'):
		self.db[var] = sol[var][i]
		if var == 'ξ':
			self.US_Xi() # update calibration of Xi based on new value of ξ.
		if ωi is not None:
			self.db['ω'] = sol['ω'][i]
		if βi is not None:
			self.db['βi'] = self.US_β(sol['β'][i])
		self._getCalF(log = log)(t0);
		# self.USCalSimple_ESC_log_FH(t0);
		sol['ω'][i], sol['β'][i] = self.db['ω'].copy(), self.US_βinv()
		if full_output:
			sol['sol'][i] = self._getSolF(log = log)()
			# sol['sol'][i] = self.ESC_log_FH()
			sol['obj'][i] = self.US_κ()-sol['sol'][i]['κ'].xs(t0)
		else:
			sol['obj'][i] = self.US_κ()-self._getSolF(log = log)()['κ'].xs(t0)
		return sol

	def USCal_GR_iteration(self, t0, sols, n, log = None, var = 'ξ'):
		idx = self.USCal_SCidx(sols)
		obj0, obj1 = sols['obj'][idx], sols['obj'][idx+1]
		sols = self.USCal_updateGrid(sols, idx, obj0, obj1, n)
		[self.USCal_OnGrid_i(t0, sols, i, ωi = sols['ω'], βi = sols['β'], full_output=False, log = log, var = var) for i in range(1,n+2)]
		return sols

	def USCal_SCidx(self, sol):
		return np.nonzero(np.diff(np.sign(sol['obj'])) != 0)[0][0]

	def USCal_updateGrid_i(self, k, sol, idx, obj0, obj1, n):
		v = np.linspace(sol[k][idx], sol[k][idx+1], 2+n)
		v = np.sort(np.insert(v, 0, inverseInterp1d(v[0], v[-1], obj0, obj1)))
		return np.flip(v) if sol[k][idx]-sol[k][idx+1]>0 else v

	def USCal_updateGrid(self, sol, idx, obj0, obj1, n):
		return {k: self.USCal_updateGrid_i(k, sol, idx, obj0, obj1, n) for k in sol}

	def USCal_getSol(self, sol):
		idx = abs(sol['obj']).argmin()
		return {k: sol[k][idx] for k in sol}

	def USCal_checkTol(self, sol, tol):
		if min(abs(sol['obj']))<tol:
			return self.USCal_getSol(sol)
		else:
			return False