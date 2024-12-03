import numpy as np, pandas as pd, scipy, functools
from pyDbs import is_iterable, SymMaps as sm, adj
from scipy import optimize
from copy import deepcopy
from argentina_base import BaseScalar, BaseGrid, BaseTime
from argentina_policy import PEE, LOG, inverseInterp1d, cartesianGrids
from argentinaAnalytical_base import BaseScalar_A, BaseGrid_A, BaseTime_A
from argentinaAnalytical_policy import PEE_A, LOG_A

def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

def polGrid(v0, vT, n, exp = 1):
	""" Create polynomial grid with exponent 'exp'. 
		If exp>1 there are more gridpoint in the lower end of the grid."""
	return v0+(vT-v0)*((np.arange(1,n+1)-1)/(n-1))**exp

def defaultGrid_(n, l, u, kl, ku):
	return np.insert(np.linspace(l,u,n-2), [0, n-2], [l-1/kl-1e-4, u+1/ku+1e-4])

def defaultGrid(k, db):
	return defaultGrid_(db[f'{k}_n'], db[f'{k}_l'], db[f'{k}_u'], db[f'k{k}_l'], db[f'k{k}_u'])

def interpFixedPoint(ŝ, s):
	""" Interpolation of fixed point problem with two grids"""
	Δs = ŝ-s # distance from steady state
	changeSign = np.diff(np.sign(Δs))!=0
	s1,s2 = s[:-1][changeSign], s[1:][changeSign]
	ŝ1,ŝ2 = ŝ[:-1][changeSign], ŝ[1:][changeSign]
	return ŝ1+((ŝ2-ŝ1)*(ŝ1-s1))/(s2-s1-(ŝ2-ŝ1)) # lin. approximation

class Model:
	def __init__(self, nj = 4, T = 10, ngrid = 50, ns0 = 10, pars = None, gridkwargs = None):
		""" Fixed namespace """
		self.nj, self.T = nj, T
		self.ni = self.nj-1
		self.db = {}
		self.parTypes = self._parTypes.copy()
		self.initIdxs() # add relevnat pandas indices to database
		self.addNamespaces() # aux classes that help nagivate stacked vectors + lag/lead symbols.
		self.x0 = self.defaultInitials
		self.B = BaseScalar(self)
		self.BG = BaseGrid(self)
		self.BT = BaseTime(self)
		self.initPars(pars = pars) # add parameters and targets
		self.initArgentina()
		self.updateAuxPars()
		self.initGrids(ngrid, ns0, **noneInit(gridkwargs, {}))
		self.PEE = PEE(self)
		self.LOG = LOG(self)

	# Some basic methods for navigating symbols:
	def leadSym(self, symbol, lead = -1, opt = None, ns = 'exo'):
		if isinstance(symbol, pd.Series):
			return self.ns[ns].getShift(symbol, lead, opt = noneInit(opt, {'useLoc':'nn'}))
		elif isinstance(symbol, pd.DataFrame):
			return self.ns[ns].getShift(symbol.stack(), lead, opt = noneInit(opt, {'useLoc': 'nn'})).unstack()
		else:
			return pd.Series(symbol).shift(lead, fill_value=symbol[-1]).values

	def __call__(self, x, name, ns = 'exo', **kwargs):
		return self.ns[ns](x, name, **kwargs)

	def get(self, x, name, ns = 'exo'):
		return self.ns[ns].get(x, name)

	#######################################################################
	##########					1. INIT METHODS				 	###########
	#######################################################################
	def createCopyFromt0(self, t0):
		""" Return a copy of Model instance 'm' with time starting from t0 """
		mt0 = deepcopy(self)
		mt0.T = self.T-t0
		for k,v in self.db.items():
			if isinstance(v, (pd.Series, pd.DataFrame, pd.Index)):
				mt0.db[k] = adj.rc_pd(v, self.db['t'][t0:])
		mt0.db['ν'] = self.db['ν'][t0:]
		mt0.addNamespaces() # reset definition of namespaces
		[baseIns.__setattr__('t0', mt0.db['t'][0]) for baseIns in (mt0.B, mt0.BG, mt0.BT)];
		mt0.PEE.x0 = {t: self.PEE.x0[t] for t in mt0.db['t']}
		mt0.LOG.x0 = {t: self.LOG.x0[t] for t in mt0.db['t']}
		for ns in ('EE_FH','EE_IH','EE_FH_PEE','EE_FH_LOG'):
			mt0.x0[ns] = np.hstack([adj.rc_pd(self.ns[ns].get(self.x0[ns],k), mt0.db['t']) for k in self.ns[ns].symbols])
		return mt0

	def initIdxs(self):
		self.db['t'] = pd.Index(range(self.T), name = 't')
		self.db['txE'] = pd.Index(range(self.T-1), name = 't') # Time index without terminal period
		self.db['t_']= self.db['t'].append(pd.Index([-1], name = 't')).sort_values() # Time index with -1 included
		self.db['j'] = pd.Index(range(self.nj), name = 'j')
		self.db['i'] = self.db['j'][1:]
		self.db['u'] = self.db['j'][0:1]
		self.db['tj'] = pd.MultiIndex.from_product([self.db['t'], self.db['j']])
		self.db['ti'] = pd.MultiIndex.from_product([self.db['t'], self.db['i']])

	@property
	def _parTypes(self):
		return {'0D': list(self.default0DParams), '1D': list(self.default1Dparams)+self.aux1DParams+self.paramsFromFuncs, '2D': list(self.default2Dparams)+self.aux2DParams}
	@property
	def default0DParams(self):
		return {'τ0': .142, 'RR0': 0.678/0.803, 's0': 0.184, 'RRGroups': (1,2), 't0': 2}
	@property
	def default1Dparams(self):
		return {'α': .43, 'ν': 1, 'ξ' : .35, 'ρ': 1.2, 'ω': 1.25, 'α0': .5, 'χ': 1}
	@property
	def default2Dparams(self):
		return {'γj': np.full(self.nj, 1/self.ni)} | {k: np.full(self.nj, 1) for k in ('pj','μj','Xj','ηj','zxj','zηj','βj')}
	@property
	def aux2DParams(self):
		return [f"{k[:-1]}i" for k in self.default2Dparams]
	@property
	def aux1DParams(self):
		return [f"{k[:-1]}0" for k in self.default2Dparams]
	@property
	def paramsFromFuncs(self):
		return ['αr','θ','eps','κ','Γh']
	@property
	def aux_αr(self):
		return (1-self.db['α'])/self.db['α']
	@property
	def aux_θ(self):
		return pd.Series(self.getθ(), index = self.db['t'])
	@property
	def aux_eps(self):
		return pd.Series(self.getEps(), index = self.db['t'])
	@property
	def aux_κ(self):
		return (self.db['p']+self.db['eps[t+1]']*self.db['γ0']*self.db['p0'])*(1+self.db['γ0'])/(1+self.db['γ0[t+1]'])
	@property
	def aux_p(self):
		return (self.db['γi'] * self.db['pi']).sum(axis=1)
	@property
	def aux_Γh(self):
		return self.BT.Γh()

	def addNamespaces(self):
		if not hasattr(self, 'ns'):
			self.ns = {}
		self.ns['EE_IH'] = sm(symbols = {x: self.db['t'] for x in ('s','h','Γs')}) # solve EE given policy
		self.ns['EE_FH'] = sm(symbols = {'h': self.db['t'], 's': self.db['txE'], 'Γs': self.db['txE']}) # solve EE given policy, finite horizon
		self.ns['EE_FH_PEE'] = sm(symbols = {'h': self.db['t'], 's': self.db['txE'], 'Γs': self.db['txE'], 's0/s': self.db['txE']}) # solve EE with endogenous policy, finite horizon.
		self.ns['EE_FH_LOG'] = sm(symbols = {'τ': self.db['t']})
		self.ns['exo'] = sm(symbols = (dict.fromkeys(self.default1Dparams, self.db['t']) |
									   dict.fromkeys(self.default2Dparams, self.db['tj'])| 
									   dict.fromkeys(self.aux1DParams, self.db['t']) |
									   dict.fromkeys(self.aux2DParams, self.db['ti']) |
									   dict.fromkeys(self.paramsFromFuncs, self.db['t'])))
		[ns.compile() for ns in self.ns.values()];
		self.ns['EE_FH'].addLaggedSym('h[t+1]','h',-1, c= ('not',self.db['t'][-1:]))
		self.ns['EE_FH_PEE'].addLaggedSym('h[t+1]','h',-1, c= ('not',self.db['t'][-1:]))
		
	def initPars(self, pars = None):
		self.db.update(self.defaultParameters) # default parameters and targets
		self.addDefaultHeterogeneity # default heterogeneity
		[self.db.update(self.adjPar(k,v)) for k,v in noneInit(pars, {}).items()];

	@property
	def defaultParameters(self):
		return functools.reduce(lambda x,y: x|y, [self.adjPar(k,v) for k,v in self.default0DParams.items()] + [self.adjPar(k,v) for k,v in self.default1Dparams.items()])
	@property
	def addDefaultHeterogeneity(self):
		[self.db.update(self.adjPar(k,v)) for k,v in self.default2Dparams.items()]; 
		# return functools.reduce(lambda x,y: x|y, [self.adjPar(k,v) for k,v in self.default2Dparams.items()])
	def updateAuxPars(self):
		[self.db.update(self.addLeadAndLags(k, getattr(self, f'aux_{k}'))) for k in self.paramsFromFuncs]

	def adjPar(self, k, vals, t = None):
		if k == 'β':
			return self.adj2Dj_ll('βj', self.simpleβ(vals), t = t)
		elif k == 'pj':
			return self.adjpj('pj', vals, t = t)
		elif k in self.parTypes['0D']:
			return {k: vals}
		elif k in self.parTypes['1D']:
			return self.adj1D_ll(k, vals, t = t)
		elif k in self.aux2DParams:
			return self.adj2Di_ll(k, vals, t = t)
		elif k in self.parTypes['2D']:
			return self.adj2Dj_ll(k, vals, t = t)

	def simpleβ(self, β):
		return β * self.db['pj']
	def simpleβinv(self):
		return self.db['βj'].iloc[0,0]/self.db['pj'].iloc[0,0]
	def adjpj(self, k, vals, t = None):
		d = self.adjDf_tj(k,vals,t=t)
		d['p'] = (self.db['γi'] * d['pi']).sum(axis=1)
		return self.addLeadAndLags(d)

	def adj1D_ll(self, k, vals, t = None):
		return self.addLeadAndLags(k, self.adjVec_t(k, vals, t = t))
	def adj2Dj_ll(self, k , vals, t = None):
		return self.addLeadAndLags(self.adjDf_tj(k, vals ,t = t))
	def adj2Di_ll(self, k , vals, t = None):
		return self.addLeadAndLags(self.adjDf_ti(k, vals ,t = t))

	def addLeadAndLags(self, k, s = None):
		if isinstance(k, dict):
			return functools.reduce(lambda x,y: x|y, [self.addLeadAndLags(i,v) for i,v in k.items()])
		else:
			return {k: s, f'{k}[t+1]': self.leadSym(s, lead = {'t':-1} if isinstance(s, pd.DataFrame) else -1), f'{k}[t-1]': self.leadSym(s, lead = {'t': 1} if isinstance(s, pd.DataFrame) else  1)}
	def adjVec_t(self, k, vals, t = None):
		if t is None:
			return pd.Series(vals, index = self.db['t'])
		else:
			x = self.db[k]
			x.loc[t] = vals
			return x
	def adjDf_ti(self, k, vals, t = None):
		if vals.ndim == 1:
			if t is None: 
				xi = pd.DataFrame(np.tile(vals, (self.T,1)), index = self.db['t'], columns = self.db['i'])
			else:
				xi = self.db[k]
				k.loc[t] = vals
		else:
			xi = pd.DataFrame(vals, index = self.db['t'], columns = self.db['i'])
		return {f'{k[:-1]}j': pd.concat([self.db[f'{k[:-1]}0'], xi], axis = 1), k: xi}
	def adjDf_tj(self, k, vals, t = None):
		if vals.ndim == 1:
			if t is None:
				xj =  pd.DataFrame(np.tile(vals, (self.T, 1)), index = self.db['t'], columns = self.db['j'])
			else:
				xj = self.db[k]
				xj.loc[t] = vals
		else:
			xj = pd.DataFrame(vals, index = self.db['t'], columns = self.db['j'])
		return {k: xj, f'{k[:-1]}i': xj[self.db['i']], f'{k[:-1]}0': xj[0]}

	def updateSolGrids(self, ngrid, ns0, update_x0 = True, **kwargs):
		self.initGrids(ngrid, ns0, **kwargs)
		if update_x0:
			self.PEE.x0 = self.PEE.defaultInitials

	def initGrids(self, ngrid, ns0, sgridExp = 1, s0gridExp = 1, **kwargs):
		self.ngrid, self.ns0 = ngrid, ns0
		self.nss0grid = self.ngrid * self.ns0
		d = self.defaultGridSettings | kwargs
		d['sGrid'] = polGrid(d['s_l'], d['s_u'], self.ngrid, sgridExp) # grid value 1d, nonlinear for state 's' 
		d['s0Grid'] = polGrid(d['s0_l'], d['s0_u'], self.ns0, s0gridExp) # grid value 1d, nonlinear for state 's0'
		# Stacked 2d grids as cartesian products of the two:
		idx1ds, d['ss0Idx'], gridsnd = cartesianGrids({'s0': d['s0Grid'], 's': d['sGrid']})
		d['sIdx'], d['s0Idx'] = idx1ds['s'], idx1ds['s0'] 
		d['sGrid_ss0'], d['s0Grid_ss0'] = gridsnd['s'], gridsnd['s0']
		self.db.update(d)
		self.ns['PEEpol'] = sm(symbols = {k: pd.MultiIndex.from_product([self.db['sIdx'], self.db['s0Idx']]) for k in ('s','s0/s', 'τ')})
		self.ns['LOGpol'] = sm(symbols = {k: self.db['s0Idx'] for k in ('s0/s', 'τ')})
		self.ns['PEEpol'].compile()
		self.ns['LOGpol'].compile()


	@property
	def defaultInitials(self):
		return {'EE_FH': np.full(self.ns['EE_FH'].len, .2), 
				'EE_IH': np.full(self.ns['EE_IH'].len, .2),
				'EE_FH_PEE': np.full(self.ns['EE_FH_PEE'].len, .2),
				'EE_FH_LOG': np.full(self.T, .2),
				'SS_Scalar': np.full(self.ni+1, .2)}

	def initArgentina(self):
		""" Apply simple calibration methods + adhoc adjustments relevant for specific Argentina model version """
		# i. ηj, Xj stuff:
		self.addEigenVectors()
		ηi = self.getηi()
		η0 = 0.3 * ηi[0] * self.db['zη0'].xs(self.db['t0'])/self.db['zηi'].xs(self.db['t0'])[1] # initial guess for η0 based on the rest of the vector
		self.db.update(self.adjPar('ηj', np.hstack([η0,ηi])))
		xj = self.db['zxj'].xs(self.db['t0'])
		yx = np.hstack([self.db['yx'][0]*xj[0]/xj[1], self.db['yx']]) # inital guess for yx vector.
		self.db.update(self.adjPar('Xj', self.db['ηj'].values/(np.tile(yx, (self.T,1))**(1/self.db['ξ'].values.reshape(self.T,1)))))

	@property
	def defaultGridSettings(self):
		d = {f'k{x}_l': 10 for x in ('τ','θ', 'eps')}
		d.update({f'k{x}_u': 10 for x in ('τ','θ','eps')})
		d.update({f'{x}_l': 1e-4 for x in ('τ','θ','eps')})
		d.update({f'{x}_n': 101 for x in ('τ','θ','eps')})
		d.update({f'{x}_l': 1e-4 for x in ('τ','θ','eps')})
		d.update({f'{x}_u': 1-1e-4 for x in ('τ','θ','eps')})
		d['s_l'] = 1e-3 # stuff and stuff
		d['s0_l'] = 0
		d['s0_u'] = 1 # s0/s
		d['s_u'] = self.SS_Scalar_solve(0, t = 0)['s'] # set upper bound to 2 times the steady state level with zero taxes (largest possible savings rate)
		# d['s_u'] = 0.02
		return d

	#######################################################################
	##########			2. Simple calibration methods			###########
	#######################################################################

	def addEigenVectors(self):
		valx, vecx = scipy.sparse.linalg.eigs(self.db['zxi'].xs(self.db['t0']).values.reshape(self.ni,1) * self.db['γi'].xs(self.db['t0']).values.reshape(1, self.ni), k = 1)
		valη, vecη = scipy.sparse.linalg.eigs(self.db['zηi'].xs(self.db['t0']).values.reshape(self.ni,1) * self.db['γi'].xs(self.db['t0']).values.reshape(1, self.ni), k = 1)
		self.db['yx'], self.db['yη'] = abs(np.real(vecx)).reshape(self.ni), abs(np.real(vecη).reshape(self.ni))
	def getηi(self):
		return self.db['yη']/(self.db['yx']*sum(self.db['γi'].xs(self.db['t0']).values*self.db['yη']))
	# Calibration given ξ:
	def getXi(self):
		return self.db['ηi']/self.db['yx']**(1/self.db['ξ'])
	def getθ(self):
		i,ii = self.db['RRGroups'][0], self.db['RRGroups'][1]
		ξ= self.db['ξ'].xs(self.db['t0'])
		h1,h2 = (self.db['Xi'][i]**ξ/self.db['ηi'][i]**(1+ξ)).xs(self.db['t0']), (self.db['Xi'][ii]**ξ/self.db['ηi'][ii]**(1+ξ)).xs(self.db['t0'])
		return (self.db['RR0']*h1-h2)/(1-h2-self.db['RR0']*(1-h1))
	def getEps(self, coverageRate = 0.7):
		return coverageRate * (1-self.db['θ'].xs(self.db['t0'])) * (self.simpleβinv()**(5/30)*9.45/14.45+self.simpleβinv()**(10/30)*12.55/22.55)/2


	#######################################################################
	##########				3. Steady state methods 			###########
	#######################################################################
	def SS_solveVector(self, τ, x0 = None, t = None):
		""" Solve for steady state on grid of τ; if length = 1 then use scalar optimization. """
		length = len(τ)
		x0 = noneInit(x0, np.full(int((self.ni+1)*length), .5))
		sol = optimize.root(lambda x: self.BG.steadyStateEqs(x[length:].reshape(length, self.ni), x[:length], τ, t = t), x0)
		assert sol['success'], f""" Couldn't identify steady state with τ = {τ}"""
		return self.SS_report(sol['x'][length:].reshape(length, self.ni), sol['x'][:length], τ, t = t)

	def SS_solveLoop(self, τ, x0_from_loop = True, x0 = None, t = None):
		""" Solve with τ being a vector """
		Bi, Γs = np.empty((len(τ),self.ni)), np.empty(len(τ))
		sol = optimize.root(lambda x: self.B.steadyStateEqs(x[1:], x[0], τ[0], t = t), x0[:self.ni+1])
		Γs[0], Bi[0,:] = sol['x'][0], sol['x'][1:]
		for i in range(1,len(τ)):
			sol = optimize.root(lambda x: self.B.steadyStateEqs(x[1:], x[0], τ[i], t = t), sol['x'] if x0_from_loop else x0[i*(self.ni+1):(i+1)*(self.ni+1)])
			Γs[i], Bi[i,:] = sol['x'][0], sol['x'][1:]
			assert sol['success'], f""" Couldn't identify steady state with τ = {τ}, iteration {i}"""
		return self.SS_report(Bi, Γs, τ, t = t)

	def SS_Scalar_solve(self, τ, t = None, **kwargs):
		sol = optimize.root_scalar(lambda x: self.B.steadyStateScalarEq(x, τ, t = t), bracket = (1e-7,1-1e-7), **kwargs)
		assert sol['converged'], f""" Couldn't identify steady state with τ = {τ}"""
		Bi = self.B.steadyStateScalar_Bi(sol['root'], τ, t = t)
		return self.SS_report(Bi, sol['root'], τ, t = t)

	def LOG_SS_Scalar(self, τ, t = None):
		return self.SS_report(self.B.get('βi',t), self.B.LOG_steadyState_Γs(τ, t = t), τ, t = t)

	def SS_report(self, Bi, Γs, τ, t = None):
		d = {'Bi': Bi, 'Γs': Γs, 'τ': τ, 's': np.nan_to_num(self.B.steadyState_s(Γs, τ, t = t), nan = 0)}
		d['h'] = self.BG.backOutH(s = d['s'], Γs = d['Γs'], t = t)
		d['B0']= self.BG.B0(s_ = d['s'], h = d['h'], t = t)
		d['Θs']= self.BG.backOutΘs(s_ = d['s'], s = d['s'], t = t)
		d['s0/s'] = self.BG.s0_s(B0 = d['B0'], Θs = d['Θs'], τp = τ, t = t)
		return d

	def steadyStatePEE(self, policyFunction, τ0 = None, t = None):
		""" Identify initial tuple of states """
		def fixedPointCriteria(τ):
			ss = self.SS_Scalar_solve(τ, t = t)
			return τ-np.clip(policyFunction(np.array([ss['s'], ss['s0/s']]).T), self.db['τ_l'], self.db['τ_u'])
		sol = optimize.root(fixedPointCriteria, noneInit(τ0, 0.25))
		assert sol['success'], f"""Couldn't identify steady state fixed point """
		return self.SS_Scalar_solve(sol['x'], t = t)

	def steadyStateLOG(self, policyFunction, t = None):
		def fixedPointCriteria(τ):
			ss = self.LOG_SS_Scalar(τ, t = t)
			return τ-np.clip(policyFunction(ss['s0/s']), self.db['τ_l'], self.db['τ_u'])
		sol = optimize.root_scalar(fixedPointCriteria, bracket = (1e-7, 1-1e-7))
		assert sol['converged'], f""" Couldn't identify steady state fixed point """
		return self.LOG_SS_Scalar(sol['root'], t = t)

	#######################################################################
	##########				4. Economic Equilibrium 			###########
	#######################################################################
	def EE_FH_solve(self, τ, s0, x0 = None, update = True):
		τp = self.leadSym(τ)
		sol = optimize.root(lambda x: self.EE_FH_objective(x, τ, τp, s0), noneInit(x0, self.x0['EE_FH']))
		assert sol['success'], f""" Could not identify economic equilibrium (self.EE_FH_solve) with parameter inputs: 
		τ: {τ}, s0: {s0}"""
		if update:
			self.x0['EE_FH'] = sol['x']
		return self.EE_FH_report(sol, s0, τ)

	def EE_FH_objective(self, x, τ, τp, s0):
		Γs, h, s, s_ = self(x, 'Γs',ns = 'EE_FH'), self(x, 'h',ns='EE_FH'), self(x,'s',ns='EE_FH'), self.FH_sLag(x, s0)
		return np.hstack([self.BT.FH_h(s_ = s_, τ = τ, τp = τp, Γs = Γs)-h,
						  self.BT.FH_s(h = h, Γs = Γs)-s,
						  self.BT.FH_Γs(s = s, hp = h[1:], τp = τp)-Γs])

	def FH_sLag(self, x, s0, ns = 'EE_FH'):
		return np.insert(self(x,'s', ns = ns), 0, s0) # get lagged version of s inserting exogenous value as s_(t0).

	def EE_FH_report(self, sol, s0, τ):
		d = self.ns['EE_FH'].unloadSol(sol['x'])
		d['EE_FH_x'] = sol['x']
		d['s[t-1]'] = pd.Series(self.FH_sLag(sol['x'], s0), index = self.db['t'])
		d['τ'] = pd.Series(τ, index = self.db['t'])
		return d

	# With LOG policy functions:
	def EE_FH_LOG_solve(self, gridSol, z0 = None, x0 = None, update = True):
		policyFunction = self.LOG.vectorPolicy(gridSol)
		if z0 is None:
			ss = self.steadyStateLOG(self.LOG.gridPolicy(gridSol[self.db['t'][0]]['τ']), t = self.db['t'][0])
			z0 = [ss['s'], ss['s0/s']]
			if x0 is None:
				x0 = np.full(self.T, ss['τ'])
		sol = optimize.root(lambda x: self.EE_FH_LOG_objective(x, policyFunction, z0), noneInit(x0, self.x0['EE_FH_LOG']))
		assert sol['success'], f""" Couldn't identify economic equilibrium"""
		if update:
			self.x0['EE_FH_LOG'] = sol['x']
		return self.EE_FH_LOG_report(sol['x'], policyFunction, z0)

	def EE_FH_LOG_objective(self, τ, policyFunction, z0):
		return policyFunction(self.LOG_EE_FH(τ, s0 = z0[0], s0_s = z0[1])['s0/s[t-1]'])-τ

	def LOG_EE_FH(self, τ, s0 = None, s0_s = None):
		sol = {'τ': τ, 'τ[t+1]': self.leadSym(τ)}
		sol['Γs'] = self.BT.FH_LOG_Γs(τ = sol['τ'], τp = sol['τ[t+1]'])
		sol['Θh'] = self.BT.FH_LOG_Θh(τ = sol['τ'], τp = sol['τ[t+1]'], Γs = sol['Γs'])
		sol['Θs'] = self.BT.FH_LOG_Θs(Θh = sol['Θh'], Γs = sol['Γs'])
		sol['s']  = self.BT.FH_LOG_s(Θs = sol['Θs'], s0 = s0)
		sol['s[t-1]'] = np.insert(sol['s'], 0, s0)
		sol['h']  = self.BT.FH_LOG_h(Θh = sol['Θh'], s_ = sol['s[t-1]'])
		sol['s0/s'] = self.BT.FH_LOG_s0_s(Θs = sol['Θs'], τp = sol['τ[t+1]'])
		sol['s0/s[t-1]'] = np.insert(sol['s0/s'], 0, s0_s)
		return sol

	def EE_FH_LOG_report(self, x, policyFunction, z0):
		d = self.LOG_EE_FH(x, s0 = z0[0], s0_s = z0[1])
		[d.__setitem__(k, pd.Series(d[k], index = self.db['t'])) for k in ('τ','τ[t+1]','Θh','s[t-1]','h','s0/s[t-1]')];
		[d.__setitem__(k, pd.Series(d[k], index = self.db['txE'])) for k in ('Γs','Θs','s','s0/s')];
		return d

	# With PEE policy functions:
	def EE_FH_PEE_solveRobust(self, gridSol, z0 = None, τ0 = None, update = True):
		""" Get initial guess from approximate solution """
		approx, z0 = self.EE_FH_PEE_approx(gridSol, z0 = z0, τ0 = τ0)
		return self.EE_FH_PEE_solve(gridSol, z0 = z0, x0 = approx, update = update)

	def EE_FH_PEE_solve(self, gridSol, z0 = None, x0 = None, τ0 = None, update = True):
		policyFunction = self.PEE.vectorPolicy(gridSol)
		if z0 is None:
			ss = self.steadyStatePEE(self.PEE.gridPolicy(gridSol[self.db['t'][0]]['τ']), τ0 = τ0, t = self.db['t'][0])
			z0 = [ss['s'], ss['s0/s']]
		sol = optimize.root(lambda x: self.EE_FH_PEE_objective(x, policyFunction, z0), noneInit(x0, self.x0['EE_FH_PEE']))
		assert sol['success'], f""" Could not identify economic equilibrium (self.EE_FH_PEE_solve) with parameter inputs: 
		state: {z0}"""
		if update:
			self.x0['EE_FH_PEE'] = sol['x']
		return self.EE_FH_PEE_report(sol['x'], policyFunction, z0)

	def EE_FH_PEE_objective(self, x, policyFunction, z0):
		Γs, h, s, s_, s0_s = self(x, 'Γs',ns = 'EE_FH_PEE'), self(x, 'h',ns='EE_FH_PEE'), self(x,'s',ns='EE_FH_PEE'), self.FH_sLag(x, z0[0], ns = 'EE_FH_PEE'), self(x, 's0/s',ns = 'EE_FH_PEE')
		s0_s_ = np.insert(s0_s, 0, z0[1]) # add initial state s0/s[0] to the vector of states.
		τ = policyFunction(np.vstack([s_, s0_s_]).T) # evaluate policy
		τp = self.leadSym(τ) # get leaded version
		return np.hstack([self.BT.FH_h(s_ = s_, τ = τ, τp = τp, Γs = Γs)-h,
						  self.BT.FH_s(h = h, Γs = Γs)-s,
						  self.BT.FH_Γs(s = s, hp = h[1:], τp = τp)-Γs,
						  self.BT.FH_s0_s(s_ = s_, s = s, hp = h[1:], τp = τp)-s0_s])

	def EE_FH_PEE_report(self, x, policyFunction, z0):
		d = self.ns['EE_FH_PEE'].unloadSol(x)
		d['EE_FH_PEE_x'] = x
		d['s[t-1]'] = pd.Series(self.FH_sLag(x, z0[0], ns = 'EE_FH_PEE'), index = self.db['t'])
		d['s0/s[t-1]'] = pd.Series(np.insert(self(x, 's0/s', ns = 'EE_FH_PEE'), 0, z0[1]), index = self.db['t'])
		d['τ'] = pd.Series(policyFunction(np.vstack([d['s[t-1]'].values, d['s0/s[t-1]'].values]).T), index = self.db['t'])
		d['Θh'] = self.BT.FH_BackOutΘh(s_ = d['s[t-1]'], h = d['h'])
		d['Θs'] = self.BT.FH_BackOutΘs(s_ = d['s[t-1]'], s = d['s'])
		return d

	### Approximate PEE solution from grid
	def EE_FH_PEE_approx(self, gridSol, z0 = None, τ0 = None, idx = None, **kwargs):
		syms = list(self.ns['EE_FH_PEE'].symbols)
		idx = noneInit(idx, (syms.index('s'), syms.index('s0/s')))
		sol = dict.fromkeys(self.db['t'])
		sol[self.db['t'][0]], z0 = self.EE_FH_PEE_approx_t0(gridSol[self.db['t'][0]], z0 = z0, τ0 = τ0, **kwargs)
		[sol.__setitem__(t, self.EE_FH_PEE_approx_tx0(gridSol[t], self.getZ0(sol[t-1], idx = idx), t), **kwargs) for t in self.db['t'][1:]]
		return self.approxStackVector(sol), z0[0]

	def EE_FH_PEE_approx_t0(self, gridSol0, z0 = None, τ0 = None, **kwargs):
		if z0 is None:
			ss = self.steadyStatePEE(self.PEE.gridPolicy(gridSol0['τ'], **kwargs), τ0 = τ0, t = self.db['t'][0])
			z0 = np.array([ss['s'], ss['s0/s']]).T
		return np.array([self.PEE.gridPolicy(gridSol0[k], **kwargs)(z0)[0] for k in self.ns['EE_FH_PEE'].symbols]), z0
	def getZ0(self, sol_, idx = (1,3)):
		return np.array([sol_[idx[0]], sol_[idx[1]] ]).T
	def EE_FH_PEE_approx_tx0(self, gridSolt, z0, t, **kwargs):
		return np.array([self.PEE.gridPolicy(gridSolt[k], **kwargs)(z0)[0] for k in self.ns['EE_FH_PEE'].symbols])
	def approxStackVector(self, sol):
		""" This selecst the right elements corresponding to the solution vector in problem EE_FH_PEE. """
		arr = np.vstack(list(sol.values()))
		return np.hstack([arr[:,0], arr[:-1,1], arr[1:,2], arr[:-1, 3]])

	#######################################################################
	##########					5. PEE methods	 				###########
	#######################################################################
	def PEE_initialsFromSmallGrid(self, ngrid, ns0, **kwargs):
		""" Temporarily create smaller grids to get a rough solution - add to self.PEE.x0 """
		ngrid_0, ns0_0 = self.ngrid, self.ns0
		self.updateSolGrids(ngrid, ns0, **kwargs)
		sols = self.PEE.FH(**kwargs)
		idx, grids = self.db['ss0Idx'], (self.db['sGrid'], self.db['s0Grid'])
		# revert grids
		self.updateSolGrids(ngrid_0, ns0_0)
		self.PEE.interpInitialsFromSols(sols, idx, grids)
		return sols, idx, grids

	#######################################################################
	##########				6. Calibration methods	 			###########
	#######################################################################
	def calibLOG(self, x0 = None, update = True, **kwargs):
		sol = optimize.root(lambda x: self.calibLOG_objective(x, update = update), noneInit(x0, self.calibGetx0()), **kwargs)
		assert sol['success'], f""" Couldn't calibrate model """
		return sol['x']

	def calibGetx0(self):
		return np.hstack([self.simpleβinv(), self.db['ω'].xs(self.db['t0']), self.db['η0'].xs(self.db['t0']), self.db['X0'].xs(self.db['t0'])])

	def calibLOG_objective(self, x, update = True):
		self.calibUpdateParameters(x)
		sol = self.LOG.FH()
		path = self.EE_FH_LOG_solve(sol)
		η0 = self.B.calib_η0(τ = path['τ'].xs(self.db['t0']), Θh = path['Θh'].xs(self.db['t0']))
		return np.hstack([path['τ'].xs(self.db['t0'])-self.db['τ0'],
						  self.B.calib_savingsRate(Θs = path['Θs'].xs(self.db['t0']), Θh = path['Θh'].xs(self.db['t0']))-self.db['s0'],
						  η0-x[2],
						  self.B.calib_X0(η0 = η0, Θh = path['Θh'].xs(self.db['t0']))-x[3]])

	def calibPEE(self, x0  = None, update = True, **kwargs):
		""" Simple update of parameters based on simulated PEE Path"""
		sol = optimize.root(lambda x: self.calibPEE_objective(x, update = update), noneInit(x0, self.calibGetx0()), **kwargs)
		assert sol['success'], f"""Couldn't calibrate model"""
		return sol['x']

	def calibUpdateParameters(self, x):
		self.db.update(self.adjPar('β', x[0])) # update beta estimate - this makes sure that entire β matrix and subcomponents are updated
		self.db.update(self.adjPar('ω', x[1])) # update omega estimate.
		ηj = self.db['ηj'].iloc[0].values.copy()
		ηj[0] = x[2]
		self.db.update(self.adjPar('ηj',ηj))
		Xj = self.db['Xj'].iloc[0].values.copy()
		Xj[0] = x[3]
		self.db.update(self.adjPar('Xj',Xj))
		self.updateAuxPars() # update auxiliary parameters

	def calibPEE_objective(self, x, update = True):
		""" Search for β, ω that results in desired savings rate + PEE tax rate. """
		self.calibUpdateParameters(x)
		sol = self.PEE.FH()
		PEE = self.EE_FH_PEE_solveRobust(sol, update = update)
		η0 = self.B.calib_η0(τ = PEE['τ'].xs(self.db['t0']), Θh = PEE['Θh'].xs(self.db['t0']))
		return np.hstack([PEE['τ'].xs(self.db['t0'])-self.db['τ0'],
						 self.B.calib_savingsRate(Θs = PEE['Θs'].xs(self.db['t0']), Θh = PEE['Θh'].xs(self.db['t0']))-self.db['s0'],
						  η0-x[2],
						  self.B.calib_X0(η0 = η0, Θh = PEE['Θh'].xs(self.db['t0']))-x[3]])

class Model_A(Model):
	def __init__(self, nj = 4, T = 10, ngrid = 50, pars = None, gridkwargs = None):
		""" Fixed namespace """
		self.nj, self.T = nj, T
		self.ni = self.nj-1
		self.db = {}
		self.parTypes = self._parTypes.copy()
		self.initIdxs() # add relevnat pandas indices to database
		self.addNamespaces() # aux classes that help nagivate stacked vectors + lag/lead symbols.
		self.x0 = self.defaultInitials
		self.B = BaseScalar_A(self)
		self.BG = BaseGrid_A(self)
		self.BT = BaseTime_A(self)
		self.initPars(pars = pars) # add parameters and targets
		self.initArgentina()
		self.updateAuxPars()
		self.initGrids(ngrid, **noneInit(gridkwargs, {}))
		self.PEE = PEE_A(self)
		self.LOG = LOG_A(self)

	def addNamespaces(self):
		self.ns = {}
		self.ns['EE_IH'] = sm(symbols = {x: self.db['t'] for x in ('s','h','Γs')}) # solve EE given policy
		self.ns['EE_FH'] = sm(symbols = {'h': self.db['t'], 's': self.db['txE'], 'Γs': self.db['txE']}) # solve EE given policy, finite horizon
		self.ns['exo'] = sm(symbols = (dict.fromkeys(self.default1Dparams, self.db['t']) |
									   dict.fromkeys(self.default2Dparams, self.db['tj'])| 
									   dict.fromkeys(self.aux1DParams, self.db['t']) |
									   dict.fromkeys(self.aux2DParams, self.db['ti']) |
									   dict.fromkeys(self.paramsFromFuncs, self.db['t'])))
		[ns.compile() for ns in self.ns.values()];
		self.ns['EE_FH'].addLaggedSym('h[t+1]','h',-1, c= ('not',self.db['t'][-1:]))

	@property
	def defaultInitials(self):
		return {'EE_FH': np.full(self.ns['EE_FH'].len, .2), 
				'EE_IH': np.full(self.ns['EE_IH'].len, .2),
				'EE_FH_LOG': np.full(self.T, .2),
				'SS_Scalar': np.full(self.ni+1, .2)}

	def initGrids(self, ngrid, sgridExp = 1, s0gridExp = 1, **kwargs):
		self.ngrid = ngrid
		d = self.defaultGridSettings | kwargs
		d['sGrid'] = polGrid(d['s_l'], d['s_u'], self.ngrid, sgridExp) # grid value 1d, nonlinear for state 's'
		d['τGrid'] = defaultGrid('τ', d) # grid value 1d, policy grid for τ
		# Stacked 2d grids as cartesian products of the two:
		idx1ds, d['sτIdx'], gridsnd = cartesianGrids({'τ': d['τGrid'], 's': d['sGrid']})
		d['sIdx'], d['τIdx'] = idx1ds['s'], idx1ds['τ'] 
		d['sGrid_sτ'], d['τGrid_sτ'] = gridsnd['s'], gridsnd['τ']
		self.db.update(d)

	@property
	def defaultGridSettings(self):
		d = {f'k{x}_l': 10 for x in ('τ','θ', 'eps')}
		d.update({f'k{x}_u': 10 for x in ('τ','θ','eps')})
		d.update({f'{x}_l': 1e-4 for x in ('τ','θ','eps')})
		d.update({f'{x}_n': 101 for x in ('τ','θ','eps')})
		d.update({f'{x}_l': 1e-4 for x in ('τ','θ','eps')})
		d.update({f'{x}_u': 1-1e-4 for x in ('τ','θ','eps')})
		d['s_l'] = .001 # stuff and stuff
		# d['s_u'] = 2 * self.SS_Scalar_solve(0, t = 0)['s'] # set upper bound to 2 times the steady state level with zero taxes (largest possible savings rate)
		d['s_u'] = 0.05
		return d
