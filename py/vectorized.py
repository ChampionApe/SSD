import numpy as np, pandas as pd
from scipy import optimize

def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

def pss(x, l =-1):
	return x.shift(l, fill_value = x.iloc[-1])

def pssa(x, l = -1):
	return pd.Series(x).shift(-1, fill_value=x[-1]).values

def addLevelToUtil(x, par, ν, s_):
	return x if s_ is None else x+par*np.log(s_/ν)

class infHorizon:
	def __init__(self, ni = 11, T = 10, **kwargs):
		""" Fixed namespace """
		self.ni = ni
		self.T = T
		self.db = self.defaultParameters | kwargs
		self.db['t'] = pd.Index(range(self.T), name = 't')
		self.db['t_']= self.db['t'].append(pd.Index([-1], name = 't')).sort_values() # Time index with -1 included
		self.db['i'] = pd.Index(range(self.ni), name = 'i')
		self.db['coreVars'] = pd.Index(['τ', 'Υ', 'Θh', 'Θs', 'dlnΥ', 'dlnΘh', 'dlnΘs'], name = 'variable').sort_values()
		self.mainIndex = pd.MultiIndex.from_product([self.db['t'], self.db['coreVars']]).sort_values()
		self.eeIndex = self.mainIndex[self.mainIndex.get_level_values('variable').isin(['Θh','Θs','Υ'])]
		self.lnDevIndex = self.mainIndex[self.mainIndex.get_level_values('variable').isin(['dlnΘh','dlnΘs','dlnΥ'])]
		self.db['Γ'] = self.auxΓ
		self.mainLinIndex = self.linIndex(self.mainIndex)

	@property
	def defaultParameters(self):
		return {'α': .5, 
				'A': np.ones(self.T), 
				'ν': np.ones(self.T),
				'η': np.linspace(1,2,self.ni),
				'γ': np.full((self.ni,), 1/self.ni),
				'X': np.ones(self.ni),
				'β': np.full((self.ni,), .32),
				'βu': .25, 
				'ξ' : .25,
				'epsilon' : .5, 
				'θ' : .5, 
				'γu': .05, 
				'χ1': .1, 
				'χ2': .05,
				'ω': .5}

	@property
	def power_s(self):
		return self.db['α']*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])
	@property
	def power_h(self):
		return self.db['α']*self.db['ξ']/(1+self.db['α']*self.db['ξ'])
	@property
	def power_p(self):
		return self.power_s**2

	def linIndex(self, index):
		return pd.Series(range(len(index)), index = index)

	def getIndex(self, name, index, l = None):
		""" Return linear index needed to subset symbol 'name'; use l = -1 to get the x_{t+1} vector. """
		return index.xs(name, level = 'variable') if l is None else pss(index.xs(name, level = 'variable'), l = l)		

	def get(self, x, name, l = None):
		return x[self.getIndex(name, self.mainLinIndex, l = l)]

	def get_(self, x, name, index, l = None):
		return x[self.getIndex(name, self.linIndex(index), l= l)]

	def getEE(self, x, name, l = None):
		return self.get_(x, name, self.eeIndex, l = l)

	def getLnDev(self, x, name, l = None):
		return self.get_(x, name, self.lnDevIndex, l = l)

	## Auxiliary parameter functions:
	@property
	def auxΓ(self):
		return sum(self.db['γ'] * np.power(self.db['η'], 1+self.db['ξ']) / np.power(self.db['X'], self.db['ξ']))

	@property
	def auxΓβ1(self):
		return sum( (self.db['β']/(1+self.db['β'])) * self.db['γ'] * np.power(self.db['η'], 1+self.db['ξ'])/np.power(self.db['X'], self.db['ξ']))

	@property
	def auxΓβ2(self):
		return sum( (1/(1+self.db['β']))  * self.db['γ'] * np.power(self.db['η'], 1+self.db['ξ'])/np.power(self.db['X'], self.db['ξ']))

	@property
	def auxΓβ3(self):
		return sum( self.db['γ'] / (1+self.db['β']))

	@property
	def auxPenDen(self):
		return 1-self.db['γu']+self.db['γu']*(1-self.db['epsilon'])*(1-self.db['θ'])

	def auxIncome(self, τ, Υ):
		return (1-self.db['α'])*(1-τ)*self.db['A']*np.power(Υ, self.db['ξ']*(1-self.db['α'])/(1+self.db['ξ']))/(self.db['Γ']**(self.db['α']))

	def auxSavings(self, τ, Υ):
		return self.auxIncome(τ, Υ)-(self.db['ξ']/(1+self.db['ξ']))*Υ

	def auxΥ(self, Θh, Θs, τ, τp):
		return ((1-self.db['α'])*(1-τ)*self.db['A']*np.power(Θh,1-self.db['α']) + ((1-self.db['α'])/self.db['α'])*self.db['θ'] * τp * Θs /self.auxPenDen)/self.db['Γ']

	def auxΘh(self, Υ):
		return self.db['Γ'] * np.power(Υ,self.db['ξ']/(1+self.db['ξ']))

	def auxΘs(self, Υ, τ, τp):
		return self.auxSavings(τ, Υ)*self.auxΓβ1/(1+((1-self.db['α'])/self.db['α'])*(τp/self.auxPenDen)*((self.db['θ']/self.db['Γ'])*self.auxΓβ2+(1-self.db['θ'])*self.auxΓβ3))

	## Core Economic Equilibrium functions:
	def savingsSpread(self, τp):
		""" Return vector of Θsi/Θs """
		x1 = ((self.db['β'] * np.power(self.db['η'], 1+self.db['ξ'])/(np.power(self.db['X'], self.db['ξ'])*(1+self.db['β']))) / self.auxΓβ1).reshape(self.ni,1)
		x2 = (self.db['θ']/self.db['Γ'])*self.auxΓβ2+(1-self.db['θ'])*self.auxΓβ3
		x3 = ((1-self.db['α'])/self.db['α'])*τp/self.auxPenDen
		x4 = ((self.db['θ']*np.power(self.db['η'], 1+self.db['ξ'])/(self.db['Γ']*np.power(self.db['X'], self.db['ξ']))+1-self.db['θ'])/(1+self.db['β'])).reshape(self.ni,1)
		return x1*(1+x2*x3)-x3*x4

	def economicEquilibriumEqs(self, Θh, Θs, Υ, τ, τp):
		""" Core equations for economic equilibrium in given year given parameters. x is a vector of Θh, Θs, Υ"""
		return np.hstack([self.auxΥ(Θh, Θs, τ, τp)-Υ,
						  self.auxΘh(Υ)-Θh,
						  self.auxΘs(Υ, τ, τp)-Θs])

	def solveCoreEE(self, τ, τp, x0 = None):
		""" Given a vector of τ, τ_{t+1} (and parameters), solve for economic equilibrium """
		sol, _, ier, msg = optimize.fsolve(lambda x: self.economicEquilibriumEqs(self.getEE(x, 'Θh'),
																				 self.getEE(x, 'Θs'),
																				 self.getEE(x, 'Υ'),
																				 τ, τp),
							noneInit(x0, [0.5]*(self.T*3)), full_output=True)
		if ier == 1:
			return pd.Series(sol, index = self.eeIndex)
		else:
			print(f"solveCoreEE couldn't identify an equilibrium - fsolve returns {msg}")

	def auxLnDevΥ(self, dlnΘh, dlnΘs, Θh, Θs, Υ, τ, τp):
		""" returns ∂ln(Υ)/∂τ """
		return (1/(Υ*self.db['Γ'])) * ((1-self.db['α'])*self.db['A']*np.power(Θh, 1-self.db['α'])*((1-τ)*(1-self.db['α'])*dlnΘh-1)+((1-self.db['α'])/self.db['α'])*self.db['θ']*τp * Θs * dlnΘs/self.auxPenDen)

	def auxLnDevΘh(self, dlnΥ):
		return (self.db['ξ']/(1+self.db['ξ'])) * dlnΥ

	def auxLnDevΘs(self, dlnΥ, Υ, τ):
		return (dlnΥ*((self.db['ξ']*(1-self.db['α'])/(1+self.db['ξ']))*(1-self.db['α'])*(1-τ)*self.db['A']*np.power(Υ, self.db['ξ']*(1-self.db['α'])/(1+self.db['ξ']))/(self.db['Γ']**self.db['α'])- self.db['ξ']*Υ/(1+self.db['ξ']))-(1-self.db['α'])*self.db['A']*np.power(Υ, self.db['ξ']*(1-self.db['α'])/(1+self.db['ξ']))/(self.db['Γ']**self.db['α']))/self.auxSavings(τ, Υ)

	def lnDevsEqs(self, dlnΘh, dlnΘs, dlnΥ, Θh, Θs, Υ, τ, τp):
		""" Equations that determine dlnΘh, dlnΘs, dlnΥ, given Θh, Θs, Υ, τ, τp. """
		return np.hstack([self.auxLnDevΘh(dlnΥ)-dlnΘh,
						  self.auxLnDevΘs(dlnΥ, Υ, τ)-dlnΘs,
						  self.auxLnDevΥ(dlnΘh, dlnΘs, Θh, Θs, Υ, τ, τp)-dlnΥ])

	def solveLnDevs(self, Θh, Θs, Υ, τ, τp, x0 = None):
		""" Solve dlnΘh, dlnΘs, dlnΥ. """
		sol, _, ier, msg = optimize.fsolve(lambda x: self.lnDevsEqs(self.getLnDev(x, 'dlnΘh'),
																	self.getLnDev(x, 'dlnΘs'),
																	self.getLnDev(x, 'dlnΥ'),
																	Θh, Θs, Υ, τ, τp),
							noneInit(x0, [0.5]*(self.T*3)), full_output=True)
		if ier == 1:
			return pd.Series(sol, index = self.lnDevIndex)
		else:
			print(f"solveLnDevs couldn't identify an equilibrium - fsolve returns {msg}")

	## PEE Functions:
	@property
	def auxPensionRate(self):
		return (self.db['Γ']*(1-self.db['θ'])+self.db['θ']*np.power(self.db['η'], 1+self.db['ξ'])/np.power(self.db['X'], self.db['ξ']))/(self.db['Γ']*self.auxPenDen)

	def polSupportRetirees(self, dlnΘh, τ):
		return self.db['ω']*(self.db['γu']*self.polSupportUnemployedRetiree(dlnΘh, τ)+(1-self.db['γu'])*np.matmul(self.db['γ'],self.polSupportRetireeVector(dlnΘh, τ)))

	def polSupportUnemployedRetiree(self, dlnΘh, τ):
		return (1-self.db['α'])*(dlnΘh+(1-self.db['epsilon'])*(1-self.db['θ'])/(self.db['χ2'] * self.auxPenDen/self.db['ν']+(1-self.db['α'])*(1-self.db['epsilon'])*(1-self.db['θ'])*τ))

	def polSupportRetireeVector(self, dlnΘh, τ):
		""" Return matrix with rows = types, columns = years """
		x = self.auxPensionRate.reshape(self.ni, 1)
		return (1-self.db['α'])*(dlnΘh+x/(self.db['α']*self.savingsSpread(τ)+(1-self.db['α'])*x*τ))

	def polSupportUnemployedYoung(self, dlnΘh, dlnΘs):
		return (1-self.db['α'])*dlnΘh+self.db['βu']*self.db['α']*dlnΘs*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])

	def polSupportWorkerVector(self, dlnΘs):
		return dlnΘs * (1+self.db['β'].reshape(self.ni,1)*self.power_s)

	def polSupportYoung(self, dlnΘh, dlnΘs):
		return self.db['ν']*(self.db['γu']*self.polSupportUnemployedYoung(dlnΘh, dlnΘs)+(1-self.db['γu'])*np.matmul(self.db['γ'], self.polSupportWorkerVector(dlnΘs)))

	def PEECondition(self, dlnΘh, dlnΘs, τ):
		return self.polSupportYoung(dlnΘh, dlnΘs)+self.polSupportRetirees(dlnΘh,τ)

	def corePEEEqs(self, dlnΘh, dlnΘs, dlnΥ, Θh, Θs, Υ, τ, τp):
		return np.hstack([self.economicEquilibriumEqs(Θh, Θs, Υ, τ, τp),
						  self.lnDevsEqs(dlnΘh, dlnΘs, dlnΥ, Θh, Θs, Υ, τ, τp),
						  self.PEECondition(dlnΘh, dlnΘs, τ)])

	def solveCorePEE(self, x0 = None):
		""" Given parameters, solve PEE """
		sol, _, ier, msg = optimize.fsolve(lambda x: self.corePEEEqs(self.get(x, 'dlnΘh'),
																	 self.get(x, 'dlnΘs'),
																	 self.get(x, 'dlnΥ'),
																	 self.get(x, 'Θh'),
																	 self.get(x, 'Θs'),
																	 self.get(x, 'Υ'),
																	 self.get(x, 'τ'),
																	 self.get(x, 'τ', l=-1)),
							noneInit(x0, [0.5]*(self.T*7)), full_output=True)
		if ier == 1:
			return pd.Series(sol, index = self.mainIndex)
		else:
			print(f"solveCorePEE couldn't identify an equilibrium - fsolve returns {msg}")

	def updateAndSolve(self, x0 = None, **kwargs):
		""" Update parameters with dictionary kwargs and resolve """
		self.db.update(kwargs)
		self.sol = self.solveCorePEE(x0 = x0)
		return self.sol

	def calibrateω(self, τ0, t0, x0=None):
		sol, _, ier, msg = optimize.fsolve(lambda x: self.updateAndSolve(**{'ω': x}).xs((t0,'τ'))-τ0, noneInit(x0, self.db['ω']), full_output=True)
		if ier == 1:
			self.db['ω'] = sol
			return sol
		else:
			print(f"Error in calibrateω: {msg}")

	## REPORTING FUNCTIONS - MAP TO FULL SYSTEM OF EQUATIONS AND COMPUTE LEVELS IN VARIABLES:
	def reportAll(self, sol = None):
		self.unloadCorePEE(sol = noneInit(sol, self.sol))
		self.reportCoefficients()
		self.reportLevels()
		self.reportUtils()		

	def unloadCorePEE(self, sol):
		""" Unload to database as separate pandas symbols; 'sol' is the ouptut from self.solveCorePEE"""
		[self.db.__setitem__(k, sol.xs(k,level='variable')) for k in self.db['coreVars']];

	def reportCoefficients(self):
		""" Assumes that self.unloadCorePEE has been run """
		[self.db.__setitem__(k, getattr(self,'aux'+k)()) for k in ('Θhi','Θsi','Θc1i','Θc2i','Θc2pi','Θ̃c1i','Θc1u','Θc2u','Θc2pu')];

	def reportLevels(self):
		""" Assumes self.reportCoefficients has been run"""
		self.db['s'] = self.levels_s()
		[self.db.__setitem__(k, getattr(self,'levels_'+k)(self.db['s'])) for k in ('c1i','c2i','c1u','c2u','̃c1i','c2pi','c2pu','hi','h')];

	def reportUtils(self):
		""" Assumes self.reportLevels has been run"""
		[self.db.__setitem__(k, getattr(self,'aux_'+k)()) for k in ('util1i','util1u','util2i','util2u')];
		self.db['utilPol'] = self.aux_utilPol()

	# Coefficient functions:
	def auxΘhi(self, Υ = None):
		Υ = noneInit(Υ, self.db['Υ'].values)
		return pd.DataFrame((np.power(Υ, self.db['ξ']/(1+self.db['ξ']))*(np.power(self.db['η'], 1+self.db['ξ'])/np.power(self.db['X'],self.db['ξ'])).reshape(self.ni,1)).T,
				index = self.db['t'], columns = self.db['i'])

	def auxΘsi(self, Θs = None, Υ=None, τ=None):
		Θs, Υ, τ = noneInit(Θs, self.db['Θs'].values), noneInit(Υ, self.db['Υ'].values), noneInit(τ, self.db['τ'].values)
		return pd.DataFrame(((1/(1+self.db['β'])).reshape(self.ni,1)*((self.db['β']*np.power(self.db['η'], 1+self.db['ξ'])/np.power(self.db['X'], self.db['ξ'])).reshape(self.ni,1) * self.auxSavings(τ, Υ)-((1-self.db['α'])/self.db['α'])*self.auxPensionRate.reshape(self.ni,1)*pssa(τ)*Θs)).T,
				index = self.db['t'], columns = self.db['i'])

	def auxΘc1i(self, Θs = None, Υ = None, τ = None):
		Θs, Υ, τ = noneInit(Θs, self.db['Θs'].values), noneInit(Υ, self.db['Υ'].values), noneInit(τ, self.db['τ'].values)
		return pd.DataFrame(((1/(1+self.db['β'])).reshape(self.ni,1)*((np.power(self.db['η'], 1+self.db['ξ'])/np.power(self.db['X'], self.db['ξ'])).reshape(self.ni,1)*(self.auxIncome(τ, Υ)+self.db['β'].reshape(self.ni,1)*self.db['ξ']*Υ/(1+self.db['ξ'])) +((1-self.db['α'])/self.db['α'])*self.auxPensionRate.reshape(self.ni,1)*pssa(τ)*Θs)).T,
				index = self.db['t'], columns = self.db['i'])

	def auxΘc2i(self, Θh=None, τ=None):
		Θh, τ = noneInit(Θh, self.db['Θh'].values), noneInit(τ, self.db['τ'].values)
		return pd.DataFrame((self.db['α']*self.db['A']*self.db['ν']*np.power(Θh, 1-self.db['α'])*(self.savingsSpread(τ)+((1-self.db['α'])/self.db['α'])*self.auxPensionRate.reshape(self.ni,1)*τ)).T,
				index = self.db['t'], columns = self.db['i'])

	def auxΘc2pi(self, Θh =None, Θs = None, τ=None):
		Θhp, Θs, τp = pssa(noneInit(Θh, self.db['Θh'].values)), noneInit(Θs, self.db['Θs'].values), pssa(noneInit(τ, self.db['τ'].values))
		return pd.DataFrame((self.db['α'] * pssa(self.db['A']) * pssa(self.db['ν']) * np.power(Θhp, 1-self.db['α']) * np.power(Θs/pssa(self.db['ν']), self.power_s) * (self.savingsSpread(τp)+((1-self.db['α'])/self.db['α'])*self.auxPensionRate.reshape(self.ni,1)*τp)).T,
				index = self.db['t'], columns = self.db['i'])

	def auxΘ̃c1i(self, Θs = None, Υ = None, τ = None):
		Θs, Υ, τ = noneInit(Θs, self.db['Θs'].values), noneInit(Υ, self.db['Υ'].values), noneInit(τ, self.db['τ'].values)
		return pd.DataFrame(((1/(1+self.db['β'])).reshape(self.ni,1)*((np.power(self.db['η'], 1+self.db['ξ'])/np.power(self.db['X'], self.db['ξ'])).reshape(self.ni,1) * self.auxSavings(τ, Υ)+((1-self.db['α'])/self.db['α'])*self.auxPensionRate.reshape(self.ni,1)*pssa(τ)*Θs)).T,
				index = self.db['t'], columns =self.db['i'])

	def auxΘc1u(self, Θh = None):
		Θh = noneInit(Θh, self.db['Θh'].values)
		return pd.Series(self.db['χ1']*self.db['A']*np.power(Θh,1-self.db['α']), 
				index = self.db['t'])

	def auxΘc2u(self, Θh = None, τ = None):
		Θh, τ = noneInit(Θh, self.db['Θh'].values), noneInit(τ, self.db['τ'].values)
		return pd.Series(self.db['α']*self.db['A']*self.db['ν']*np.power(Θh,1-self.db['α'])*(self.db['χ2']/(self.db['α']*self.db['ν'])+τ*((1-self.db['α'])/self.db['α'])*(1-self.db['epsilon'])*(1-self.db['θ'])/self.auxPenDen),
				index = self.db['t'])

	def auxΘc2pu(self, Θh = None, Θs = None, τ = None):
		Θhp, Θs, τp = pssa(noneInit(Θh, self.db['Θh'].values)), noneInit(Θs, self.db['Θs'].values), pssa(noneInit(τ, self.db['τ'].values))
		return pd.Series(self.db['α'] * pssa(self.db['A']) * pssa(self.db['ν']) * np.power(Θhp, 1-self.db['α']) * np.power(Θs/pssa(self.db['ν']), self.power_s) * (self.db['χ2']/(self.db['α']*pssa(self.db['ν'])+τp*((1-self.db['α'])/self.db['α'])*(1-self.db['epsilon'])*(1-self.db['θ'])/self.auxPenDen)),
				index = self.db['t'])

	# Levels in variables - we need to define Θ variables first
	def steadyStateSavings(self, Θs, ν):
		return Θs**((1+self.db['α']*self.db['ξ'])/(1-self.db['α'])) * (ν**(-(self.db['α']*(1+self.db['ξ'])/(1-self.db['α']))))

	def levels_s(self, Θs = None, s_ = None):
		""" Return vector of savings in levels """
		Θs = noneInit(Θs, self.db['Θs'].values)
		s = [None]*(self.T+1)
		s[0] = noneInit(s_, self.steadyStateSavings(Θs[0], self.db['ν'][0])) # initialize from steady state level if nothing else is provided
		for t in self.db['t']:
			s[t+1] = Θs[t]*(s[t]/self.db['ν'][t])**(self.power_s)
		return pd.Series(s, index = self.db['t_'])

	def auxMainState(self, s):
		return pss(s, l=1).loc[0:]/pd.Series(self.db['ν'], index = self.db['t'])

	def auxLevel(self, s, par):
		return self.auxMainState(s).apply(lambda x: np.power(x, par))

	def levels_c1i(self, s, Θc1i = None):
		Θc1i = noneInit(Θc1i, self.db['Θc1i'])
		return Θc1i.mul(self.auxLevel(s, self.power_s),axis=0)

	def levels_̃c1i(self, s, Θ̃c1i = None):
		Θ̃c1i = noneInit(Θ̃c1i, self.db['Θ̃c1i'])
		return Θ̃c1i.mul(self.auxLevel(s, self.power_s), axis=0)

	def levels_hi(self, s, Θhi= None):
		Θhi = noneInit(Θhi, self.db['Θhi'])
		return Θhi.mul(self.auxLevel(s, self.power_h), axis =0)

	def levels_h(self, s, Θh = None):
		Θh = noneInit(Θh, self.db['Θh'])
		return Θh*self.auxLevel(s, self.power_h)

	def levels_c2i(self, s, Θc2i = None):
		Θc2i = noneInit(Θc2i, self.db['Θc2i'])
		return Θc2i.mul(self.auxLevel(s, self.power_s),axis=0)

	def levels_c2pi(self, s, Θc2pi = None):
		Θc2pi = noneInit(Θc2pi, self.db['Θc2pi'])
		return Θc2pi.mul(self.auxLevel(s, self.power_p), axis =0)

	def levels_c1u(self, s, Θc1u =None):
		Θc1u = noneInit(Θc1u, self.db['Θc1u'])
		return Θc1u*self.auxLevel(s, self.power_s)

	def levels_c2u(self, s, Θc2u = None):
		Θc2u = noneInit(Θc2u, self.db['Θc2u'])
		return Θc2u*self.auxLevel(s, self.power_s)

	def levels_c2pu(self, s, Θc2pu=None):
		Θc2pu = noneInit(Θc2pu, self.db['Θc2pu'])
		return Θc2pu*self.auxLevel(s, self.power_p)

	# Political support for various households:
	def aux_util1i(self):
		""" Indirect utility for young consumers """
		return self.db['c1i'].apply(np.log)+self.db['c2pi'].apply(np.log)*self.db['β']

	def aux_util1u(self):
		""" Utility for young hand-to-mouth"""
		return self.db['c1u'].apply(np.log)+self.db['c2pu'].apply(np.log)*self.db['βu']

	def aux_util2i(self):
		""" Utility for retired households """
		return self.db['c2i'].apply(np.log)

	def aux_util2u(self):
		""" Utility for old hand-to-mouth"""
		return self.db['c2u'].apply(np.log)

	def aux_utilPol(self):
		""" Political objective function """
		return (self.db['ω']*((1-self.db['γu'])*(self.db['util2i']*self.db['γ']).sum(axis=1)+self.db['γu']*self.db['util2u'])
			+ self.db['ν']*((1-self.db['γu'])*(self.db['util1i']*self.db['γ']).sum(axis=1)+self.db['γu']*self.db['util1u']))