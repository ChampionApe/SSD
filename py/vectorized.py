import numpy as np, pandas as pd
from scipy import optimize

def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

##########################################################################
##################	 			1. Base 				##################
##########################################################################

class infHorizon:
	def __init__(self, ni = 11, T = 10, **kwargs):
		""" Fixed namespace """
		self.ni = ni
		self.T = T
		self.db = self.defaultParameters | kwargs
		self.mainIndex = pd.MultiIndex.from_product([pd.Index(range(self.T), name = 't'), 
													 pd.Index(['τ', 'Υ', 'Θh', 'Θs', 'dlnΥ', 'dlnΘh', 'dlnΘs'], name = 'variable')]).sort_values()
		self.eeIndex = self.mainIndex[self.mainIndex.get_level_values('variable').isin(['Θh','Θs','Υ'])]
		self.lnDevIndex = self.mainIndex[self.mainIndex.get_level_values('variable').isin(['dlnΘh','dlnΘs','dlnΥ'])]
		self.db['Γ'] = self.auxΓ
		self.mainLinIndex = self.linIndex(self.mainIndex)

	def linIndex(self, index):
		return pd.Series(range(len(index)), index = index)

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
				'ε' : .5, 
				'θ' : .5, 
				'γu': .05, 
				'χ1': .1, 
				'χ2': .05,
				'ω': 1.5}

	def pss(self, x, l = -1):
		return x.shift(l, fill_value = x.iloc[-1])

	def getIndex(self, name, index, l = None):
		""" Return linear index needed to subset symbol 'name'; use l = -1 to get the x_{t+1} vector. """
		return index.xs(name, level = 'variable') if l is None else self.pss(index.xs(name, level = 'variable'), l = l)		

	def get(self, x, name, l = None):
		return x[self.getIndex(name, self.mainLinIndex, l = l)]

	def get_(self, x, name, index, l = None):
		return x[self.getIndex(name, self.linIndex(index), l= l)]

	def getEE(self, x, name, l = None):
		return self.get_(x, name, self.eeIndex, l = l)

	def getLnDev(self, x, name, l = None):
		return self.get_(x, name, self.lnDevIndex, l = l)

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
		return 1-self.db['γu']+self.db['γu']*(1-self.db['ε'])*(1-self.db['θ'])

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

	# Political support functions:
	@property
	def auxPensionRate(self):
		return (self.db['Γ']*(1-self.db['θ'])+self.db['θ']*np.power(self.db['η'], 1+self.db['ξ'])/np.power(self.db['X'], self.db['ξ']))/(self.db['Γ']*self.auxPenDen)

	def polSupportRetirees(self, dlnΘh, τ):
		return self.db['ω']*(self.db['γu']*self.polSupportUnemployedRetiree(dlnΘh, τ)+(1-self.db['γu'])*np.matmul(self.db['γ'],self.polSupportRetireeVector(dlnΘh, τ)))

	def polSupportUnemployedRetiree(self, dlnΘh, τ):
		return (1-self.db['α'])*(dlnΘh+(1-self.db['ε'])*(1-self.db['θ'])/(self.db['χ2'] * self.auxPenDen/self.db['ν']+(1-self.db['α'])*(1-self.db['ε'])*(1-self.db['θ'])*τ))

	def polSupportRetireeVector(self, dlnΘh, τ):
		""" Return matrix with rows = types, columns = years """
		x = self.auxPensionRate.reshape(self.ni, 1)
		return (1-self.db['α'])*(dlnΘh+x/(self.db['α']*self.savingsSpread(τ)+(1-self.db['α'])*x*τ))

	def polSupportUnemployedYoung(self, dlnΘh, dlnΘs):
		return (1-self.db['α'])*dlnΘh+self.db['βu']*self.db['α']*dlnΘs*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])

	def polSupportWorkerVector(self, dlnΘs):
		return dlnΘs * (1+self.db['β'].reshape(self.ni,1)*self.db['α']*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))

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
