import numpy as np, pandas as pd, pyDbs
from pyDbs import SymMaps as sm
from scipy import optimize

def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

def addLevelToUtil(x, par, ν, s_):
	return x if s_ is None else x+par*np.log(s_/ν)

def argentinaCalEps(θ, β):
	return 0.7 * (1-θ) * (β**(5/30)*9.45/14.45+β**(10/30)*12.55/22.55)/2

class infHorizon:
	def __init__(self, ni = 11, T = 10, epsilon = 0.1, θ = 0.5, **kwargs):
		""" Fixed namespace """
		self.ni = ni
		self.T = T
		self.db = self.defaultParameters | kwargs
		self.db['t'] = pd.Index(range(self.T), name = 't')
		self.db['t_']= self.db['t'].append(pd.Index([-1], name = 't')).sort_values() # Time index with -1 included
		self.db['i'] = pd.Index(range(self.ni), name = 'i')
		self.ns = {}
		self.addNamespaces()
		self.db.update(self.initSC(epsilon, 'epsilon'))
		self.db.update(self.initSC(θ, 'θ'))

	def addNamespaces(self):
		self.ns['ESC'] = sm(symbols = {k: self.db['t'] for k in ['epsilon','θ','τ']})
		self.ns['PEE'] = sm(symbols = {k: self.db['t'] for k in ['τ']})
		self.ns['EV'] = sm(symbols = {f'transfer_{k}':  pd.MultiIndex.from_product([self.db['t'], self.db['i']]) for k in ('Y','O')}
									|{f'transfer_{k}U': self.db['t'] for k in ('Y','O')}
									|{'transfer_Pol': self.db['t']})
		[ns.compile() for ns in self.ns.values()];
		self.ns['PEE'].addShiftedSym('τ[t+1]','τ', -1, opt = {'useLoc':'nn'})
		[self.ns['ESC'].addShiftedSym(f'{k}[t+1]',f'{k}', -1, opt = {'useLoc':'nn'}) for k in ['epsilon','θ','τ']];

	def __call__(self, x, name, ns = 'ESC', **kwargs):
		return self.ns[ns](x, name, **kwargs)

	def get(self, x, name, ns = 'ESC'):
		return self.ns[ns].get(x, name)

	def leadSym(self, symbol, lead = -1, opt = None):
		return self.ns['ESC'].getShift(symbol, lead, opt = noneInit(opt, {'useLoc':'nn'})) if isinstance(symbol, pd.Series) else pd.Series(symbol).shift(lead, fill_value=symbol[-1]).values

	@property
	def defaultParameters(self):
		return {'α': .5, 
				'A': np.ones(self.T), 
				'ν': np.ones(self.T),
				'η': np.linspace(1,2,self.ni),
				'γ': np.full(self.ni, 1/self.ni),
				'X': np.ones(self.ni),
				'β': np.full(self.ni, .32),
				'βu': .25, 
				'ξ' : .1,
				'γu': .05, 
				'χ1': .1, 
				'χ2': .05,
				'ω': 1}

	def initSC(self, sc, name):
		""" Define relevant epsilon or θ parameters"""
		sc = pd.Series(sc, index = self.db['t'], name = name) if not pyDbs.is_iterable(sc) else sc
		return {name: sc, f'{name}[t+1]': self.leadSym(sc)}

	################ Auxiliary functions:
	@property
	def power_s(self):
		return self.db['α']*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])
	@property
	def power_h(self):
		return self.db['α']*self.db['ξ']/(1+self.db['α']*self.db['ξ'])
	@property
	def power_p(self):
		return self.power_s**2

	@property
	def aux_Prod(self):
		return np.power(self.db['η'], 1+self.db['ξ'])/np.power(self.db['X'], self.db['ξ'])

	@property
	def auxΓβ1(self):
		return sum( (self.db['β']/(1+self.db['β'])) * self.db['γ'] * self.aux_Prod)

	@property
	def auxΓβ2(self):
		return sum( self.db['γ'] / (1+self.db['β']))

	def auxPen(self, τp, epsilonp):
		return τp/(1+self.db['γu']*epsilonp/(1-self.db['γu']))

	def aux_Γs(self, τp, epsilonp, θp):
		return (1/(1+self.db['ξ'])) * self.auxΓβ1 / (1+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilonp)*(θp +(1-θp)*self.auxΓβ2))

	def savingsRate(self, Θs, Θh):
		return Θs/((1-self.db['α'])*(Θh**(1-self.db['α'])))

	################ EE:
	def solve_EE(self, τ, τp, epsilonp, θp):
		Υ = self.EE_Υ(τ, τp, epsilonp, θp)
		return {'Υ':  pd.Series(Υ, index = self.db['t'], name = ' Υ'),
				'Θs': pd.Series(self.EE_Θs(Υ = Υ, τp = τp, epsilonp = epsilonp, θp = θp), index = self.db['t'], name = 'Θs'),
				'Θh': pd.Series(self.EE_Θh(Υ = Υ), index = self.db['t'], name = 'Θh')}

	def EE_Υ(self, τ, τp, epsilonp, θp):
		return np.power((1-self.db['α'])*(1-τ)*self.db['A'] / (1-((1-self.db['α'])/self.db['α'])*θp*self.auxPen(τp, epsilonp)*self.aux_Γs(τp, epsilonp, θp)), 
							(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))

	def EE_Θs(self, Υ = None, τp = None, epsilonp = None, θp = None):
		return Υ * self.aux_Γs(τp, epsilonp, θp)

	def EE_Θh(self, Υ = None):
		return np.power(Υ, self.db['ξ']/(1+self.db['ξ']))

	################ PEE:
	def updateSolve_PEE(self, x0 = None, **kwargs):
		""" Update parameters with dictionary kwargs and resolve """
		self.db.update(kwargs)
		self.db.update(self.solve_PEE(τ0 = self.db['τ'].values if 'τ' in self.db else None))
		return self.db

	def solve_PEE(self, τ0 = None, epsilon = None, θ = None):
		epsilon, θ = noneInit(epsilon, self.db['epsilon'].values), noneInit(θ, self.db['θ'].values)
		τ  = self.PEE_τ(τ0 = τ0, epsilon = epsilon, θ = θ)
		τp = self.leadSym(τ)
		return {'τ': pd.Series(τ, index = self.db['t'], name = 'τ'), 
				'τ[t+1]': pd.Series(τp, index = self.db['t'], name = 'τ[t+1]')} | self.solve_EE(τ, τp, self.leadSym(epsilon), self.leadSym(θ))

	def PEE_τ(self, τ0 = None, epsilon = None, θ = None):
		epsilon, θ = noneInit(epsilon, self.db['epsilon'].values), noneInit(θ, self.db['θ'].values)
		sol, _, ier, msg = optimize.fsolve(lambda x: self.aux_PEE_polObj(x, epsilon, θ),
			noneInit(τ0, [0.5]*self.ns['PEE'].len), full_output=True)
		if ier == 1:
			return sol
		else:
			return print(f"PEE couldn't identify an equilibrium - fsolve returns {msg}")

	def aux_PEE_polObj(self, τ, epsilon, θ):
		return (self.db['ω'] * (self.db['γu']*self.aux_PEE_HtM_old(τ, epsilon)+(1-self.db['γu'])*np.matmul(self.db['γ'], self.aux_PEE_retirees(τ, epsilon, θ)))
				+self.db['ν']* (self.db['γu']*self.aux_PEE_HtM_young(τ)+(1-self.db['γu'])*np.matmul(self.db['γ'], self.aux_PEE_workers(τ))))

	def aux_PEE_retirees(self, τ, epsilon, θ):
		x = (1-self.db['α'])*(1-θ+θ*self.aux_Prod.reshape(self.ni,1))/(1+self.db['γu']*epsilon/(1-self.db['γu']))
		return -(1-self.db['α'])*self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-τ))+x / (self.db['α']*self.savingsSpread(τ, epsilon, θ)+x*τ)

	def aux_PEE_HtM_old(self, τ, epsilon):
		return -(1-self.db['α'])*self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-τ))+epsilon/((1+self.db['γu']*epsilon/(1-self.db['γu']))*self.db['χ2']/self.db['ν']+epsilon*τ)

	def aux_PEE_HtM_young(self, τ):
		return -((1-self.db['α'])*self.db['ξ']/(1+self.db['α']*self.db['ξ'])+self.db['βu']*self.db['α']*((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))**2)/(1-τ)

	def aux_PEE_workers(self, τ):
		return -(((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*(1+self.db['β']*self.db['α']*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))).reshape(self.ni,1) / (1-τ)

	def savingsSpread(self, τp, epsilonp, θp):
		x1 = ((self.db['β'] * np.power(self.db['η'], 1+self.db['ξ'])/(np.power(self.db['X'], self.db['ξ'])*(1+self.db['β']))) / self.auxΓβ1).reshape(self.ni,1)
		return x1+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilonp) * (θp   * (x1-self.aux_Prod.reshape(self.ni,1)) 
																			+(1-θp)* (x1*self.auxΓβ2-1/(1+self.db['β']).reshape(self.ni,1)))

	################ ESC - endogenous system characteristics
	def solve_ESC(self, τ = None, epsilon = None, θ = None):
		""" Given parameters, solve PEE """
		sol, _, ier, msg = optimize.fsolve(lambda x: self.ESC_eqs(self(x, 'τ'),
																  self(x, 'epsilon'),
																  self(x, 'θ')),
							np.hstack([noneInit(τ, np.full(self.T, .5)),
									   noneInit(epsilon, np.full(self.T, .1)),
									   noneInit(θ, np.full(self.T,.5))]), full_output=True)
		if ier == 1:
			solDict = self.ns['ESC'].unloadSol(sol)
			return solDict | self.solve_EE(solDict['τ'].values, solDict['τ[t+1]'].values, solDict['epsilon[t+1]'].values, solDict['θ[t+1]'])
		else:
			print(f"solve_ESC couldn't identify an equilibrium - fsolve returns {msg}")

	def solve_ESC_bounded(self, τ = None, epsilon = None, θ = None, epsilonMax = 1):
		""" Given parameters, solve PEE """
		sol, _, ier, msg = optimize.fsolve(lambda x: self.ESC_eqs_bounded(self(x, 'τ'),
																  self(x, 'epsilon'),
																  self(x, 'θ'), epsilonMax = epsilonMax),
							np.hstack([noneInit(τ, np.full(self.T, .5)),
									   noneInit(epsilon, np.full(self.T, .1)),
									   noneInit(θ, np.full(self.T,.5))]), full_output=True)
		if ier == 1:
			solDict = self.ns['ESC'].unloadSol(sol)
			return solDict | self.solve_EE(solDict['τ'].values, solDict['τ[t+1]'].values, solDict['epsilon[t+1]'].values, solDict['θ[t+1]'])
		else:
			print(f"solve_ESC_bounded couldn't identify an equilibrium - fsolve returns {msg}")


	def ESC_eqs(self, τ, epsilon, θ):
		return np.hstack([self.aux_PEE_polObj(τ, epsilon, θ),
						  self.aux_ESC_epsilon(τ, epsilon, θ),
						  self.aux_ESC_θ(τ, epsilon, θ)])

	def ESC_eqs_bounded(self, τ, epsilon, θ, epsilonMax = 1):
		return np.hstack([self.aux_PEE_polObj(τ, np.minimum(epsilon, epsilonMax), θ),
						  self.aux_ESC_epsilon_bounded(τ, np.minimum(epsilon, epsilonMax), θ, epsilonMax = epsilonMax),
						  self.aux_ESC_θ(τ, np.minimum(epsilon, epsilonMax), θ)])

	def aux_ESC_epsilon_bounded(self, τ, epsilon, θ, epsilonMax = 1):
		""" Condition for optimal ε: If epsilon = epsilonMax then bounded  """
		bounded = np.minimum(self.aux_ESC_epsilon(τ, epsilonMax, θ), 0) # if epsilonMax --> ">0", then use 0 (as if criteria fulfilled)
		unbounded = self.aux_ESC_epsilon(τ, epsilon, θ)
		unbounded[epsilon == epsilonMax] = bounded[epsilon == epsilonMax]
		return unbounded

	def aux_ESC_epsilon(self, τ, epsilon, θ):
		""" Condition for optimal ε """
		return self.aux_ESC_eps_OU(τ, epsilon)+self.aux_ESC_eps_O(τ, epsilon, θ)

	def aux_ESC_θ(self, τ, epsilon, θ):
		""" Condition for optimal θ """
		return np.matmul(self.db['γ'], (self.aux_Prod-1).reshape(self.ni,1) / (self.db['α'] * self.savingsSpread(τ,epsilon,θ)+(1-self.db['α'])*self.auxPen(τ,epsilon)*(θ*self.aux_Prod.reshape(self.ni,1)+1-θ)))

	def aux_ESC_eps_OU(self, τ, epsilon):
		return 1/((1+self.db['γu'] * epsilon/(1-self.db['γu']))*self.db['χ2']/self.db['ν']+epsilon*τ)

	def aux_ESC_eps_O(self, τ, epsilon, θ):
		return -np.matmul(self.db['γ'], (1-self.db['α'])*(θ*self.aux_Prod.reshape(self.ni,1)+1-θ)/(self.db['α'] * self.savingsSpread(τ,epsilon,θ)+(1-self.db['α'])*self.auxPen(τ,epsilon)*(θ*self.aux_Prod.reshape(self.ni,1)+1-θ)))


	################ Reporting functions:

	def reportAll(self, s_= None):
		""" Based on the PEE solution, report host of other relevant variables"""
		self.reportCoefficients()
		self.reportLevels(s_=s_)
		self.reportUtils()

	def reportCoefficients(self):
		""" Assumes that self.solve_PEE has been run and unloaded to the self.db """
		[self.db.__setitem__(k, getattr(self,'aux_'+k)) for k in ('Θhi','Θsi','Θc1i','Θc2i','Θc2pi','Θ̃c1i','Θc1u','Θc2u','Θc2pu')];

	def reportLevels(self, s_ = None):
		""" Assumes self.reportCoefficients has been run"""
		self.db['s'] = self.levels_s(s_=s_)
		[self.db.__setitem__(k, getattr(self,'levels_'+k)(self.db['s'])) for k in ('c1i','c2i','c1u','c2u','̃c1i','c2pi','c2pu','hi','h')];
		self.db['R'] = self.levels_R(self.db['s'])

	def reportUtils(self):
		""" Assumes self.reportLevels has been run"""
		[self.db.__setitem__(k, getattr(self,'aux_'+k)(self.db)) for k in ('util1i','util1u','util2i','util2u', 'utilPol')];


	# Coefficient functions: 
	@property
	def aux_Θhi(self):
		return pd.DataFrame((self.db['Θh'].values * self.aux_Prod.reshape(self.ni,1)).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θsi(self):
		return pd.DataFrame((self.db['Θs'].values*self.savingsSpread(self.db['τ[t+1]'].values, self.db['epsilon[t+1]'].values, self.db['θ[t+1]'].values)).T, 
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc1i(self):
		return pd.DataFrame((self.db['Υ'].values * (self.aux_Prod*(1-self.db['β']/((1+self.db['ξ'])*(1+self.db['β'])))).reshape(self.ni,1)+self.db['Θs'].values*((1-self.db['α'])/self.db['α']) * (1-self.db['θ[t+1]'].values)*self.auxPen(self.db['τ[t+1]'].values, self.db['epsilon[t+1]'].values)/((1+self.db['β']).reshape(self.ni,1))).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc2i(self):
		return pd.DataFrame((self.db['α']*self.db['A']*self.db['ν']*(self.db['Θh'].values**(1-self.db['α']))*(self.savingsSpread(self.db['τ'].values, self.db['epsilon'].values,self.db['θ'].values)+((1-self.db['α'])/self.db['α'])*self.auxPen(self.db['τ'].values, self.db['epsilon'].values)*(self.db['θ'].values*self.aux_Prod.reshape(self.ni,1)+(1-self.db['θ']).values))).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc2pi(self):
		return pd.DataFrame((self.db['α']*self.leadSym(self.db['A']*self.db['ν']*(self.db['Θh'].values)**(1-self.db['α']))*np.power(self.db['Θs'].values/self.leadSym(self.db['ν']), self.power_s)*(self.savingsSpread(self.db['τ[t+1]'].values, self.db['epsilon[t+1]'].values,self.db['θ[t+1]'].values)+((1-self.db['α'])/self.db['α'])*self.auxPen(self.db['τ[t+1]'].values, self.db['epsilon[t+1]'].values)*(self.db['θ[t+1]'].values*self.aux_Prod.reshape(self.ni,1)+(1-self.db['θ[t+1]'].values)))).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θ̃c1i(self):
		return pd.DataFrame((self.db['Υ'].values/((1+self.db['ξ'])*(1+self.db['β'])).reshape(self.ni,1) * (
							self.aux_Prod.reshape(self.ni,1)+
							(1-self.db['θ[t+1]'].values)*((1-self.db['α'])/self.db['α'])*self.auxPen(self.db['τ[t+1]'].values, self.db['epsilon[t+1]'].values) * self.auxΓβ1 / (1+((1-self.db['α'])/self.db['α'])*self.auxPen(self.db['τ[t+1]'].values, self.db['epsilon[t+1]'].values)*(self.db['θ[t+1]'].values+(1-self.db['θ[t+1]'].values)*self.auxΓβ2))
							)).T, index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc1u(self):
		return pd.Series(self.db['χ1']*self.db['A']*(1-self.db['α'])*(self.db['Θh'].values**(1-self.db['α'])), 
				index = self.db['t'])

	@property
	def aux_Θc2u(self):
		return pd.Series((1-self.db['α'])*self.db['A']*self.db['ν']*(self.db['Θh'].values**(1-self.db['α']))*(self.db['χ2']/self.db['ν']+self.db['epsilon'].values*self.auxPen(self.db['τ'].values, self.db['epsilon'].values)),
				index = self.db['t'])

	@property
	def aux_Θc2pu(self):
		return pd.Series((1-self.db['α']) * self.leadSym(self.db['A']) * self.leadSym(self.db['ν']) * (self.leadSym(self.db['Θh']).values**(1-self.db['α'])) * np.power(self.db['Θs'].values/self.leadSym(self.db['ν']), self.power_s) * (self.db['χ2']/self.leadSym(self.db['ν'])+self.db['epsilon[t+1]'].values*self.auxPen(self.db['τ[t+1]'].values, self.db['epsilon[t+1]'].values)),
				index = self.db['t'])

	def levels_s(self, s_ = None):
		""" Return vector of savings in levels """
		s = [None]*(self.T+1)
		s[0] = noneInit(s_, self.steadyStateSavings(self.db['Θs'].values[0], self.db['ν'][0])) # initialize from steady state level if nothing else is provided
		for t in self.db['t']:
			s[t+1] = self.db['Θs'].values[t]*(s[t]/self.db['ν'][t])**(self.power_s)
		return pd.Series(s, index = self.db['t_'], name = 's')

	def steadyStateSavings(self, Θs, ν):
		return Θs**((1+self.db['α']*self.db['ξ'])/(1-self.db['α'])) * (ν**(-(self.db['α']*(1+self.db['ξ'])/(1-self.db['α']))))

	def auxMainState(self, s):
		return self.leadSym(s, lead=1).loc[0:]/pd.Series(self.db['ν'], index = self.db['t'])

	def auxLevel(self, s, par):
		return self.auxMainState(s).apply(lambda x: np.power(x, par))

	def levels_R(self, s):
		return self.db['α'] * pd.Series(self.db['A'], index = self.db['t'])*(self.auxMainState(s)/self.db['h'])**(self.db['α']-1)

	def levels_c1i(self, s):
		return self.db['Θc1i'].mul(self.auxLevel(s, self.power_s),axis=0)

	def levels_̃c1i(self, s):
		return self.db['Θ̃c1i'].mul(self.auxLevel(s, self.power_s), axis=0)

	def levels_hi(self, s):
		return self.db['Θhi'].mul(self.auxLevel(s, self.power_h), axis =0)

	def levels_h(self, s):
		return self.db['Θh']*self.auxLevel(s, self.power_h)

	def levels_c2i(self, s):
		return self.db['Θc2i'].mul(self.auxLevel(s, self.power_s),axis=0)

	def levels_c2pi(self, s):
		return self.db['Θc2pi'].mul(self.auxLevel(s, self.power_p), axis =0)

	def levels_c1u(self, s):
		return self.db['Θc1u']*self.auxLevel(s, self.power_s)

	def levels_c2u(self, s):
		return self.db['Θc2u']*self.auxLevel(s, self.power_s)

	def levels_c2pu(self, s):
		return self.db['Θc2pu']*self.auxLevel(s, self.power_p)

	# Political support for various households:
	def aux_util1i(self, db, Δy = 0, Δo = 0):
		""" Indirect utility for young consumers """
		return (db['̃c1i']+Δy).apply(np.log)+(db['c2pi']+Δo).apply(np.log)*db['β']

	def aux_util1u(self, db, Δy = 0, Δo = 0):
		""" Utility for young hand-to-mouth"""
		return (db['c1u']+Δy).apply(np.log)+db['c2pu'].apply(np.log)*db['βu']

	def aux_util2i(self, db, Δ = 0):
		""" Utility for retired households """
		return (db['c2i']+Δ).apply(np.log)

	def aux_util2u(self, db, Δ = 0):
		""" Utility for old hand-to-mouth"""
		return (db['c2u']+Δ).apply(np.log)

	def aux_utilPol(self, db, Δy = 0, Δy2 = 0, Δyu = 0, Δyu2 = 0, Δo = 0, Δou = 0):
		""" Political objective function """
		return (db['ω']*( ((1-db['γu'])*self.aux_util2i(db, Δo)*db['γ']).sum(axis=1)+db['γu']*self.aux_util2u(db, Δou)) 
			+   db['ν']*( ((1-db['γu'])*self.aux_util1i(db, Δy, Δy2)*db['γ']).sum(axis=1) +db['γu']*self.aux_util1u(db, Δyu, Δyu2))
			)
	################ EV methods:
	def EV_solInPercentages(self, db, sol):
		""" Report transfers relative to current."""
		relativeTransfers = {'transfer_Y': (sol['transfer_Y'].unstack('i')/(db['c1i'])).stack(),
							 'transfer_O': (sol['transfer_O'].unstack('i')/db['c2i']).stack(),
							 'transfer_YU': sol['transfer_YU']/db['c1u'],
							 'transfer_OU': sol['transfer_OU']/db['c2u']}
		averageConsumption = (db['ν'] * ((1-db['γu'])*(db['c1i'] * db['γ']).sum(axis=1)+db['γu'] * db['c1u'])
								+		 (1-db['γu'])*(db['c2i'] * db['γ']).sum(axis=1)+db['γu'] * db['c2u']) / (1+db['ν'])
		relativeTransfers['transfer_Pol'] = sol['transfer_Pol']/averageConsumption
		return relativeTransfers

	def solve_EV_Permanent(self, db0, db1, x0 = None, ftol = 1e-10):
		x0 = noneInit(x0, np.zeros(self.ns['EV'].len))
		f  = lambda x: self.EV_Permanent_Eqs(db0, db1, self.get(x, 'transfer_Y', ns = 'EV'),
													   self.get(x, 'transfer_O', ns = 'EV'),
													   self.get(x, 'transfer_YU', ns = 'EV'),
													   self.get(x, 'transfer_OU', ns = 'EV'),
													   self.get(x, 'transfer_Pol', ns = 'EV'))
		if max(abs(f(x0)))<ftol:
			return x0
		else:
			sol, _, ier, msg = optimize.fsolve(f, x0, full_output=True)
			if (ier == 1) | (max(abs(f(sol)))<ftol):
				return sol
			else:
				print(f"solve_EV_Permanent couldn't identify a vector of transfers that establishes equivalent variation - fsolve returns {msg}")

	######## Permanent, anticipated transfers that consumers may use smooth out over time
	def EV_Permanent_Eqs(self, db0, db1, Δy, Δo, Δyu, Δou, Δpol):
		""" Equations used to solve for the equivalent variation """
		return np.hstack([(self.EV_Permanent_Y(db1, Δy)-db0['util1i'].stack()).values,
						  (self.EV_Permanent_YU(db1, Δyu)-db0['util1u']).values,
						  (self.EV_Permanent_O(db1, Δo)-db0['util2i'].stack()).values,
						  (self.EV_Permanent_OU(db1, Δou)-db0['util2u']).values,
						  (self.EV_Permanent_Pol(db1, Δpol)-db0['utilPol']).values])


	def EV_Permanent_Pol(self, db, transfer):
		return self.aux_utilPol(db, Δy  = transfer.values.reshape(self.T,1) / (1+db['β']), Δy2 = (transfer.values * self.leadSym(db['R']).values).reshape(self.T,1) * (db['β']/(1+db['β'])),
									Δyu = transfer, Δyu2 = transfer * self.leadSym(db['R']),
									Δo  = transfer.values.reshape(self.T,1),
									Δou = transfer
								)

	def EV_Permanent_Y(self, db, transfer):
		return self.aux_util1i(db,  Δy = 2*transfer.unstack(level='i') / (1+db['β']), 
									Δo = (2*transfer.unstack(level='i')*db['β']/(1+db['β'])).mul(self.leadSym(db['R']), axis = 0)
								).stack()

	def EV_Permanent_YU(self, db, transfer):
		return self.aux_util1u(db,  Δy = transfer, Δo = transfer * self.leadSym(db['R']))

	def EV_Permanent_O(self, db, transfer):
		return self.aux_util2i(db, Δ = transfer.unstack(level='i')).stack()

	def EV_Permanent_OU(self, db, transfer):
		return self.aux_util2u(db, Δ = transfer)

	################ Calibration, Argentina:
	def argentinaCalibrateEqs(self, x, τ0, s0, t0):
		self.db['ω'] = x[0]
		self.db['β'], self.db['βu'] = np.full(self.ni, x[1]), x[1]
		self.db.update(self.initSC(argentinaCalEps(self.db['θ'].values[0], x[1]), 'epsilon'))
		sol = self.solve_PEE()
		return np.hstack([sol['τ'].xs(t0)-τ0,
						  self.savingsRate(sol['Θs'].xs(t0), sol['Θh'].xs(t0))-s0])

	def argentinaCalibrate(self, τ0, s0, t0, x0 = None):
		sol, _, ier, msg = optimize.fsolve(lambda x: self.argentinaCalibrateEqs(x, τ0, s0, t0), noneInit(x0, [self.db['ω'], self.db['β'][0]]), full_output=True)
		if ier == 1:
			self.db['ω'], self.db['β'] = sol[0], np.full(self.ni, sol[1])
			self.db.update(self.initSC(argentinaCalEps(self.db['θ'].values[0], sol[1]), 'epsilon'))
			return sol
		else:
			print(f"Error in argentinaCalibrate: {msg}")



