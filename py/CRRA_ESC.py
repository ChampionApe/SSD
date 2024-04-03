import numpy as np, pandas as pd, pyDbs
from pyDbs import SymMaps as sm
from scipy import optimize
_numtypes = (int,float,np.generic)

def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

def addLevelToUtil(x, par, ν, s_):
	return x if s_ is None else x+par*np.log(s_/ν)

def polGrid(v0, vT, n, exp = 1):
	""" Create polynomial grid with exponent 'exp'. 
		If exp>1 there are more gridpoint in the lower end of the grid."""
	return v0+(vT-v0)*((np.arange(1,n+1)-1)/(n-1))**exp

def cleanSol(x, keep):
	""" Keep is a boolean array, x is a vector/matrix to be subsetted"""
	return x[keep] if x.ndim == 1 else x[:,keep]

def interpSol(x, xp, fp):
	""" linear interpolation where x, xp are 1d vectors and fp may be 1d or 2d (simply repeats interpolation over the 2d)"""
	if isinstance(fp, _numtypes):
		return fp
	elif fp.ndim == 1:
		return np.interp(x,xp,fp)
	else:
		return np.vstack([np.interp(x,xp,fp[i,:]) for i in range(fp.shape[0])])

def aux_τMeshGrid(sGrid, τGrid_1d):
	return np.meshgrid(sGrid, τGrid_1d)[1]

def argentinaCalEps(θ, β):
	return 0.7 * (1-θ) * (β**(5/30)*9.45/14.45+β**(10/30)*12.55/22.55)/2

class finiteHorizon:
	def __init__(self, ni = 11, T = 10, ngrid = 50, epsilon = 0.1, θ = 0.5, **kwargs):
		""" Fixed namespace """
		self.ni = ni
		self.T = T
		self.db = self.defaultParameters | kwargs
		self.db['t'] = pd.Index(range(self.T), name = 't')
		self.db['t_']= self.db['t'].append(pd.Index([-1], name = 't')).sort_values() # Time index with -1 included
		self.db['txE'] = self.db['t'][:-1] # all years except terminal year
		self.db['txE_'] = self.db['t_'][:-1] # all years including -1 except terminal year
		self.db['i'] = pd.Index(range(self.ni), name = 'i')
		self.db['sgrid'] = pd.Index(range(ngrid), name = 'sgrid') # grid of s[t-1] to identify policy function on
		self.ns = {}
		self.addNamespaces()
		self.db.update(self.initSC(epsilon, 'epsilon'))
		self.db.update(self.initSC(θ, 'θ'))


	def addNamespaces(self):
		self.ns['ESC[t]'] = sm(symbols = {x: self.db['sgrid'] for x in ('τ', 'θ', 'epsilon')}) # namespace used in policy function identification
		self.ns['EE'] = sm(symbols = {'s': self.db['txE'], 'Γs': self.db['txE_'], 'h': self.db['t']})
		[ns.compile() for ns in self.ns.values()];
		self.ns['EE'].addLaggedSym('s[t-1]', 's', 1, c = ('not', self.db['t'][0:1])) # this will not include the s0 value as it is not endogenous in the optimization

	def initSC(self, sc, name):
		""" Define relevant epsilon or θ parameters"""
		sc = pd.Series(sc, index = self.db['t'], name = name) if not pyDbs.is_iterable(sc) else sc
		return {name: sc, f'{name}[t+1]': sc.iloc[1:].set_axis(sc.index[0:-1])}

	def __call__(self, x, name, ns = 'ESC[t]', **kwargs):
		return self.ns[ns](x, name, **kwargs)

	def get(self, x, name, ns = 'ESC[t]'):
		return self.ns[ns].get(x, name)

	def leadSym(self, symbol, lead = -1, opt = None):
		return self.ns['EE'].getShift(symbol, lead, opt = noneInit(opt, {'useLoc':'nn'})) if isinstance(symbol, pd.Series) else pd.Series(symbol).shift(lead, fill_value=symbol[-1]).values

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
				'ω': 1,
				'ωu': .4,
				'ωη': .9,
				'ρ': .5} # 1/CRRA parameter

	@property
	def ω2u(self):
		return self.db['ω']*self.db['ωu']
	@property
	def ω2i(self):
		return self.db['ω']*(1+self.db['ωη']*(self.aux_Prod-1))
	@property
	def ω1u(self):
		return self.db['ωu']
	@property
	def ω1i(self):
		return 1+self.db['ωη']*(self.aux_Prod-1)

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

	def auxΓB1(self, B):
		return np.matmul(self.aux_Prod * self.db['γ'], B/(1+B))

	def auxΓB2(self,B):
		return np.matmul(self.db['γ'], 1/(1+B))

	def auxΓB3(self,B):
		return np.matmul(self.aux_Prod * self.db['γ'], B/((1+B)**2))

	def auxΓB4(self,B):
		return np.matmul(self.db['γ'], B/((1+B)**2))

	def auxPen(self, τp, epsilonp):
		return τp/(1+self.db['γu']*epsilonp/(1-self.db['γu']))

	def aux_R(self, s, h, ν, A = 1):
		return self.db['α'] * A * (ν*h/s)**(1-self.db['α'])

	def aux_B(self, s, h, ν, A = 1):
		return self.db['β'].reshape(self.ni,1)**self.db['ρ'] * (self.aux_R(s,h, ν, A = A))**(self.db['ρ']-1)

	def aux_B_scalar(self, s, h, ν, A =1):
		return self.db['β']**self.db['ρ'] * (self.aux_R(s,h,ν, A = A)**(self.db['ρ']-1))

	def aux_Γs(self, Bp, τp, θp, epsilonp):
		""" τp, Bp, θp, epsilonp are vectors of the same length"""
		return (1/(1+self.db['ξ']))*np.matmul(self.db['γ'] * self.aux_Prod, Bp/(1+Bp)) /(1+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilonp)*(θp+(1-θp)*np.matmul(self.db['γ'], 1/(1+Bp))))

	def aux_Γs_scalar(self, Bp, τp, θp, epsilonp):
		""" τ is a scalar (no grid or time dimensions)"""
		return (1/(1+self.db['ξ']))*sum(self.db['γ'] * self.aux_Prod * Bp/(1+Bp)) /(1+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilonp)*(θp+(1-θp)*sum(self.db['γ']/(1+Bp))))

	def auxPen2(self, θ, epsilon):
		return (1-θ+θ*self.aux_Prod)/(1+self.db['γu']*epsilon/(1-self.db['γu']))

	def savingsRate(self, Θs, Θh):
		return Θs/((1-self.db['α'])*(Θh**(1-self.db['α'])))

	################ Economic Equilibrium given s0 and policy
	def solve_EE(self, τ, τp, s0, x0 = None):
		sol, _, ier, msg = optimize.fsolve(lambda x: self.aux_solve_EE(x,τ, τp, s0)-x,
			noneInit(x0, np.full(self.ns['EE'].len, 0.5)), full_output=True)
		assert ier==1, f"""self.solve_EE couldn't identify en equilibrium. fsolve returns: 
		"{msg}" """
		d = self.ns['EE'].unloadSol(sol) | {'s[t-1]': pd.Series(self.get_s(sol, s0), index = self.db['t'])}
		return d | {'Θs': d['s']/(((d['s[t-1]']/self.db['ν'])[:-1])**self.power_s),
					'Θh': d['h']/((d['s[t-1]']/self.db['ν'])**self.power_h)}

	def EE_x0_fromPEE(self, sol):
		return np.hstack([sol['s'].values, sol['Γs'].values, sol['h'].values])

	def aux_solve_EE(self, x, τ, τp, s0):
		return np.hstack([self.EE_s(self(x, 's', ns = 'EE'), self.get_sLag(x, s0), self(x,'Γs', ns = 'EE'), τ, τp).values,
						  self.EE_Γs(self.get_s(x, s0), self(x, 'h', ns = 'EE'), τ.values),
						  self.EE_h(self(x, 's', ns = 'EE'), self(x, 'Γs', ns = 'EE'), τ)])
		
	def get_s(self, x, s0):
		return np.insert(self(x, 's', ns = 'EE'), 0, s0) # insert s0 to the numpy array s

	def get_sLag(self, x, s0):
		return np.insert(self(x,'s[t-1]', ns = 'EE'), 0, s0) # insert s0 to the numpy array s[t-1]

	def EE_s(self, s, sLag, Γs, τ, τp):
		""" Condition holds for all t except terminal T. sLag needs to include initial value. """
		return ((1-self.db['α'])*(1-τ.iloc[:-1])*self.db['A'][:-1]/(1-((1-self.db['α'])/self.db['α'])*self.db['θ'].values[1:]*self.auxPen(τp, self.db['epsilon'].values[1:])*Γs[1:]))**((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])) * Γs[1:] *(sLag/self.db['ν'][:-1])**self.power_s

	def EE_h_nonT(self, s, Γs):
		""" Condition holds for all t except T. """
		return (s/Γs[1:])**(self.db['ξ']/(1+self.db['ξ']))

	def EE_h(self, s, Γs, τ):
		""" Condition for all t. s needs to include intial constant s0"""
		return np.hstack([self.EE_h_nonT(s,Γs), self.aux_h_T(τ.iloc[-1], s[-1])])

	def EE_Γs(self, sLag, h, τ):
		""" Condition holds for all txE_ """
		return self.aux_Γs(self.aux_B(sLag,h,self.db['ν']), τ, self.db['θ'], self.db['epsilon'])

	################ Steady state functions:
	def solve_steadyState(self, τ, ν, θ, epsilon, x0 = None, A = 1):
		sol, _, ier, msg = optimize.fsolve(lambda x: np.hstack([self.steadyState_B(x[-1], τ, ν, θ, epsilon)-x[0:-1],
		 														self.aux_Γs_scalar(x[0:-1], τ, θ, epsilon)-x[-1]]),
			noneInit(x0, np.full(self.ni+1, 0.5)), full_output=True)
		if ier == 1:
			return {'B': sol[0:-1], 'Γs': sol[-1], 's': self.aux_steadyState_s(sol[-1], τ, ν,θ,epsilon, A = A)}
		else:
			return print(f"solve_steadyState couldn't identify an equilibrium - fsolve returns {msg}")

	def aux_steadyState_s(self, Γs, τ, ν, θ, epsilon, A = 1):
		return ( (((1-self.db['α'])*(1-τ)*A)/(1-((1-self.db['α'])/self.db['α'])*(θ*self.auxPen(τ, epsilon))*Γs))**(1+self.db['ξ'])*Γs**(1+self.db['α']*self.db['ξ'])/(ν**(self.db['α']*(1+self.db['ξ']))) )**(1/(1-self.db['α']))

	def steadyState_B(self, Γs, τ, ν, θ, epsilon):
		return self.db['β']**self.db['ρ']*( (self.db['α']-(1-self.db['α'])*(θ*τ/(1+self.db['γu']*epsilon/(1-self.db['γu'])))*Γs*ν**(self.db['α']))/((1-self.db['α'])*(1-τ)*Γs))**(self.db['ρ']-1)

	def approximateSteadyStateFromGrid(self, τ, Γs, ν, s, θ, epsilon):
		""" Given grids of τ, Γs, s - and ν scalar - this interpolates steady state savings """
		ŝ  = self.aux_steadyState_s(Γs, τ, ν, θ, epsilon) # Steady state values based on grids
		Δs = ŝ-s # distance from steady state
		id1, id2 = Δs[Δs>0].argmin(), Δs[Δs<0].argmax() # identify grid points closest to steady state
		s1, ŝ1 = s[Δs>0][id1], ŝ[Δs>0][id1]
		s2, ŝ2 = s[Δs<0][id2], ŝ[Δs<0][id2]
		return ŝ1+((ŝ2-ŝ1)*(ŝ1-s1))/(s2-s1-(ŝ2-ŝ1)) # lin. approximation

	################ Simulate PEE path given policies
	def updateAndSolve_PEE(self, sGrid, gridOption = 'resample', s0 = None, **kwargs):
		""" Update parameters with dictionary kwargs and resolve """
		self.db.update(kwargs)
		self.db.update(self.solve_PEE(sGrid, gridOption=gridOption, s0 = s0))
		return self.db

	def updateAndSolve_ESC(self, sGrid, gridOption = 'resample', s0 = None, **kwargs):
		""" Update parameters with dictionary kwargs and resolve """
		self.db.update(kwargs)
		self.db.update(self.solve_ESC(sGrid, gridOption=gridOption, s0 = s0))
		return self.db

	def solve_ESC(self, sGrid, gridOption = 'resample', s0 = None):
		policy = self.solve_ESC_policy(sGrid, gridOption=gridOption)
		sols   = self.solve_ESC_givenPolicy(policy, s0 = s0)
		return self.reportESC_main(sols)

	def solve_PEE(self, sGrid, gridOption = 'resample', s0 = None):
		policy = self.solve_PEE_policy(sGrid, gridOption=gridOption)
		sols   = self.solve_PEE_givenPolicy(policy, s0 = s0)
		return self.reportPEE_main(sols)

	def reportESC_main(self, sol):
		""" Given solution from self.solve_ESC, report main symbols required to characterize full economy. """
		pee = self.reportPEE_main(sol)
		pee.update({k: pd.Series([sol_i[k] for sol_i in sol.values()], index = self.db['t']) for k in ('θ','epsilon','∂θ/∂s','∂epsilon/∂s')})
		return pee

	def reportPEE_main(self, sol):
		""" Given solution from self.solve_PEE, report main symbols required to characterize full economy. 
		This includes taxes, savings, labor supply, wages, interest rates. """
		d = {k: pd.Series([sol_i[k] for sol_i in sol.values()], index = self.db['t']) for k in ('τ','h','s[t-1]','∂τ/∂s')}
		d['B'] = pd.DataFrame(np.hstack([sol_i['B'] for sol_i in sol.values()]).T, index = self.db['t'], columns = self.db['i'])
		d['Γs'] = pd.Series([sol_i['Γs'] for sol_i in sol.values()], index = self.db['txE_'])
		d['s'] = d['s[t-1]'][1:].set_axis(self.db['txE'])
		d['τ[t+1]'] = self.leadSym(d['τ']).iloc[0:-1]
		d['Θs'] = d['s']/(((d['s[t-1]']/self.db['ν'])[:-1])**self.power_s)
		d['Θh'] = d['h']/((d['s[t-1]']/self.db['ν'])**self.power_h)
		return d

	def solve_ESC_givenPolicy(self, policy, s0 = None):
		""" Simulate solution given dict of policies for each t"""
		if s0 is None:
			s0 = self.approximateSteadyStateFromGrid(policy[0]['τ'], policy[0]['Γs'], self.db['ν'][0], policy[0]['s[t-1]'], policy[0]['θ'], policy[0]['epsilon'])
		sol = dict.fromkeys(self.db['t'])
		sol[0] = {k: interpSol(s0, policy[0]['s[t-1]'], policy[0][k]) for k in policy[0]}
		for t in self.db['t'][1:]:
			sol[t] = {k: interpSol(sol[t-1]['s'], policy[t]['s[t-1]'], policy[t][k]) for k in policy[t]}
		return sol

	def solve_PEE_givenPolicy(self, policy, s0 = None):
		""" Simulate solution given dict of policies for each t"""
		if s0 is None:
			s0 = self.approximateSteadyStateFromGrid(policy[0]['τ'], policy[0]['Γs'], self.db['ν'][0], policy[0]['s[t-1]'], self.db['θ'].iloc[0], self.db['epsilon'].iloc[0])
		sol = dict.fromkeys(self.db['t'])
		sol[0] = {k: interpSol(s0, policy[0]['s[t-1]'], policy[0][k]) for k in policy[0]}
		for t in self.db['t'][1:]:
			sol[t] = {k: interpSol(sol[t-1]['s'], policy[t]['s[t-1]'], policy[t][k]) for k in policy[t]}
		return sol

	################  Solution on grid of savings
	def solve_ESC_policy(self, s, gridOption = 'resample'):
		sols = {self.T-1: self.solve_ESC_T(s)}
		if gridOption == 'resample':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.resampleSolution(self.solve_ESC_t(sols[t+1], t), s)
		elif gridOption == 'clean':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.cleanSolution(self.solve_ESC_t(sols[t+1], t), s)
			else:
				sols[t] = self.solve_ESC_t(sols[t+1], t)
		return sols

	def solve_PEE_policy(self, s, gridOption = 'resample'):
		sols = {self.T-1: self.solve_PEE_T(s)}
		if gridOption == 'resample':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.resampleSolution(self.solve_PEE_t(sols[t+1], t, x0 = sols[t+1]['τ']), s)
		elif gridOption == 'clean':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.cleanSolution(self.solve_PEE_t(sols[t+1], t, x0 = sols[t+1]['τ']), s)
			else:
				sols[t] = self.solve_PEE_t(sols[t+1], t, x0 = sols[t+1]['τ'])
		return sols

	def resampleSolution(self, sol, s):
		""" Redraw the solution on the grid 's' using linear interpolation """
		return {k: interpSol(s, sol['s[t-1]'], v) if  k != 's[t-1]' else s for k,v in sol.items()}

	def cleanSolution(self, sol, s):
		keep = sol['s[t-1]']<max(s)
		return {k: cleanSol(sol[k],keep) for k in sol}

	################ NEW t functions
	def solve_ESC_t(self, sol_p, t, x0 = None):
		sol = self.aux_ESC_precomputations(sol_p,t)
		x, _, ier, msg = optimize.fsolve(lambda x: self.aux_ESC_polObj_t(self(x,'τ'), self(x, 'θ'), self(x, 'epsilon'), sol, sol_p, t), 
			noneInit(x0, np.hstack([sol_p['τ'], sol_p['θ'], sol_p['epsilon']])), full_output=True)
		if ier == 1:
			return self.aux_ESC_t_solve(x, sol, sol_p, t)
		else:
			return print(f"""solve_ESC_t couldn't identify an equilibrium - fsolve returns: '{msg}' """)

	def aux_ESC_t_solve(self, x, sol, sol_p, t):
		""" Return solution dictionary given vector of taxes"""
		sol['τ'], sol['θ'], sol['epsilon'] = self(x, 'τ'), self(x, 'θ'), self(x,'epsilon')
		sol['s[t-1]'] = sol['s_τ0']*(1-sol['τ'])**(-1/self.db['α'])
		sol['B']  = self.aux_B(sol['s[t-1]'], sol['h'], self.db['ν'][t])
		sol['Γs'] = self.aux_Γs(sol['B'], sol['τ'], sol['θ'], sol['epsilon'])
		sol.update({f'∂{k}/∂s': np.gradient(sol[f'{k}'], sol['s[t-1]']) for k in ('τ','θ','epsilon')})
		sol['dln(h)/dln(s)'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s)'] = self.db['α']*self.db['ξ']*(1+sol['Ψ']*(1-sol_p['∂ln(h)/∂ln(s)']))/((1+self.db['α']*self.db['ξ'])*sol['σ'])
		sol.update(self.aux_derivatives_τ(sol['τ'], sol, sol_p))
		return sol

	def solve_PEE_t(self, sol_p, t, x0 = None):
		sol = self.aux_PEE_precomputations(sol_p,t)
		τ, _, ier, msg = optimize.fsolve(lambda τ: self.aux_PEE_polObj_t(τ, sol, sol_p, t), 
			noneInit(x0, sol_p['τ']), full_output=True)
		if ier == 1:
			return self.aux_PEE_t_solve(τ, sol, sol_p, t)
		else:
			return print(f"""solve_PEE_t couldn't identify an equilibrium - fsolve returns: '{msg}' """)

	def aux_PEE_t_solve(self, τ, sol, sol_p, t):
		""" Return solution dictionary given vector of taxes"""
		sol['τ'] = τ
		sol['s[t-1]'] = sol['s_τ0']*(1-τ)**(-1/self.db['α'])
		sol['B']  = self.aux_B(sol['s[t-1]'], sol['h'], self.db['ν'][t])
		sol['Γs'] = self.aux_Γs(sol['B'], sol['τ'], self.db['θ'][t], self.db['epsilon'][t])
		sol['∂τ/∂s'] = np.gradient(τ, sol['s[t-1]'])
		sol['dln(h)/dln(s)'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s)'] = self.db['α']*self.db['ξ']*(1+sol['Ψ']*(1-sol_p['∂ln(h)/∂ln(s)']))/((1+self.db['α']*self.db['ξ'])*sol['σ'])
		sol.update(self.aux_derivatives_τ(τ, sol, sol_p))
		return sol

	### ESC functions:
	def aux_ESC_polObj_t(self, τ, θ, epsilon, sol, sol_p, t):
		""" Returns stacked FOCs for τ, θ, epsilon. """
		funcOfτ = self.aux_ESC_funcOfτ(τ, θ, epsilon, sol, sol_p, t)
		young = self.db['ν'][t]*(self.db['γu']*self.ω1u*self.aux_ESC_HtM_young_t(sol, sol_p, funcOfτ, t)+(1-self.db['γu'])*np.matmul(self.ω1i*self.db['γ'], self.aux_PEE_workers_t(sol, sol_p, funcOfτ, t)))
		euler = self.aux_ESC_retirees_t(τ, θ, epsilon, sol, sol_p, funcOfτ, t)
		htm   = self.aux_ESC_HtM_old_t(τ, epsilon, sol, sol_p, funcOfτ, t)
		return np.hstack([young+(1-self.db['γu'])*np.matmul(self.ω2i*self.db['γ'], euler['τ'])+self.db['γu']*self.ω2u*htm['τ'],
						  np.matmul(self.ω2i*self.db['γ'], euler['θ']),
						  (1-self.db['γu'])*np.matmul(self.ω2i*self.db['γ'], euler['epsilon'])+self.db['γu']*self.ω2u*htm['epsilon']])

	##### Tax effect on indirect utility
	def aux_PEE_polObj_t(self, τ, sol, sol_p, t):
		funcOfτ = self.aux_PEE_funcOfτ(τ, sol, sol_p, t)
		return (self.db['γu']*self.ω2u*self.aux_PEE_HtM_old_t(τ, sol, sol_p, funcOfτ, t)+(1-self.db['γu'])*np.matmul(self.ω2i * self.db['γ'], self.aux_PEE_retirees_t(τ, sol, sol_p, funcOfτ, t))
				+self.db['ν'][t]*(self.db['γu']*self.ω1u*self.aux_PEE_HtM_young_t(sol, sol_p, funcOfτ, t)+(1-self.db['γu'])*np.matmul(self.ω1i*self.db['γ'], self.aux_PEE_workers_t(sol, sol_p, funcOfτ, t))))

	###### Contribution to political objectives: 
	### Euler retirees
	def aux_PEE_retirees_t(self, τ, sol, sol_p, funcOfτ, t):
		c2i_coeff = self.aux_c2i_coeff(funcOfτ['si/s'], τ, self.db['θ'][t], self.db['epsilon'][t])
		c2i = self.aux_c2i_t(τ, funcOfτ['s[t-1]'], sol['h'], self.db['ν'][t], c2i_coeff)
		return self.aux_PEE_retirees(self.db['θ'][t], self.db['epsilon'][t], funcOfτ['dln(h)/dτ'], c2i, c2i_coeff)

	def aux_ESC_retirees_t(self, τ, θ, epsilon, sol, sol_p, funcOfτ, t):
		return self.aux_ESC_retirees(τ, θ, epsilon, funcOfτ['s[t-1]'], sol['h'], self.db['ν'][t], funcOfτ['s[t-1]'], funcOfτ['dln(h)/dτ'])

	### HtM old
	def aux_PEE_HtM_old_t(self, τ, sol, sol_p, funcOfτ, t):
		""" Contribution to FOC for PEE only (not ESC)"""
		c2u_coeff = self.aux_c2u_coeff(τ, self.db['epsilon'][t], self.db['ν'][t])
		c2u = self.aux_c2u_t(funcOfτ['s[t-1]'], sol['h'], self.db['ν'][t], c2u_coeff)
		return self.aux_PEE_HtM_old(self.db['epsilon'][t], funcOfτ['dln(h)/dτ'], c2u, c2u_coeff)

	def aux_ESC_HtM_old_t(self, τ, epsilon, sol, sol_p, funcOfτ, t):
		return self.aux_ESC_HtM_old(τ, epsilon, funcOfτ['s[t-1]'], sol['h'], self.db['ν'][t], funcOfτ['dln(h)/dτ'])

	### Euler young
	def aux_PEE_workers_t(self, sol, sol_p, funcOfτ, t):
		k = ((1-self.db['α'])/self.db['α'])*(1/(1+self.db['γu']*self.db['epsilon'][t+1]/(1-self.db['γu'])))*(1-self.db['θ'][t+1])*(1+self.db['ξ'])*sol_p['Γs']
		return self.aux_ĉ1i_t(sol_p['τ'], self.db['θ'][t+1], self.db['epsilon'][t+1], sol['h'], sol_p['B'], sol_p['Γs'])**(1-1/self.db['ρ']) * (
					((1+self.db['ξ'])/self.db['ξ'])*funcOfτ['dln(h)/dτ']
				+	(sol_p['B']/(1+sol_p['B']))*(1-self.db['α'])*(sol_p['dln(h)/dln(s)']-1)*funcOfτ['dln(s)/dτ']
				+	k * (funcOfτ['∂τp/∂τ']+sol_p['τ']*funcOfτ['dln(Γs)/dτ']) / (self.aux_Prod.reshape(self.ni,1)+sol_p['τ']*k)
				)

	def aux_ESC_workers_t(self, sol_p, sol, funcOfτ, t):
		k = ((1-self.db['α'])/self.db['α'])*(1/(1+self.db['γu']*sol_p['epsilon']/(1-self.db['γu'])))*(1+self.db['ξ'])*sol_p['Γs']
		return self.aux_ĉ1i_t(sol_p['τ'], sol_p['θ'], sol_p['epsilon'], sol['h'], sol_p['B'], sol_p['Γs'])**(1-1/self.db['ρ']) * (
					((1+self.db['ξ'])/self.db['ξ'])*funcOfτ['dln(h)/dτ']
				+	(sol_p['B']/(1+sol_p['B']))*(1-self.db['α'])*(sol_p['dln(h)/dln(s)']-1)*funcOfτ['dln(s)/dτ']
				+	k * ((1-sol_p['θ'])*funcOfτ['∂τp/∂τ']-sol_p['τ']*funcOfτ['∂θ/∂τ']+(1-sol_p['θ'])*sol_p['τ']*(funcOfτ['dln(Γs)/dτ']-(self.db['γu']/(1-self.db['γu']))*(1/(1+self.db['γu']*sol_p['epsilon']/(1-self.db['γu'])))*funcOfτ['∂epsilon/∂τ']))/(self.aux_Prod.reshape(self.ni,1)+sol_p['τ']*(1-sol_p['θ'])*k)
				)

	def aux_ĉ1i_t(self, τp, θp, epsilonp, h, B, Γs):
		return (h**((1+self.db['ξ'])/self.db['ξ'])/(1+self.db['ξ'])) * (1+B)**(1/(self.db['ρ']-1)) * (self.aux_Prod.reshape(self.ni,1)+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilonp)*(1-θp)*(1+self.db['ξ'])*Γs)

	### HtM young
	def aux_PEE_HtM_young_t(self, sol, sol_p, funcOfτ, t):
		k = self.db['epsilon'][t+1]/(1+self.db['γu']*self.db['epsilon'][t+1]/(1-self.db['γu']))
		return (self.aux_c1u_t(funcOfτ['s[t-1]'], sol['h'],self.db['ν'][t])**(1-1/self.db['ρ'])*(1-self.db['α'])*funcOfτ['dln(h)/dτ']
				+self.db['βu']*self.aux_c2pu_t(sol_p['τ'], self.db['epsilon'][t+1], sol_p['s[t-1]'], sol_p['h'], self.db['ν'][t+1])**(1-1/self.db['ρ'])*(
					(self.db['α']+(1-self.db['α'])*sol_p['dln(h)/dln(s)'])*funcOfτ['dln(s)/dτ']+funcOfτ['∂τp/∂τ']*k/(self.db['χ2']/self.db['ν'][t+1]+k*sol_p['τ'])
				))

	def aux_ESC_HtM_young_t(self, sol, sol_p, funcOfτ, t):
		k = 1/(1+self.db['γu']*sol_p['epsilon']/(1-self.db['γu']))
		return (self.aux_c1u_t(funcOfτ['s[t-1]'], sol['h'],self.db['ν'][t])**(1-1/self.db['ρ'])*(1-self.db['α'])*funcOfτ['dln(h)/dτ']
				+self.db['βu']*self.aux_c2pu_t(sol_p['τ'], sol_p['epsilon'], sol_p['s[t-1]'], sol_p['h'], self.db['ν'][t+1])**(1-1/self.db['ρ'])*(
					(self.db['α']+(1-self.db['α'])*sol_p['dln(h)/dln(s)'])*funcOfτ['dln(s)/dτ']
					+funcOfτ['∂τp/∂τ']*sol_p['epsilon']*k/(self.db['χ2']/self.db['ν'][t+1]+k*sol_p['epsilon']*sol_p['τ'])
					+funcOfτ['∂epsilon/∂τ']*sol_p['τ']*k**2/(self.db['χ2']/self.db['ν'][t+1]+k*sol_p['epsilon']*sol_p['τ'])
				))

	def aux_c2pu_t(self, τp, epsilonp, sp, hp, νp, A=1):
		return (1-self.db['α'])*A*(sp/νp)**(self.db['α'])*hp**(1-self.db['α'])*(self.db['χ2']/νp+epsilonp*self.auxPen(τp, epsilonp))

	###### Functions of τ and s
	def aux_PEE_funcOfτ(self, τ, sol, sol_p, t):
		""" Return functions of τ on the grid of s""" 
		return self.aux_funcOfτ(τ, self.db['θ'][t], self.db['epsilon'][t], sol, sol_p, t)

	def aux_ESC_funcOfτ(self, τ, θ, epsilon, sol, sol_p, t):
		funcOfτ = self.aux_funcOfτ(τ, θ, epsilon, sol, sol_p, t)
		funcOfτ['∂θ/∂τ'] = sol_p['∂θ/∂s'] * funcOfτ['dln(s)/dτ'] * sol_p['s[t-1]']
		funcOfτ['∂epsilon/∂τ'] = sol_p['∂epsilon/∂s'] * funcOfτ['dln(s)/dτ'] * sol_p['s[t-1]']
		return funcOfτ

	def aux_funcOfτ(self, τ, θ, epsilon, sol, sol_p, t):
		""" Return functions of τ on the grid of s"""
		funcOfτ = {'s[t-1]': sol['s_τ0']*(1-τ)**(-1/self.db['α'])}
		funcOfτ['B'] = self.aux_B(funcOfτ['s[t-1]'], sol['h'], self.db['ν'][t])
		funcOfτ['Γs'] = self.aux_Γs(funcOfτ['B'], τ, θ, epsilon)
		funcOfτ['si/s'] = self.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, θ, epsilon)
		funcOfτ.update(self.aux_derivatives_τ(τ, sol, sol_p))
		funcOfτ['dln(s)/dτ'] = self.aux_PEE_logDev(sol, funcOfτ, 's')
		funcOfτ.update({f'dln({k})/dτ': self.aux_PEE_logDev(sol, funcOfτ, k) for k in ('Γs','h')})
		funcOfτ['∂τp/∂τ'] = sol_p['∂τ/∂s'] * funcOfτ['dln(s)/dτ'] * sol_p['s[t-1]']
		return funcOfτ

	def aux_ESC_precomputations(self, sol_p, t):
		""" sol_p is the solution from t+1 """
		sol = {'s': sol_p['s[t-1]'],
					 'h': (sol_p['s[t-1]']/sol_p['Γs'])**(self.db['ξ']/(1+self.db['ξ'])),
					 'Ω': self.aux_𝛀(sol_p['τ'], sol_p['Γs'], sol_p['θ'], sol_p['epsilon']),
					 'Ψ': self.aux_Ψ(sol_p['τ'], sol_p['B'], sol_p['θ'], sol_p['epsilon'])}
		sol['s_τ0'] = self.db['ν'][t]*sol['h']**(1/self.power_h)*((1-((1-self.db['α'])/self.db['α'])*sol_p['θ']*self.auxPen(sol_p['τ'],  sol_p['epsilon'])*sol_p['Γs'])/((1-self.db['α'])*self.db['A'][t]))**(1/self.db['α'])
		sol.update(self.aux_laggedDerivatives_τ(sol, sol_p, sol_p['θ'], sol_p['epsilon']))
		sol['σ'] = self.aux_σ(sol_p['τ'], sol['Ω'], sol['Ψ'], sol_p['∂ln(h)/∂ln(s)'])
		sol.update(self.aux_laggedDerivatives_θ(sol, sol_p))
		sol.update(self.aux_laggedDerivatives_eps(sol, sol_p))
		sol.update({f'{k}_strategy': self.aux_ESC_strategy(sol, sol_p, k) for k in ('s','Γs','h')})
		return sol

	def aux_PEE_logDev(self, sol, funcOfτ, k):
		if k == 's':
			return funcOfτ[f'∂ln({k})/∂τ']/(1-sol[f'{k}_strategy'])
		else:
			return funcOfτ[f'∂ln({k})/∂τ']+sol[f'{k}_strategy'] * funcOfτ['dln(s)/dτ']

	def aux_PEE_strategy(self, sol, sol_p, k):
		return sol[f'∂ln({k})/∂τ[t+1]'] * sol_p['∂τ/∂s'] * sol['s']

	def aux_ESC_strategy(self, sol, sol_p, k):
		return self.aux_PEE_strategy(sol, sol_p, k) + (sol[f'∂ln({k})/∂θ[t+1]']*sol_p['∂θ/∂s']+sol[f'∂ln({k})/∂epsilon[t+1]']*sol_p['∂epsilon/∂s'])*sol['s']

	def aux_PEE_precomputations(self, sol_p, t):
		""" sol_p is the solution from t+1 """
		sol = {'s': sol_p['s[t-1]'],
					 'h': (sol_p['s[t-1]']/sol_p['Γs'])**(self.db['ξ']/(1+self.db['ξ'])),
					 'Ω': self.aux_𝛀(sol_p['τ'], sol_p['Γs'], self.db['θ'][t+1], self.db['epsilon'][t+1]),
					 'Ψ': self.aux_Ψ(sol_p['τ'], sol_p['B'], self.db['θ'][t+1], self.db['epsilon'][t+1])}
		sol['s_τ0'] = self.db['ν'][t]*sol['h']**(1/self.power_h)*((1-((1-self.db['α'])/self.db['α'])*self.db['θ'][t+1]*self.auxPen(sol_p['τ'], self.db['epsilon'][t+1])*sol_p['Γs'])/((1-self.db['α'])*self.db['A'][t]))**(1/self.db['α'])
		sol.update(self.aux_laggedDerivatives_τ(sol, sol_p, self.db['θ'][t+1], self.db['epsilon'][t+1]))
		sol['σ'] = self.aux_σ(sol_p['τ'], sol['Ω'], sol['Ψ'], sol_p['∂ln(h)/∂ln(s)'])
		sol.update({f'{k}_strategy': self.aux_PEE_strategy(sol, sol_p, k) for k in ('s','Γs','h')})
		return sol

	def aux_laggedDerivatives_τ(self, sol, sol_p, θp, epsilonp):
		k1 = sol['Ω']*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])
		k2 = (θp+(1-θp)*self.auxΓB2(sol_p['B']))*((1-self.db['α'])/self.db['α'])/(1+self.db['γu']*epsilonp/(1-self.db['γu']))
		k3 = k2/(1+k2*sol_p['τ'])
		dlns_dτ  = (1/(1+sol['Ψ']*(1+k1*sol_p['τ'])))*(k1+(1+k1*sol_p['τ'])*(sol['Ψ']*sol_p['∂ln(h)/∂τ']-k3))
		dlnΓs_dτ = sol['Ψ']*(sol_p['∂ln(h)/∂τ']-dlns_dτ)-k3
		return {'∂ln(s)/∂τ[t+1]': dlns_dτ,
				'∂ln(Γs)/∂τ[t+1]': dlnΓs_dτ,
				'∂ln(h)/∂τ[t+1]': self.db['ξ']*(dlns_dτ-dlnΓs_dτ)/(1+self.db['ξ'])}


	def aux_laggedDerivatives_eps(self, sol, sol_p):
		k1 = (sol_p['θ']+(1-sol_p['θ'])*self.auxΓB2(sol_p['B']))*((1-self.db['α'])/self.db['α'])*self.auxPen(sol_p['τ'], sol_p['epsilon'])
		k2 = (self.db['γu']/(1-self.db['γu']))/(1+self.db['γu']*sol_p['epsilon']/(1-self.db['γu']))
		k3 = ((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*sol_p['τ']*sol['Ω']
		dlns_deps  = ((1+k3)*k2*k1/(1+k1)-k3*k2)/sol['σ']
		dlnΓs_deps = sol['Ψ']*dlns_deps*(sol_p['∂ln(h)/∂ln(s)']-1)+k1*k2/(1+k1)
		return {'∂ln(s)/∂epsilon[t+1]': dlns_deps,
				'∂ln(Γs)/∂epsilon[t+1]': dlnΓs_deps,
				'∂ln(h)/∂epsilon[t+1]': self.db['ξ']*(dlns_deps-dlnΓs_deps)/(1+self.db['ξ'])}

	def aux_laggedDerivatives_θ(self, sol, sol_p):
		k1 = ((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*sol_p['τ']*sol['Ω']
		k2 = ((1-self.db['α'])/self.db['α'])*self.auxPen(sol_p['τ'], sol_p['epsilon'])
		k3 = k2*(1-self.auxΓB2(sol_p['B']))/(1+k2*(sol_p['θ']+(1-sol_p['θ'])*self.auxΓB2(sol_p['B'])))
		dlns_dθ = (k1/sol_p['θ']-(1+k1)*k3)/sol['σ']
		dlnΓs_dθ= sol['Ψ']*(sol_p['∂ln(h)/∂ln(s)']-1)*dlns_dθ-k3
		return {'∂ln(s)/∂θ[t+1]': dlns_dθ, 
				'∂ln(Γs)/∂θ[t+1]': dlnΓs_dθ, 
				'∂ln(h)/∂θ[t+1]': self.db['ξ']*(dlns_dθ-dlnΓs_dθ)/(1+self.db['ξ'])}

	def aux_derivatives_τ(self, τ, sol, sol_p):
		dlns_dτ  = -(1+self.db['ξ'])/((1+self.db['α']*self.db['ξ'])*(1-τ)*sol['σ'])
		dlnΓs_dτ = sol['Ψ']*(sol_p['∂ln(h)/∂ln(s)']-1)*dlns_dτ
		return {'∂ln(s)/∂τ': dlns_dτ,
				'∂ln(Γs)/∂τ': dlnΓs_dτ,
				'∂ln(h)/∂τ': self.db['ξ']*(dlns_dτ-dlnΓs_dτ)/(1+self.db['ξ'])}

	def aux_σ(self, τp, Ω, Ψ, dlnh_dlns):
		return 1+(1+τp*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*Ψ*(1-dlnh_dlns)

	def aux_𝛀(self, τp, Γs, θp, epsilonp):
		k = Γs*((1-self.db['α'])/self.db['α'])*θp/(1+self.db['γu']*epsilonp/(1-self.db['γu']))
		return k/(1-τp*k)

	def aux_Ψ(self, τp, Bp, θp, epsilonp):
		return (1-self.db['α'])*(self.db['ρ']-1)*(self.auxΓB3(Bp)/self.auxΓB1(Bp)+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilonp)*(1-θp)*self.auxΓB4(Bp)/(1+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilonp)*(θp+(1-θp)*self.auxΓB2(Bp))))


	################ NEW terminal period functions
	### ESC functions:
	def solve_ESC_T(self, s, θT = .5, epsilonT = .5, x0 = None):
		sol, _, ier, msg = optimize.fsolve(lambda τ: self.aux_PEE_polObj_T(τ,θT, epsilonT,s),
				noneInit(x0, np.full(len(s),.5)), full_output=True)
		sol = np.hstack([sol, np.full(len(s), θT), np.full(len(s), epsilonT)])
		# sol, _, ier, msg = optimize.fsolve(lambda x: self.aux_ESC_polObj_T(self(x, 'τ'), self(x,'θ'), self(x, 'epsilon'), s), 
		# 	noneInit(x0, np.full(self.ns['ESC[t]'].len, 0.5)), full_output=True)
		if ier == 1:
			return self.aux_ESC_T_solve(s, **self.ns['ESC[t]'].unloadSol(sol))
		else:
			return print(f"solve_ESC_T couldn't identify an equilibrium - fsolve returns {msg}")

	def aux_ESC_polObj_T(self, τ, θ, epsilon, s):
		""" Returns stacked FOCs for τ, θ, epsilon. """
		young = self.db['ν'][-1]*(self.db['γu']*self.ω1u*self.aux_PEE_HtM_young_T(τ, s)+(1-self.db['γu'])*np.matmul(self.ω1i*self.db['γ'], self.aux_PEE_workers_T(τ, θ, epsilon, s)))
		euler = self.aux_ESC_retirees_T(τ, θ, epsilon, s)
		htm   = self.aux_ESC_HtM_old_T(τ, epsilon, s)
		return np.hstack([young+(1-self.db['γu'])*np.matmul(self.ω2i*self.db['γ'], euler['τ'])+self.db['γu']*self.ω2u*htm['τ'],
						  np.matmul(self.ω2i*self.db['γ'], euler['θ']),
						  (1-self.db['γu'])*np.matmul(self.ω2i*self.db['γ'], euler['epsilon'])+self.db['γu']*self.ω2u*htm['epsilon']])

	def aux_ESC_T_solve(self,s, τ, θ, epsilon):
		solDict = {'τ': τ.values, 'θ': θ.values, 'epsilon': epsilon.values, 'h': self.aux_h_T(τ.values, s), 's[t-1]': s}
		solDict['B']  = self.aux_B(solDict['s[t-1]'], solDict['h'], self.db['ν'][-1]) # this is B[t]
		solDict['Γs'] = self.aux_Γs(solDict['B'], solDict['τ'], solDict['θ'], solDict['epsilon']) # this is Γs[t-1] 
		solDict['∂τ/∂s'] = np.gradient(solDict['τ'], solDict['s[t-1]'])
		solDict['∂θ/∂s'] = np.gradient(solDict['θ'], solDict['s[t-1]'])
		solDict['∂epsilon/∂s'] = np.gradient(solDict['epsilon'], solDict['s[t-1]'])
		solDict['dln(h)/dln(s)'] = np.gradient(solDict['h'], solDict['s[t-1]']) * (solDict['s[t-1]']/solDict['h'])
		solDict['∂ln(h)/∂ln(s)'] = self.db['α'] * self.db['ξ']/(1+self.db['α']*self.db['ξ'])
		solDict['∂ln(h)/∂τ'] = self.aux_dlnh_dτ_T(τ.values)
		return solDict

	### PEE only functions
	def solve_PEE_T(self, s, x0 = None):
		sol, _, ier, msg = optimize.fsolve(lambda τ: self.aux_PEE_polObj_T(τ, self.db['θ'].iloc[-1], self.db['epsilon'].iloc[-1], s), 
			noneInit(x0, np.full(len(s),.5)), full_output=True)
		if ier == 1:
			return self.aux_PEE_T_solve(sol, s)
		else:
			return print(f"solve_PEE_T couldn't identify an equilibrium - fsolve returns {msg}")

	def aux_PEE_T_solve(self, τ, s):
		""" Return solution dictionary given vector of taxes"""
		solDict = {'τ': τ, 'h': self.aux_h_T(τ,s), 's[t-1]': s}
		solDict['B']  = self.aux_B(solDict['s[t-1]'], solDict['h'], self.db['ν'][-1]) # this is B[t]
		solDict['Γs'] = self.aux_Γs(solDict['B'], solDict['τ'], self.db['θ'].iloc[-1], self.db['epsilon'].iloc[-1]) # this is Γs[t-1] 
		solDict['∂τ/∂s'] = np.gradient(solDict['τ'], solDict['s[t-1]'])
		solDict['dln(h)/dln(s)'] = np.gradient(solDict['h'], solDict['s[t-1]']) * solDict['s[t-1]']/solDict['h']
		solDict['∂ln(h)/∂ln(s)'] = self.db['α'] * self.db['ξ']/(1+self.db['α']*self.db['ξ'])
		solDict['∂ln(h)/∂τ'] = self.aux_dlnh_dτ_T(τ)
		return solDict

	def aux_PEE_polObj_T(self, τ, θ, epsilon, s):
		return (self.db['γu']*self.ω2u*self.aux_PEE_HtM_old_T(τ, θ, epsilon,s)+(1-self.db['γu'])*np.matmul(self.ω2i*self.db['γ'], self.aux_PEE_retirees_T(τ, θ, epsilon, s))
				+self.db['ν'][-1]*(self.db['γu']*self.ω1u*self.aux_PEE_HtM_young_T(τ, s)+(1-self.db['γu'])*np.matmul(self.ω1i*self.db['γ'], self.aux_PEE_workers_T(τ, θ, epsilon, s)))
				)

	def aux_PEE_HtM_old_T(self, τ, θ, epsilon, s):
		""" Contribution to FOC for PEE only (not ESC)"""
		c2u_coeff = self.aux_c2u_coeff(τ, epsilon, self.db['ν'][-1])
		c2u = self.aux_c2u_t(s, self.aux_h_T(τ ,s), self.db['ν'][-1], c2u_coeff)
		return self.aux_PEE_HtM_old(epsilon, self.aux_dlnh_dτ_T(τ), c2u, c2u_coeff)

	def aux_PEE_retirees_T(self, τ, θ, epsilon, s):
		""" Contribution to FOC for PEE only (not ESC)"""
		B = self.aux_B(s, self.aux_h_T(τ,s), self.db['ν'][-1])
		sSpread = self.savingsSpread(B, self.aux_Γs(B, τ, θ, epsilon), τ, θ, epsilon)
		c2i_coeff = self.aux_c2i_coeff(sSpread, τ, θ, epsilon)
		c2i = self.aux_c2i_t(τ, s, self.aux_h_T(τ ,s), self.db['ν'][-1], c2i_coeff)
		return self.aux_PEE_retirees(θ, epsilon, self.aux_dlnh_dτ_T(τ), c2i, c2i_coeff)

	def aux_PEE_HtM_young_T(self, τ, s):
		return -self.aux_c1u_t(s, self.aux_h_T(τ,s), self.db['ν'][-1])**(1-1/self.db['ρ'])*(1-self.db['α'])*self.db['ξ'] /((1+self.db['ξ']*self.db['α'])*(1-τ))

	def aux_PEE_workers_T(self, τ, θ, epsilon, s):
		return -self.aux_c̃1i_T(τ, s)**(1-1/self.db['ρ']) * (1+self.db['ξ']) /((1+self.db['ξ']*self.db['α'])*(1-τ))

	def aux_PEE_HtM_old(self, epsilon, dlnh_dτ, c2u, c2u_coeff):
		return c2u**(1-1/self.db['ρ'])*((1-self.db['α'])*dlnh_dτ+(epsilon/(1+self.db['γu']*epsilon/(1-self.db['γu'])))/c2u_coeff)

	def aux_PEE_retirees(self, θ, epsilon, dlnh_dτ, c2i, c2i_coeff):
		return c2i**(1-1/self.db['ρ'])*((1-self.db['α'])*dlnh_dτ+((θ*self.aux_Prod.reshape(self.ni,1)+1-θ)*((1-self.db['α'])/self.db['α'])/(1+self.db['γu']*epsilon/(1-self.db['γu'])))/c2i_coeff)

	### Auxiliary Functions
	def aux_dlnh_dτ_T(self, τ):
		return -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-τ))

	def aux_Θh_T(self, τ):
		return ((1-self.db['α'])*(1-τ)*self.db['A'][-1])**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))

	def aux_h_T(self, τ, s):
		return self.aux_Θh_T(τ) * (s/self.db['ν'][-1])**self.power_h

	# HtM retirees
	def aux_c2u_coeff(self, τ, epsilon, ν):
		return self.db['χ2']/ν+self.auxPen(τ,epsilon)*epsilon

	def aux_c2u_t(self, s, h, ν, c2u_coeff, A = 1):
		""" Holds for all t, including T"""
		return (1-self.db['α'])*A*ν*(s/ν)**(self.db['α'])*h**(1-self.db['α'])*c2u_coeff

	# young households
	def aux_c1u_t(self,s,h,ν,A =1):
		""" Holds for all t, including T"""
		return self.db['χ1']*(1-self.db['α'])*A*(s/ν)**(self.db['α'])*h**(1-self.db['α'])

	def aux_c̃1i_T(self, τ, s):
		return self.aux_Prod.reshape(self.ni,1)*(self.aux_h_T(τ, s)**((1+self.db['ξ'])/self.db['ξ']))/(1+self.db['ξ'])

	# Euler retirees
	def aux_c2i_coeff(self, sSpread, τ, θ, epsilon):
		return sSpread + ((1-self.db['α'])/self.db['α'])*self.auxPen(τ,epsilon)*(θ*self.aux_Prod.reshape(self.ni,1)+1-θ)

	def aux_c2i_t(self, τ, s, h, ν, c2i_coeff, A = 1):
		return self.db['α']*A*ν*(s/ν)**(self.db['α'])*h**(1-self.db['α'])*c2i_coeff

	### HtM RETIREES FUNCTIONS:
	def aux_ESC_HtM_old_T(self, τ, epsilon, s):
		""" Contribution to FOC for full ESC"""
		return self.aux_ESC_HtM_old(τ, epsilon, s, self.aux_h_T(τ,s), self.db['ν'][-1], self.aux_dlnh_dτ_T(τ))

	def aux_ESC_HtM_old(self, τ, epsilon, s, h, ν, dlnh_dτ):
		c2u_coeff = self.aux_c2u_coeff(τ, epsilon, ν)
		c2u = self.aux_c2u_t(s, h, ν, c2u_coeff)
		return {'τ': self.aux_PEE_HtM_old(epsilon, dlnh_dτ, c2u, c2u_coeff),
				'epsilon': self.aux_ESC_HtM_old_eps(c2u, c2u_coeff)}

	def aux_ESC_HtM_old_eps(self, c2u, c2u_coeff):
		return c2u**(1-1/self.db['ρ']) * 1/c2u_coeff
	
	### EULER RETIREES FUNCTIONS
	def aux_ESC_retirees_T(self, τ, θ, epsilon, s):
		""" Contribution to FOC for full ESC"""
		B = self.aux_B(s, self.aux_h_T(τ,s), self.db['ν'][-1])
		sSpread = self.savingsSpread(B, self.aux_Γs(B, τ, θ, epsilon), τ, θ, epsilon)
		return self.aux_ESC_retirees(τ, θ, epsilon, s, self.aux_h_T(τ,s), self.db['ν'][-1], sSpread, self.aux_dlnh_dτ_T(τ))

	def aux_ESC_retirees(self, τ, θ, epsilon, s, h, ν, sSpread, dlnh_dτ):
		c2i_coeff = self.aux_c2i_coeff(sSpread, τ, θ, epsilon)
		c2i = self.aux_c2i_t(τ, s, h, ν, c2i_coeff)
		return {'τ': self.aux_PEE_retirees(θ, epsilon, dlnh_dτ, c2i, c2i_coeff),
				'θ': self.aux_ESC_retirees_θ(c2i, c2i_coeff),
				'epsilon':self.aux_ESC_retirees_eps(θ, c2i, c2i_coeff)}

	def aux_ESC_retirees_eps(self, θ, c2i, c2i_coeff):
		return -c2i**(1-1/self.db['ρ']) * ((1-self.db['α'])/self.db['α'])*(θ*self.aux_Prod.reshape(self.ni,1)+1-θ)/c2i_coeff

	def aux_ESC_retirees_θ(self, c2i, c2i_coeff):
		return c2i**(1-1/self.db['ρ']) * (self.aux_Prod-1).reshape(self.ni,1) / c2i_coeff

	################ EE functions, non-terminal:
	def savingsSpread(self, Bp, Γs, τp, θp, epsilonp):
		""" τp is a vector of the same length as Bp"""
		return self.aux_Prod.reshape(self.ni,1) * Bp / ((1+Bp)*(1+self.db['ξ'])*Γs)-((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilonp)*(θp*self.aux_Prod.reshape(self.ni,1)+(1-θp)/(1+Bp))

	################ PEE functions:
	def aux_util1i(self, db, Δy = 0, Δo = 0):
		return ((db['̃c1i']+Δy)**(1-1/self.db['ρ'])).add((db['c2pi']+Δo)**(1-1/self.db['ρ'])*self.db['β'], fill_value = 0)/(1-1/self.db['ρ'])

	def aux_util1u(self, db, Δy = 0, Δo = 0):
		""" Utility for young hand-to-mouth"""
		return ((db['c1u']+Δy)**(1-1/self.db['ρ'])).add((db['c2pu']+Δo)**(1-1/self.db['ρ'])*self.db['βu'], fill_value = 0)/(1-1/self.db['ρ'])

	def aux_util2i(self, db, Δ = 0):
		""" Utility for retired households """
		return (db['c2i']+Δ)**(1-1/self.db['ρ'])/(1-1/self.db['ρ'])

	def aux_util2u(self, db, Δ = 0):
		""" Utility for old hand-to-mouth"""
		return (db['c2u']+Δ)**(1-1/self.db['ρ'])/(1-1/self.db['ρ'])

	def aux_utilPol(self, db, Δy = 0, Δy2 = 0, Δyu = 0, Δyu2 = 0, Δo = 0, Δou = 0):
		""" Political objective function """
		return (self.db['ω']*( (1-self.db['γu'])*np.matmul(self.aux_util2i(db, Δo),self.db['γ'])+self.db['γu']*self.aux_util2u(db, Δou) )
			+   self.db['ν']*( (1-self.db['γu'])*np.matmul(self.aux_util1i(db, Δy, Δy2),self.db['γ'])+self.db['γu']*self.aux_util1u(db, Δyu, Δyu2) )
			)

	############ Reporting functions:
	def reportAll(self):
		""" Based on the PEE solution, report host of other relevant variables"""
		self.reportCoefficients()
		self.reportLevels()
		self.reportUtils()

	def reportCoefficients(self):
		""" Assumes that self.solve_PEE has been run and unloaded to the self.db """
		[self.db.__setitem__(k, getattr(self,'aux_'+k)) for k in ('Θhi','Θsi','Θc1i','Θc2i','Θc2pi','Θ̃c1i','Θc1u','Θc2u','Θc2pu')];

	def reportLevels(self):
		""" Assumes self.reportCoefficients has been run"""
		[self.db.__setitem__(k, getattr(self,'levels_'+k)) for k in ('c1i','c2i','c1u','c2u','̃c1i','c2pi','c2pu','hi','w','R','b̄')];
		[self.db.__setitem__(k, getattr(self,'levels_'+k)) for k in ('bu','bi')];

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
		return pd.DataFrame((self.db['Θs'].values*self.savingsSpread(self.db['B'].values[1:].T, self.db['Γs'].values[1:], self.db['τ[t+1]'].values, self.db['θ'], self.db['epsilon'])).T,
			index = self.db['txE'], columns = self.db['i'])

	@property
	def aux_Θc1i(self):
		return pd.DataFrame(np.vstack([self.aux_Θc1i_txE, self.aux_Θc1i_T]), index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc1i_txE(self):
		return ((self.db['Θh'].values[:-1]**((1+self.db['ξ'])/self.db['ξ'])).reshape(self.T-1,1) * (self.aux_Prod*(1-self.db['B'].values[1:]/((1+self.db['ξ'])*(1+self.db['B'].values[1:]))))+
				(self.db['Θs'].values*((1-self.db['α'])/self.db['α']) * (1-self.db['θ'])*self.auxPen(self.db['τ[t+1]'].values, self.db['epsilon'])).reshape(self.T-1,1)/((1+self.db['B'].values[1:]))
				)
	@property
	def aux_Θc1i_T(self):
		return self.aux_Prod * ((1-self.db['α'])*(1-self.db['τ'].values[-1])*self.db['A'][-1])**((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))

	@property
	def aux_Θc2i(self):
		return pd.DataFrame( (self.db['α']*self.db['A']*self.db['ν']*(self.db['Θh'].values**(1-self.db['α']))).reshape(self.T,1)*(self.savingsSpread(self.db['B'].values.T, self.db['Γs'].values, self.db['τ'].values, self.db['θ'], self.db['epsilon']).T+((1-self.db['α'])/self.db['α'])*self.auxPen(self.db['τ'].values, self.db['epsilon']).reshape(self.T,1)*(self.db['θ']*self.aux_Prod+(1-self.db['θ']))),
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc2pi(self):
		return pd.DataFrame((self.db['α']*self.db['A'][1:]*self.db['ν'][1:]*(self.db['Θh'].values[1:])**(1-self.db['α'])*(self.db['Θs'].values/self.db['ν'][1:])**self.power_s).reshape(self.T-1,1) * (self.savingsSpread(self.db['B'].values[1:].T, self.db['Γs'].values[1:], self.db['τ'].values[1:], self.db['θ'], self.db['epsilon']).T+((1-self.db['α'])/self.db['α'])*self.auxPen(self.db['τ[t+1]'].values, self.db['epsilon']).reshape(self.T-1,1)*(self.db['θ']*self.aux_Prod+1-self.db['θ'])),
					index = self.db['txE'], columns = self.db['i'])

	@property
	def aux_Θ̃c1i_txE(self):
		return ((self.db['Θh'].values[:-1]**((1+self.db['ξ'])/self.db['ξ'])).reshape(self.T-1,1) /((1+self.db['ξ'])*(1+self.db['B'].values[1:]))) * (self.aux_Prod+((1-self.db['α'])/self.db['α'])*(self.auxPen(self.db['τ[t+1]'].values, self.db['epsilon'])*(1-self.db['θ'])*(1+self.db['ξ'])*self.db['Γs'].values[1:]).reshape(self.T-1,1))

	@property
	def aux_Θ̃c1i_T(self):
		return self.aux_Θc1i_T/(1+self.db['ξ'])

	@property
	def aux_Θ̃c1i(self):
		return pd.DataFrame(np.vstack([self.aux_Θ̃c1i_txE, self.aux_Θ̃c1i_T]), index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc1u(self):
		return pd.Series(self.db['χ1']*self.db['A']*(1-self.db['α'])*(self.db['Θh'].values**(1-self.db['α'])), 
				index = self.db['t'])

	@property
	def aux_Θc2u(self):
		return pd.Series((1-self.db['α'])*self.db['A']*self.db['ν']*(self.db['Θh'].values**(1-self.db['α']))*(self.db['χ2']/self.db['ν']+self.db['epsilon']*self.auxPen(self.db['τ'], self.db['epsilon'])),
				index = self.db['t'])

	@property
	def aux_Θc2pu(self):
		return pd.Series((1-self.db['α']) * self.db['A'][1:]*self.db['ν'][1:] * self.db['Θh'].values[1:]**(1-self.db['α']) * (self.db['Θs'].values/self.db['ν'][1:])**self.power_s * (self.db['χ2']/self.db['ν'][1:]+self.db['epsilon']*self.auxPen(self.db['τ[t+1]'].values, self.db['epsilon'])),
				index = self.db['txE'])

	def auxLevel(self, par):
		return (self.db['s[t-1]']/self.db['ν'])**par

	@property
	def levels_R(self):
		return self.aux_R(self.db['s[t-1]'], self.db['h'], self.db['ν'], A = self.db['A'])

	@property
	def levels_w(self):
		return (1-self.db['α'])*self.db['A']*self.auxLevel(self.db['α']) * self.db['h']**(-self.db['α'])

	@property
	def levels_b̄(self):
		return (self.db['ν']*self.db['w']*self.db['h']*self.db['τ'])[1:] / (self.db['h'].values[:-1]*(1+self.db['γu']*self.db['epsilon']/(1-self.db['γu']*self.db['epsilon'])))

	@property
	def levels_bu(self):
		return self.db['epsilon'] * self.db['h'].values[:-1]* self.db['b̄']

	@property
	def levels_bi(self):
		return self.db['θ']*self.db['hi'].iloc[:-1].add((1-self.db['θ'])*self.db['h'].iloc[:-1], axis = 0).set_axis(self.db['b̄'].index).mul(self.db['b̄'], axis = 0)

	@property
	def levels_c1i(self):
		return self.db['Θc1i'].mul(self.auxLevel(self.power_s),axis=0)

	@property
	def levels_̃c1i(self):
		return self.db['Θ̃c1i'].mul(self.auxLevel(self.power_s), axis=0)

	@property
	def levels_hi(self):
		return self.db['Θhi'].mul(self.auxLevel(self.power_h), axis=0)

	@property
	def levels_c2i(self):
		return self.db['Θc2i'].mul(self.auxLevel(self.power_s),axis=0)

	@property
	def levels_c2pi(self):
		return self.db['Θc2pi'].mul(self.auxLevel(self.power_p), axis=0).dropna()

	@property
	def levels_c1u(self):
		return self.db['Θc1u']*self.auxLevel(self.power_s)

	@property
	def levels_c2u(self):
		return self.db['Θc2u']*self.auxLevel(self.power_s)

	@property
	def levels_c2pu(self):
		return self.db['Θc2pu']*self.auxLevel(self.power_p).dropna()

	################ Calibration, Argentina:
	def argentinaCalibrate_preReformEqs(self, x, τ0, s0, θ0, t0, sGrid):
		""" Calibrate model to reflect choice of τ, s, θ, epsilon"""
		self.db['ω'] = x[0]
		self.db['ωu'] = x[1]
		self.db['ωη'] = x[2]
		self.db['β'], self.db['βu'] = np.full(self.ni, x[3]), x[3]
		sol = self.solve_ESC(sGrid)
		return np.hstack([sol['τ'].xs(t0)-τ0,
						  sol['θ'].xs(t0)-θ0,
						  sol['epsilon'].xs(t0)-argentinaCalEps(θ0, x[3]),
						  self.savingsRate(sol['Θs'].xs(t0), sol['Θh'].xs(t0))-s0])

	def argentinaCalibrate_preReform(self, τ0, s0, θ0, t0, sGrid, x0 = None):
		sol, _, ier, msg = optimize.fsolve(lambda x: self.argentinaCalibrate_preReformEqs(x, τ0, s0, θ0, t0, sGrid), noneInit(x0, [self.db['ω'], self.db['ωu'], self.db['ωη'], self.db['β'][0]]), full_output = True)
		assert ier == 1, f"""Error in argentinaCalibrate_preReform. fsolve returns: "{msg}" """
		self.db['ω'], self.db['ωu'], self.db['ωη'], self.db['β'], self.db['βu'] = sol[0], sol[1], sol[2], np.full(self.ni, sol[3]), sol[3]
		return sol

	def argentinaCalibrate_postReformEqs(self, x, θ, epsilon, t0, sGrid):
		""" Ensuret that the model replicates x, θ, epsilon"""
		self.db['ωu'] = x[0]
		self.db['ωη'] = x[1]
		sol = self.solve_ESC(sGrid)
		return np.hstack([sol['θ'].xs(t0)-θ,
						  sol['epsilon'].xs(t0)-epsilon])

	def argentinaCalibrate_postReform(self, θ, epsilon, t0, sGrid, x0 = None):
		sol, _, ier, msg = optimize.fsolve(lambda x: self.argentinaCalibrate_postReformEqs(x, θ, epsilon, t0, sGrid), noneInit(x0, [self.db['ωu'], self.db['ωη']]), full_output = True)
		assert ier == 1, f"""Error in argentinaCalibrate_postReform. fsolve returns: "{msg}" """
		self.db['ωu'], self.db['ωη'] = sol[0], sol[1]
		return sol

	def argentinaCalibrateEqs(self, x, τ0, s0, t0, sGrid, **kwargs):
		self.db['ω'] = x[0]
		self.db['β'], self.db['βu'] = np.full(self.ni, x[1]), x[1]
		self.db['epsilon'] = argentinaCalEps(self.db['θ'], x[1])
		sol = self.solve_PEE(sGrid, **kwargs)
		return np.hstack([sol['τ'].xs(t0)-τ0,
						  self.savingsRate(sol['Θs'].xs(t0), sol['Θh'].xs(t0))-s0])

	def argentinaCalibrate(self, τ0, s0, t0, sGrid, x0 = None, **kwargs):
		sol, _, ier, msg = optimize.fsolve(lambda x: self.argentinaCalibrateEqs(x, τ0, s0, t0, sGrid,**kwargs), noneInit(x0, [self.db['ω'], self.db['β'][0]]), full_output=True)
		if ier == 1:
			self.db['ω'], self.db['β'] = sol[0], np.full(self.ni, sol[1])
			self.db['epsilon'] = argentinaCalEps(self.db['θ'], self.db['β'][0])
			return sol
		else:
			print(f"Error in argentinaCalibrate: {msg}")
