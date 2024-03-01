import numpy as np, pandas as pd
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
	def __init__(self, ni = 11, T = 10, **kwargs):
		""" Fixed namespace """
		self.ni = ni
		self.T = T
		self.db = self.defaultParameters | kwargs
		self.db['t'] = pd.Index(range(self.T), name = 't')
		self.db['t_']= self.db['t'].append(pd.Index([-1], name = 't')).sort_values() # Time index with -1 included
		self.db['txE'] = self.db['t'][:-1] # all years except terminal year
		self.db['txE_'] = self.db['t_'][:-1] # all years including -1 except terminal year
		self.db['i'] = pd.Index(range(self.ni), name = 'i')
		self.ns = {}
		self.addNamespaces()

	def addNamespaces(self):
		self.ns['EE'] = sm(symbols = {'s': self.db['txE'], 'Γs': self.db['txE_'], 'h': self.db['t']})
		[ns.compile() for ns in self.ns.values()];
		self.ns['EE'].addLaggedSym('s[t-1]', 's', 1, c = ('not', self.db['t'][0:1])) # this will not include the s0 value as it is not endogenous in the optimization

	def __call__(self, x, name, ns = 'EE', **kwargs):
		return self.ns[ns](x, name, **kwargs)

	def get(self, x, name, ns = 'EE'):
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
				'epsilon': .1,
				'θ' : .5, 
				'γu': .05, 
				'χ1': .1, 
				'χ2': .05,
				'ω': 1,
				'ρ': .5} # 1/CRRA parameter

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

	def auxPen(self, τp, epsilon):
		return τp/(1+self.db['γu']*epsilon/(1-self.db['γu']))

	def aux_R(self, s, h, ν, A = 1):
		return self.db['α'] * A * (ν*h/s)**(1-self.db['α'])

	def aux_B(self, s, h, ν, A = 1):
		return self.db['β'].reshape(self.ni,1)**self.db['ρ'] * (self.aux_R(s,h, ν, A = A))**(self.db['ρ']-1)

	def aux_B_scalar(self, s, h, ν, A =1):
		return self.db['β']**self.db['ρ'] * (self.aux_R(s,h,ν, A = A)**(self.db['ρ']-1))

	def aux_B_grid(self, s, h, ν, A = 1):
		""" h is defined over a grid (τ, s)"""
		return ((self.aux_R(s,h,ν, A=A))**(self.db['ρ']-1)).reshape(h.shape+(1,))*self.db['β']**self.db['ρ']

	def aux_Γs(self, Bp, τp, θ, epsilon):
		""" τ and Bp are vectors of the same length"""
		return (1/(1+self.db['ξ']))*np.matmul(self.db['γ'] * self.aux_Prod, Bp/(1+Bp)) /(1+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilon)*(θ+(1-θ)*np.matmul(self.db['γ'], 1/(1+Bp))))

	def aux_Γs_grid(self, Bp, τp, θ, epsilon):
		""" τ is defined over 2d grid, Bp 3d grid (including type index 'i'') """
		return (1/(1+self.db['ξ']))*np.matmul(Bp/(1+Bp), self.db['γ'] * self.aux_Prod) /(1+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilon)*(θ+(1-θ)*np.matmul(1/(1+Bp),self.db['γ'])))

	def aux_Γs_scalar(self, Bp, τp, θ, epsilon):
		""" τ is a scalar (no grid or time dimensions)"""
		return (1/(1+self.db['ξ']))*sum(self.db['γ'] * self.aux_Prod * Bp/(1+Bp)) /(1+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilon)*(θ+(1-θ)*sum(self.db['γ']/(1+Bp))))

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
						  self.EE_Γs(self.get_s(x, s0), self(x, 'h'), τ.values),
						  self.EE_h(self(x, 's', ns = 'EE'), self(x, 'Γs', ns = 'EE'), τ)])
		
	def get_s(self, x, s0):
		return np.insert(self(x, 's', ns = 'EE'), 0, s0) # insert s0 to the numpy array s

	def get_sLag(self, x, s0):
		return np.insert(self(x,'s[t-1]', ns = 'EE'), 0, s0) # insert s0 to the numpy array s[t-1]

	def EE_s(self, s, sLag, Γs, τ, τp):
		""" Condition holds for all t except terminal T. sLag needs to include initial value. """
		return ((1-self.db['α'])*(1-τ.iloc[:-1])*self.db['A'][:-1]/(1-((1-self.db['α'])/self.db['α'])*self.db['θ']*self.auxPen(τp, self.db['epsilon'])*Γs[1:]))**((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])) * Γs[1:] *(sLag/self.db['ν'][:-1])**self.power_s

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

	def solve_steadyState_grid(self, τ, ν, x0 = None):
		""" x[:len(τ)] = s, x[len(τ):] = Γs"""
		sol, _, ier, msg = optimize.fsolve(lambda x: np.hstack([self.aux_steadyState_s(x[len(τ):], τ, ν, self.db['θ'], self.db['epsilon'])-x[:len(τ)],
																self.aux_Γs(self.steadyState_B_grid(x[len(τ):], τ,ν), τ, self.db['θ'], self.db['epsilon'])-x[len(τ):]]),
								noneInit(x0, np.full(2*len(τ), 0.1)), full_output=True)
		if ier == 1:
			return {'s': sol[:len(τ)], 'Γs': sol[len(τ):]}
		else:
			return print(f"fsolve in self.solve_steadyState_grid returns {msg}")

	def aux_steadyState_s(self, Γs, τ, ν, θ, epsilon, A = 1):
		return ( (((1-self.db['α'])*(1-τ)*A)/(1-((1-self.db['α'])/self.db['α'])*(θ*self.auxPen(τ, epsilon))*Γs))**(1+self.db['ξ'])*Γs**(1+self.db['α']*self.db['ξ'])/(ν**(self.db['α']*(1+self.db['ξ']))) )**(1/(1-self.db['α']))

	def steadyState_B(self, Γs, τ, ν, θ, epsilon):
		return self.db['β']**self.db['ρ']*( (self.db['α']-(1-self.db['α'])*(θ*τ/(1+self.db['γu']*epsilon/(1-self.db['γu'])))*Γs*ν**(self.db['α']))/((1-self.db['α'])*(1-τ)*Γs))**(self.db['ρ']-1)

	def steadyState_B_grid(self, Γs, τ, ν):
		""" τ, Γs are vectors of the same length; ν can either be a vector the same length or scalar. """
		return (self.db['β']**self.db['ρ']).reshape(self.ni,1)*( (self.db['α']-(1-self.db['α'])*(self.db['θ']*τ/(1+self.db['γu']*self.db['epsilon']/(1-self.db['γu'])))*Γs*ν**(self.db['α']))/((1-self.db['α'])*(1-τ)*Γs))**(self.db['ρ']-1)

	def approximateSteadyStateFromGrid(self, τ, Γs, ν, s):
		""" Given grids of τ, Γs, s - and ν scalar - this interpolates steady state savings """
		ŝ  = self.aux_steadyState_s(Γs, τ, ν, self.db['θ'], self.db['epsilon']) # Steady state values based on grids
		Δs = ŝ-s # distance from steady state
		id1, id2 = Δs[Δs>0].argmin(), Δs[Δs<0].argmax() # identify grid points closest to steady state
		s1, ŝ1 = s[Δs>0][id1], ŝ[Δs>0][id1]
		s2, ŝ2 = s[Δs<0][id2], ŝ[Δs<0][id2]
		return ŝ1+((ŝ2-ŝ1)*(ŝ1-s1))/(s2-s1-(ŝ2-ŝ1))

	################ Simulate PEE path given policies
	def updateAndSolve(self, sGrid, gridOption = 'resample', s0 = None, **kwargs):
		""" Update parameters with dictionary kwargs and resolve """
		self.db.update(kwargs)
		self.db.update(self.solve_PEE(sGrid, gridOption=gridOption, s0 = s0))
		return self.db

	def solve_PEE(self, sGrid, gridOption = 'resample', s0 = None):
		policy = self.solve_PEE_policy(sGrid, gridOption=gridOption)
		sols   = self.solve_PEE_givenPolicy(policy, s0 = s0)
		return self.reportPEE_main(sols)

	def reportPEE_main(self, sol):
		""" Given solution from self.solve_PEE, report main symbols required to characterize full economy. 
		This includes taxes, savings, labor supply, wages, interest rates. """
		d = {k: pd.Series([sol_i[k] for sol_i in sol.values()], index = self.db['t']) for k in ('τ','h','s[t-1]','∂τ/∂s')}
		d['B'] = pd.DataFrame(np.hstack([sol_i['B'] for sol_i in sol.values()]).T, index = self.db['t'], columns = self.db['i'])
		d['Γs'] = pd.Series([sol_i['Γs'] for sol_i in sol.values()], index = self.db['txE_'])
		d['∂ln(s)/∂τ'] = pd.Series([sol_i['∂ln(s)/∂τ'] for t,sol_i in sol.items() if t < self.T-1], index = self.db['txE'])
		d['s'] = d['s[t-1]'][1:].set_axis(self.db['txE'])
		d['τ[t+1]'] = self.leadSym(d['τ']).iloc[0:-1]
		d['Θs'] = d['s']/(((d['s[t-1]']/self.db['ν'])[:-1])**self.power_s)
		d['Θh'] = d['h']/((d['s[t-1]']/self.db['ν'])**self.power_h)
		### ONLY INCLUDED THESE FOR TROUBLESHOOTING PURPOSES / TO DECOMPOSE EFFECTS
		# d.update({k: pd.Series([sol_i[k] for t,sol_i in sol.items() if t < self.T-1], index = self.db['txE']) for k in ('dln(s)/dτ','dln(Γs)/dτ','dln(h)/dτ','∂τp/∂τ','dln(h)/dln(s)')})
		# d['si/s'] = pd.DataFrame(np.hstack([sol_i['si/s'] for t,sol_i in sol.items() if t< self.T-1]).T, index = self.db['txE'], columns=self.db['i'])
		return d

	def solve_PEE_givenPolicy(self, policy, s0 = None):
		""" Simulate solution given dict of policies for each t"""
		if s0 is None:
			s0 = self.approximateSteadyStateFromGrid(policy[0]['τ'], policy[0]['Γs'], self.db['ν'][0], policy[0]['s[t-1]'])
			# ss = self.solve_steadyState_grid(policy[0]['τ'], self.db['ν'][0], x0 = np.hstack([policy[0]['s[t-1]'], policy[0]['Γs']]))
			# s0 = ss['s']
		sol = dict.fromkeys(self.db['t'])
		sol[0] = {k: interpSol(s0, policy[0]['s[t-1]'], policy[0][k]) for k in policy[0]}
		for t in self.db['t'][1:]:
			sol[t] = {k: interpSol(sol[t-1]['s'], policy[t]['s[t-1]'], policy[t][k]) for k in policy[t]}
		return sol

	################  Solution on grid of savings
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

	def solve_PEE_t(self, solDict, t, x0 = None):
		solDict_t = self.aux_PEE_precomputations(solDict,t)
		sol, _, ier, msg = optimize.fsolve(lambda τ: self.aux_PEE_polObj_t(τ, solDict, solDict_t, t), 
			noneInit(x0, solDict['τ']), full_output=True)
		if ier == 1:
			return self.aux_PEE_t_solve(solDict, solDict_t, sol, t)
		else:
			return print(f"""solve_PEE_t couldn't identify an equilibrium - fsolve returns: '{msg}' """)

	def aux_PEE_t_solve(self, solDict, solDict_t, τ, t):
		""" Return solution dictionary given vector of taxes"""
		solDict_t['τ'] = τ
		solDict_t['s[t-1]'] = solDict_t['s_τ0']*(1-τ)**(-1/self.db['α'])
		solDict_t['∂τ/∂s'] = np.gradient(τ, solDict_t['s[t-1]'])
		solDict_t['dln(h)/dln(s)'] = np.gradient(solDict_t['h'], solDict_t['s[t-1]']) * solDict_t['s[t-1]']/solDict_t['h']
		solDict_t['∂ln(h)/∂ln(s)'] = self.db['α']*self.db['ξ']*(1+solDict_t['Ψ']*(1-solDict['∂ln(h)/∂ln(s)']))/(1+self.db['α']*self.db['ξ']+solDict_t['Ψ']*(1-solDict['∂ln(h)/∂ln(s)'])*(1+self.db['α']*self.db['ξ']+(1+self.db['ξ'])*solDict['τ']*solDict_t['Ω']))
		solDict_t['B']  = self.aux_B(solDict_t['s[t-1]'], solDict_t['h'], self.db['ν'][t])
		solDict_t['Γs'] = self.aux_Γs(solDict_t['B'], solDict_t['τ'], self.db['θ'], self.db['epsilon'])
		solDict_t['∂ln(s)/∂τ'] = self.aux_PEE_dlns_dτ_EE(τ, solDict['τ'], solDict_t['Ω'], solDict_t['Ψ'], solDict['∂ln(h)/∂ln(s)']) # we only include this for reporting purposes after the solution have been found!
		#### WE ONLY INCLUDE THE NEXT LINES HERE FOR TROUBLESHOOTING PURPOSES / TO DECOMPOSE EFFECTS:
		# solDict_t['si/s'] = self.savingsSpread(solDict_t['B'], solDict_t['Γs'], τ, self.db['θ'], self.db['epsilon'])
		# solDict_t['∂τp/∂τ'] = solDict['∂τ/∂s'] * solDict_t['∂ln(s)/∂τ'] * solDict['s[t-1]']
		# solDict_t.update(self.aux_PEE_logDevs(τ, solDict['τ'], solDict_t['Ω'], solDict_t['Ψ'], solDict['dln(h)/dln(s)'], solDict_t['∂τp/∂τ'], solDict['B']))
		return solDict_t

	def aux_𝛀(self, τp, Γs):
		k = Γs*((1-self.db['α'])/self.db['α'])*self.db['θ']/(1+self.db['γu']*self.db['epsilon']/(1-self.db['γu']))
		return k/(1-τp*k)

	def aux_Ψ(self, τp, Bp):
		return (1-self.db['α'])*(self.db['ρ']-1)*(self.auxΓB3(Bp)/self.auxΓB1(Bp)+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, self.db['epsilon'])*(1-self.db['θ'])*self.auxΓB4(Bp)/(1+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, self.db['epsilon'])*(self.db['θ']+(1-self.db['θ'])*self.auxΓB2(Bp))))

	###### PRECOMPUTATIONS
	def aux_PEE_precomputations(self, solDict, t):
		""" solDict is the solution from t+1 """
		solDict_t = {'s': solDict['s[t-1]'],
					 'h': (solDict['s[t-1]']/solDict['Γs'])**(self.db['ξ']/(1+self.db['ξ'])),
					 'Ω': self.aux_𝛀(solDict['τ'], solDict['Γs']),
					 'Ψ': self.aux_Ψ(solDict['τ'], solDict['B'])}
		solDict_t['s_τ0'] = self.db['ν'][t]*solDict_t['h']**(1/self.power_h)*((1-((1-self.db['α'])/self.db['α'])*self.db['θ']*self.auxPen(solDict['τ'], self.db['epsilon'])*solDict['Γs'])/((1-self.db['α'])*self.db['A'][t]))**(1/self.db['α'])
		return solDict_t

	###### Functions of τ and s
	def aux_PEE_funcOfτ(self, τ, solDict, solDict_t, t):
		""" Return functions of τ on the grid of s""" 
		funcOfτ = {'∂ln(s)/∂τ': self.aux_PEE_dlns_dτ_EE(τ, solDict['τ'], solDict_t['Ω'], solDict_t['Ψ'], solDict['∂ln(h)/∂ln(s)'])}
		funcOfτ['∂τp/∂τ'] = solDict['∂τ/∂s'] * funcOfτ['∂ln(s)/∂τ'] * solDict['s[t-1]']
		funcOfτ.update(self.aux_PEE_logDevs(τ, solDict['τ'], solDict_t['Ω'], solDict_t['Ψ'], solDict['dln(h)/dln(s)'], funcOfτ['∂τp/∂τ'], solDict['B']))
		funcOfτ['s[t-1]'] = solDict_t['s_τ0']*(1-τ)**(-1/self.db['α'])
		funcOfτ['B'] = self.aux_B(funcOfτ['s[t-1]'], solDict_t['h'], self.db['ν'][t])
		funcOfτ['Γs'] = self.aux_Γs(funcOfτ['B'], τ, self.db['θ'], self.db['epsilon'])
		funcOfτ['si/s'] = self.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, self.db['θ'], self.db['epsilon'])
		return funcOfτ

	def aux_PEE_dlns_dτ_EE(self, τ, τp, Ω, Ψ, dlnh_dlns):
		""" Returns ∂ln(s)/∂ln(τ), relies on ∂ln(h)/∂ln(s) from solution in t+1"""
		return -((1+self.db['ξ'])/(1-τ))/(1+self.db['α']*self.db['ξ']+Ψ*(1+self.db['α']*self.db['ξ']+(1+self.db['ξ'])*Ω*τp)*(1-dlnh_dlns))

	def aux_PEE_logDevs(self, τ, τp, Ω, Ψ, dlnh_dlns, dτp_dτ, B):
		""" Defines dln(s)/dln(τ) on a grid of s, as a function of τ """
		denom = 1+self.db['α']*self.db['ξ']+Ψ*(1+self.db['α']*self.db['ξ']+(1+self.db['ξ'])*Ω*τp)*(1-dlnh_dlns)
		k = ((1-self.db['α'])/self.db['α'])*(1/(1+self.db['γu']*self.db['epsilon']/(1-self.db['γu'])))*(self.db['θ']+(1-self.db['θ'])*self.auxΓB2(B))
		dlns_dτ = (-(1+self.db['ξ'])/(1-τ)+ dτp_dτ * ((1+self.db['ξ'])*Ω-(1+self.db['α']*self.db['ξ']+(1+self.db['ξ'])*τp*Ω)*k/(1+k*τp)))/denom
		dlnΓ_dτ = Ψ*(dlnh_dlns-1)*dlns_dτ - dτp_dτ * k/(1+k*τp)
		return {'dln(s)/dτ': dlns_dτ, 
				'dln(Γs)/dτ': dlnΓ_dτ, 
				'dln(h)/dτ': (self.db['ξ']/(1+self.db['ξ']))*(dlns_dτ-dlnΓ_dτ)}


	###### Tax effect on indirect utility
	def aux_PEE_polObj_t(self, τ, solDict, solDict_t, t):
		funcOfτ = self.aux_PEE_funcOfτ(τ, solDict, solDict_t, t)
		return (self.db['ω'] * (self.db['γu']*self.aux_PEE_HtM_old_t(τ, solDict, solDict_t, funcOfτ, t)+(1-self.db['γu'])*np.matmul(self.db['γ'], self.aux_PEE_retirees_t(τ, solDict, solDict_t, funcOfτ, t)))
				+self.db['ν'][t]* (self.db['γu']*self.aux_PEE_HtM_young_t(solDict, solDict_t, funcOfτ, t)+(1-self.db['γu'])*np.matmul(self.db['γ'], self.aux_PEE_workers_t(solDict, solDict_t, funcOfτ))))

	def aux_PEE_workers_t(self, solDict, solDict_t, funcOfτ):
		k = ((1-self.db['α'])/self.db['α'])*(1/(1+self.db['γu']*self.db['epsilon']/(1-self.db['γu'])))*(1-self.db['θ'])*(1+self.db['ξ'])*solDict['Γs']
		return self.aux_ĉ1i_t(solDict_t['h'], solDict['B'], solDict['τ'], solDict['Γs'])**(1-1/self.db['ρ']) * (
					((1+self.db['ξ'])/self.db['ξ'])*funcOfτ['dln(h)/dτ']
				+	(solDict['B']/(1+solDict['B']))*(1-self.db['α'])*(solDict['dln(h)/dln(s)']-1)*funcOfτ['dln(s)/dτ']
				+	k * (funcOfτ['∂τp/∂τ']+solDict['τ']*funcOfτ['dln(Γs)/dτ']) / (self.aux_Prod.reshape(self.ni,1)+solDict['τ']*k)
				)

	def aux_PEE_retirees_t(self, τ, solDict, solDict_t, funcOfτ, t):
		k = ((1-self.db['α'])/self.db['α']) * (1/(1+self.db['γu']*self.db['epsilon']/(1-self.db['γu']))) * (1-self.db['θ']+self.db['θ']*self.aux_Prod.reshape(self.ni,1))
		return self.aux_c2i_t(τ, funcOfτ['si/s'], funcOfτ['s[t-1]'], solDict_t['h'], t)**(1-1/self.db['ρ']) * (
					(1-self.db['α'])*funcOfτ['dln(h)/dτ']
				+	k/(funcOfτ['si/s']+k*τ)
			)

	def aux_PEE_HtM_young_t(self, solDict, solDict_t, funcOfτ, t):
		k = (self.db['epsilon']/(1+self.db['γu']*self.db['epsilon']/(1-self.db['γu'])))
		return (self.aux_c1u_t(funcOfτ['s[t-1]'], solDict_t['h'], t)**(1-1/self.db['ρ'])*(1-self.db['α'])*funcOfτ['dln(h)/dτ']
			+	self.db['βu']*self.aux_c2pu_t(solDict['s[t-1]'], solDict['h'], solDict['τ'], t)**(1-1/self.db['ρ'])*( (self.db['α']+(1-self.db['α'])*solDict['dln(h)/dln(s)'])*funcOfτ['dln(s)/dτ']+k*funcOfτ['∂τp/∂τ']/(self.db['χ2']/self.db['ν'][t+1]+k*solDict['τ']))
		)

	def aux_PEE_HtM_old_t(self, τ, solDict, solDict_t, funcOfτ, t):
		k = (self.db['epsilon']/(1+self.db['γu']*self.db['epsilon']/(1-self.db['γu'])))
		return self.aux_c2u_t(τ, funcOfτ['s[t-1]'], solDict_t['h'], t)**(1-1/self.db['ρ'])*((1-self.db['α'])*funcOfτ['dln(h)/dτ']+k/(self.db['χ2']/self.db['ν'][t]+τ*k))

	def aux_ĉ1i_t(self, h, B, τp, Γs):
		return (h**((1+self.db['ξ'])/self.db['ξ'])/(1+self.db['ξ'])) * (1+B)**(1/(self.db['ρ']-1)) * (self.aux_Prod.reshape(self.ni,1)+((1-self.db['α'])/self.db['α'])*self.auxPen(τp, self.db['epsilon'])*(1-self.db['θ'])*(1+self.db['ξ'])*Γs)

	def aux_c2i_t(self, τ, sSpread, s, h, t):
		return self.db['α']*self.db['A'][t]*self.db['ν'][t]*(s/self.db['ν'][t])**(self.db['α'])*h**(1-self.db['α'])*(sSpread+((1-self.db['α'])/self.db['α'])*self.auxPen(τ, self.db['epsilon'])*(self.db['θ']*self.aux_Prod.reshape(self.ni,1)+1-self.db['θ']))

	def aux_c1u_t(self,s,h,t):
		return self.db['χ1']*(1-self.db['α'])*self.db['A'][t]*(s/self.db['ν'][t])**(self.db['α'])*h**(1-self.db['α'])

	def aux_c2pu_t(self, sp, hp, τp, t):
		return self.db['χ2']*(1-self.db['α'])*self.db['A'][t+1]*(sp/self.db['ν'][t+1])**(self.db['α'])*hp**(1-self.db['α'])*(self.db['χ2']/self.db['ν'][t+1]+self.db['epsilon']*self.auxPen(τp, self.db['epsilon']))

	def aux_c2u_t(self,τ, s, h, t):
		return self.db['χ2']*(1-self.db['α'])*self.db['A'][t]*self.db['ν'][t]*(s/self.db['ν'][t])**(self.db['α'])*h**(1-self.db['α'])*(self.db['χ2']/self.db['ν'][t]+self.db['epsilon']*self.auxPen(τ, self.db['epsilon']))

	################ Terminal period functions:
	def solve_PEE_T(self, s, x0 = None):
		sol, _, ier, msg = optimize.fsolve(lambda τ: self.aux_PEE_polObj_T(s, τ, self.db['epsilon'], self.db['θ']), 
			noneInit(x0, np.full(len(s),.5)), full_output=True)
		if ier == 1:
			return self.aux_PEE_T_solve(sol, s)
		else:
			return print(f"solve_PEE_T couldn't identify an equilibrium - fsolve returns {msg}")

	def solve_PEE_T_grid(self, s, τ, x0 = None, reportSol = True):
		τArgMax = abs(self.aux_PEE_polObj_T_grid(s,τ, self.db['epsilon'], self.db['θ'])).argmin(axis= 0,keepdims=True)
		τOpt = np.take_along_axis(τ, τArgMax, 0)[0]
		return self.aux_PEE_T_solve(τOpt, s) if reportSol else τOpt

	def aux_PEE_T_solve(self, τ, s):
		""" Return solution dictionary given vector of taxes"""
		solDict = {'τ': τ, 'h': self.aux_h_T(τ,s), 's[t-1]': s}
		solDict['∂τ/∂s'] = np.gradient(solDict['τ'], solDict['s[t-1]'])
		solDict['dln(h)/dln(s)'] = np.gradient(solDict['h'], solDict['s[t-1]']) * solDict['s[t-1]']/solDict['h']
		solDict['∂ln(h)/∂ln(s)'] = self.db['α'] * self.db['ξ']/(1+self.db['α']*self.db['ξ']) 
		solDict['B']  = self.aux_B(solDict['s[t-1]'], solDict['h'], self.db['ν'][-1]) # this is B[t]
		solDict['Γs'] = self.aux_Γs(solDict['B'], solDict['τ'], self.db['θ'], self.db['epsilon']) # this is Γs[t-1] 
		return solDict

	def aux_Θh_T(self, τ):
		return ((1-self.db['α'])*(1-τ)*self.db['A'][-1])**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))

	def aux_h_T(self, τ, s):
		return self.aux_Θh_T(τ) * (s/self.db['ν'][-1])**self.power_h

	def aux_c̃1i_T(self, τ, s):
		return self.aux_Prod.reshape(self.ni,1)*(self.aux_h_T(τ, s)**((1+self.db['ξ'])/self.db['ξ']))/(1+self.db['ξ'])

	def aux_c̃1i_T_grid(self, τ, s):
		""" τ is a 2d grid """
		return ((self.aux_h_T(τ, s)**((1+self.db['ξ'])/self.db['ξ']))).reshape(τ.shape+(1,))*self.aux_Prod/(1+self.db['ξ'])

	def aux_c2i_T(self, τ, s, θ, epsilon, Θh, savingsSpread):
		return self.db['α'] * self.db['A'][-1] * self.db['ν'][-1] * (savingsSpread+((1-self.db['α'])/self.db['α'])*self.auxPen(τ, epsilon)*(θ*self.aux_Prod.reshape(self.ni,1)+1-θ))* Θh **(1-self.db['α']) * (s/self.db['ν'][-1])**self.power_s

	def aux_c2i_T_grid(self, τ, s, θ, epsilon, Θh, savingsSpread):
		return self.db['α'] * self.db['A'][-1] * self.db['ν'][-1] * (savingsSpread+((1-self.db['α'])/self.db['α'])*self.auxPen(τ, epsilon).reshape(τ.shape+(1,))*(θ*self.aux_Prod+1-θ))* Θh.reshape(Θh.shape+(1,)) **(1-self.db['α']) * (s.reshape(len(s),1)/self.db['ν'][-1])**self.power_s

	def aux_c1u_T(self, s, Θh):
		return self.db['χ1']*(1-self.db['α'])*self.db['A'][-1]*Θh**(1-self.db['α'])*(s/self.db['ν'][-1])**self.power_s

	def aux_c2u_T(self, τ, s, epsilon, Θh):
		return (1-self.db['α'])*self.db['A'][-1]*self.db['ν'][-1]*Θh**(1-self.db['α'])*(self.db['χ2']/self.db['ν'][-1]+self.auxPen(τ, epsilon)*epsilon)*(s/self.db['ν'][-1])**self.power_s

	def aux_PEE_polObj_T(self, s, τ, epsilon, θ):
		Θh = self.aux_Θh_T(τ)
		return (self.db['ω'] * (self.db['γu']*self.aux_PEE_HtM_old_T(τ, s, epsilon, Θh)+(1-self.db['γu'])*np.matmul(self.db['γ'], self.aux_PEE_retirees_T(τ, s, θ, epsilon, Θh)))
				+self.db['ν'][-1]* (self.db['γu']*self.aux_PEE_HtM_young_T(τ, s, Θh)+(1-self.db['γu'])*np.matmul(self.db['γ'], self.aux_PEE_workers_T(τ, s, θ, epsilon))))

	def aux_PEE_polObj_T_grid(self, s, τ, epsilon, θ):
		Θh = self.aux_Θh_T(τ)
		return (self.db['ω'] * (self.db['γu']*self.aux_PEE_HtM_old_T(τ, s, epsilon, Θh)+(1-self.db['γu'])*np.matmul(self.aux_PEE_retirees_T_grid(τ, s, θ, epsilon, Θh),self.db['γ']))
				+self.db['ν'][-1] * (self.db['γu']*self.aux_PEE_HtM_young_T(τ, s, Θh)+(1-self.db['γu'])*np.matmul(self.aux_PEE_workers_T_grid(τ, s, θ, epsilon),self.db['γ'])))

	def aux_PEE_workers_T(self, τ, s, θ, epsilon):
		return -self.aux_c̃1i_T(τ, s)**(1-1/self.db['ρ']) * (1+self.db['ξ']) /((1+self.db['ξ']*self.db['α'])*(1-τ))

	def aux_PEE_workers_T_grid(self, τ, s, θ, epsilon):
		return -self.aux_c̃1i_T_grid(τ, s)**(1-1/self.db['ρ']) * (1+self.db['ξ']) /((1+self.db['ξ']*self.db['α'])*(1-τ.reshape(τ.shape+(1,))))

	def aux_PEE_retirees_T(self, τ, s, θ, epsilon, Θh):
		B = self.aux_B(s, self.aux_h_T(τ,s), self.db['ν'][-1])
		savingsSpread = self.savingsSpread(B, self.aux_Γs(B, τ, θ, epsilon), τ, θ, epsilon)
		return self.aux_c2i_T(τ, s, θ, epsilon, Θh, savingsSpread)**(1-1/self.db['ρ'])*((1-self.db['α'])*self.auxPen2(θ, epsilon).reshape(self.ni,1)/(self.db['α']*savingsSpread+(1-self.db['α'])*self.auxPen2(θ,epsilon).reshape(self.ni,1)*τ)-(1-self.db['α'])*self.db['ξ'] /((1+self.db['ξ']*self.db['α'])*(1-τ)))

	def aux_PEE_retirees_T_grid(self, τ, s, θ, epsilon, Θh):
		B = self.aux_B_grid(s, self.aux_h_T(τ,s), self.db['ν'][-1])
		savingsSpread = self.savingsSpread_grid(B, self.aux_Γs_grid(B,τ,θ,epsilon), τ, θ, epsilon)
		return self.aux_c2i_T_grid(τ, s, θ, epsilon, Θh, savingsSpread)**(1-1/self.db['ρ'])*((1-self.db['α'])*self.auxPen2(θ, epsilon)/(self.db['α']*savingsSpread+(1-self.db['α'])*self.auxPen2(θ,epsilon)*τ.reshape(τ.shape+(1,)))-(1-self.db['α'])*self.db['ξ'] /((1+self.db['ξ']*self.db['α'])*(1-τ.reshape(τ.shape+(1,)))))

	def aux_PEE_HtM_young_T(self, τ, s, Θh):
		return -self.aux_c1u_T(s, Θh)**(1-1/self.db['ρ'])*(1-self.db['α'])*self.db['ξ'] /((1+self.db['ξ']*self.db['α'])*(1-τ))

	def aux_PEE_HtM_old_T(self, τ, s, epsilon, Θh):
		return self.aux_c2u_T(τ, s, epsilon, Θh)**(1-1/self.db['ρ'])*( (epsilon/(1+self.db['γu']*epsilon/(1-self.db['γu'])))/(self.db['χ2']/self.db['ν'][-1]+τ*epsilon/(1+self.db['γu']*epsilon/(1-self.db['γu'])))-(1-self.db['α'])*self.db['ξ'] /((1+self.db['ξ']*self.db['α'])*(1-τ)))

	################ EE functions, non-terminal:
	def savingsSpread(self, Bp, Γs, τp, θ, epsilon):
		""" τp is a vector of the same length as Bp"""
		return self.aux_Prod.reshape(self.ni,1) * Bp / ((1+Bp)*(1+self.db['ξ'])*Γs)-((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilon)*(θ*self.aux_Prod.reshape(self.ni,1)+(1-θ)/(1+Bp))

	def savingsSpread_grid(self, Bp, Γs, τp, θ, epsilon):
		""" Γs, τp are defined over 2d grids, Bp is 3d grid (including type index 'i') """
		return self.aux_Prod * Bp / ((1+Bp)*(1+self.db['ξ'])*Γs.reshape(Γs.shape+(1,)))-((1-self.db['α'])/self.db['α'])*self.auxPen(τp, epsilon).reshape(τp.shape+(1,))*(θ*self.aux_Prod+(1-θ)/(1+Bp))

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
