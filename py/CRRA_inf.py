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

def interpFixedPoint(ŝ, s):
	""" Interpolation of fixed point problem with two grids"""
	Δs = ŝ-s # distance from steady state
	id1, id2 = Δs[Δs>0].argmin(), Δs[Δs<0].argmax() # identify grid points closest to steady state
	s1, ŝ1 = s[Δs>0][id1], ŝ[Δs>0][id1]
	s2, ŝ2 = s[Δs<0][id2], ŝ[Δs<0][id2]
	return ŝ1+((ŝ2-ŝ1)*(ŝ1-s1))/(s2-s1-(ŝ2-ŝ1)) # lin. approximation

def aux_τMeshGrid(sGrid, τGrid_1d):
	return np.meshgrid(sGrid, τGrid_1d)[1]

def argentinaCalEps(θ, β):
	return 0.7 * (1-θ) * (β**(5/30)*9.45/14.45+β**(10/30)*12.55/22.55)/2

class infHorizon:
	def __init__(self, ni = 11, T = 10, ngrid = 50, eps = 0.1, θ = 0.5, **kwargs):
		""" Fixed namespace """
		self.ni = ni
		self.T = T
		self.ngrid = ngrid
		self.db = self.defaultParameters | kwargs
		self.db['αr'] = (1-self.db['α'])/self.db['α']
		self.db['t'] = pd.Index(range(self.T), name = 't')
		self.db['t_']= self.db['t'].append(pd.Index([-1], name = 't')).sort_values() # Time index with -1 included
		self.db['i'] = pd.Index(range(self.ni), name = 'i')
		self.db['sgrid'] = pd.Index(range(ngrid), name = 'sgrid') # grid of s[t-1] to identify policy function on
		self.ns = {}
		self.addNamespaces()
		self.db.update(self.initSC(eps, 'eps'))
		self.db.update(self.initSC(θ, 'θ'))

	def addNamespaces(self):
		self.ns['ESC'] = sm(symbols = {k: self.db['t'] for k in ('τ', 'θ', 'eps')})
		self.ns['ESC[t]'] = sm(symbols = {x: self.db['sgrid'] for x in ('τ', 'θ', 'eps')}) # namespace used in policy function identification
		self.ns['EE'] = sm(symbols = {x: self.db['t'] for x in ('s','h','Γs')}) # solve EE given policy
		[ns.compile() for ns in self.ns.values()];
		# Define auxiliary lagged/leaded symbols
		self.ns['EE'].addShiftedSym('h[t+1]','h',-1,opt = {'useLoc':'nn'})
		[self.ns['ESC'].addShiftedSym(f'{k}[t+1]',f'{k}', -1, opt = {'useLoc':'nn'}) for k in ('τ', 'θ','eps')];
		self.ns['EE'].addLaggedSym('s[t-1]', 's', 1, c = ('not', self.db['t'][0:1])) # this will not include the s0 value as it is not endogenous in the optimization

	def initSC(self, sc, name, ns = 'ESC'):
		""" Define relevant eps or θ parameters"""
		sc = pd.Series(sc, index = self.db['t'], name = name) if not pyDbs.is_iterable(sc) else sc
		return {name: sc, f'{name}[t+1]': self.leadSym(sc, ns = ns)}

	def __call__(self, x, name, ns = 'ESC[t]', **kwargs):
		return self.ns[ns](x, name, **kwargs)

	def get(self, x, name, ns = 'ESC[t]'):
		return self.ns[ns].get(x, name)

	def leadSym(self, symbol, lead = -1, opt = None, ns = 'ESC'):
		return self.ns[ns].getShift(symbol, lead, opt = noneInit(opt, {'useLoc':'nn'})) if isinstance(symbol, pd.Series) else pd.Series(symbol).shift(lead, fill_value=symbol[-1]).values

	@property
	def defaultParameters(self):
		return {'α': .5, 
				'A': np.ones(self.T), 
				'ν': np.ones(self.T),
				'η': np.linspace(1,2,self.ni),
				'γ': np.full(self.ni, 1/self.ni),
				'X': np.ones(self.ni),
				'β': np.full(self.ni, 1),
				'βu': 1, 
				'ξ' : .1,
				'γu': .05, 
				'χ1': .1, 
				'χ2': .05,
				'ω': 5,
				'ωu': .2,
				'ωη': 1.65,
				'ρ': .5,
				'kθ_l': 10,
				'kθ_u': 10,
				'keps_l': 10,
				'keps_u': 10}

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
	@property
	def power_s(self):
		return self.db['α']*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])
	@property
	def power_h(self):
		return self.db['α']*self.db['ξ']/(1+self.db['α']*self.db['ξ'])
	@property
	def power_p(self):
		return self.power_s**2

	###########################################################
	##########			Auxiliary functions 		###########
	###########################################################

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

	def auxPen(self, τ, eps):
		return τ/(1+self.db['γu']*eps/(1-self.db['γu']))

	def aux_R(self, s, h, ν, A = 1):
		return self.db['α'] * A * (ν*h/s)**(1-self.db['α'])

	def aux_B(self, s, h, ν, A = 1):
		return self.db['β'].reshape(self.ni,1)**self.db['ρ'] * (self.aux_R(s,h, ν, A = A))**(self.db['ρ']-1)

	def aux_Γs(self, Bp, τp, θp, epsp):
		""" τp, Bp, θp, epsp are vectors of the same length"""
		return (1/(1+self.db['ξ']))*np.matmul(self.db['γ'] * self.aux_Prod, Bp/(1+Bp)) /(1+self.db['αr']*self.auxPen(τp, epsp)*(θp+(1-θp)*np.matmul(self.db['γ'], 1/(1+Bp))))

	def savingsSpread(self, Bp, Γs, τp, θp, epsp):
		""" τp is a vector of the same length as Bp"""
		return self.aux_Prod.reshape(self.ni,1) * Bp / ((1+Bp)*(1+self.db['ξ'])*Γs)-self.db['αr']*self.auxPen(τp, epsp)*(θp*self.aux_Prod.reshape(self.ni,1)+(1-θp)/(1+Bp))

	def aux_Ω(self, Γs, τp, θp, epsp):
		k = Γs*self.db['αr']/(1+self.db['γu']*epsp/(1-self.db['γu']))
		return k/(1-τp*k*θp)

	def aux_Ψ(self, Bp, τp, θp, epsp):
		return (1-self.db['α'])*(self.db['ρ']-1)*(self.auxΓB3(Bp)/self.auxΓB1(Bp)+self.db['αr']*self.auxPen(τp, epsp)*(1-θp)*self.auxΓB4(Bp)/(1+self.db['αr']*self.auxPen(τp, epsp)*(θp+(1-θp)*self.auxΓB2(Bp))))

	def aux_σ(self, Ω, Ψ, dlnh_dlns, τp, θp):
		return 1+(1+τp*θp*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*Ψ*(1-dlnh_dlns)

	### Economic equilibrium derivatives:
	def aux_derivatives(self, Ψ, σ, dlnh_dlns, τ):
		dlns_dτ  = -(1+self.db['ξ'])/((1+self.db['α']*self.db['ξ'])*(1-τ)*σ)
		dlnΓs_dτ = Ψ*(dlnh_dlns-1)*dlns_dτ
		return {'∂ln(s)/∂τ': dlns_dτ,
				'∂ln(Γs)/∂τ': dlnΓs_dτ,
				'∂ln(h)/∂τ': self.db['ξ']*(dlns_dτ-dlnΓs_dτ)/(1+self.db['ξ'])}

	def aux_laggedDerivatives_τ(self, Ω, Ψ, Bp, dlnh_dτp, τp, θp, epsp):
		""" The derivative used here dlnh_dτp = ∂ln(h[t+1])/∂τ[t+1]"""
		k1 = θp*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])
		k2 = (θp+(1-θp)*self.auxΓB2(Bp))*self.db['αr']/(1+self.db['γu']*epsp/(1-self.db['γu']))
		k3 = k2/(1+k2*τp)
		dlns_dτ  = (1/(1+Ψ*(1+k1*τp)))*(k1+(1+k1*τp)*(Ψ*dlnh_dτp-k3))
		dlnΓs_dτ = Ψ*(dlnh_dτp-dlns_dτ)-k3
		return {'∂ln(s)/∂τ[t+1]': dlns_dτ,
				'∂ln(Γs)/∂τ[t+1]': dlnΓs_dτ,
				'∂ln(h)/∂τ[t+1]': self.db['ξ']*(dlns_dτ-dlnΓs_dτ)/(1+self.db['ξ'])}

	def aux_laggedDerivatives_θ(self, Ω, Ψ, σ, Bp, dlnh_dlns, τp, θp, epsp):
		k1 = ((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*τp*Ω
		k2 = self.db['αr']*self.auxPen(τp, epsp)
		k3 = k2*(1-self.auxΓB2(Bp))/(1+k2*(θp+(1-θp)*self.auxΓB2(Bp)))
		dlns_dθ = (k1-(1+k1*θp)*k3)/σ
		dlnΓs_dθ= Ψ*(dlnh_dlns-1)*dlns_dθ-k3
		return {'∂ln(s)/∂θ[t+1]': dlns_dθ, 
				'∂ln(Γs)/∂θ[t+1]': dlnΓs_dθ, 
				'∂ln(h)/∂θ[t+1]': self.db['ξ']*(dlns_dθ-dlnΓs_dθ)/(1+self.db['ξ'])}

	def aux_laggedDerivatives_eps(self, Ω, Ψ, σ, Bp, dlnh_dlns, τp, θp, epsp):
		k1 = (θp+(1-θp)*self.auxΓB2(Bp))*self.db['αr']*self.auxPen(τp, epsp)
		k2 = (self.db['γu']/(1-self.db['γu']))/(1+self.db['γu']*epsp/(1-self.db['γu']))
		k3 = ((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*τp*Ω*θp
		dlns_deps  = ((1+k3)*k2*k1/(1+k1)-k3*k2)/σ
		dlnΓs_deps = Ψ*dlns_deps*(dlnh_dlns-1)+k1*k2/(1+k1)
		return {'∂ln(s)/∂eps[t+1]': dlns_deps,
				'∂ln(Γs)/∂eps[t+1]': dlnΓs_deps,
				'∂ln(h)/∂eps[t+1]': self.db['ξ']*(dlns_deps-dlnΓs_deps)/(1+self.db['ξ'])}

	def savingsRate(self, Θs, Θh):
		return Θs/((1-self.db['α'])*(Θh**(1-self.db['α'])))

	def resampleSolution(self, sol, s):
		""" Redraw the solution on the grid 's' using linear interpolation """
		return {k: interpSol(s, sol['s[t-1]'], v) if  k != 's[t-1]' else s for k,v in sol.items()}

	def cleanSolution(self, sol, s):
		keep = sol['s[t-1]']<max(s)
		return {k: cleanSol(sol[k],keep) for k in sol}

	###########################################################
	##########		1. Identifying PEE paths  		###########
	###########################################################

	###### 1.1.  Identify policy functions and then identify PEE paths
	###  PEE with time dependent functions; terminal steady state
	def solve_PEE(self, s, gridOption = 'resample', s0 = None, returnPols = False, kwargs_ee = None, **kwargs):
		policy = self.solve_PEE_policy(s, gridOption=gridOption, **kwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_polFunc(policy, self.db['θ'].values, self.db['eps'].values, s0, **noneInit(kwargs_ee, {}))
		return (self.reportMain_PEE(sols), policy) if returnPols else self.reportMain_PEE(sols)

	###  ESC with time dependent functions; terminal steady state
	def solve_ESC(self, s, gridOption = 'resample', s0 = None, returnPols = False, kwargs_ee = None, **kwargs):
		policy = self.solve_ESC_policy(s, gridOption=gridOption, **kwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_ESC(policy, s0, **noneInit(kwargs_ee, {}))
		return (self.reportMain_ESC(sols), policy) if returnPols else self.reportMain_ESC(sols)

	def solve_ESCBound(self, s, gridOption = 'resample', s0 = None, returnPols = False, kwargs_ee = None, **kwargs):
		policy = self.solve_ESCBound_policy(s, gridOption=gridOption, **kwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_ESC(policy, s0, **noneInit(kwargs_ee, {}))
		return (self.reportMain_ESC(sols), policy) if returnPols else self.reportMain_ESC(sols)

	###  PEE steady state functions
	def solve_PEE_ss(self, s, s0 = None, returnPols = False, kwargs_ee = None, **kwargs):
		policy = self.solve_PEE_policy_ss(s, **kwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_polFunc(policy, self.db['θ'].values, self.db['eps'].values, s0, **noneInit(kwargs_ee, {}))
		return (self.reportMain_PEE(sols), policy) if returnPols else self.reportMain_PEE(sols)

	###  ESC steady state functions
	def solve_ESC_ss(self, s, s0 = None, returnPols = False, kwargs_ee = None, **kwargs):
		policy = self.solve_ESC_policy_ss(s, **kwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_ESC(policy, s0, **noneInit(kwargs_ee, {}))
		return (self.reportMain_ESC(sols), policy) if returnPols else self.reportMain_ESC(sols)

	def reportMain_PEE(self, sol):
		sol['B'] = pd.DataFrame(self.aux_B(sol['s[t-1]'].values, sol['h'].values, self.db['ν']).T, columns = self.db['i'], index = self.db['t'])
		sol['Γs[t-1]'] = pd.Series(self.aux_Γs(sol['B'].values.T, sol['τ'].values, self.db['θ'].values, self.db['eps'].values), index = self.db['t'])
		sol['B[t+1]'] = pd.DataFrame(self.aux_B(sol['s'].values, sol['h[t+1]'].values, self.leadSym(self.db['ν'])).T, columns = self.db['i'], index = self.db['t'])
		sol['τ[t+1]'] = self.leadSym(sol['τ'])
		return sol

	def reportMain_ESC(self, sol):
		sol['B'] = pd.DataFrame(self.aux_B(sol['s[t-1]'].values, sol['h'].values, self.db['ν']).T, columns = self.db['i'], index = self.db['t'])
		sol['Γs[t-1]'] = pd.Series(self.aux_Γs(sol['B'].values.T, sol['τ'].values, sol['θ'].values, sol['eps'].values), index = self.db['t'])
		sol['B[t+1]'] = pd.DataFrame(self.aux_B(sol['s'].values, sol['h[t+1]'].values, self.leadSym(self.db['ν'])).T, columns = self.db['i'], index = self.db['t'])
		sol.update({f'{k}[t+1]': self.leadSym(sol[f'{k}']) for k in ('τ', 'θ','eps')})
		return sol

	###### 1.2. Collection of policy functions over time 
	###  PEE path with time dependent policy functions; terminal steady state function
	def solve_PEE_policy(self, s, gridOption = 'resample', **kwargs):
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1] = self.solve_steadyState_PEE(self.db['θ'].values[-1], self.db['eps'].values[-1], self.db['ν'][-1], s, path =False,  **kwargs)
		if gridOption == 'resample':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.resampleSolution(self.solve_PEE_t(self.db['θ'].values[t], self.db['eps'].values[t], self.db['ν'][t], 
																 self.db['θ'].values[t+1], self.db['eps'].values[t+1], self.db['ν'][t+1], sols[t+1]), s)
		elif gridOption == 'clean':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.cleanSolution(self.solve_PEE_t(self.db['θ'].values[t], self.db['eps'].values[t], self.db['ν'][t], 
															  self.db['θ'].values[t+1], self.db['eps'].values[t+1], self.db['ν'][t+1], sols[t+1]), s)
			else:
				sols[t] = self.solve_PEE_t(self.db['θ'].values[t], self.db['eps'].values[t], self.db['ν'][t], 
										   self.db['θ'].values[t+1], self.db['eps'].values[t+1], self.db['ν'][t+1], sols[t+1])
		return sols

	### ESC path with time dependent policy functions; terminal steady state function
	def solve_ESC_policy(self, s, gridOption = 'resample', **kwargs):
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1]= self.solve_steadyState_ESC(self.db['ν'][-1], s, path = False, **kwargs)
		if gridOption == 'resample':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.resampleSolution(self.solve_ESC_t(self.db['ν'][t], self.db['ν'][t+1], sols[t+1]), s)
		elif gridOption == 'clean':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.cleanSolution(self.solve_ESC_t(self.db['ν'][t], self.db['ν'][t+1], sols[t+1]), s)
			else:
				sols[t] = self.solve_ESC_t(self.db['ν'][t], self.db['ν'][t+1], sols[t+1])
		return sols

	def solve_ESCBound_policy(self, s, gridOption = 'resample', **kwargs):
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1]= self.solve_steadyState_ESCBound(self.db['ν'][-1], s, path = False, **kwargs)
		if gridOption == 'resample':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.resampleSolution(self.solve_ESCBound_t(self.db['ν'][t], self.db['ν'][t+1], sols[t+1]), s)
		elif gridOption == 'clean':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.cleanSolution(self.solve_ESCBound_t(self.db['ν'][t], self.db['ν'][t+1], sols[t+1]), s)
			else:
				sols[t] = self.solve_ESCBound_t(self.db['ν'][t], self.db['ν'][t+1], sols[t+1])
		return sols

	### PEE path from steady state policy functions
	def solve_PEE_policy_ss(self, s, **kwargs):
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1] = self.solve_steadyState_PEE(self.db['θ'].values[-1], self.db['eps'].values[-1], self.db['ν'][-1], s, path = False, **kwargs)
		for t in range(self.T-2,-1,-1):
			sols[t] = self.solve_steadyState_PEE(self.db['θ'].values[t], self.db['eps'].values[t], self.db['ν'][t], s, τ0 = sols[t+1]['τ'], path = False, **kwargs)
		return sols

	### ESC path from steady state policy functions
	def solve_ESC_policy_ss(self, s, **kwargs):
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1] = self.solve_steadyState_ESC(self.db['ν'][-1], s, path = False,**kwargs)
		for t in range(self.T-2,-1,-1):
			sols[t] = self.solve_steadyState_ESC(self.db['ν'][t], s, path = False,**kwargs)
		return sols

	###########################################################
	##########		2. Steady state functions 		###########
	###########################################################

	########## 2.1. Scalar version of the steady state
	def ss_B_eq(self, B, Γs, τ, θ, eps, ν):
		""" Requirement = 0 for B to be in steady state - this does not return ss level of B"""
		if self.db['ρ']<1:
			return self.db['β']**(self.db['ρ']/(1-self.db['ρ']))-B**(1/(1-self.db['ρ']))*(ν*(self.db['α']-(1-self.db['α'])*self.auxPen(τ, eps)*θ*Γs))/((1-self.db['α'])*(1-τ)*Γs)
		elif self.db['ρ']>1:
			return B**(1/(self.db['ρ']-1))-self.db['β']**(self.db['ρ']/(self.db['ρ']-1))*(ν*(self.db['α']-(1-self.db['α'])*self.auxPen(τ, eps)*θ*Γs))/((1-self.db['α'])*(1-τ)*Γs)

	def ss_Γs_eq(self, B, Γs, τ, θ, eps, ν):
		""" Requirement = 0 for Γs to be in steady state - this does not return ss level of Γs """
		return self.auxΓB1(B)-Γs*(1+self.db['ξ'])*(1+self.db['αr']*self.auxPen(τ,eps)*(θ+(1-θ)*self.auxΓB2(B)))

	def ss_eqs(self, B, Γs, τ, θ, eps, ν):
		""" Stacked requirements = 0 for steady state B and Γs """
		return np.hstack([self.ss_B_eq(B, Γs, τ, θ, eps, ν), self.ss_Γs_eq(B,Γs,τ,θ,eps,ν)])

	def solve_ss(self, τ, θ, eps, ν, A = 1, x0 = None, **kwargs):
		sol = optimize.root(lambda x: self.ss_eqs(x[1:], x[0], τ, θ, eps, ν), noneInit(x0, np.full(self.ni+1, 0.5)), **kwargs)
		assert sol['success'], f""" Could not identify steady state (self.solve_ss) with parameter inputs: 
		τ: {τ}, θ: {θ}, ε: {eps}, ν: {ν}"""
		return self.report_ss(sol['x'], τ, θ, eps, ν, A = A)

	def report_ss(self, sol, τ, θ, eps, ν, A = 1):
		return {'B': sol[1:], 'Γs': sol[0], 's': self.aux_ss_savings(sol[0], τ, θ, eps, ν, A = A)}

	def aux_ss_savings(self, Γs, τ, θ, eps, ν, A = 1):
		return ( ((1-self.db['α'])*(1-τ)*A/(1-self.db['αr']*self.auxPen(τ, eps)*θ*Γs))**(1+self.db['ξ'])*Γs**(1+self.db['α']*self.db['ξ'])/(ν**(self.db['α']*(1+self.db['ξ']))) )**(1/(1-self.db['α']))

	def aux_ss_dlnh_ds(self, Ω, Ψ, τ, θ, eps):
		""" get steady state level of ∂ln(h)/∂ln(s) given solution to solve_ss (ss) and parameters """
		a = Ψ*(1+τ*θ*Ω*(1+self.db['ξ'])/(1+self.db['ξ']*self.db['α']))
		b = -(1+Ψ*(1+self.db['α']*self.db['ξ']+τ*θ*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])))
		c = self.db['α']*self.db['ξ']*(1+Ψ)
		return (-b-np.sqrt(b**2-4*a*c))/(2*a)

	########## 2.2. τ is a vector of inputs:
	### NB: This is seemingly faster than a multivariate function that identifies B, Γs simultaneously,
	### 	because fsolve does not handle data in a sparse way.
	def solve_ssVec(self, τ, θ, eps, ν, A =1, x0 = None, **kwargs):
		""" solve_ss with grid of taxes """
		sol = optimize.root(lambda x: self.ss_eqs(x[1:], x[0], τ[0], θ, eps, ν), noneInit(x0, np.full(self.ni+1, 0.5)),**kwargs)
		B  = np.empty((self.ni, self.ngrid))
		Γs = np.empty(self.ngrid)
		Γs[0], B[:,0] = sol['x'][0], sol['x'][1:]
		for i in range(1,self.ngrid):
			sol = optimize.root(lambda x: self.ss_eqs(x[1:], x[0], τ[i], θ, eps, ν), sol['x'], **kwargs)
			Γs[i], B[:,i] = sol['x'][0], sol['x'][1:]
		return {'B': B, 'Γs': Γs, 's': self.aux_ss_savings(Γs, τ, θ, eps, ν, A = A)}

	########## 2.3. τ, θ, eps are vectors of inputs (same length) 
	def solve_ss_ESC(self, τ, θ, eps, ν, A =1, x0 = None, **kwargs):
		""" solve_ss with grid of taxes and system characteristics """
		sol = optimize.root(lambda x: self.ss_eqs(x[1:], x[0], τ[0], θ[0], eps[0], ν), noneInit(x0, np.full(self.ni+1, 0.5)),**kwargs)
		B  = np.empty((self.ni, self.ngrid))
		Γs = np.empty(self.ngrid)
		Γs[0], B[:,0] = sol['x'][0], sol['x'][1:]
		for i in range(1,self.ngrid):
			sol = optimize.root(lambda x: self.ss_eqs(x[1:], x[0], τ[i], θ[i], eps[i], ν), sol['x'], **kwargs)
			Γs[i], B[:,i] = sol['x'][0], sol['x'][1:]
		return {'B': B, 'Γs': Γs, 's': self.aux_ss_savings(Γs, τ, θ, eps, ν, A = A)}

	###########################################################
	##########		3. Economic equilibrium 		###########
	###########################################################

	### 3.1. Economic Equilibrium given policies
	def solve_EE(self, τ, θ, eps, s0, x0 = None):
		τp, θp, epsp = self.leadSym(τ), self.leadSym(θ), self.leadSym(eps)
		sol = optimize.root(lambda x: self.EE_eqs(x, τ, θ, eps, τp, θp, epsp, s0), noneInit(x0, np.full(self.ns['EE'].len, 0.2)))
		assert sol['success'], f""" Could not identify economic equilibrium (self.solve_EE) with parameter inputs: 
		τ: {τ}, θ: {θ}, ε: {eps}, s0: {s0}"""
		return self.report_EE(sol, s0)

	def report_EE(self, sol, s0):
		d = self.ns['EE'].unloadSol(sol['x'])
		d['s[t-1]'].loc[0] = s0
		d['s[t-1]'] = d['s[t-1]'].sort_index()
		d['Θs'] = d['s']/((d['s[t-1]']/self.db['ν'])**self.power_s)
		d['Θh'] = d['h']/((d['s[t-1]']/self.db['ν'])**self.power_h)
		return d

	def EE_eqs(self, x, τ, θ, eps, τp, θp, epsp, s0):
		""" Stacked requirements = 0 for economic equilibrium """
		return np.hstack([self.EE_h_eq(self(x, 'Γs', ns = 'EE'), self.get_sLag(x, s0), τ, τp, θp, epsp, self.db['ν'])-self(x,'h',ns='EE'),
						  self.EE_s_eq(self(x, 'h', ns = 'EE'), self(x,'Γs', ns = 'EE'))-self(x,'s',ns='EE'),
						  self.EE_Γs_eq(self(x,'s',ns='EE'), self(x,'h[t+1]', ns = 'EE'), self.leadSym(self.db['ν']), τp, θp, epsp)-self(x,'Γs',ns='EE')])

	def aux_Θh(self, τ, Γs, τp, θp, epsp):
		return ((1-self.db['α'])*(1-τ)/((1-self.db['αr']*self.auxPen(τp, epsp)*θp*Γs)))**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))

	def EE_h_eq(self, Γs, s_, τ, τp, θp, epsp, ν):
		return self.aux_Θh(τ, Γs, τp, θp, epsp)*(s_/ν)**self.power_h

	def EE_s_eq(self, h, Γs):
		return h**((1+self.db['ξ'])/self.db['ξ'])*Γs

	def EE_Γs_eq(self, s, hp, νp, τp, θp, epsp):
		return self.aux_Γs(self.aux_B(s, hp, νp), τp, θp, epsp)

	def get_sLag(self, x, s0):
		return np.insert(self(x,'s[t-1]', ns = 'EE'), 0, s0) # insert s0 to the numpy array s[t-1]

	## 3.2. EE with policy functions for each t
	def solve_EE_polFunc(self, gridsol, θ, eps, s0, x0 = None):
		θp, epsp = self.leadSym(θ), self.leadSym(eps)
		policyFunction = self.aux_vecPolFunction(gridsol)
		sol = optimize.root(lambda x: self.EE_pol_eqs(x, policyFunction, θ, eps, θp, epsp, s0), noneInit(x0, np.full(self.ns['EE'].len, 0.2)))
		assert sol['success'], f""" Could not identify economic equilibrium (self.solve_EE_polFunc) with parameter inputs: 
		θ: {θ}, ε: {eps}, s0: {s0}"""
		return self.report_EE_polFunc(sol, policyFunction, s0)

	def EE_pol_eqs(self, x, policyFunction, θ, eps, θp, epsp, s0):
		τ = policyFunction(self.get_sLag(x,s0))
		τp = self.leadSym(τ)
		return self.EE_eqs(x, τ, θ, eps, τp, θp, epsp, s0)

	def report_EE_polFunc(self, sol, policyFunction, s0):
		d = self.ns['EE'].unloadSol(sol['x'])
		d['s[t-1]'].loc[0] = s0
		d['s[t-1]'] = d['s[t-1]'].sort_index()
		d['Θs'] = d['s']/((d['s[t-1]']/self.db['ν'])**self.power_s)
		d['Θh'] = d['h']/((d['s[t-1]']/self.db['ν'])**self.power_h)
		d['τ'] = pd.Series(policyFunction(d['s[t-1]']), index = self.db['t'])
		return d

	def aux_vecPolFunction(self, griddedSolution, x = 's[t-1]', y = 'τ'):
		""" Return vectorized policy function from dict of gridded solutions (keys = t and values = dict of gridded policy)"""
		return lambda k: np.array([np.interp(k[t], griddedSolution[t][x], griddedSolution[t][y]) for t in self.db['t']])

	## 3.3. EE with policy functions for each t for τ, θ, ε
	def solve_EE_ESC(self, gridsol, s0, x0 = None):
		τf = self.aux_vecPolFunction(gridsol, y = 'τ')
		θf = self.aux_vecPolFunction(gridsol, y = 'θ')
		epsf = self.aux_vecPolFunction(gridsol, y = 'eps')
		sol = optimize.root(lambda x: self.EE_ESC_eqs(x, τf, θf, epsf, s0), noneInit(x0, np.full(self.ns['EE'].len, 0.2)))
		assert sol['success'], f""" Could not identify economic equilibrium (self.solve_EE_ESC)."""
		return self.report_EE_ESC(sol, τf, θf, epsf, s0)

	def EE_ESC_eqs(self, x, τf, θf, epsf, s0):
		s_ = self.get_sLag(x, s0)
		τ, θ, eps = τf(s_), θf(s_), epsf(s_)
		τp, θp, epsp = self.leadSym(τ), self.leadSym(θ), self.leadSym(eps)
		return self.EE_eqs(x, τ, θ, eps, τp, θp, epsp, s0)

	def report_EE_ESC(self, sol, τf, θf, epsf, s0):
		d = self.ns['EE'].unloadSol(sol['x'])
		d['s[t-1]'].loc[0] = s0
		d['s[t-1]'] = d['s[t-1]'].sort_index()
		d['Θs'] = d['s']/((d['s[t-1]']/self.db['ν'])**self.power_s)
		d['Θh'] = d['h']/((d['s[t-1]']/self.db['ν'])**self.power_h)
		d['τ'] = pd.Series(τf(d['s[t-1]']), index = self.db['t'])
		d['θ'] = pd.Series(θf(d['s[t-1]']), index = self.db['t'])
		d['eps'] = pd.Series(epsf(d['s[t-1]']), index = self.db['t'])
		return d

	###########################################################
	##########		4. Terminal state solution 		###########
	###########################################################
	
	### 4.1. Given a guess on policy function τ(s), define dict
	### 	with sufficient data to solve year t problem.
	def aux_h_t(self, τ, τp, θp, epsp, Γs, s_, ν):
		return ((1-self.db['α'])*(1-τ)/((1-self.db['αr']*self.auxPen(τp, epsp)*θp*Γs)))**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))*(s_/ν)**self.power_h

	def steadyState_solp(self, τ, θ, eps, ν, s):
		""" This establishes solution in t = T+1. """
		ss = self.solve_ssVec(τ, θ, eps, ν)
		return self.steadyState_solp_aux(ss, τ, θ, eps, ν, s)

	def steadyState_solp_aux(self, ss, τ, θ, eps, ν,s):
		sol = {'τ': τ, 'θ': θ, 'eps': eps, 's[t-1]': s, 's':ss['s']}
		sol['h'] = (sol['s']/ss['Γs'])**(self.db['ξ']/(1+self.db['ξ'])) # impose ass assumption on policy, interest rate, and labor supply
		sol['B'] = self.aux_B(sol['s'], sol['h'], ν) # B[t+1] is steady state insofar as s, h are.
		sol['Γs'] = self.aux_Γs(sol['B'], sol['τ'], θ, eps) # Γs[t] is in steady state, insofar as B and τ is.
		# sol['h'] = self.aux_h_t(τ, τ, θ, eps, ss['Γs'], s, ν) # impose ss assumption on policy and interest rates, but not h
		# sol['B']  = self.aux_B(sol['s[t-1]'], sol['h'], ν) # this is B[t]
		# sol['Γs'] = self.aux_Γs(sol['B'], sol['τ'], θ, eps) # this is Γs[t-1] 
		sol['∂τ/∂s'] = np.gradient(sol['τ'], sol['s[t-1]'])
		sol['dln(h)/dln(s)'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['Ω'] = self.aux_Ω(ss['Γs'], τ, θ, eps)
		sol['Ψ'] = self.aux_Ψ(ss['B'], τ, θ, eps)
		sol['∂ln(h)/∂ln(s)'] = self.aux_ss_dlnh_ds(sol['Ω'], sol['Ψ'], τ, θ, eps)
		sol['σ'] = self.aux_σ(sol['Ω'], sol['Ψ'], sol['∂ln(h)/∂ln(s)'], sol['τ'], θ)
		sol.update(self.aux_derivatives(sol['Ψ'], sol['σ'], sol['∂ln(h)/∂ln(s)'], sol['τ']))
		return sol

	### 4.2. Given a guess on policy function x(s) = (τ, θ, ε)(s), define dict
	### 	with sufficient data to solve year t problem.
	def steadyState_solp_ESC(self, τ, θ, eps, ν,s):
		ss = self.solve_ss_ESC(τ, θ, eps, ν) # solve steady state on grids
		sol = self.steadyState_solp_aux(ss, τ, θ, eps, ν, s)
		sol.update({f'∂{k}/∂s': np.gradient(sol[f'{k}'], sol['s[t-1]']) for k in ('θ','eps')})
		return sol

	def steadyState_solp_ESCBound(self, τ, θ, eps, ν,s):
		sol = self.steadyState_solp_ESC(τ, θ, eps, ν,s)
		sol['θ_notBound']= sol['θ']
		sol['eps_notBound'] = sol['eps']
		return sol

	### 4.3. Iterate until convergence in policy function
	def solve_steadyState_PEE(self, θ, eps, ν, s, τ0 = None, tol_SA = None, iterMax = 100, tol_NK = 1e-5, path = True, **kwargs):
		""" Solve for the steady state PEE policy. 
			tol_SA::: tolerance for when to quit successive approximation (SA) approach.
			tol_NK::: tolerance for when to quit newton-kantorovich (NK) iterations. If tol_NK>= tol_SA the NK step is ignored."""
		if tol_SA:
			sol = self.steadyState_SA_PEE(noneInit(τ0, np.full(len(s), .2)), θ, eps, ν, s, tol = tol_SA, iterMax = iterMax)
			if tol_SA>tol_NK:
				fullSol = optimize.root(lambda x: self.steadyState_eq_PEE(x, θ, eps, ν, s), sol, tol = tol_NK, **kwargs)
				assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_PEE)"""
				sol = fullSol['x']
		else:
			fullSol = optimize.root(lambda x: self.steadyState_eq_PEE(x, θ, eps, ν, s), noneInit(τ0, np.full(len(s), .2)), tol = tol_NK, **kwargs)
			assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_PEE)"""
			sol = fullSol['x']
		return self.report_steadyState_PEE(sol, θ, eps, ν, s, path = path)

	def steadyState_SA_PEE(self, τ, θ, eps, ν, s, tol = 1e-5, iterMax = 100):
		""" Successive approximation of the policy function until tolerance is reached"""
		i = 0
		sol_p = self.steadyState_solp(τ, θ, eps, ν, s) # initial solution guess
		sol = self.resampleSolution(self.solve_PEE_t(θ, eps, ν, θ, eps, ν, sol_p), s)
		if max(sol['τ'])>1:
			print(f"""WARNING: It looks like the solution has values >1 for variable 'τ'. This may cause unexpected behavior in certain functions.""")
		elif min(sol['τ'])<0:
			print(f"""WARNING: It looks like the solution has values <0 for variable 'τ'. This may cause unexpected behavior in certain functions.""")
		while i<iterMax:
			if max(abs(sol['τ']-sol_p['τ']))<tol:
				break
			else:
				i += 1
				sol_p = self.steadyState_solp(sol['τ'], θ, eps, ν, s)
				sol = self.resampleSolution(self.solve_PEE_t(θ, eps, ν, θ, eps, ν, sol_p), s)
		if i == iterMax:
			raise ValueError('iterMax in successive approximations without reaching exit tolerance.')
		else:
			return sol['τ']

	def report_steadyState_PEE(self, τ, θ, eps, ν, s, path = True):
		""" sol::: Dictionary with PEE solution. 
			policy::: Dictionary with policy grids in steady state"""
		policy = self.steadyState_solp(τ, θ, eps, ν,s)
		if path:
			sInterp = interpFixedPoint(policy['s'], s)
			sols = optimize.root(lambda x: self.solve_ss(np.interp(x, policy['s[t-1]'], policy['τ']), θ, eps,ν)['s']-np.interp(x, policy['s[t-1]'], policy['s']), 
								x0 = sInterp)
			assert sols['success'], f""" Couldn't identify steady state (self.report_steadyState_PEE)"""
			sol = self.solve_ss(np.interp(sols['x'], policy['s[t-1]'], policy['τ']), θ, eps,ν)
			sol.update({k: interpSol(sol['s'], policy['s[t-1]'], policy[k]) for k in policy if k not in sol})
			return sol, policy
		else:
			return policy			

	def steadyState_eq_PEE(self, τ, θ, eps, ν, s):
		sol_p = self.steadyState_solp(τ, θ, eps, ν, s)
		sol = self.resampleSolution(self.solve_PEE_t(θ, eps, ν, θ, eps, ν, sol_p), s)
		return sol['τ']-sol_p['τ']

	### 4.4. Iterate until convergence in policy functions, ESC
	def solve_steadyState_ESC(self, ν, s, x0 = None, tol_SA = None, iterMax = 100, tol_NK = 1e-5, path = True, **kwargs):
		""" Solve for the steady state ESC policy. 
			tol_SA::: tolerance for when to quit successive approximation (SA) approach.
			tol_NK::: tolerance for when to quit newton-kantorovich (NK) iterations. If tol_NK>= tol_SA the NK step is ignored."""
		if tol_SA:
			sol = self.steadyState_SA_ESC(noneInit(x0, np.full(self.ns['ESC[t]'].len, .2)), ν, s, tol = tol_SA, iterMax = iterMax)
			if tol_SA>tol_NK:
				fullSol = optimize.root(lambda x: self.steadyState_eq_ESC(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s), sol, tol = tol_NK, **kwargs)
				assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_ESC)"""
				sol = fullSol['x']
		else:
			fullSol = optimize.root(lambda x: self.steadyState_eq_ESC(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s), noneInit(x0, np.full(self.ns['ESC[t]'].len, .2)), tol = tol_NK, **kwargs)
			assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_ESC)"""
			sol = fullSol['x']
		return self.report_steadyState_ESC(self(sol,'τ'), self(sol,'θ'), self(sol,'eps'), ν, s, path = path)

	def steadyState_SA_ESC(self, x, ν, s, tol = 1e-5, iterMax = 100):
		""" Successive approximation of the policy function until tolerance is reached"""
		i = 0
		sol_p = self.steadyState_solp_ESC(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s) # initial solution guess
		sol = self.resampleSolution(self.solve_ESC_t(ν, ν, sol_p), s)
		for k in ('τ','θ','eps'):
			if max(sol[k])>1:
				print(f"""WARNING: It looks like the solution has values >1 for variable {k}. This may cause unexpected behavior in certain functions.""")
			elif min(sol[k])<0:
				print(f"""WARNING: It looks like the solution has values <0 for variable {k}. This may cause unexpected behavior in certain functions.""")
		while i<iterMax:
			if max(abs(np.hstack([sol[k]-sol_p[k] for k in ('τ', 'θ', 'eps')])))<tol:
				break
			else:
				i += 1
				sol_p = self.steadyState_solp_ESC(sol['τ'], sol['θ'], sol['eps'], ν, s)
				sol = self.resampleSolution(self.solve_ESC_t(ν, ν, sol_p), s)
		if i == iterMax:
			raise ValueError('iterMax in successive approximations without reaching exit tolerance.')
		else:
			return np.hstack([sol_p['τ'], sol_p['θ'], sol_p['eps']])

	def report_steadyState_ESC(self, τ, θ, eps, ν, s, path = True):
		""" sol::: Dictionary with ESC solution path.
			policy::: Dictionary with policy grids in steady state.
			path::: Boolean indicating whether or not to include 'sol' in output"""
		policy = self.steadyState_solp_ESC(τ, θ, eps, ν,s)
		if path:
			sInterp = interpFixedPoint(policy['s'], s)
			sols = optimize.root(lambda x: self.solve_ss(np.interp(x, policy['s[t-1]'], policy['τ']), 
														 np.interp(x, policy['s[t-1]'], policy['θ']), 
														 np.interp(x, policy['s[t-1]'], policy['eps']), ν)['s']-np.interp(x, policy['s[t-1]'], policy['s']), 
								x0 = sInterp)
			assert sols['success'], f""" Couldn't identify steady state (self.report_steadyState_ESC)"""
			sol = self.solve_ss(np.interp(sols['x'], policy['s[t-1]'], policy['τ']), 
								np.interp(sols['x'], policy['s[t-1]'], policy['θ']), 
								np.interp(sols['x'], policy['s[t-1]'], policy['eps']),ν)
			sol.update({k: interpSol(sol['s'], policy['s[t-1]'], policy[k]) for k in policy if k not in sol})
			return sol, policy
		else:
			return policy

	def report_steadyState_ESCBound(self, τ, θ, eps, ν, s, path = True):
		if path: 
			sol, policy = self.report_steadyState_ESC(τ, θ, eps, ν, s, path = path)
			policy['θ_notBound'] = policy['θ']
			policy['eps_notBound'] = policy['eps']
			return sol, policy
		else:
			policy = self.report_steadyState_ESC(τ, θ, eps, ν, s, path = path)
			policy['θ_notBound'] = policy['θ']
			policy['eps_notBound'] = policy['eps']
			return policy

	def steadyState_eq_ESC(self, τ, θ, eps, ν, s):
		sol_p = self.steadyState_solp_ESC(τ, θ, eps, ν, s)
		sol = self.resampleSolution(self.solve_ESC_t(ν, ν, sol_p),s)
		return np.hstack([(sol[k]-sol_p[k]) for k in ('τ', 'θ', 'eps')])

	### 4.5. Iterate until convergence in policy functions, ESC, bounded version
	def solve_steadyState_ESCBound(self, ν, s, x0 = None, tol_SA = 1e-7, iterMax = 100, tol_NK = 1e-5, path = True, **kwargs):
		""" Solve for the steady state ESC policy. 
			tol_SA::: tolerance for when to quit successive approximation (SA) approach. If None then this is skipped.
			tol_NK::: tolerance for when to quit newton-kantorovich (NK) iterations. If tol_NK>= tol_SA the NK step is ignored."""
		if tol_SA:
			sol = self.steadyState_SA_ESCBound(noneInit(x0, np.full(self.ns['ESC[t]'].len, .2)), ν, s, tol = tol_SA, iterMax = iterMax)
			if tol_SA>tol_NK:
				fullSol = optimize.root(lambda x: self.steadyState_eq_ESCBound(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s), sol, tol = tol_NK, **kwargs)
				assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_ESCBound)"""
				sol = fullSol['x']
		else:
			fullSol = optimize.root(lambda x: self.steadyState_eq_ESCBound(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s), noneInit(x0, np.full(self.ns['ESC[t]'].len, .2)), tol = tol_NK, **kwargs)
			assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_ESCBound)"""
			sol = fullSol['x']
		return self.report_steadyState_ESCBound(self(sol,'τ'), self(sol,'θ'), self(sol,'eps'), ν, s, path = path)

	def steadyState_SA_ESCBound(self, x, ν, s, tol = 1e-5, iterMax = 100):
		""" Successive approximation of the policy function until tolerance is reached"""
		i = 0
		sol_p = self.steadyState_solp_ESCBound(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s) # initial solution guess
		sol = self.resampleSolution(self.solve_ESCBound_t(ν, ν, sol_p), s)
		while i<iterMax:
			if max(abs(np.hstack([sol[k]-sol_p[k] for k in ('τ', 'θ', 'eps')])))<tol:
				break
			else:
				i += 1
				sol_p = self.steadyState_solp_ESCBound(sol['τ'], sol['θ'], sol['eps'], ν, s)
				sol = self.resampleSolution(self.solve_ESCBound_t(ν, ν, sol_p), s)
		if i == iterMax:
			raise ValueError('iterMax in successive approximations without reaching exit tolerance.')
		else:
			return np.hstack([sol['τ'], sol['θ'], sol['eps']])

	def steadyState_eq_ESCBound(self, τ, θ, eps, ν, s):
		sol_p = self.steadyState_solp_ESCBound(τ, θ, eps, ν, s)
		sol = self.resampleSolution(self.solve_ESCBound_t(ν, ν, sol_p),s)
		return np.hstack([(sol[k]-sol_p[k]) for k in ('τ', 'θ', 'eps')])

	###########################################################
	##########			5. Policy function  		###########
	###########################################################

	### 5.1. PEE: Given a solution from t+1, solve for the
	###		 solution for period t. 
	def solve_PEE_t(self, θ, eps, ν, θp, epsp, νp, sol_p, x0 = None, **kwargs):
		sol = self.PEE_precomputations(θp, epsp, ν, sol_p)
		τ = optimize.root(lambda τ: self.PEE_polObj_t(τ, θ, eps, ν, θp, epsp, νp, sol, sol_p), noneInit(x0, sol_p['τ']), **kwargs)
		assert τ['success'], f"""Could not identify PEE solution (self.solve_PEE_t) with parameters:
		θ: {θ}, ε: {eps}, ν: {ν}, θ[t+1]: {θp}, ε[t+1]: {epsp}, ν[t+1]: {νp}, and solution from t+1 with taxes
		τ[t+1]: {sol_p['τ']}"""
		return self.report_PEE_t(τ['x'], θ, eps, ν, sol, sol_p)

	def report_PEE_t(self, τ, θ, eps, ν, sol, sol_p):
		""" Return solution dictionary given vector of taxes"""
		sol['τ'] = τ
		sol['s[t-1]'] = sol['s_τ0']*(1-τ)**(-1/self.db['α'])
		sol['B']  = self.aux_B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.aux_Γs(sol['B'], sol['τ'], θ, eps)
		sol['∂τ/∂s'] = np.gradient(τ, sol['s[t-1]'])
		sol['dln(h)/dln(s)'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s)'] = self.db['α']*self.db['ξ']*(1+sol['Ψ']*(1-sol_p['∂ln(h)/∂ln(s)']))/((1+self.db['α']*self.db['ξ'])*sol['σ'])
		sol.update(self.aux_derivatives(sol['Ψ'], sol['σ'], sol_p['∂ln(h)/∂ln(s)'], τ))
		return sol

	##### Tax effect on indirect utility
	def PEE_polObj_t(self, τ, θ, eps, ν, θp, epsp, νp, sol, sol_p):
		funcOfτ = self.funcOfτ(τ, θ, eps, ν, sol, sol_p)
		return (self.db['γu']*self.ω2u*self.aux_PEE_HtM_old_t(τ, eps, ν, sol, sol_p, funcOfτ)+(1-self.db['γu'])*np.matmul(self.ω2i * self.db['γ'], self.aux_PEE_retirees_t(τ, θ, eps, ν, sol, sol_p, funcOfτ))
				+ν*(self.db['γu']*self.ω1u*self.aux_PEE_HtM_young_t(epsp, ν, νp, sol, sol_p, funcOfτ)+(1-self.db['γu'])*np.matmul(self.ω1i*self.db['γ'], self.aux_PEE_workers_t(θp, epsp, sol, sol_p, funcOfτ))))

	def PEE_precomputations(self, θp, epsp, ν, sol_p, A = 1):
		""" sol_p is the solution from t+1 """
		sol = {'s': sol_p['s[t-1]'],
					'h': (sol_p['s[t-1]']/sol_p['Γs'])**(self.db['ξ']/(1+self.db['ξ'])),
					'Ω': self.aux_Ω(sol_p['Γs'], sol_p['τ'], θp, epsp),
					'Ψ': self.aux_Ψ(sol_p['B'], sol_p['τ'], θp, epsp)}
		sol['s_τ0'] = ν*sol['h']**(1/self.power_h)*((1-self.db['αr']*θp*self.auxPen(sol_p['τ'], epsp)*sol_p['Γs'])/((1-self.db['α'])*A))**(1/self.db['α'])
		sol.update(self.aux_laggedDerivatives_τ(sol['Ω'], sol['Ψ'], sol_p['B'], sol_p['∂ln(h)/∂τ'], sol_p['τ'], θp, epsp))
		sol['σ'] = self.aux_σ(sol['Ω'], sol['Ψ'], sol_p['∂ln(h)/∂ln(s)'], sol_p['τ'], θp)
		sol.update({f'{k}_strategy': self.aux_PEE_strategy(sol, sol_p, k) for k in ('s','Γs','h')})
		return sol

	def funcOfτ(self, τ, θ, eps, ν, sol, sol_p):
		""" Return functions of τ on the grid of s"""
		funcOfτ = {'s[t-1]': sol['s_τ0']*(1-τ)**(-1/self.db['α'])}
		funcOfτ['B'] = self.aux_B(funcOfτ['s[t-1]'], sol['h'], ν)
		funcOfτ['Γs'] = self.aux_Γs(funcOfτ['B'], τ, θ, eps)
		funcOfτ['si/s'] = self.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, θ, eps)
		funcOfτ.update(self.aux_derivatives(sol['Ψ'], sol['σ'], sol_p['∂ln(h)/∂ln(s)'], τ))
		funcOfτ['dln(s)/dτ'] = self.aux_PEE_logDev(sol, funcOfτ, 's')
		funcOfτ.update({f'dln({k})/dτ': self.aux_PEE_logDev(sol, funcOfτ, k) for k in ('Γs','h')})
		funcOfτ['∂τp/∂τ'] = sol_p['∂τ/∂s'] * funcOfτ['dln(s)/dτ'] * sol_p['s[t-1]']
		return funcOfτ

	### 5.2. ESC: Given a solution from t+1, solve for the
	###		 solution for period t. 
	def solve_ESC_t(self, ν, νp, sol_p, x0 = None, **kwargs):
		sol = self.ESC_precomputations(ν, sol_p)
		esc = optimize.root(lambda x: self.ESC_polObj_t(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, νp, sol, sol_p), noneInit(x0, np.hstack([sol_p['τ'], sol_p['θ'], sol_p['eps']])), **kwargs)
		assert esc['success'], f"""Could not identify ESC solution (self.solve_ESC_t) with parameters:
		ν: {ν}, ν[t+1]: {νp}, and solution from t+1 with:
		τ[t+1]: {sol_p['τ']}, 
		θ[t+1]: {sol_p['θ']},
		ε[t+1]: {sol_p['eps']}"""
		return self.report_ESC_t(esc['x'], ν, sol, sol_p)

	def report_ESC_t(self, x, ν, sol, sol_p):
		""" Return solution dictionary given vector of taxes"""
		sol['τ'], sol['θ'], sol['eps'] = self(x, 'τ'), self(x, 'θ'), self(x,'eps')
		sol['s[t-1]'] = sol['s_τ0']*(1-sol['τ'])**(-1/self.db['α'])
		sol['B']  = self.aux_B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.aux_Γs(sol['B'], sol['τ'], sol['θ'], sol['eps'])
		sol.update({f'∂{k}/∂s': np.gradient(sol[f'{k}'], sol['s[t-1]']) for k in ('τ','θ','eps')})
		sol['dln(h)/dln(s)'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s)'] = self.db['α']*self.db['ξ']*(1+sol['Ψ']*(1-sol_p['∂ln(h)/∂ln(s)']))/((1+self.db['α']*self.db['ξ'])*sol['σ'])
		sol.update(self.aux_derivatives(sol['Ψ'], sol['σ'], sol_p['∂ln(h)/∂ln(s)'], sol['τ']))
		return sol

	def ESC_polObj_t(self, τ, θ, eps, ν, νp, sol, sol_p):
		""" Returns stacked FOCs for τ, θ, eps. """
		funcOfτ = self.ESC_funcOfτ(τ, θ, eps, ν, sol, sol_p)
		young = ν*(self.db['γu']*self.ω1u*self.aux_ESC_HtM_young_t(ν, νp, sol, sol_p, funcOfτ)+(1-self.db['γu'])*np.matmul(self.ω1i*self.db['γ'], self.aux_PEE_workers_t(sol_p['θ'], sol_p['eps'], sol, sol_p, funcOfτ)))
		euler = self.aux_ESC_retirees(τ, θ, eps, funcOfτ['s[t-1]'], sol['h'], ν, funcOfτ['si/s'], funcOfτ['dln(h)/dτ'])
		htm   = self.aux_ESC_HtM_old(τ, eps, funcOfτ['s[t-1]'], sol['h'], ν, funcOfτ['dln(h)/dτ'])
		return np.hstack([young+(1-self.db['γu'])*np.matmul(self.ω2i*self.db['γ'], euler['τ'])+self.db['γu']*self.ω2u*htm['τ'],
						  np.matmul(self.ω2i*self.db['γ'], euler['θ']),
						  (1-self.db['γu'])*np.matmul(self.ω2i*self.db['γ'], euler['eps'])+self.db['γu']*self.ω2u*htm['eps']])

	def solve_ESCBound_t(self, ν, νp, sol_p, x0 = None, **kwargs):
		sol = self.ESC_precomputations(ν, sol_p)
		esc = optimize.root(lambda x: self.ESCBound_polObj_t(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, νp, sol, sol_p), noneInit(x0, np.hstack([sol_p['τ'], sol_p['θ_notBound'], sol_p['eps_notBound']])), **kwargs)
		assert esc['success'], f"""Could not identify ESC solution (self.solve_ESCBound_t) with parameters:
		ν: {ν}, ν[t+1]: {νp}, and solution from t+1 with:
		τ[t+1]: {sol_p['τ']}, 
		θ[t+1]: {sol_p['θ']},
		ε[t+1]: {sol_p['eps']}"""
		return self.report_ESCBound_t(esc['x'], ν, sol, sol_p)

	def report_ESCBound_t(self, x, ν, sol, sol_p):
		""" Return solution dictionary given vector of taxes"""
		sol = self.report_ESC_t(np.hstack([self(x,'τ'), np.clip(self(x,'θ'),0,1), np.clip(self(x,'eps'),0,1)]), ν, sol, sol_p)
		sol['θ_notBound'] = self(x,'θ')
		sol['eps_notBound'] = self(x, 'eps')
		return sol

	def ESCBound_polObj_t(self, τ, θ, eps, ν, νp, sol, sol_p):
		""" Returns stacked FOCs for τ, θ, eps. θ, eps from sol_p are bounded versions (main solution), but inputs θ,eps are not yet bounded:"""
		θBound, epsBound = np.clip(θ, 0, 1), np.clip(eps, 0,1)
		funcOfτ = self.ESC_funcOfτ(τ, θBound, epsBound, ν, sol, sol_p)
		young = ν*(self.db['γu']*self.ω1u*self.aux_ESC_HtM_young_t(ν, νp, sol, sol_p, funcOfτ)+(1-self.db['γu'])*np.matmul(self.ω1i*self.db['γ'], self.aux_PEE_workers_t(sol_p['θ'], sol_p['eps'], sol, sol_p, funcOfτ)))
		euler = self.aux_ESC_retirees(τ, θBound, epsBound, funcOfτ['s[t-1]'], sol['h'], ν, funcOfτ['si/s'], funcOfτ['dln(h)/dτ'])
		htm   = self.aux_ESC_HtM_old(τ, epsBound, funcOfτ['s[t-1]'], sol['h'], ν, funcOfτ['dln(h)/dτ'])
		return np.hstack([young+(1-self.db['γu'])*np.matmul(self.ω2i*self.db['γ'], euler['τ'])+self.db['γu']*self.ω2u*htm['τ'],
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γ'], euler['θ']), θ, kl = self.db['kθ_l'], ku = self.db['kθ_u']),
						  self.adjustFocMultiplicative((1-self.db['γu'])*np.matmul(self.ω2i*self.db['γ'], euler['eps'])+self.db['γu']*self.ω2u*htm['eps'], eps, kl = self.db['keps_l'], ku = self.db['kθ_u'])])

	def adjustFocMultiplicative(self, z, x, l = 0, u = 1, kl = 10, ku = 10):
		""" Adjust FOC in a multiplicative way: z is the "original" marginal effect, x is the bounded variable """
		return z-abs(z)*(kl*(np.clip(x, None, l)-l)+ku*(np.clip(x,u,None)-u))

	def ESC_precomputations(self, ν, sol_p, A = 1):
		""" sol_p is the solution from t+1 """
		sol = {'s': sol_p['s[t-1]'],
					'h': (sol_p['s[t-1]']/sol_p['Γs'])**(self.db['ξ']/(1+self.db['ξ'])),
					'Ω': self.aux_Ω(sol_p['Γs'], sol_p['τ'], sol_p['θ'], sol_p['eps']),
					'Ψ': self.aux_Ψ(sol_p['B'], sol_p['τ'], sol_p['θ'], sol_p['eps'])}
		sol['s_τ0'] = ν*sol['h']**(1/self.power_h)*((1-self.db['αr']*sol_p['θ']*self.auxPen(sol_p['τ'], sol_p['eps'])*sol_p['Γs'])/((1-self.db['α'])*A))**(1/self.db['α'])
		sol.update(self.aux_laggedDerivatives_τ(sol['Ω'], sol['Ψ'], sol_p['B'], sol_p['∂ln(h)/∂τ'], sol_p['τ'], sol_p['θ'], sol_p['eps']))
		sol['σ'] = self.aux_σ(sol['Ω'], sol['Ψ'], sol_p['∂ln(h)/∂ln(s)'], sol_p['τ'], sol_p['θ'])
		sol.update(self.aux_laggedDerivatives_θ(sol['Ω'], sol['Ψ'], sol['σ'], sol_p['B'], sol_p['∂ln(h)/∂ln(s)'], sol_p['τ'], sol_p['θ'], sol_p['eps']))
		sol.update(self.aux_laggedDerivatives_eps(sol['Ω'], sol['Ψ'], sol['σ'], sol_p['B'], sol_p['∂ln(h)/∂ln(s)'], sol_p['τ'], sol_p['θ'], sol_p['eps']))
		sol.update({f'{k}_strategy': self.aux_ESC_strategy(sol, sol_p, k) for k in ('s','Γs','h')})
		return sol

	def ESC_funcOfτ(self, τ, θ, eps, ν, sol, sol_p):
		""" Return functions of τ, θ, ε on grid of s"""
		funcOfτ = self.funcOfτ(τ, θ, eps, ν, sol, sol_p)
		funcOfτ['∂θ/∂τ'] = sol_p['∂θ/∂s'] * funcOfτ['dln(s)/dτ'] * sol_p['s[t-1]']
		funcOfτ['∂eps/∂τ'] = sol_p['∂eps/∂s'] * funcOfτ['dln(s)/dτ'] * sol_p['s[t-1]']
		return funcOfτ

	def aux_PEE_logDev(self, sol, funcOfτ, k):
		if k == 's':
			return funcOfτ[f'∂ln({k})/∂τ']/(1-sol[f'{k}_strategy'])
		else:
			return funcOfτ[f'∂ln({k})/∂τ']+sol[f'{k}_strategy'] * funcOfτ['dln(s)/dτ']

	def aux_PEE_strategy(self, sol, sol_p, k):
		return sol[f'∂ln({k})/∂τ[t+1]'] * sol_p['∂τ/∂s'] * sol['s']

	def aux_ESC_strategy(self, sol, sol_p, k):
		return self.aux_PEE_strategy(sol, sol_p, k) + (sol[f'∂ln({k})/∂θ[t+1]']*sol_p['∂θ/∂s']+sol[f'∂ln({k})/∂eps[t+1]']*sol_p['∂eps/∂s'])*sol['s']

	###########################################################
	##########	6. Auxiliary household functions 	###########
	###########################################################	
	###### 6.1. Euler retirees
	def aux_ESC_retirees(self, τ, θ, eps, s, h, ν, sSpread, dlnh_dτ):
		""" Contribution to political objective in ESC problem"""
		c2i_coeff = self.aux_c2i_coeff(sSpread, τ, θ, eps)
		c2i = self.aux_c2i_t(τ, s, h, ν, c2i_coeff)
		return {'τ': self.aux_PEE_retirees(θ, eps, dlnh_dτ, c2i, c2i_coeff),
				'θ': self.aux_ESC_retirees_θ(c2i, c2i_coeff),
				'eps':self.aux_ESC_retirees_eps(θ, c2i, c2i_coeff)}

	def aux_PEE_retirees_t(self, τ, θ, eps, ν, sol, sol_p, funcOfτ):
		c2i_coeff = self.aux_c2i_coeff(funcOfτ['si/s'], τ, θ, eps)
		c2i = self.aux_c2i_t(τ, funcOfτ['s[t-1]'], sol['h'], ν, c2i_coeff)
		return self.aux_PEE_retirees(θ, eps, funcOfτ['dln(h)/dτ'], c2i, c2i_coeff)

	def aux_PEE_retirees(self, θ, eps, dlnh_dτ, c2i, c2i_coeff):
		""" Contribution to political objective in PEE problem """
		return c2i**(1-1/self.db['ρ'])*((1-self.db['α'])*dlnh_dτ+((θ*self.aux_Prod.reshape(self.ni,1)+1-θ)*self.db['αr']/(1+self.db['γu']*eps/(1-self.db['γu'])))/c2i_coeff)

	def aux_ESC_retirees_eps(self, θ, c2i, c2i_coeff):
		return -c2i**(1-1/self.db['ρ']) * self.db['αr']*(θ*self.aux_Prod.reshape(self.ni,1)+1-θ)/c2i_coeff

	def aux_ESC_retirees_θ(self, c2i, c2i_coeff):
		return c2i**(1-1/self.db['ρ']) * (self.aux_Prod-1).reshape(self.ni,1) / c2i_coeff
	
	def aux_c2i_t(self, τ, s, h, ν, c2i_coeff, A = 1):
		return self.db['α']*A*ν*(s/ν)**(self.db['α'])*h**(1-self.db['α'])*c2i_coeff

	def aux_c2i_coeff(self, sSpread, τ, θ, eps):
		return sSpread + self.db['αr']*self.auxPen(τ,eps)*(θ*self.aux_Prod.reshape(self.ni,1)+1-θ)

	###### 6.2. HtM Retirees
	def aux_ESC_HtM_old(self, τ, eps, s, h, ν, dlnh_dτ):
		""" Contribution to political objective in ESC problem """
		c2u_coeff = self.aux_c2u_coeff(τ, eps, ν)
		c2u = self.aux_c2u_t(s, h, ν, c2u_coeff)
		return {'τ': self.aux_PEE_HtM_old(eps, dlnh_dτ, c2u, c2u_coeff),
				'eps': self.aux_ESC_HtM_old_eps(c2u, c2u_coeff)}

	def aux_PEE_HtM_old_t(self, τ, eps, ν, sol, sol_p, funcOfτ):
		""" Contribution to FOC for PEE only (not ESC)"""
		c2u_coeff = self.aux_c2u_coeff(τ, eps, ν)
		c2u = self.aux_c2u_t(funcOfτ['s[t-1]'], sol['h'], ν, c2u_coeff)
		return self.aux_PEE_HtM_old(eps, funcOfτ['dln(h)/dτ'], c2u, c2u_coeff)

	def aux_PEE_HtM_old(self, eps, dlnh_dτ, c2u, c2u_coeff):
		""" Contribution to political objective in PEE problem """
		return c2u**(1-1/self.db['ρ'])*((1-self.db['α'])*dlnh_dτ+(eps/(1+self.db['γu']*eps/(1-self.db['γu'])))/c2u_coeff)

	def aux_ESC_HtM_old_eps(self, c2u, c2u_coeff):
		return c2u**(1-1/self.db['ρ']) * 1/c2u_coeff

	def aux_c2u_t(self, s, h, ν, c2u_coeff, A = 1):
		return (1-self.db['α'])*A*ν*(s/ν)**(self.db['α'])*h**(1-self.db['α'])*c2u_coeff

	def aux_c2u_coeff(self, τ, eps, ν):
		return self.db['χ2']/ν+self.auxPen(τ,eps)*eps

	###### 6.3. Euler workers
	def aux_PEE_workers_t(self, θp, epsp, sol, sol_p, funcOfτ):
		k = self.db['αr']*(1/(1+self.db['γu']*epsp/(1-self.db['γu'])))*(1+self.db['ξ'])*sol_p['Γs']
		return self.aux_ĉ1i_t(sol_p['τ'], θp, epsp, sol['h'], sol_p['B'], sol_p['Γs'])**(1-1/self.db['ρ']) * (
					((1+self.db['ξ'])/self.db['ξ'])*funcOfτ['dln(h)/dτ']
				+	(sol_p['B']/(1+sol_p['B']))*(1-self.db['α'])*(sol_p['dln(h)/dln(s)']-1)*funcOfτ['dln(s)/dτ']
				+	k * (funcOfτ['∂τp/∂τ']+sol_p['τ']*funcOfτ['dln(Γs)/dτ']) / (self.aux_Prod.reshape(self.ni,1)+sol_p['τ']*k)
				)

	def aux_ESC_workers_t(self, sol_p, sol, funcOfτ, t):
		k = self.db['αr']*(1/(1+self.db['γu']*sol_p['eps']/(1-self.db['γu'])))*(1+self.db['ξ'])*sol_p['Γs']
		return self.aux_ĉ1i_t(sol_p['τ'], sol_p['θ'], sol_p['eps'], sol['h'], sol_p['B'], sol_p['Γs'])**(1-1/self.db['ρ']) * (
					((1+self.db['ξ'])/self.db['ξ'])*funcOfτ['dln(h)/dτ']
				+	(sol_p['B']/(1+sol_p['B']))*(1-self.db['α'])*(sol_p['dln(h)/dln(s)']-1)*funcOfτ['dln(s)/dτ']
				+	k * ((1-sol_p['θ'])*funcOfτ['∂τp/∂τ']-sol_p['τ']*funcOfτ['∂θ/∂τ']+(1-sol_p['θ'])*sol_p['τ']*(funcOfτ['dln(Γs)/dτ']-(self.db['γu']/(1-self.db['γu']))*(1/(1+self.db['γu']*sol_p['eps']/(1-self.db['γu'])))*funcOfτ['∂eps/∂τ']))/(self.aux_Prod.reshape(self.ni,1)+sol_p['τ']*(1-sol_p['θ'])*k)
				)

	def aux_ĉ1i_t(self, τp, θp, epsp, h, B, Γs):
		return (h**((1+self.db['ξ'])/self.db['ξ'])/(1+self.db['ξ'])) * (1+B)**(1/(self.db['ρ']-1)) * (self.aux_Prod.reshape(self.ni,1)+self.db['αr']*self.auxPen(τp, epsp)*(1-θp)*(1+self.db['ξ'])*Γs)

	###### 6.4. HtM young
	def aux_PEE_HtM_young_t(self, epsp, ν, νp, sol, sol_p, funcOfτ):
		k = epsp/(1+self.db['γu']*epsp/(1-self.db['γu']))
		return (self.aux_c1u_t(funcOfτ['s[t-1]'], sol['h'], ν)**(1-1/self.db['ρ'])*(1-self.db['α'])*funcOfτ['dln(h)/dτ']
				+self.db['βu']*self.aux_c2pu_t(sol_p['τ'], epsp, sol_p['s[t-1]'], sol_p['h'], νp)**(1-1/self.db['ρ'])*(
					(self.db['α']+(1-self.db['α'])*sol_p['dln(h)/dln(s)'])*funcOfτ['dln(s)/dτ']+funcOfτ['∂τp/∂τ']*k/(self.db['χ2']/νp+k*sol_p['τ'])
				))

	def aux_ESC_HtM_young_t(self, ν, νp, sol, sol_p, funcOfτ):
		k = 1/(1+self.db['γu']*sol_p['eps']/(1-self.db['γu']))
		return (self.aux_c1u_t(funcOfτ['s[t-1]'], sol['h'], ν)**(1-1/self.db['ρ'])*(1-self.db['α'])*funcOfτ['dln(h)/dτ']
				+self.db['βu']*self.aux_c2pu_t(sol_p['τ'], sol_p['eps'], sol_p['s[t-1]'], sol_p['h'], ν)**(1-1/self.db['ρ'])*(
					(self.db['α']+(1-self.db['α'])*sol_p['dln(h)/dln(s)'])*funcOfτ['dln(s)/dτ']
					+funcOfτ['∂τp/∂τ']*sol_p['eps']*k/(self.db['χ2']/νp+k*sol_p['eps']*sol_p['τ'])
					+funcOfτ['∂eps/∂τ']*sol_p['τ']*k**2/(self.db['χ2']/νp+k*sol_p['eps']*sol_p['τ'])
				))

	def aux_c1u_t(self,s,h,ν,A =1):
		return self.db['χ1']*(1-self.db['α'])*A*(s/ν)**(self.db['α'])*h**(1-self.db['α'])

	def aux_c2pu_t(self, τp, epsp, sp, hp, νp, A=1):
		return (1-self.db['α'])*A*(sp/νp)**(self.db['α'])*hp**(1-self.db['α'])*(self.db['χ2']/νp+epsp*self.auxPen(τp, epsp))

	###########################################################
	##########			7. Reporting module		 	###########
	###########################################################	
	def reportAll(self):
		""" Based on the solution report host of other relevant variables"""
		self.reportCoefficients()
		self.reportLevels()
		self.reportUtils()

	def reportCoefficients(self):
		""" Assumes that self.solve_PEE has been run and unloaded to the self.db """
		[self.db.__setitem__(k, getattr(self,'aux_'+k)) for k in ('Θhi','Θsi','Θc1i','Θc2i','Θc2pi','Θ̃c1i','Θc1u','Θc2u','Θc2pu')];

	def reportLevels(self):
		""" Assumes self.reportCoefficients has been run"""
		[self.db.__setitem__(k, getattr(self,'levels_'+k)) for k in ('c1i','c2i','c1u','c2u','̃c1i','c2pi','c2pu','hi','w','R')];

	def reportUtils(self):
		""" Assumes self.reportLevels has been run"""
		[self.db.__setitem__(k, getattr(self,'aux_'+k)(self.db)) for k in ('util1i','util1u','util2i','util2u', 'utilPol')];

	@property
	def aux_Θhi(self):
		return pd.DataFrame((self.db['Θh'].values * self.aux_Prod.reshape(self.ni,1)).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θsi(self):
		return pd.DataFrame((self.db['Θs'].values*self.savingsSpread(self.db['B[t+1]'].values.T, self.db['Γs'].values, self.db['τ[t+1]'].values, self.db['θ[t+1]'].values, self.db['eps[t+1]'].values)).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc1i(self):
		return pd.DataFrame((self.db['Θh'].values**((1+self.db['ξ'])/self.db['ξ'])*(self.aux_Prod.reshape(self.ni,1) * (self.db['B[t+1]'].values.T/(1+self.db['B[t+1]'].values.T))) + self.db['αr'] * self.auxPen(self.db['τ[t+1]'].values, self.db['eps[t+1]'].values) * (1-self.db['θ[t+1]'].values) * self.db['Θs'].values / (1+self.db['B[t+1]'].values.T)).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc2i(self):
		return pd.DataFrame((self.db['α']*self.db['A']*self.db['ν']*self.db['Θh'].values**(1-self.db['α'])*(self.savingsSpread(self.db['B'].values.T, self.db['Γs[t-1]'].values, self.db['τ'].values, self.db['θ'].values, self.db['eps'].values)+self.db['αr']*self.auxPen(self.db['τ'].values, self.db['eps'].values)*(self.db['θ'].values*self.aux_Prod.reshape(self.ni,1)+(1-self.db['θ'].values)))).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc2pi(self):
		return pd.DataFrame((self.db['α']*self.leadSym(self.db['A']*self.db['ν']*self.db['Θh'].values**(1-self.db['α']))*(self.savingsSpread(self.db['B[t+1]'].values.T, self.db['Γs'].values, self.db['τ[t+1]'].values, self.db['θ[t+1]'].values, self.db['eps[t+1]'].values)+self.db['αr']*self.auxPen(self.db['τ[t+1]'].values, self.db['eps[t+1]'].values)*(self.db['θ[t+1]'].values*self.aux_Prod.reshape(self.ni,1)+(1-self.db['θ[t+1]'].values)))).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θ̃c1i(self):
		return pd.DataFrame(((self.db['Θh'].values**((1+self.db['ξ'])/self.db['ξ']))/((1+self.db['ξ'])*(1+self.db['B[t+1]'].values.T))*(self.aux_Prod.reshape(self.ni,1) + self.db['αr']*self.auxPen(self.db['τ[t+1]'].values,self.db['eps[t+1]'].values)*(1-self.db['θ[t+1]'].values)*(1+self.db['ξ'])*self.db['Γs'].values)).T, 
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc1u(self):
		return pd.Series(self.db['χ1']*self.db['A']*(1-self.db['α'])*(self.db['Θh'].values**(1-self.db['α'])), 
				index = self.db['t'])

	@property
	def aux_Θc2u(self):
		return pd.Series((1-self.db['α'])*self.db['A']*self.db['ν']*(self.db['Θh'].values**(1-self.db['α']))*(self.db['χ2']/self.db['ν']+self.db['eps'].values*self.auxPen(self.db['τ'].values, self.db['eps'].values)),
				index = self.db['t'])

	@property
	def aux_Θc2pu(self):
		return pd.Series((1-self.db['α']) * self.leadSym(self.db['A'] * self.db['ν']) * (self.leadSym(self.db['Θh']).values**(1-self.db['α'])) * np.power(self.db['Θs'].values/self.leadSym(self.db['ν']), self.power_s) * (self.db['χ2']/self.leadSym(self.db['ν'])+self.db['eps[t+1]'].values*self.auxPen(self.db['τ[t+1]'].values, self.db['eps[t+1]'].values)),
				index = self.db['t'])

	def auxLevel(self, par):
		return (self.db['s[t-1]']/self.db['ν'])**par

	@property
	def levels_R(self):
		return self.aux_R(self.db['s[t-1]'], self.db['h'], self.db['ν'], A = self.db['A'])

	@property
	def levels_w(self):
		return (1-self.db['α'])*self.db['A']*self.auxLevel(self.db['α']) * self.db['h']**(-self.db['α'])

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
		return ( (1-self.db['γu'])*np.matmul(self.aux_util2i(db, Δo),self.ω2i*self.db['γ'])+self.db['γu']*self.ω2u*self.aux_util2u(db, Δou)
			+   self.db['ν']*( (1-self.db['γu'])*np.matmul(self.aux_util1i(db, Δy, Δy2),self.ω1i*self.db['γ'])+self.db['γu']*self.ω1u*self.aux_util1u(db, Δyu, Δyu2) )
			)

	###########################################################
	##########		8. Calibration Argentina		 ###########
	###########################################################	
	def argentinaCal_simple_PEE(self, τ0, s0, t0, sGrid, par0 = None, **kwargs):
		""" Match savings rate and PEE tax - not ESC"""
		sol = optimize.root(lambda x: self.argentinaCal_simple_PEE_eqs(x, τ0, s0, t0, sGrid, **kwargs), noneInit(par0, [self.db['ω'], self.db['β'][0]]))
		assert sol['success'], f""" Couldn't calibrate model"""
		return sol['x']

	def argentinaCal_simple_ESC(self, τ0, s0, t0, sGrid, par0 = None, **kwargs):
		""" Match savings rate and PEE tax - not ESC"""
		sol = optimize.root(lambda x: self.argentinaCal_simple_ESC_eqs(x, τ0, s0, t0, sGrid, **kwargs), noneInit(par0, [self.db['ω'], self.db['β'][0]]))
		assert sol['success'], f""" Couldn't calibrate model"""
		return sol['x']

	def argentinaCal_simple_ESC_eqs(self, x, τ0, s0, t0, sGrid, **kwargs):
		""" Target savings rate and policy level in t0 - solve with PEE """
		self.db['ω'] = x[0]
		self.db['β'], self.db['βu'] = np.full(self.ni, x[1]), x[1]
		sol = self.solve_ESCBound(sGrid, **kwargs)
		return np.hstack([sol['τ'].xs(t0)-τ0,
						  self.savingsRate(sol['Θs'].xs(t0), sol['Θh'].xs(t0))-s0])

	def argentinaCal_simple_PEE_eqs(self, x, τ0, s0, t0, sGrid, **kwargs):
		""" Target savings rate and policy level in t0 - solve with PEE """
		self.db['ω'] = x[0]
		self.db['β'], self.db['βu'] = np.full(self.ni, x[1]), x[1]
		self.db['eps'] = argentinaCalEps(self.db['θ'], x[1])
		self.db['eps[t+1]'] = self.leadSym(self.db['eps'])
		sol = self.solve_PEE(sGrid, **kwargs)
		return np.hstack([sol['τ'].xs(t0)-τ0,
						  self.savingsRate(sol['Θs'].xs(t0), sol['Θh'].xs(t0))-s0])


	def argentinaCalibrate_preReformEqs(self, x, τ0, s0, θ0, t0, sGrid, **kwargs):
		""" Calibrate model to reflect choice of τ, s, θ, epsilon"""
		self.db['ω'] = x[0]
		self.db['ωu'] = x[1]
		self.db['ωη'] = x[2]
		self.db['β'], self.db['βu'] = np.full(self.ni, x[3]), x[3]
		sol = self.solve_ESCBound(sGrid, **kwargs)
		return np.hstack([sol['τ'].xs(t0)-τ0,
						  sol['θ'].xs(t0)-θ0,
						  sol['eps'].xs(t0)-argentinaCalEps(θ0, x[3]),
						  self.savingsRate(sol['Θs'].xs(t0), sol['Θh'].xs(t0))-s0])

	def argentinaCalibrate_preReform(self, τ0, s0, θ0, t0, sGrid, par0 = None, **kwargs):
		sol = optimize.root(lambda x: self.argentinaCalibrate_preReformEqs(x, τ0, s0, θ0, t0, sGrid, **kwargs), noneInit(par0, [self.db['ω'], self.db['ωu'], self.db['ωη'], self.db['β'][0]]))
		assert sol['success'], f""" Couldn't calibrate model """
		return sol['x']