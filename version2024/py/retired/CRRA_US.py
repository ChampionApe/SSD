import numpy as np, pandas as pd, pyDbs, scipy
from pyDbs import is_iterable, SymMaps as sm
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

def x0vec(x0, i = None):
	return x0 if ((x0.ndim==1) | (i is None)) else x0[i]

def interpFixedPoint(ŝ, s):
	""" Interpolation of fixed point problem with two grids"""
	Δs = ŝ-s # distance from steady state
	changeSign = np.diff(np.sign(Δs))!=0
	s1,s2 = s[:-1][changeSign], s[1:][changeSign]
	ŝ1,ŝ2 = ŝ[:-1][changeSign], ŝ[1:][changeSign]
	return ŝ1+((ŝ2-ŝ1)*(ŝ1-s1))/(s2-s1-(ŝ2-ŝ1)) # lin. approximation

def aux_τMeshGrid(sGrid, τGrid_1d):
	return np.meshgrid(sGrid, τGrid_1d)[1]

class infHorizon:
	def __init__(self, ni = 11, T = 10, ngrid = 50, **kwargs):
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
		self.inferPars(**kwargs)
		self.initUS()
		self.x0 = self.defaultInitials # dictionary that is used to pass initial guesses for solutions
		self.f0 = self.defaultFunctions # dictionary that is used to specify what functions are called per default

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

	def j(self, x):
		""" return full vector of symbol x by combining 0 and i types"""
		return np.hstack([self.db[f'{x}0'], self.db[f'{x}i']])

	def inferPars(self, **kwargs):
		""" Split full vectors syntax 'xj' into subsets """
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

	### US specific functions:
	def initUS(self):
		self.db.update(self.initSC(self.US_eps(), 'eps')) # start at calibration target
		self.US_addEigenVectors()
		self.db['ηi'] = self.US_ηi()

		# Parameters that depends on ξ:
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
		# self.db['X0'] = (self.db['η0']**(1+self.db['ξ'])*(1-self.db['γ0'])/self.db['zη0'])**(1/self.db['ξ'])
	# Calibration given ξ and equilibrium level ht0:
	def US_X0(self, ht0):
		return ((1-self.db['γ0'])/(self.db['zη0']*(1-self.db['τ0'])*ht0))**(1/self.db['ξ'])*self.db['η0']**((1+self.db['ξ'])/self.db['ξ'])*(1-self.db['α'])*(1-self.db['τ0'])*(self.db['α']/self.db['R0'])**((self.db['α']/(1+self.db['α'])))

	# Targets in calibration:
	def US_eps(self):
		return (self.db['UShare0']/(1-self.db['UShare0']))*(self.db['p']/self.db['p̄'])
	def US_θ(self):
		h1, h2 = self.db['Xi'][0]**(self.db['ξ'])/self.db['ηi'][0]**(1+self.db['ξ']), self.db['Xi'][1]**(self.db['ξ'])/self.db['ηi'][1]**(1+self.db['ξ'])
		return (self.db['RR0']*h1-h2)/(1-self.db['γ0']-h2-self.db['RR0']*(1-self.db['γ0']-h1))
	
	@property
	def defaultParameters(self):
		return {'α': .5, 
				'A': np.ones(self.T), 
				'ν': np.ones(self.T),
				'ξ' : .1,
				'ρ': .5,
				'ω': 2,
				'τ0': .158,
				'RR0': 39.4/50.1,
				'UShare0': 3.4/15.8,
				'R0': 2.443,
				'kθ_l': 10,
				'kθ_u': 10,
				'keps_l': 10,
				'keps_u': 10,
				'kτ_l': 10,
				'kτ_u': 10}

	def addNamespaces(self):
		self.ns['ESC'] = sm(symbols = {k: self.db['t'] for k in ('τ', 'θ', 'eps')})
		self.ns['ESC[t]'] = sm(symbols = {x: self.db['sgrid'] for x in ('τ', 'θ', 'eps')}) # namespace used in policy function identification
		self.ns['EE'] = sm(symbols = {x: self.db['t'] for x in ('s','h','Γs')}) # solve EE given policy
		self.ns['ssVec'] = sm(symbols = {'B': pd.MultiIndex.from_product([self.db['i'], self.db['sgrid']]), 'Γs': self.db['sgrid']})
		self.ns['EV'] = sm(symbols = {f'transfer_{k}':  pd.MultiIndex.from_product([self.db['t'], self.db['i']]) for k in ('Y','O')}
									|{f'transfer_{k}U': self.db['t'] for k in ('Y','O')}
									|{'transfer_Pol': self.db['t']})
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
	def defaultInitials(self):
		""" Default initial values used in numerical problems below."""
		return {'ss': np.full(self.ni+1, 0.5),
				'ssVec': dict.fromkeys(range(self.ngrid), np.full(self.ni+1,.5)),
				'ssVec_v2': np.full(self.ns['ssVec'].len, .5),
				'ss_ESC': dict.fromkeys(range(self.ngrid), np.full(self.ni+1, .5)),
				'EE': np.full(self.ns['EE'].len, 0.2),
				'EE_polFunc': np.full(self.ns['EE'].len, 0.2),
				'EE_ESC': np.full(self.ns['EE'].len, 0.2),
				'PEE_t': np.full(self.ngrid, .2),
				'ESC_t': np.full((self.ngrid, 3), 0.2),
				'ESCB_t': np.full((self.ngrid, 3), 0.2),
				'PEEvec_t': np.full(self.ngrid, .2),
				'ESCvec_t': np.full(self.ns['ESC[t]'].len, 0.2),
				'ESCBvec_t': np.full(self.ns['ESC[t]'].len, 0.2),
				'steadyState_PEE': np.full(self.ngrid, .2),
				'steadyState_ESC': np.full(self.ns['ESC[t]'].len, .2),
				'steadyState_ESCB': np.full(self.ns['ESC[t]'].len, .2),
				'EV': np.full(self.ns['EV'].len, 0)}

	def getx0(self, solve, i =None, t= None):
		x0 = self.x0[solve] if t is None else self.x0[solve][t]
		return x0 if i is None else x0[i]

	@property
	def defaultFunctions(self):
		return {'SA_PEE': self.solve_PEEvec_t,
				'NK_PEE': self.solve_PEEvec_t,
				'PEE_t': self.solve_PEEvec_t,
				'SA_ESC': self.solve_ESC_t,
				'NK_ESC': self.solve_ESC_t,
				'ESC_t': self.solve_ESC_t,
				'SA_ESCB': self.solve_ESCB_t,
				'NK_ESCB': self.solve_ESCB_t,
				'ESCB_t': self.solve_ESCB_t}


	@property
	def ω20(self):
		return self.db['ω'] * self.db['p0'] * self.db['μ0']
	@property
	def ω2i(self):
		return self.db['ω'] * self.db['pi'] * self.db['μi']
	@property
	def ω2j(self):
		return self.db['ω'] * self.j('p') * self.j('μ')
	@property
	def ω10(self):
		return self.db['μ0']
	@property
	def ω1i(self):
		return self.db['μi']
	@property
	def ω1j(self):
		return self.j('μ')
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
		return np.power(self.db['ηi'], 1+self.db['ξ'])/np.power(self.db['Xi'], self.db['ξ'])

	def auxΓB1(self, B):
		return np.matmul(self.aux_Prod * self.db['γi'], B/(1+B))

	def auxΓB2(self,B):
		return np.matmul(self.db['γi'], 1/(1+B))

	def auxΓB3(self,B):
		return np.matmul(self.aux_Prod * self.db['γi'], B/((1+B)**2))

	def auxΓB4(self,B):
		return np.matmul(self.db['γi'], B/((1+B)**2))

	def auxPen(self, τ, eps, epsc = 1):
		return self.db['p']*τ/(eps*self.db['p̄']+epsc*self.db['p'])

	def aux_R(self, s, h, ν, A = 1):
		return self.db['α'] * A * (ν*h/s)**(1-self.db['α'])

	def aux_B(self, s, h, ν, A = 1):
		return self.db['βi'].reshape(self.ni,1)**self.db['ρ'] * (self.aux_R(s,h, ν, A = A)/self.db['p'])**(self.db['ρ']-1)

	def aux_Γs(self, Bp, τp, θp, epsp, epsc = 1):
		""" τp, Bp, θp, epsp are vectors of the same length"""
		return (1/(1+self.db['ξ']))*np.matmul(self.db['γi'] * self.aux_Prod, Bp/(1+Bp)) /(1+self.db['αr']*self.auxPen(τp, epsp, epsc = epsc)*(epsc*θp+(epsp+epsc*(1-θp)/(1-self.db['γ0']))*np.matmul(self.db['γi'], 1/(1+Bp))))

	def savingsSpread(self, Bp, Γs, τp, θp, epsp, epsc=1):
		""" τp is a vector of the same length as Bp"""
		return self.aux_Prod.reshape(self.ni,1) * Bp / ((1+Bp)*(1+self.db['ξ'])*Γs)-self.db['αr']*self.auxPen(τp, epsp, epsc=epsc)*(epsc*θp*self.aux_Prod.reshape(self.ni,1)+(epsp+epsc*(1-θp)/(1-self.db['γ0']))/(1+Bp))

	def aux_h_t(self, τ, τp, θp, epsp, Γs, s_, ν, epsc = 1):
		return ((1-self.db['α'])*(1-τ)/((1-self.db['αr']*epsc*self.auxPen(τp, epsp, epsc = epsc)*θp*Γs)))**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))*(s_/ν)**self.power_h

	def aux_Ω(self, Γs, τp, θp, epsp, epsc = 1):
		k = Γs*self.db['αr']*epsc*self.db['p']/(epsp*self.db['p̄']+epsc*self.db['p'])
		return k/(1-τp*k*θp)

	def aux_Ψ(self, Bp, τp, θp, epsp, epsc = 1):
		return (1-self.db['α'])*(self.db['ρ']-1)*(self.auxΓB3(Bp)/self.auxΓB1(Bp)+self.db['αr']*self.auxPen(τp, epsp, epsc=epsc)*(epsp+epsc*(1-θp)/(1-self.db['γ0']))*self.auxΓB4(Bp)/(1+self.db['αr']*self.auxPen(τp, epsp,epsc=epsc)*(epsc*θp+(epsp+epsc*(1-θp)/(1-self.db['γ0'])))*self.auxΓB2(Bp)))

	def aux_σ(self, Ω, Ψ, dlnh_dlns, τp, θp):
		return 1+(1+τp*θp*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*Ψ*(1-dlnh_dlns)

	### Economic equilibrium derivatives:
	def aux_derivatives(self, Ψ, σ, dlnh_dlns, τ):
		dlns_dτ  = -(1+self.db['ξ'])/((1+self.db['α']*self.db['ξ'])*(1-τ)*σ)
		dlnΓs_dτ = Ψ*(dlnh_dlns-1)*dlns_dτ
		return {'∂ln(s)/∂τ': dlns_dτ,
				'∂ln(Γs)/∂τ': dlnΓs_dτ,
				'∂ln(h)/∂τ': self.db['ξ']*(dlns_dτ-dlnΓs_dτ)/(1+self.db['ξ'])}

	def aux_laggedDerivatives_τ(self, Ω, Ψ, Bp, dlnh_dτp, τp, θp, epsp, epsc = 1):
		""" The derivative used here dlnh_dτp = ∂ln(h[t+1])/∂τ[t+1]"""
		k1 = θp*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])
		k2 = self.db['αr']*(epsc*θp+(epsp+epsc*(1-θp)/(1-self.db['γ0']))*self.auxΓB2(Bp))*self.db['p']/(epsp*self.db['p̄']+epsc*self.db['p'])
		k3 = k2/(1+k2*τp)
		dlns_dτ  = (1/(1+Ψ*(1+k1*τp)))*(k1+(1+k1*τp)*(Ψ*dlnh_dτp-k3))
		dlnΓs_dτ = Ψ*(dlnh_dτp-dlns_dτ)-k3
		return {'∂ln(s)/∂τ[t+1]': dlns_dτ,
				'∂ln(Γs)/∂τ[t+1]': dlnΓs_dτ,
				'∂ln(h)/∂τ[t+1]': self.db['ξ']*(dlns_dτ-dlnΓs_dτ)/(1+self.db['ξ'])}

	def aux_laggedDerivatives_θ(self, Ω, Ψ, σ, Bp, dlnh_dlns, τp, θp, epsp, epsc = 1):
		k1 = ((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*τp*Ω
		k2 = self.db['αr']*epsc*self.auxPen(τp, epsp, epsc = epsc)
		k3 = k2*(1-self.auxΓB2(Bp)/(1-self.db['γ0']))/(1+k2*(epsc*θp+(epsp+epsc*(1-θp)/(1-self.db['γ0']))*self.auxΓB2(Bp)))
		dlns_dθ = (k1-(1+k1*θp)*k3)/σ
		dlnΓs_dθ= Ψ*(dlnh_dlns-1)*dlns_dθ-k3
		return {'∂ln(s)/∂θ[t+1]': dlns_dθ, 
				'∂ln(Γs)/∂θ[t+1]': dlnΓs_dθ, 
				'∂ln(h)/∂θ[t+1]': self.db['ξ']*(dlns_dθ-dlnΓs_dθ)/(1+self.db['ξ'])}

	def aux_laggedDerivatives_eps(self, Ω, Ψ, σ, Bp, dlnh_dlns, τp, θp, epsp, epsc = 1):
		k1 = ((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*τp*Ω*θp
		k2 = self.db['αr']*self.auxPen(τp, epsp, epsc = epsc)
		k3 = epsc*θp*self.db['p̄']+(epsc*self.db['p̄']*(1-θp)/(1-self.db['γ0'])-epsc*self.db['p'])*self.auxΓB2(Bp)
		k4 = epsc*θp+(epsp+epsc*(1-θp)/(1-self.db['γ0']))*self.auxΓB2(Bp)
		k5 = (k2/(epsp*self.db['p̄']+epsc*self.db['p']))*k3/(1+k2*k4)
		dlns_deps = (1/σ)*((1+k1)*k5-k1*self.db['p̄']/(epsp*self.db['p̄']+epsc*self.db['p']))
		dlnΓs_deps = Ψ*dlns_deps*(dlnh_dlns-1)+k5
		return {'∂ln(s)/∂eps[t+1]': dlns_deps,
				'∂ln(Γs)/∂eps[t+1]': dlnΓs_deps,
				'∂ln(h)/∂eps[t+1]': self.db['ξ']*(dlns_deps-dlnΓs_deps)/(1+self.db['ξ'])}

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
	def solve_PEE(self, s, gridOption = 'resample', s0 = None, returnPols = False, sskwargs = None, x0_t = False, tkwargs = None):
		policy = self.solve_PEE_policy(s, gridOption=gridOption, sskwargs = sskwargs, x0_t = x0_t, tkwargs = tkwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_polFunc(policy, self.db['θ'].values, self.db['eps'].values, s0)
		return (self.reportMain_PEE(sols), policy) if returnPols else self.reportMain_PEE(sols)

	def solve_ESC(self, s, gridOption = 'resample', s0 = None, returnPols = False, sskwargs = None, x0_t = False, tkwargs = None):
		""" ESC with time dependent functions; terminal steady state"""
		policy = self.solve_ESC_policy(s, gridOption=gridOption, sskwargs = sskwargs, x0_t = x0_t, tkwargs = tkwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_ESC(policy, s0)
		return (self.reportMain_ESC(sols), policy) if returnPols else self.reportMain_ESC(sols)

	def solve_ESCB(self, s, gridOption = 'resample', s0 = None, returnPols = False, sskwargs = None, x0_t = False, tkwargs = None):
		""" ESCB with time dependent functions; terminal steady state"""
		policy = self.solve_ESCB_policy(s, gridOption=gridOption, sskwargs = sskwargs, x0_t = x0_t, tkwargs = tkwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_ESC(policy, s0)
		return (self.reportMain_ESC(sols), policy) if returnPols else self.reportMain_ESC(sols)

	def solve_PEE_ss(self, s, s0 = None, returnPols = False, ite = True, sskwargs = None, tkwargs = None):
		""" PEE steady state approxmations"""
		policy = self.solve_PEE_policy_ss(s, ite = ite, sskwargs = sskwargs, tkwargs = tkwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_polFunc(policy, self.db['θ'].values, self.db['eps'].values, s0)
		return (self.reportMain_PEE(sols), policy) if returnPols else self.reportMain_PEE(sols)

	def solve_ESC_ss(self, s, s0 = None, returnPols = False, ite = True, sskwargs = None, tkwargs = None):
		""" PEE steady state approxmations"""
		policy = self.solve_ESC_policy_ss(s, ite = ite, sskwargs = sskwargs, tkwargs = tkwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_ESC(policy, s0)
		return (self.reportMain_ESC(sols), policy) if returnPols else self.reportMain_ESC(sols)

	def solve_ESCB_ss(self, s, s0 = None, returnPols = False, ite = True, sskwargs = None, tkwargs = None):
		""" PEE steady state approxmations"""
		policy = self.solve_ESCB_policy_ss(s, ite = ite, sskwargs = sskwargs, tkwargs = tkwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_ESC(policy, s0)
		return (self.reportMain_ESC(sols), policy) if returnPols else self.reportMain_ESC(sols)

	def reportMain_PEE(self, sol):
		sol['B'] = pd.DataFrame(self.aux_B(sol['s[t-1]'].values, sol['h'].values, self.db['ν']).T, columns = self.db['i'], index = self.db['t'])
		sol['Γs[t-1]'] = pd.Series(self.aux_Γs(sol['B'].values.T, sol['τ'].values, self.db['θ'].values, self.db['eps'].values), index = self.db['t'])
		sol['B[t+1]'] = pd.DataFrame(self.aux_B(sol['s'].values, sol['h[t+1]'].values, self.leadSym(self.db['ν'])).T, columns = self.db['i'], index = self.db['t'])
		sol['τ[t+1]'] = self.leadSym(sol['τ'])
		sol['R'] = pd.Series(self.aux_R(sol['s[t-1]'].values, sol['h'].values, self.db['ν']), index = self.db['t'])
		return sol

	def reportMain_ESC(self, sol):
		sol['B'] = pd.DataFrame(self.aux_B(sol['s[t-1]'].values, sol['h'].values, self.db['ν']).T, columns = self.db['i'], index = self.db['t'])
		sol['Γs[t-1]'] = pd.Series(self.aux_Γs(sol['B'].values.T, sol['τ'].values, sol['θ'].values, sol['eps'].values), index = self.db['t'])
		sol['B[t+1]'] = pd.DataFrame(self.aux_B(sol['s'].values, sol['h[t+1]'].values, self.leadSym(self.db['ν'])).T, columns = self.db['i'], index = self.db['t'])
		sol.update({f'{k}[t+1]': self.leadSym(sol[f'{k}']) for k in ('τ', 'θ','eps')})
		sol['R'] = pd.Series(self.aux_R(sol['s[t-1]'].values, sol['h'].values, self.db['ν']), index = self.db['t'])
		return sol

	###### 1.2. Collection of policy functions over time 
	def solve_PEE_policy(self, s, gridOption = 'resample', sskwargs = None, x0_t = False, tkwargs = None):
		""" PEE path with time dependent policy functions; terminal steady state function"""
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1] = self.solve_steadyState_PEE(self.db['θ'].values[-1], self.db['eps'].values[-1], self.db['ν'][-1], s, path =False, **noneInit(sskwargs, {}))
		if gridOption == 'resample':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.resampleSolution(self.f0['PEE_t'](self.db['θ'].values[t], self.db['eps'].values[t], self.db['ν'][t], 
																  self.db['θ'].values[t+1], self.db['eps'].values[t+1], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs, {})), s)
		elif gridOption == 'clean':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.cleanSolution(self.f0['PEE_t'](self.db['θ'].values[t], self.db['eps'].values[t], self.db['ν'][t], 
															  self.db['θ'].values[t+1], self.db['eps'].values[t+1], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs, {})), s)
		else:
			sols[t] = self.f0['PEE_t'](self.db['θ'].values[t], self.db['eps'].values[t], self.db['ν'][t], 
									   self.db['θ'].values[t+1], self.db['eps'].values[t+1], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs, {}))
		return sols

	def solve_ESC_policy(self, s, gridOption = 'resample', sskwargs = None, x0_t = False, tkwargs = None):
		""" ESC path with time dependent policy functions; terminal steady state function """
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1]= self.solve_steadyState_ESC(self.db['ν'][-1], s, path = False, **noneInit(sskwargs, {}))
		if gridOption == 'resample':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.resampleSolution(self.f0['ESC_t'](self.db['ν'][t], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs,{})), s)
		elif gridOption == 'clean':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.cleanSolution(self.f0['ESC_t'](self.db['ν'][t], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs, {})), s)
		else:
			sols[t] = self.f0['ESC_t'](self.db['ν'][t], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs, {}))
		return sols

	def solve_ESCB_policy(self, s, gridOption = 'resample', sskwargs = None, x0_t =False, tkwargs = None):
		""" ESCB path with time dependent policy functions; terminal steady state function """
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1]= self.solve_steadyState_ESCB(self.db['ν'][-1], s, path = False, **noneInit(sskwargs, {}))
		if gridOption == 'resample':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.resampleSolution(self.f0['ESCB_t'](self.db['ν'][t], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs,{})), s)
		elif gridOption == 'clean':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.cleanSolution(self.f0['ESCB_t'](self.db['ν'][t], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs,{})), s)
		else:
			sols[t] = self.f0['ESCB_t'](self.db['ν'][t], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs,{}))
		return sols

	def solve_PEE_policy_ss(self, s, ite = True, sskwargs = None, tkwargs = None):
		""" PEE path from steady state policy functions"""
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1] = self.solve_steadyState_PEE(self.db['θ'].values[-1], self.db['eps'].values[-1], self.db['ν'][-1], s, path =False, **noneInit(sskwargs, {}))
		if ite:
			self.x0['steadyState_PEE'] = sols[self.T-1]['τ'] # update initial guess
		for t in range(self.T-2,-1,-1):
			sols[t] = self.solve_steadyState_PEE(self.db['θ'].values[t], self.db['eps'].values[t], self.db['ν'][t], s, path = False, **noneInit(tkwargs, {}))
			if ite:
				self.x0['steadyState_PEE'] = sols[t]['τ'] # update initial guess
		return sols

	def solve_ESC_policy_ss(self, s, ite = True, sskwargs = None, tkwargs = None):
		""" ESC path from steady state policy functions """
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1] = self.solve_steadyState_ESC(self.db['ν'][-1], s, path = False, **noneInit(sskwargs,{}))
		if ite:
			self.x0['steadyState_ESC'] = np.hstack([sols[self.T-1][k] for k in self.ns['ESC[t]'].symbols])
		for t in range(self.T-2,-1,-1):
			sols[t] = self.solve_steadyState_ESC(self.db['ν'][t], s, path = False, **noneInit(tkwargs,{}))
			if ite:
				self.x0['steadyState_ESC'] = np.hstack([sols[t][k] for k in self.ns['ESC[t]'].symbols])
		return sols

	def solve_ESCB_policy_ss(self, s, ite = True, sskwargs = None, tkwargs = None):
		""" ESC path from steady state policy functions """
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1] = self.solve_steadyState_ESCB(self.db['ν'][-1], s, path = False, **noneInit(sskwargs,{}))
		if ite:
			self.x0['steadyState_ESCB'] = np.hstack([sols[self.T-1][k] for k in self.ns['ESC[t]'].symbols])
		for t in range(self.T-2,-1,-1):
			sols[t] = self.solve_steadyState_ESCB(self.db['ν'][t], s, path = False, **noneInit(tkwargs,{}))
			if ite:
				self.x0['steadyState_ESCB'] = np.hstack([sols[t][k] for k in self.ns['ESC[t]'].symbols])
		return sols

	###########################################################
	##########		2. Steady state functions 		###########
	###########################################################

	########## 2.1. Steady state, scalar version
	def solve_ss(self, τ, θ, eps, ν, A = 1, x0_ss = None, **kwargs):
		""" Given policy and worker-to-retiree ratio, solve for steady state B and Γs; from this compute  's' as well and report"""
		sol = optimize.root(lambda x: self.ss_eqs(x[1:], x[0], τ, θ, eps, ν), noneInit(x0_ss, self.getx0('ss')), **kwargs)
		assert sol['success'], f""" Could not identify steady state (self.solve_ss) with parameter inputs: 
		τ: {τ}, θ: {θ}, ε: {eps}, ν: {ν}"""
		return self.report_ss(sol['x'], τ, θ, eps, ν, A = A)

	def report_ss(self, sol, τ, θ, eps, ν, A = 1):
		return {'B': sol[1:], 'Γs': sol[0], 's': np.nan_to_num(self.aux_ss_savings(sol[0], τ, θ, eps, ν, A = A), nan = 0)}

	def ss_B_eq(self, B, Γs, τ, θ, eps, ν, epsc = 1):
		""" Requirement = 0 for B to be in steady state - this does not return ss level of B"""
		if self.db['ρ']<1:
			return self.db['βi']**(self.db['ρ']/(1-self.db['ρ']))*(1-self.db['α'])*(1-τ)-B**(1/(1-self.db['ρ']))*(ν*(self.db['α']-(1-self.db['α'])*epsc*self.auxPen(τ, eps, epsc = epsc)*θ*Γs))/Γs
		elif self.db['ρ']>1:
			return B**(1/(self.db['ρ']-1))*(1-self.db['α'])*(1-τ)-self.db['βi']**(self.db['ρ']/(self.db['ρ']-1))*(ν*(self.db['α']-(1-self.db['α'])*eps**self.auxPen(τ, eps, epsc = epsc)*θ*Γs))/Γs

	def ss_Γs_eq(self, B, Γs, τ, θ, eps, ν, epsc = 1):
		""" Requirement = 0 for Γs to be in steady state - this does not return ss level of Γs """
		return self.auxΓB1(B)-Γs*(1+self.db['ξ'])*(1+self.db['αr']*self.auxPen(τ,eps, epsc = epsc)*(epsc*θ+(eps+epsc*(1-θ)/(1-self.db['γ0']))*self.auxΓB2(B)))

	def ss_eqs(self, B, Γs, τ, θ, eps, ν, epsc = 1):
		""" Stacked requirements = 0 for steady state B and Γs """
		return np.hstack([self.ss_B_eq(B, Γs, τ, θ, eps, ν, epsc=epsc), self.ss_Γs_eq(B,Γs,τ,θ,eps,ν,epsc=epsc)])

	def aux_ss_savings(self, Γs, τ, θ, eps, ν, A = 1, epsc = 1):
		return ( ((1-self.db['α'])*(1-τ)*A/(1-self.db['αr']*epsc*self.auxPen(τ, eps, epsc = epsc)*θ*Γs))**(1+self.db['ξ'])*Γs**(1+self.db['α']*self.db['ξ'])/(ν**(self.db['α']*(1+self.db['ξ']))) )**(1/(1-self.db['α']))

	def aux_ss_dlnh_ds(self, Ω, Ψ, τ, θ, eps):
		""" get steady state level of ∂ln(h)/∂ln(s) given solution to solve_ss (ss) and parameters """
		a = Ψ*(1+τ*θ*Ω*(1+self.db['ξ'])/(1+self.db['ξ']*self.db['α']))
		b = -(1+Ψ*(1+self.db['α']*self.db['ξ']+τ*θ*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])))
		c = self.db['α']*self.db['ξ']*(1+Ψ)
		return (-b-np.sqrt(b**2-4*a*c))/(2*a)

	########## 2.2. τ is a vector of inputs:
	### NB: This is seemingly faster than vectorized function self.solve_ssVec_v2,
	### 	because fsolve does not handle data in a sparse way.
	def solve_ssVec(self, τ, θ, eps, ν, A=1, x0_ssVec = None, ite_ssVec = True, **kwargs):
		""" solve_ss with grid of taxes """
		sol = optimize.root(lambda x: self.ss_eqs(x[1:], x[0], τ[0], θ, eps, ν), noneInit(x0_ssVec, self.getx0('ssVec', i= 0)),**kwargs)
		B  = np.empty((self.ni, self.ngrid))
		Γs = np.empty(self.ngrid)
		Γs[0], B[:,0] = sol['x'][0], sol['x'][1:]
		for i in range(1,self.ngrid):
			sol = optimize.root(lambda x: self.ss_eqs(x[1:], x[0], τ[i], θ, eps, ν), sol['x'] if ite_ssVec else self.getx0('ssVec',i=i), **kwargs)
			Γs[i], B[:,i] = sol['x'][0], sol['x'][1:]
		return {'B': B, 'Γs': Γs, 's': self.aux_ss_savings(Γs, τ, θ, eps, ν, A = A)}

	def solve_ssVec_v2(self, τ, θ, eps, ν, A = 1, x0_ssVec_v2 = None, **kwargs):
		""" solve solve_ss with grid of taxes, vectorized solution""" 
		sol = optimize.root(lambda x: self.ssVec_eqs(self.get(x, 'B', ns = 'ssVec').unstack(level='sgrid').values, self(x, 'Γs', ns = 'ssVec'), τ, θ, eps,ν),
							noneInit(x0_ssVec_v2, self.getx0('ssVec_v2')), **kwargs)
		assert sol['success'], f""" Could not identify steady state (self.solve_ssVec_v2) with parameter inputs: 
		τ: {τ}, θ: {θ}, ε: {eps}, ν: {ν}"""
		unload = self.ns['ssVec'].unloadSol(sol['x'])
		return {'B': unload['B'].unstack('sgrid').values, 'Γs': unload['Γs'].values, 's': self.aux_ss_savings(unload['Γs'].values, τ, θ, eps, ν, A = A)}

	def ss_Bvec_eq(self, B, Γs, τ, θ, eps, ν, epsc = 1):
		""" Requirement = 0 for B to be in steady state - this does not return ss level of B"""
		if self.db['ρ']<1:
			return self.db['βi'].reshape(self.ni,1)**(self.db['ρ']/(1-self.db['ρ']))*(1-self.db['α'])*(1-τ)-B**(1/(1-self.db['ρ']))*(ν*(self.db['α']-(1-self.db['α'])*epsc*self.auxPen(τ, eps, epsc = epsc)*θ*Γs))/Γs
		elif self.db['ρ']>1:
			return B**(1/(self.db['ρ']-1))*(1-self.db['α'])*(1-τ)-self.db['βi'].reshape(self.ni,1)**(self.db['ρ']/(self.db['ρ']-1))*(ν*(self.db['α']-(1-self.db['α'])*epsc*self.auxPen(τ, eps, epsc = epsc)*θ*Γs))/Γs

	def ssVec_eqs(self, B, Γs, τ, θ, eps, ν, epsc = 1):
		""" Stacked requirements = 0 for steady state B and Γs """
		return np.hstack([self.ss_Bvec_eq(B, Γs, τ, θ, eps, ν, epsc = epsc).reshape(self.ni*self.ngrid), self.ss_Γs_eq(B,Γs,τ,θ,eps,ν, epsc = epsc)])

	########## 2.3. τ, θ, eps are vectors of inputs (same length) 
	def solve_ss_ESC(self, τ, θ, eps, ν, A =1, x0_ss_ESC = None, ite_ss_ESC = True,**kwargs):
		""" solve_ss with grid of taxes and system characteristics """
		sol = optimize.root(lambda x: self.ss_eqs(x[1:], x[0], τ[0], θ[0], eps[0], ν), noneInit(x0_ss_ESC, self.getx0('ss_ESC',i=0)),**kwargs)
		B  = np.empty((self.ni, self.ngrid))
		Γs = np.empty(self.ngrid)
		Γs[0], B[:,0] = sol['x'][0], sol['x'][1:]
		for i in range(1,self.ngrid):
			sol = optimize.root(lambda x: self.ss_eqs(x[1:], x[0], τ[i], θ[i], eps[i], ν), sol['x'] if ite_ss_ESC else self.getx0('ss_ESC',i=i), **kwargs)
			Γs[i], B[:,i] = sol['x'][0], sol['x'][1:]
		return {'B': B, 'Γs': Γs, 's': self.aux_ss_savings(Γs, τ, θ, eps, ν, A = A)}

	###########################################################
	##########		3. Economic equilibrium 		###########
	###########################################################

	### 3.1. Economic Equilibrium given policies
	def solve_EE(self, τ, θ, eps, s0, x0_EE = None):
		τp, θp, epsp = self.leadSym(τ), self.leadSym(θ), self.leadSym(eps)
		sol = optimize.root(lambda x: self.EE_eqs(x, τ, θ, eps, τp, θp, epsp, s0), noneInit(x0_EE, self.getx0('EE')))
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
	def solve_EE_polFunc(self, gridsol, θ, eps, s0, x0_EE_polFunc = None):
		θp, epsp = self.leadSym(θ), self.leadSym(eps)
		policyFunction = self.aux_vecPolFunction(gridsol)
		sol = optimize.root(lambda x: self.EE_pol_eqs(x, policyFunction, θ, eps, θp, epsp, s0), noneInit(x0_EE_polFunc, self.getx0('EE_polFunc')))
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
	def solve_EE_ESC(self, gridsol, s0, x0_EE_ESC = None):
		τf = self.aux_vecPolFunction(gridsol, y = 'τ')
		θf = self.aux_vecPolFunction(gridsol, y = 'θ')
		epsf = self.aux_vecPolFunction(gridsol, y = 'eps')
		sol = optimize.root(lambda x: self.EE_ESC_eqs(x, τf, θf, epsf, s0), noneInit(x0_EE_ESC, self.getx0('EE_ESC')))
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
	def steadyState_solp(self, τ, θ, eps, ν, s, **kwargs):
		""" This establishes solution in t = T+1 - using loop and scalar optimization. """
		ss = self.solve_ssVec(τ, θ, eps, ν, **kwargs)
		return self.steadyState_solp_aux(ss, τ, θ, eps, ν, s)

	def steadyState_solp_aux(self, ss, τ, θ, eps, ν,s):
		sol = {'τ': τ, 'θ': θ, 'eps': eps, 's[t-1]': s, 's':ss['s']}
		sol['τ_notBound'] = τ
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
	def steadyState_solp_ESC(self, τ, θ, eps, ν, s, **kwargs):
		""" Looping through scalar solutions """
		ss = self.solve_ss_ESC(τ, θ, eps, ν, **kwargs) # solve steady state on grids
		sol = self.steadyState_solp_aux(ss, τ, θ, eps, ν, s)
		sol.update({f'∂{k}/∂s': np.gradient(sol[f'{k}'], sol['s[t-1]']) for k in ('θ','eps')})
		return sol

	def steadyState_solp_ESCB(self, τ, θ, eps, ν,s, **kwargs):
		sol = self.steadyState_solp_ESC(τ, θ, eps, ν,s, **kwargs)
		sol['θ_notBound']= sol['θ']
		sol['eps_notBound'] = sol['eps']
		return sol

	### 4.3. Iterate until convergence in policy function, PEE methods
	def solve_steadyState_PEE(self, θ, eps, ν, s, tol_SA = 1e-5, kwargs_SA = None, tol_NK = 1e-5, path = True, **kwargs):
		""" Solve for the steady state PEE policy."""
		if tol_SA:
			sol = self.steadyState_SA_PEE(self.getx0('steadyState_PEE'), θ, eps, ν, s, tol = tol_SA, **noneInit(kwargs_SA, {}))
			if tol_SA>tol_NK:
				fullSol = optimize.root(lambda x: self.steadyState_eq_PEE(x, θ, eps, ν, s), sol, tol = tol_NK, **kwargs)
				assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_PEE)"""
				sol = fullSol['x']
		else:
			fullSol = optimize.root(lambda x: self.steadyState_eq_PEE(x, θ, eps, ν, s), self.getx0('steadyState_PEE'), tol = tol_NK, **kwargs)
			assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_PEE)"""
			sol = fullSol['x']
		return self.report_steadyState_PEE(sol, θ, eps, ν, s, path = path)

	def steadyState_SA_PEE(self, τ, θ, eps, ν, s, tol = 1e-5, iterMax = 100, kwargs_SA0 = None, **kwargs):
		""" Successive approximation of the policy function until tolerance is reached. """
		i = 0
		sol_p = self.steadyState_solp(τ, θ, eps, ν, s, **kwargs) # initial solution guess
		sol = self.resampleSolution(self.f0['SA_PEE'](θ, eps, ν, θ, eps, ν, sol_p, **noneInit(kwargs_SA0, {})), s)
		while i<iterMax:
			if max(abs(sol['τ']-sol_p['τ']))<tol:
				break
			else:
				i += 1
				sol_p = self.steadyState_solp(sol['τ'], θ, eps, ν, s)
				sol = self.resampleSolution(self.f0['SA_PEE'](θ, eps, ν, θ, eps, ν, sol_p), s)
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
			sols = optimize.root(lambda x: self.solve_ss(np.interp(x, policy['s[t-1]'], policy['τ']), θ, eps,ν)['s']-np.interp(x, policy['s[t-1]'], policy['s']), x0 = sInterp)
			assert sols['success'], f""" Couldn't identify steady state (self.report_steadyState_PEE)"""
			sol = self.solve_ss(np.interp(sols['x'], policy['s[t-1]'], policy['τ']), θ, eps,ν)
			sol.update({k: interpSol(sol['s'], policy['s[t-1]'], policy[k]) for k in policy if k not in sol})
			return sol, policy
		else:
			return policy

	def steadyState_eq_PEE(self, τ, θ, eps, ν, s, **kwargs):
		sol_p = self.steadyState_solp(τ, θ, eps, ν, s, **kwargs)
		sol = self.resampleSolution(self.f0['NK_PEE'](θ, eps, ν, θ, eps, ν, sol_p), s)
		return sol['τ']-sol_p['τ']

	### 4.4. Iterate until convergence in policy functions, ESC
	def solve_steadyState_ESC(self, ν, s, tol_SA = 1e-5, kwargs_SA = None, tol_NK = 1e-5, path = True, **kwargs):
		""" Solve for the steady state ESC policy. """
		if tol_SA:
			sol = self.steadyState_SA_ESC(self.getx0('steadyState_ESC'), ν, s, tol = tol_SA, **noneInit(kwargs_SA,{}))
			if tol_SA>tol_NK:
				fullSol = optimize.root(lambda x: self.steadyState_eq_ESC(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s), sol, tol = tol_NK, **kwargs)
				assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_ESC)"""
				sol = fullSol['x']
		else:
			fullSol = optimize.root(lambda x: self.steadyState_eq_ESC(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s), self.getx0('steadyState_ESC'), tol = tol_NK, **kwargs)
			assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_ESC)"""
			sol = fullSol['x']
		return self.report_steadyState_ESC(self(sol,'τ'), self(sol,'θ'), self(sol,'eps'), ν, s, path = path)

	def steadyState_SA_ESC(self, x, ν, s, tol = 1e-5, iterMax = 100, kwargs_SA0 = None, **kwargs):
		""" Successive approximation of the policy function until tolerance is reached"""
		i = 0
		sol_p = self.steadyState_solp_ESC(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s, **kwargs) # initial solution guess
		sol = self.resampleSolution(self.f0['SA_ESC'](ν, ν, sol_p, **noneInit(kwargs_SA0, {})), s)
		while i<iterMax:
			if max(abs(np.hstack([sol[k]-sol_p[k] for k in ('τ', 'θ', 'eps')])))<tol:
				break
			else:
				i += 1
				sol_p = self.steadyState_solp_ESC(sol['τ'], sol['θ'], sol['eps'], ν, s)
				sol = self.resampleSolution(self.f0['SA_ESC'](ν, ν, sol_p), s)
		if i == iterMax:
			raise ValueError('iterMax in successive approximations without reaching exit tolerance.')
		else:
			for k in ('τ','θ','eps'):
				if max(sol[k])>1:
					print(f"""WARNING: It looks like the solution has values >1 for variable {k}. This may cause unexpected behavior in certain functions.""")
				elif min(sol[k])<0:
					print(f"""WARNING: It looks like the solution has values <0 for variable {k}. This may cause unexpected behavior in certain functions.""")
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

	def steadyState_eq_ESC(self, τ, θ, eps, ν, s, **kwargs):
		sol_p = self.steadyState_solp_ESC(τ, θ, eps, ν, s, **kwargs)
		sol = self.resampleSolution(self.f0['NK_ESC'](ν, ν, sol_p),s)
		return np.hstack([(sol[k]-sol_p[k]) for k in ('τ', 'θ', 'eps')])

	### 4.5. Iterate until convergence in policy functions, ESC, bounded version
	def solve_steadyState_ESCB(self, ν, s, tol_SA = 1e-6, kwargs_SA = None, tol_NK = 1e-6, path = True, **kwargs):
		""" Solve for the steady state ESC policy. 
			tol_SA::: tolerance for when to quit successive approximation (SA) approach. If None then this is skipped.
			tol_NK::: tolerance for when to quit newton-kantorovich (NK) iterations. If tol_NK>= tol_SA the NK step is ignored."""
		if tol_SA:
			sol = self.steadyState_SA_ESCB(self.getx0('steadyState_ESCB'), ν, s, tol = tol_SA,**noneInit(kwargs_SA,{}))
			if tol_SA>tol_NK:
				fullSol = optimize.root(lambda x: self.steadyState_eq_ESCB(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s), sol, tol = tol_NK, **kwargs)
				assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_ESCB)"""
				sol = fullSol['x']
		else:
			fullSol = optimize.root(lambda x: self.steadyState_eq_ESCB(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s), self.getx0('steadyState_ESCB'), tol = tol_NK, **kwargs)
			assert fullSol['success'], f""" Couldn't identify steady state policy (self.solve_steadyState_ESCBound)"""
			sol = fullSol['x']
		return self.report_steadyState_ESCB(self(sol,'τ'), self(sol,'θ'), self(sol,'eps'), ν, s, path = path)

	def steadyState_SA_ESCB(self, x, ν, s, tol = 1e-5, iterMax = 100, kwargs_SA0 = None, **kwargs):
		""" Successive approximation of the policy function until tolerance is reached"""
		i = 0
		sol_p = self.steadyState_solp_ESCB(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, s, **kwargs) # initial solution guess
		sol = self.resampleSolution(self.f0['SA_ESCB'](ν, ν, sol_p, **noneInit(kwargs_SA0, {})), s)
		while i<iterMax:
			if max(abs(np.hstack([sol[k]-sol_p[k] for k in ('τ', 'θ', 'eps')])))<tol:
				break
			else:
				i += 1
				sol_p = self.steadyState_solp_ESCB(sol['τ'], sol['θ'], sol['eps'], ν, s)
				sol = self.resampleSolution(self.f0['SA_ESCB'](ν, ν, sol_p), s)
		if i == iterMax:
			raise ValueError('iterMax in successive approximations without reaching exit tolerance.')
		else:
			return np.hstack([sol['τ'], sol['θ'], sol['eps']])

	def report_steadyState_ESCB(self, τ, θ, eps, ν, s, path = True):
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

	def steadyState_eq_ESCB(self, τ, θ, eps, ν, s, **kwargs):
		sol_p = self.steadyState_solp_ESCB(τ, θ, eps, ν, s, **kwargs)
		sol = self.resampleSolution(self.f0['NK_ESCB'](ν, ν, sol_p),s)
		return np.hstack([(sol[k]-sol_p[k]) for k in ('τ', 'θ', 'eps')])


	###########################################################
	##########			5. Policy function  		###########
	###########################################################

	def aux_soli(self, sol, i):
		sol_i = {k: sol[k][i] if (k != 'B') and is_iterable(sol[k]) else sol[k] for k in sol}
		if 'B' in sol:
			sol_i['B'] = sol['B'][:,i]
		return sol_i

	### 5.1. PEE: Given a solution from t+1, solve for the
	###		 solution for period t. 
	def solve_PEEvec_t(self, θ, eps, ν, θp, epsp, νp, sol_p, ite_PEEvec_t = True, t = None, **kwargs):
		""" Use ite_PEE_t = True to use sol_p['τ'] as initial value, 
			use 't' to indicate if the default initial value from self.x0 is a dictionary and it should rely on the t'th entry """
		sol = self.PEE_precomputations(θp, epsp, ν, sol_p)
		τ = optimize.root(lambda τ: self.PEE_polObjVec_t(τ, θ, eps, ν, θp, epsp, νp, sol, sol_p), sol_p['τ_notBound'] if ite_PEEvec_t else self.getx0('PEEvec_t', t=t), **kwargs)
		assert τ['success'], f"""Could not identify PEE solution (self.solve_PEEvec_t) with parameters:
		θ: {θ}, ε: {eps}, ν: {ν}, θ[t+1]: {θp}, ε[t+1]: {epsp}, ν[t+1]: {νp}, and solution from t+1 with taxes
		τ[t+1]: {sol_p['τ']}"""
		return self.report_PEE_t(τ['x'], θ, eps, ν, sol, sol_p)

	def solve_PEE_t(self, θ, eps, ν, θp, epsp, νp, sol_p, ite_PEE_t = True, t = None, **kwargs):
		""" Scalar version of solve_PEEvec_t """
		sol = self.PEE_precomputations(θp, epsp, ν, sol_p)
		τ = np.empty(self.ngrid)
		τ[0] = optimize.root(lambda τ: self.PEE_polObj_t(τ, θ, eps, ν, θp, epsp, νp, self.aux_soli(sol, 0), self.aux_soli(sol_p,0)), sol_p['τ_notBound'][0] if ite_PEE_t else self.getx0('PEE_t',t=t,i=0), **kwargs)['x']
		for i in range(1, self.ngrid):
			τ[i] = optimize.root(lambda τ: self.PEE_polObj_t(τ, θ, eps, ν, θp, epsp, νp, self.aux_soli(sol, i), self.aux_soli(sol_p,i)), τ[i-1] if ite_PEE_t else self.getx0('PEE_t',t=t,i=i), **kwargs)['x']
		return self.report_PEE_t(τ, θ, eps, ν, sol, sol_p)

	def report_PEE_t(self, τ, θ, eps, ν, sol, sol_p):
		""" Return solution dictionary given vector of taxes"""
		sol['τ'] = np.clip(τ, 1e-4, 1)
		sol['τ_notBound'] = τ
		sol['s[t-1]'] = sol['s_τ0']*(1-sol['τ'])**(-1/self.db['α'])
		sol['B']  = self.aux_B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.aux_Γs(sol['B'], sol['τ'], θ, eps)
		sol['∂τ/∂s'] = np.gradient(sol['τ'], sol['s[t-1]'])
		sol['dln(h)/dln(s)'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s)'] = self.db['α']*self.db['ξ']*(1+sol['Ψ']*(1-sol_p['∂ln(h)/∂ln(s)']))/((1+self.db['α']*self.db['ξ'])*sol['σ'])
		sol.update(self.aux_derivatives(sol['Ψ'], sol['σ'], sol_p['∂ln(h)/∂ln(s)'], sol['τ']))
		return sol

	##### Tax effect on indirect utility
	def PEE_polObjVec_t(self, τ, θ, eps, ν, θp, epsp, νp, sol, sol_p):
		τBound = np.clip(τ, 1e-4, 1)
		funcOfτ = self.funcOfτ(τBound, θ, eps, ν, sol, sol_p)
		condition = (self.db['γ0']*self.ω20*self.aux_PEE_HtM_old_t(τBound, eps, ν, sol, funcOfτ)+np.matmul(self.ω2i * self.db['γi'], self.aux_PEE_retirees_t(τBound, θ, eps, ν, sol, funcOfτ))
				 +ν*(self.db['γ0']*self.ω10*self.aux_PEE_HtM_young_t(τBound, epsp, ν, νp, sol, sol_p, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_PEE_workers_t(θp, epsp, sol, sol_p, funcOfτ))))
		return self.adjustFocMultiplicative(condition, τ, l = 1e-4, kl = self.db['kτ_l'], ku = self.db['kτ_u'])

	def PEE_polObj_t(self, τ, θ, eps, ν, θp, epsp, νp, sol, sol_p):
		τBound = np.clip(τ, 1e-4, 1)
		funcOfτ = self.funcOfτ(τBound, θ, eps, ν, sol, sol_p)
		condition = (self.db['γ0']*self.ω20*self.aux_PEE_HtM_old_t(τBound, eps, ν, sol, funcOfτ)+np.matmul(self.ω2i * self.db['γi'], self.aux_PEE_retirees_t(τBound, θ, eps, ν, sol, funcOfτ))
				 +ν*(self.db['γ0']*self.ω10*self.aux_PEE_HtM_young_t(τBound, epsp, ν, νp, sol, sol_p, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_PEE_workers_scalar_t(θp, epsp, sol, sol_p, funcOfτ))))
		return self.adjustFocMultiplicative(condition, τ, l = 1e-4, kl = self.db['kτ_l'], ku = self.db['kτ_u'])

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
	###		 solution for period t - SCALAR loop version 
	def solve_ESC_t(self, ν, νp, sol_p, ite_ESC_t = True, t = None, **kwargs):
		sol = self.ESC_precomputations(ν, sol_p)
		esc = np.empty((self.ngrid, 3))
		esc[0] = optimize.root(lambda x: self.ESC_polObj_t(x[0], x[1], x[2], ν, νp, self.aux_soli(sol, 0), self.aux_soli(sol_p,0)), [sol_p[k][0] for k in ['τ','θ','eps']] if ite_ESC_t else self.getx0('ESC_t',t=t,i=0), **kwargs)['x']
		for i in range(1, self.ngrid):
			esc[i] = optimize.root(lambda x: self.ESC_polObj_t(x[0], x[1], x[2], ν, νp, self.aux_soli(sol, i), self.aux_soli(sol_p,i)), esc[i-1] if ite_ESC_t else self.getx0('ESC_t',t=t,i=i), **kwargs)['x']
		return self.report_ESC_t(np.hstack([esc[:,0], esc[:,1], esc[:,2]]), ν, sol, sol_p)

	def solve_ESCvec_t(self, ν, νp, sol_p, ite_ESCvec_t = True, t = None, **kwargs):
		sol = self.ESC_precomputations(ν, sol_p)
		esc = optimize.root(lambda x: self.ESC_polObjVec_t(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, νp, sol, sol_p),  np.hstack([sol_p['τ'], sol_p['θ'], sol_p['eps']]) if ite_ESCvec_t else self.getx0('ESCvec_t',t =t), **kwargs)
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

	def ESC_polObjVec_t(self, τ, θ, eps, ν, νp, sol, sol_p):
		""" Returns stacked FOCs for τ, θ, eps. """
		funcOfτ = self.ESC_funcOfτ(τ, θ, eps, ν, sol, sol_p)
		young = ν*(self.db['γ0']*self.ω10*self.aux_ESC_HtM_young_t(τ, ν, νp, sol, sol_p, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_ESC_workers_t(sol, sol_p, funcOfτ)))
		euler = self.aux_ESC_retirees_t(τ, θ, eps, ν, νp, sol, funcOfτ)
		htm   = self.aux_ESC_HtM_old_t(τ, eps, ν, sol, funcOfτ)
		return np.hstack([young+np.matmul(self.ω2i*self.db['γi'], euler['τ'])+self.db['γ0']*self.ω20*htm['τ'],
						  np.matmul(self.ω2i*self.db['γi'], euler['θ']),
						  np.matmul(self.ω2i*self.db['γi'], euler['eps'])+self.db['γ0']*self.ω20*htm['eps']])

	def ESC_polObj_t(self, τ, θ, eps, ν, νp, sol, sol_p):
		""" Returns stacked FOCs for τ, θ, eps. """
		funcOfτ = self.ESC_funcOfτ(τ, θ, eps, ν, sol, sol_p)
		young = ν*(self.db['γ0']*self.ω10*self.aux_ESC_HtM_young_t(τ, ν, νp, sol, sol_p, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_ESC_workers_scalar_t(sol, sol_p, funcOfτ)))
		euler = self.aux_ESC_retirees_t(τ, θ, eps, ν, νp, sol, funcOfτ)
		htm   = self.aux_ESC_HtM_old_t(τ, eps, ν, sol, funcOfτ)
		return np.hstack([young+np.matmul(self.ω2i*self.db['γi'], euler['τ'])+self.db['γ0']*self.ω20*htm['τ'],
						  np.matmul(self.ω2i*self.db['γi'], euler['θ']),
						  np.matmul(self.ω2i*self.db['γi'], euler['eps'])+self.db['γ0']*self.ω20*htm['eps']])

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

	### 5.3. ESC with bounded epsilon/theta: Given a solution from t+1, solve for the
	###		 solution for period t. Some of this reuses the stuff from the ESC_t
	def solve_ESCBvec_t(self, ν, νp, sol_p, ite_ESCBvec_t = True, t = None, **kwargs):
		sol = self.ESC_precomputations(ν, sol_p)
		esc = optimize.root(lambda x: self.ESCB_polObjVec_t(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, νp, sol, sol_p), np.hstack([sol_p['τ_notBound'], sol_p['θ_notBound'], sol_p['eps_notBound']]) if ite_ESCBvec_t else self.getx0('ESCBvec_t',t=t), **kwargs)
		assert esc['success'], f"""Could not identify ESC solution (self.solve_ESCBvec_t) with parameters:
		ν: {ν}, ν[t+1]: {νp}, and solution from t+1 with:
		τ[t+1]: {sol_p['τ']}, 
		θ[t+1]: {sol_p['θ']},
		ε[t+1]: {sol_p['eps']}"""
		return self.report_ESCB_t(esc['x'], ν, sol, sol_p)

	def solve_ESCB_t(self, ν, νp, sol_p, ite_ESCB_t = True, t = None, **kwargs):
		sol = self.ESC_precomputations(ν, sol_p)
		esc = np.empty((self.ngrid, 3))
		esc[0] = optimize.root(lambda x: self.ESCB_polObj_t(x[0], x[1], x[2], ν, νp, self.aux_soli(sol, 0), self.aux_soli(sol_p,0)), [sol_p[k][0] for k in ['τ_notBound','θ_notBound','eps_notBound']] if ite_ESCB_t else self.getx0('ESCB_t',t=t,i=0), **kwargs)['x']
		for i in range(1, self.ngrid):
			esc[i] = optimize.root(lambda x: self.ESCB_polObj_t(x[0], x[1], x[2], ν, νp, self.aux_soli(sol, i), self.aux_soli(sol_p,i)), esc[i-1] if ite_ESCB_t else self.getx0('ESCB_t',t=t,i=i), **kwargs)['x']
		return self.report_ESCB_t(np.hstack([esc[:,0], esc[:,1], esc[:,2]]), ν, sol, sol_p)

	def report_ESCB_t(self, x, ν, sol, sol_p):
		""" Return solution dictionary given vector of taxes"""
		sol = self.report_ESC_t(np.hstack([np.clip(self(x,'τ'),1e-4,1), np.clip(self(x,'θ'),0,1), np.clip(self(x,'eps'),0,1)]), ν, sol, sol_p)
		sol['τ_notBound'] = self(x, 'τ')
		sol['θ_notBound'] = self(x,'θ')
		sol['eps_notBound'] = self(x, 'eps')
		return sol

	def ESCB_polObj_t(self, τ, θ, eps, ν, νp, sol, sol_p):
		""" Returns stacked FOCs for τ, θ, eps. θ, eps from sol_p are bounded versions (main solution), but inputs θ,eps are not yet bounded:"""
		τBound, θBound, epsBound = np.clip(τ, 1e-4, 1), np.clip(θ, 0, 1), np.clip(eps, 0,1)
		funcOfτ = self.ESC_funcOfτ(τBound, θBound, epsBound, ν, sol, sol_p)
		young = ν*(self.db['γ0']*self.ω10*self.aux_ESC_HtM_young_t(τBound, ν, νp, sol, sol_p, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_ESC_workers_scalar_t(sol, sol_p, funcOfτ)))
		euler = self.aux_ESC_retirees_t(τBound, θBound, epsBound, ν, sol, funcOfτ)
		htm   = self.aux_ESC_HtM_old_t(τBound, epsBound, ν, sol, funcOfτ)
		return np.hstack([self.adjustFocMultiplicative(young+np.matmul(self.ω2i*self.db['γi'], euler['τ'])+self.db['γ0']*self.ω20*htm['τ'], τ, l = 1e-4, kl = self.db['kτ_l'], ku = self.db['kτ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['θ']), θ, kl = self.db['kθ_l'], ku = self.db['kθ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['eps'])+self.db['γ0']*self.ω20*htm['eps'], eps, kl = self.db['keps_l'], ku = self.db['kθ_u'])])

	def ESCB_polObjVec_t(self, τ, θ, eps, ν, νp, sol, sol_p):
		""" Returns stacked FOCs for τ, θ, eps. θ, eps from sol_p are bounded versions (main solution), but inputs θ,eps are not yet bounded:"""
		τBound, θBound, epsBound = np.clip(τ, 1e-4, 1), np.clip(θ, 0, 1), np.clip(eps, 0,1)
		funcOfτ = self.ESC_funcOfτ(τBound, θBound, epsBound, ν, sol, sol_p)
		young = ν*(self.db['γ0']*self.ω10*self.aux_ESC_HtM_young_t(τBound, ν, νp, sol, sol_p, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_ESC_workers_t(sol, sol_p, funcOfτ)))
		euler = self.aux_ESC_retirees_t(τBound, θBound, epsBound, ν, sol, funcOfτ)
		htm   = self.aux_ESC_HtM_old_t(τBound, epsBound, ν, sol, funcOfτ)
		return np.hstack([self.adjustFocMultiplicative(young+np.matmul(self.ω2i*self.db['γi'], euler['τ'])+self.db['γ0']*self.ω20*htm['τ'], τ, l = 1e-4, kl = self.db['kτ_l'], ku = self.db['kτ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['θ']), θ, kl = self.db['kθ_l'], ku = self.db['kθ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['eps'])+self.db['γ0']*self.ω20*htm['eps'], eps, kl = self.db['keps_l'], ku = self.db['kθ_u'])])

	def adjustFocMultiplicative(self, z, x, l = 0, u = 1, kl = 10, ku = 10):
		""" Adjust FOC in a multiplicative way: z is the "original" marginal effect, x is the bounded variable """
		return z-abs(z)*(kl*(np.clip(x, None, l)-l)+ku*(np.clip(x,u,None)-u))

	###########################################################
	##########	6. Auxiliary household functions 	###########
	###########################################################	
	###### 6.1. Euler retirees
	def aux_ESC_retirees_t(self, τ, θ, eps, ν, sol, funcOfτ):
		c2i_coeff = self.aux_c2i_coeff(funcOfτ['si/s'], τ, θ, eps)
		c2i = self.aux_c2i_t(τ, eps, funcOfτ['s[t-1]'], sol['h'], ν, c2i_coeff)
		return {'τ': self.aux_PEE_retirees_(θ, eps, funcOfτ['dln(h)/dτ'], c2i, c2i_coeff),
				'θ': self.aux_ESC_retirees_θ(τ, θ, eps, c2i, c2i_coeff),
				'eps': self.aux_ESC_retirees_eps(τ, θ, eps, c2i, c2i_coeff)}

	def aux_ESC_retirees_eps(self, τ, θ, eps, c2i, c2i_coeff):
		return c2i**(1-1/self.db['ρ'])*(self.db['αr']*self.db['p']*τ*(self.db['p']-self.db['p̄']*(θ*self.aux_Prod.reshape(self.ni,1)+(1-θ)/(1-self.db['γ0'])))/(eps*self.db['p̄']+self.db['p'])**2)/c2i_coeff

	def aux_ESC_retirees_θ(self, τ, θ, eps, c2i, c2i_coeff):
		return c2i**(1-1/self.db['ρ'])*(self.db['αr']*self.db['p']*τ*(self.aux_Prod.reshape(self.ni,1)-1/(1-self.db['γ0']))/(eps*self.db['p̄']+self.db['p']))/c2i_coeff

	def aux_PEE_retirees_t(self, τ, θ, eps, ν, sol, funcOfτ):
		c2i_coeff = self.aux_c2i_coeff(funcOfτ['si/s'], τ, θ, eps)
		return self.aux_PEE_retirees_(θ, eps, funcOfτ['dln(h)/dτ'], self.aux_c2i_t(τ, eps, funcOfτ['s[t-1]'], sol['h'], ν, c2i_coeff), c2i_coeff)

	def aux_PEE_retirees_(self, θ, eps, dlnh_dτ, c2i, c2i_coeff):
		return c2i**(1-1/self.db['ρ'])*((1-self.db['α'])*dlnh_dτ+(self.db['αr']*(self.db['p']/(eps*self.db['p̄']+self.db['p']))*(eps+θ*self.aux_Prod.reshape(self.ni,1)+(1-θ)/(1-self.db['γ0'])))/c2i_coeff)

	def aux_c2i_t(self, τ, eps, s, h, ν, c2i_coeff):
		return self.db['α']*(ν/self.db['p'])*(s/ν)**self.db['α']*h**(1-self.db['α'])*c2i_coeff

	def aux_c2i_coeff(self, sSpread, τ, θ, eps):
		return sSpread + self.db['αr']*self.auxPen(τ,eps)*(eps+θ*self.aux_Prod.reshape(self.ni,1)+(1-θ)/(1-self.db['γ0']))
	
	###### 6.2. HtM Retirees
	def aux_ESC_HtM_old_t(self, τ, eps, ν, sol, funcOfτ):
		c20 = self.aux_c20_t(funcOfτ['s[t-1]'], sol['h'],ν,τ,eps)
		return {'τ': self.aux_PEE_HtM_old_(τ, eps, funcOfτ['dln(h)/dτ'], c20),
				'eps': self.aux_ESC_HtM_old_eps(eps, c20)}

	def aux_ESC_HtM_old_eps(self, eps, c20):
		return c20**(1-1/self.db['ρ'])*(eps*self.db['p̄']+self.db['p']-self.db['p̄'])/(eps*self.db['p̄']+self.db['p'])

	def aux_PEE_HtM_old_t(self, τ, eps, ν, sol, funcOfτ):
		return self.aux_PEE_HtM_old_(τ, eps, funcOfτ['dln(h)/dτ'], self.aux_c20_t(funcOfτ['s[t-1]'], sol['h'], ν, τ, eps))

	def aux_PEE_HtM_old_(self, τ, eps, dlnh_dτ, c20):
		return c20**(1-1/self.db['ρ'])*(1/τ+(1-self.db['α'])*dlnh_dτ)

	def aux_c20_t(self, s, h, ν, τ, eps):
		return (1-self.db['α'])*ν*(s/ν)**(self.db['α'])*h**(1-self.db['α'])*eps*τ/(eps*self.db['p̄']+self.db['p'])

	###### 6.3. Euler workers
	def aux_ESC_workers_t(self, sol, sol_p, funcOfτ):
		""" """
		k1 = self.db['αr']*(1+self.db['ξ'])*sol_p['Γs']*self.db['p']/(self.db['p̄']*sol_p['eps']+self.db['p'])
		k2 = sol_p['eps']+(1-sol_p['θ'])/(1-self.db['γ0'])
		return self.aux_ĉ1i_t(sol_p['τ'], sol_p['θ'], sol_p['eps'], sol['h'], sol_p['B'], sol_p['Γs'])**(1-1/self.db['ρ']) * (
				((1+self.db['ξ'])/self.db['ξ'])*funcOfτ['dln(h)/dτ']
			+	(sol_p['B']/(1+sol_p['B']))*(1-self.db['α'])*(sol_p['dln(h)/dln(s)']-1)*funcOfτ['dln(s)/dτ']
			+(k1/(self.aux_Prod.reshape(self.ni,1)+k1*sol_p['τ']*k2))*(
				funcOfτ['dln(Γs)/dτ']*sol_p['τ']*k2+funcOfτ['∂τp/∂τ']*k2-funcOfτ['∂θ/∂τ']*sol_p['τ']/(1-self.db['γ0'])+funcOfτ['∂eps/∂τ']*sol_p['τ']*(1-self.db['p̄']*k2/(sol_p['eps']*self.db['p̄']+self.db['p'])))
			)

	def aux_ESC_workers_scalar_t(self, sol, sol_p, funcOfτ):
		""" """
		k1 = self.db['αr']*(1+self.db['ξ'])*sol_p['Γs']*self.db['p']/(self.db['p̄']*sol_p['eps']+self.db['p'])
		k2 = sol_p['eps']+(1-sol_p['θ'])/(1-self.db['γ0'])
		return self.aux_ĉ1i_scalar_t(sol_p['τ'], sol_p['θ'], sol_p['eps'], sol['h'], sol_p['B'], sol_p['Γs'])**(1-1/self.db['ρ']) * (
				((1+self.db['ξ'])/self.db['ξ'])*funcOfτ['dln(h)/dτ']
			+	(sol_p['B']/(1+sol_p['B']))*(1-self.db['α'])*(sol_p['dln(h)/dln(s)']-1)*funcOfτ['dln(s)/dτ']
			+(k1/(self.aux_Prod+k1*sol_p['τ']*k2))*(
				funcOfτ['dln(Γs)/dτ']*sol_p['τ']*k2+funcOfτ['∂τp/∂τ']*k2-funcOfτ['∂θ/∂τ']*sol_p['τ']/(1-self.db['γ0'])+funcOfτ['∂eps/∂τ']*sol_p['τ']*(1-self.db['p̄']*k2/(sol_p['eps']*self.db['p̄']+self.db['p'])))
			)

	def aux_PEE_workers_t(self, θp, epsp, sol, sol_p, funcOfτ):
		k = self.db['αr']*(1+self.db['ξ'])*sol_p['Γs']*(epsp+(1-θp)/(1-self.db['γ0']))*self.db['p']/(self.db['p̄']*epsp+self.db['p'])
		return self.aux_ĉ1i_t(sol_p['τ'], θp, epsp, sol['h'], sol_p['B'], sol_p['Γs'])**(1-1/self.db['ρ']) * (
				((1+self.db['ξ'])/self.db['ξ'])*funcOfτ['dln(h)/dτ']
			+	(sol_p['B']/(1+sol_p['B']))*(1-self.db['α'])*(sol_p['dln(h)/dln(s)']-1)*funcOfτ['dln(s)/dτ']
			+	k*(funcOfτ['∂τp/∂τ']+sol_p['τ']*funcOfτ['dln(Γs)/dτ'])/(self.aux_Prod.reshape(self.ni,1)+sol_p['τ']*k)
			)
	def aux_PEE_workers_scalar_t(self, θp, epsp, sol, sol_p, funcOfτ):
		k = self.db['αr']*(1+self.db['ξ'])*sol_p['Γs']*(epsp+(1-θp)/(1-self.db['γ0']))*self.db['p']/(self.db['p̄']*epsp+self.db['p'])
		return self.aux_ĉ1i_scalar_t(sol_p['τ'], θp, epsp, sol['h'], sol_p['B'], sol_p['Γs'])**(1-1/self.db['ρ']) * (
				((1+self.db['ξ'])/self.db['ξ'])*funcOfτ['dln(h)/dτ']
			+	(sol_p['B']/(1+sol_p['B']))*(1-self.db['α'])*(sol_p['dln(h)/dln(s)']-1)*funcOfτ['dln(s)/dτ']
			+	k*(funcOfτ['∂τp/∂τ']+sol_p['τ']*funcOfτ['dln(Γs)/dτ'])/(self.aux_Prod+sol_p['τ']*k)
			)

	def aux_ĉ1i_t(self, τp, θp, epsp, h, B, Γs):
		return h**((1+self.db['ξ'])/self.db['ξ'])*(1+B)**(1/(self.db['ρ']-1))*(self.aux_Prod.reshape(self.ni,1)+self.db['αr']*self.auxPen(τp, epsp)*(epsp+(1-θp)/1-self.db['γ0']))*Γs

	def aux_ĉ1i_scalar_t(self, τp, θp, epsp, h, B, Γs):
		return h**((1+self.db['ξ'])/self.db['ξ'])*(1+B)**(1/(self.db['ρ']-1))*(self.aux_Prod+self.db['αr']*self.auxPen(τp, epsp)*(epsp+(1-θp)/1-self.db['γ0']))*Γs

	###### 6.4. HtM young
	def aux_PEE_HtM_young_t(self, τ, epsp, ν, νp, sol, sol_p, funcOfτ):
		return (-(1+self.db['ξ'])*self.aux_c̃10_t(τ, funcOfτ['s[t-1]'], sol['h'], ν)**(1-1/self.db['ρ'])*(self.db['α']*funcOfτ['dln(h)/dτ']+(τ/(1-τ))*self.db['ξ']/(1+τ*self.db['ξ']))
			+	self.db['β0']*self.aux_c2p0_t(sol_p['τ'], epsp, sol_p['s[t-1]'], sol_p['h'], νp)**(1-1/self.db['ρ'])*(
					funcOfτ['∂τp/∂τ']+(self.db['α']+(1-self.db['α'])*sol_p['dln(h)/dln(s)'])*funcOfτ['dln(s)/dτ']
			))

	def aux_ESC_HtM_young_t(self, τ, ν, νp, sol, sol_p, funcOfτ):
		return (-(1+self.db['ξ'])*self.aux_c̃10_t(τ, funcOfτ['s[t-1]'], sol['h'], ν)**(1-1/self.db['ρ'])*(self.db['α']*funcOfτ['dln(h)/dτ']+(τ/(1-τ))*self.db['ξ']/(1+τ*self.db['ξ']))
			+	self.db['β0']*self.aux_c2p0_t(sol_p['τ'], sol_p['eps'], sol_p['s[t-1]'], sol_p['h'], νp)**(1-1/self.db['ρ'])*(
					funcOfτ['∂τp/∂τ']+(self.db['α']+(1-self.db['α'])*sol_p['dln(h)/dln(s)'])*funcOfτ['dln(s)/dτ']+(funcOfτ['∂eps/∂τ']/sol_p['eps'])*self.db['p']/(sol_p['eps']*self.db['p̄']+self.db['p'])
			))

	def aux_c̃10_t(self, τ, s, h, ν):
		return (self.db['η0']**(1+self.db['ξ'])/self.db['X0']**self.db['ξ'])*((1-self.db['α'])*(s/(ν*h))**self.db['α'])**(1+self.db['ξ'])*((1-τ)**self.db['ξ']-self.db['ξ']*(1-τ)**(1+self.db['ξ'])/(1+self.db['ξ']))

	def aux_c10_t(self, τ, s, h, ν):
		return (self.db['η0']**(1+self.db['ξ'])/self.db['X0']**self.db['ξ'])*(1-τ)**self.db['ξ']*((1-self.db['α'])*(s/(ν*h))**self.db['α'])**(1+self.db['ξ'])

	def aux_c2p0_t(self, τp, epsp, sp, hp, νp):
		return (epsp*τp/(epsp*self.db['p̄']+self.db['p']))*νp*(1-self.db['α'])*(sp/νp)**self.db['α']*hp**(1-self.db['α'])


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
		[self.db.__setitem__(k, getattr(self,'aux_'+k)) for k in ('Θhi','Θh0','Θsi','Θc1i','Θc2i','Θc2pi','Θ̃c1i','Θc10','Θc20','Θc2p0','Θc̃10')];

	def reportLevels(self):
		""" Assumes self.reportCoefficients has been run"""
		[self.db.__setitem__(k, getattr(self,'levels_'+k)) for k in ('c1i','c2i','c10','c20','̃c1i','c2pi','c2p0','hi','h0','c̃10','w','R')];

	def reportUtils(self):
		""" Assumes self.reportLevels has been run"""
		[self.db.__setitem__(k, getattr(self,'aux_'+k)(self.db)) for k in ('util1i','util10','util2i','util20', 'utilPol')];

	@property
	def aux_Θhi(self):
		return pd.DataFrame((self.db['Θh'].values * self.aux_Prod.reshape(self.ni,1)).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θh0(self):
		return pd.Series((self.db['η0']**(1+self.db['ξ'])/self.db['X0']**self.db['ξ'])*self.db['Θh'].values**(-self.db['α']*self.db['ξ'])*((1-self.db['α'])*(1-self.db['τ'].values))**self.db['ξ'], 
			index = self.db['t'])

	@property
	def aux_Θsi(self):
		return pd.DataFrame((self.db['Θs'].values*self.savingsSpread(self.db['B[t+1]'].values.T, self.db['Γs'].values, self.db['τ[t+1]'].values, self.db['θ[t+1]'].values, self.db['eps[t+1]'].values)).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc1i(self):
		return pd.DataFrame((self.db['Θh'].values**((1+self.db['ξ'])/self.db['ξ'])*(self.aux_Prod.reshape(self.ni,1)*(1-self.db['B[t+1]'].values.T/((1+self.db['B[t+1]'].values.T)*(1+self.db['ξ'])))+self.db['αr']*self.auxPen(self.db['τ[t+1]'].values, self.db['eps[t+1]'].values)*(self.db['eps[t+1]'].values+(1-self.db['θ[t+1]'].values)/(1-self.db['γ0']))*self.db['Θs'].values/(1+self.db['B[t+1]'].values.T))).T,
				index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc2i(self):
		return pd.DataFrame((self.db['α']*self.db['A']*self.db['ν']*self.db['Θh'].values**(1-self.db['α'])*(self.savingsSpread(self.db['B'].values.T, self.db['Γs[t-1]'].values, self.db['τ'].values, self.db['θ'].values, self.db['eps'].values)+self.db['αr']*self.auxPen(self.db['τ'].values, self.db['eps'].values)*(self.db['eps'].values+self.db['θ'].values*self.aux_Prod.reshape(self.ni,1)+(1-self.db['θ'].values)/(1-self.db['γ0'])))).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc2pi(self):
		return pd.DataFrame((self.db['α']*self.leadSym(self.db['A']*self.db['ν']*self.db['Θh'].values**(1-self.db['α']))*(self.savingsSpread(self.db['B[t+1]'].values.T, self.db['Γs'].values, self.db['τ[t+1]'].values, self.db['θ[t+1]'].values, self.db['eps[t+1]'].values)+self.db['αr']*self.auxPen(self.db['τ[t+1]'].values, self.db['eps[t+1]'].values)*(self.db['eps[t+1]'].values+self.db['θ[t+1]'].values*self.aux_Prod.reshape(self.ni,1)+(1-self.db['θ[t+1]'].values)/(1-self.db['γ0'])))).T,
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θ̃c1i(self):
		return pd.DataFrame(((self.db['Θh'].values**((1+self.db['ξ'])/self.db['ξ']))/((1+self.db['ξ'])*(1+self.db['B[t+1]'].values.T))*(self.aux_Prod.reshape(self.ni,1) + self.db['αr']*self.auxPen(self.db['τ[t+1]'].values,self.db['eps[t+1]'].values)*(self.db['eps[t+1]'].values+(1-self.db['θ[t+1]'].values)/(1-self.db['γ0']))*(1+self.db['ξ'])*self.db['Γs'].values)).T, 
			index = self.db['t'], columns = self.db['i'])

	@property
	def aux_Θc10(self):
		return pd.Series((self.db['η0']**(1+self.db['ξ'])/self.db['X0']**self.db['ξ'])*(1-self.db['τ'].values)**self.db['ξ']*(1-self.db['α'])**(1+self.db['ξ'])*self.db['Θh'].values**(-self.db['α']*(1+self.db['ξ'])),
			index = self.db['t'])

	@property
	def aux_Θc20(self):
		return pd.Series((self.db['eps'].values*self.db['τ'].values/(self.db['eps'].values*self.db['p̄']+self.db['p']))*self.db['ν']*(1-self.db['α'])*self.db['Θh'].values**(1-self.db['α']),
			index = self.db['t'])

	@property
	def aux_Θc2p0(self):
		return pd.Series((self.db['eps[t+1]'].values*self.db['τ[t+1]'].values/(self.db['eps[t+1]'].values*self.db['p̄']+self.db['p']))*self.leadSym(self.db['ν']*(1-self.db['α'])*self.db['Θh'].values**(1-self.db['α'])),
			index = self.db['t'])

	@property
	def aux_Θc̃10(self):
		return pd.Series((self.db['η0']**(1+self.db['ξ'])/self.db['X0']**self.db['ξ'])*(1-self.db['α'])**(1+self.db['ξ'])*self.db['Θh'].values**(-self.db['α']*(1+self.db['ξ']))*((1-self.db['τ'].values)**self.db['ξ']-(self.db['ξ']/(1+self.db['ξ']))*(1-self.db['τ'].values)**(1+self.db['ξ'])),
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
	def levels_c10(self):
		return self.db['Θc10']*self.auxLevel(self.power_s)

	@property
	def levels_c20(self):
		return self.db['Θc20']*self.auxLevel(self.power_s)

	@property
	def levels_c2p0(self):
		return self.db['Θc2p0']*self.auxLevel(self.power_p).dropna()

	@property
	def levels_h0(self):
		return self.db['Θh0']*self.auxLevel(self.power_h)

	@property
	def levels_c̃10(self):
		return self.db['Θc̃10']*self.auxLevel(self.power_s)

	def aux_util1i(self, db, Δy = 0, Δo = 0):
		return ((db['̃c1i']+Δy)**(1-1/self.db['ρ'])).add((db['c2pi']+Δo)**(1-1/self.db['ρ'])*self.db['βi'], fill_value = 0)/(1-1/self.db['ρ'])

	def aux_util10(self, db, Δy = 0, Δo = 0):
		""" Utility for young hand-to-mouth"""
		return ((db['c̃10']+Δy)**(1-1/self.db['ρ'])).add((db['c2p0']+Δo)**(1-1/self.db['ρ'])*self.db['β0'], fill_value = 0)/(1-1/self.db['ρ'])

	def aux_util2i(self, db, Δ = 0):
		""" Utility for retired households """
		return (db['c2i']+Δ)**(1-1/self.db['ρ'])/(1-1/self.db['ρ'])

	def aux_util20(self, db, Δ = 0):
		""" Utility for old hand-to-mouth"""
		return (db['c20']+Δ)**(1-1/self.db['ρ'])/(1-1/self.db['ρ'])

	def aux_utilPol(self, db, Δy = 0, Δy2 = 0, Δyu = 0, Δyu2 = 0, Δo = 0, Δou = 0):
		""" Political objective function """
		return ( np.matmul(self.aux_util2i(db, Δo),self.ω2i*self.db['γi'])+self.db['γ0']*self.ω20*self.aux_util20(db, Δou)
			+   self.db['ν']*(np.matmul(self.aux_util1i(db, Δy, Δy2), self.ω1i*self.db['γi'])+self.db['γ0']*self.ω10*self.aux_util10(db, Δyu, Δyu2) )
			)


	###########################################################
	##########			8. Calibration US			###########
	###########################################################
	def USCal_simplePEE(self, t0, sGrid, par0 = None, **kwargs):
		""" Match interest rate and PEE tax - not ESC """
		sol = optimize.root(lambda x: self.USCal_simplePEE_eqs(x, t0, sGrid, **kwargs), noneInit(par0, [self.db['ω'], self.db['β0']/self.db['p0'], self.db['X0']]))
		assert sol['success'], f""" Couldn't calibrate model """
		return sol['x']

	def USCal_simplePEE_eqs(self, x, t0, sGrid, update = True):
		""" Target interest rate leven and pension taxes in t0 - solve with PEE """
		self.db['ω'] = x[0]
		self.db['βj'] = self.US_β(x[1])
		self.db['β0'], self.db['βi'] = self.db['βj'][0], self.db['βj'][1:]
		self.db['X0'] = x[2]
		sol, pol = self.solve_PEE(sGrid, returnPols = True)
		if update:
			self.x0['steadyState_PEE'] = pol[self.T-1]['τ']
		return np.hstack([sol['τ'].xs(t0)-self.db['τ0'],
						  sol['R'].xs(t0)-self.db['R0'],
						  x[2]-self.US_X0(sol['h'].xs(t0))])

	####################################################################################
	### 								DONE UP TO HERE   							 ###
	####################################################################################


	###########################################################
	##########				8. EV 					###########
	###########################################################	

	# def EV_solInPercentages(self, db, sol):
	# 	""" Report transfers relative to current."""
	# 	relativeTransfers = {'transfer_Y': (sol['transfer_Y'].unstack('i')/(db['c1i'])).stack(),
	# 						 'transfer_O': (sol['transfer_O'].unstack('i')/db['c2i']).stack(),
	# 						 'transfer_YU': sol['transfer_YU']/db['c1u'],
	# 						 'transfer_OU': sol['transfer_OU']/db['c2u']}
	# 	averageConsumption = (db['ν'] * ((1-db['γu'])*(db['c1i'] * db['γ']).sum(axis=1)+db['γu'] * db['c1u'])
	# 							+		 (1-db['γu'])*(db['c2i'] * db['γ']).sum(axis=1)+db['γu'] * db['c2u']) / (1+db['ν'])
	# 	relativeTransfers['transfer_Pol'] = sol['transfer_Pol']/averageConsumption
	# 	return relativeTransfers

	# def solve_EV_Permanent(self, db0, db1, x0 = None, ftol = 1e-9):
	# 	x0 = noneInit(x0, np.zeros(self.ns['EV'].len))
	# 	f  = lambda x: self.EV_Permanent_Eqs(db0, db1, self.get(x, 'transfer_Y', ns = 'EV'),
	# 												   self.get(x, 'transfer_O', ns = 'EV'),
	# 												   self.get(x, 'transfer_YU', ns = 'EV'),
	# 												   self.get(x, 'transfer_OU', ns = 'EV'),
	# 												   self.get(x, 'transfer_Pol', ns = 'EV'))
	# 	if max(abs(f(x0)))<ftol:
	# 		return x0
	# 	else:
	# 		sol = optimize.root(f, x0)
	# 		if sol['success'] | (max(abs(f(sol['x'])))<ftol):
	# 			return sol['x']
	# 		else:
	# 			print(f"solve_EV_Permanent couldn't identify a vector of transfers that establishes equivalent variation")

	# ######## Permanent, anticipated transfers that consumers may use smooth out over time
	# def EV_Permanent_Eqs(self, db0, db1, Δy, Δo, Δyu, Δou, Δpol):
	# 	""" Equations used to solve for the equivalent variation """
	# 	return np.hstack([(self.EV_Permanent_Y(db1, Δy)-db0['util1i'].stack()).values,
	# 					  (self.EV_Permanent_YU(db1, Δyu)-db0['util1u']).values,
	# 					  (self.EV_Permanent_O(db1, Δo)-db0['util2i'].stack()).values,
	# 					  (self.EV_Permanent_OU(db1, Δou)-db0['util2u']).values,
	# 					  (self.EV_Permanent_Pol(db1, Δpol)-db0['utilPol']).values])


	# def EV_Permanent_Pol(self, db, transfer):
	# 	return self.aux_utilPol(db, Δy  = transfer.values.reshape(self.T,1) / (1+db['β']), Δy2 = (transfer.values * self.leadSym(db['R']).values).reshape(self.T,1) * (db['β']/(1+db['β'])),
	# 								Δyu = transfer, Δyu2 = transfer * self.leadSym(db['R']),
	# 								Δo  = transfer.values.reshape(self.T,1),
	# 								Δou = transfer
	# 							)

	# def EV_Permanent_Y(self, db, transfer):
	# 	return self.aux_util1i(db,  Δy = 2*transfer.unstack(level='i') / (1+db['β']), 
	# 								Δo = (2*transfer.unstack(level='i')*db['β']/(1+db['β'])).mul(self.leadSym(db['R']), axis = 0)
	# 							).stack()

	# def EV_Permanent_YU(self, db, transfer):
	# 	return self.aux_util1u(db,  Δy = transfer, Δo = transfer * self.leadSym(db['R']))

	# def EV_Permanent_O(self, db, transfer):
	# 	return self.aux_util2i(db, Δ = transfer.unstack(level='i')).stack()

	# def EV_Permanent_OU(self, db, transfer):
	# 	return self.aux_util2u(db, Δ = transfer)



