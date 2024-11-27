import numpy as np, pandas as pd, pyDbs, scipy
from pyDbs import is_iterable, SymMaps as sm
from scipy import optimize
_numtypes = (int,float,np.generic)

def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

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
	id1, id2 = Δs[Δs>0].argmin(), Δs[Δs<0].argmax() # identify grid points closest to steady state
	s1, ŝ1 = s[Δs>0][id1], ŝ[Δs>0][id1]
	s2, ŝ2 = s[Δs<0][id2], ŝ[Δs<0][id2]
	return ŝ1+((ŝ2-ŝ1)*(ŝ1-s1))/(s2-s1-(ŝ2-ŝ1)) # lin. approximation

def aux_τMeshGrid(sGrid, τGrid_1d):
	return np.meshgrid(sGrid, τGrid_1d)[1]

class Model:
	def __init__(self, ni = 11, T = 10, ngrid = 50, **kwargs):
		""" Fixed namespace """
		self.ni, self.T, self.ngrid = ni, T, ngrid
		self.db = self.defaultParameters | kwargs # default parameters
		self.addIdxs() # add indices for time, types, grid etc.
		self.addNamespaces() # define auxiliary class with "namespace" that helps organize move from stacked numpy arrays to sliced vectors and to pd.Series with indices.
		self.inferPars(**kwargs) # from basic parameters, infer other parameters.
		self.initUS() # initialize various settings for the specific US case
		self.x0 = self.defaultInitials # dictionary that is used to pass initial guesses for solutions
		self.f0 = self.defaultFunctions # dictionary that is used to specify what functions are called per default

	#######################################################################
	##########					1. INIT METHODS				 	###########
	#######################################################################

	@property
	def defaultParameters(self):
		return {'α': .5, 'A': np.ones(self.T), 'ν': np.ones(self.T), 'ξ' : .35, 'ρ': 1.2, 'ω': 2,  # parameters
				'τ0': .158, 'RR0': 39.4/50.1, 'UShare0': 3.4/15.8, 'R0': 2.443, # targets
				'kθ_l': 10, 'kθ_u': 10, 'keps_l': 10, 'keps_u': 10, 'kτ_l': 10, 'kτ_u': 10, # solver settings
				'τl': 1e-4, 'τu': 1, 'θl': 0, 'θu': 1, 'epsl': 1e-4, 'epsu': 2} # solver settings

	def addIdxs(self):
		self.db['t'] = pd.Index(range(self.T), name = 't')
		self.db['txE'] = pd.Index(range(self.T-1), name = 't') # Time index without terminal period
		self.db['t_']= self.db['t'].append(pd.Index([-1], name = 't')).sort_values() # Time index with -1 included
		self.db['i'] = pd.Index(range(self.ni), name = 'i')
		self.db['sgrid'] = pd.Index(range(self.ngrid), name = 'sgrid') # grid of s[t-1] to identify policy function on

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


	######### 1.2 AUXILIARY NAMESPACE CLASSES
	def addNamespaces(self):
		self.ns = {}
		self.ns['EE'] = sm(symbols = {x: self.db['t'] for x in ('s','h','Γs')}) # solve EE given policy
		self.ns['EE_FH'] = sm(symbols = {'h': self.db['t'], 's': self.db['txE'], 'Γs': self.db['txE']}) # solve EE given policy, finite horizon
		self.ns['ESC'] = sm(symbols = {k: self.db['t'] for k in ('τ', 'θ', 'eps')}) # Endogenous system characteristics solution
		self.ns['ESC[t]'] = sm(symbols = {x: self.db['sgrid'] for x in ('τ', 'θ', 'eps')}) # namespace used in policy function identification
		[ns.compile() for ns in self.ns.values()];
		# Define auxiliary lagged/leaded symbols
		self.ns['EE'].addShiftedSym('h[t+1]','h',-1, opt = {'useLoc':'nn'})
		self.ns['EE_FH'].addLaggedSym('h[t+1]','h',-1, c= ('not',self.db['t'][-1:])) 
		[self.ns['ESC'].addShiftedSym(f'{k}[t+1]',f'{k}', -1, opt = {'useLoc':'nn'}) for k in ('τ', 'θ','eps')];
		self.ns['EE'].addLaggedSym('s[t-1]', 's', 1, c = ('not', self.db['t'][0:1])) # this will not include the s0 value as it is not endogenous in the optimization

	#######################################################################
	##########	2. METHODS FOR NAVIGATING THE MODEL/SETTINGS 	###########
	#######################################################################

	######### 2.1 Default functions/values
	@property
	def defaultFunctions(self):
		""" Default lower-level methods (values) that are called in higher-level methods (keys)"""
		return {'PEE_t': self.solve_PEEvec_t,
				'ESC_t': self.solve_ESCvec_t}

	@property
	def defaultInitials(self):
		""" Default initial values used in numerical problems below."""
		return {'EE': np.full(self.ns['EE'].len, 0.2),
				'EE_FH': np.full(self.ns['EE_FH'].len, .2),
				'EE_polFunc': np.full(self.ns['EE'].len, .2),
				'EE_FH_polFunc': np.full(self.ns['EE_FH'].len, .2),
				'PEE_t': np.full(self.ngrid, .2),
				'PEE_T': np.full(self.ngrid, .2),
				'PEEvec_t': np.full(self.ngrid, .2),
				'PEEvec_T': np.full(self.ngrid, .2),
				'ESC_t': np.full((self.ngrid, 3), .2),
				'ESC_T': np.full((self.ngrid, 3), .2),
				'ESCvec_t': np.full(self.ns['ESC[t]'].len, .2),
				'ESCvec_T': np.full(self.ns['ESC[t]'].len, .2)}

	######### 2.2. Auxiliary methods
	def getx0(self, solve, i =None, t= None):
		x0 = self.x0[solve] if t is None else self.x0[solve][t]
		return x0 if i is None else x0[i]

	def aux_soli(self, sol, i):
		sol_i = {k: sol[k][i] if (k != 'B') and is_iterable(sol[k]) else sol[k] for k in sol}
		if 'B' in sol:
			sol_i['B'] = sol['B'][:,i]
		return sol_i

	def resampleSolution(self, sol, s):
		""" Redraw the solution on the grid 's' using linear interpolation """
		return {k: interpSol(s, sol['s[t-1]'], v) if  k != 's[t-1]' else s for k,v in sol.items()}

	def cleanSolution(self, sol, s):
		keep = sol['s[t-1]']<max(s)
		return {k: cleanSol(sol[k],keep) for k in sol}

	def adjustFocMultiplicative(self, z, x, l = 0, u = 1, kl = 10, ku = 10):
		""" Adjust FOC in a multiplicative way: z is the "original" marginal effect, x is the bounded variable """
		return z-abs(z)*(kl*(np.clip(x, None, l)-l)+ku*(np.clip(x,u,None)-u))


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


	def j(self, x):
		""" return full vector of symbol x by combining 0 and i types"""
		return np.hstack([self.db[f'{x}0'], self.db[f'{x}i']])
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
	##########			3. Auxiliary functions 		###########
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


	#######################################################################
	##########			4.  Solve for PEE/ESC paths				###########
	#######################################################################

	##########	3.1. PEE:
	def solve_PEE_FH(self, s, gridOption = 'resample', s0 = None, returnPols = False, termkwargs = None, x0_t = False, tkwargs = None):
		policy = self.solve_PEE_policy_FH(s, gridOption=gridOption, termkwargs = termkwargs, x0_t = x0_t, tkwargs = tkwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_FH_polFunc(policy, self.db['θ'].values, self.db['eps'].values, s0)
		return (self.reportMain_PEE_FH(sols), policy) if returnPols else self.reportMain_PEE_FH(sols)

	def solve_PEE_policy_FH(self, s, gridOption = 'resample', termkwargs = None, x0_t = False, tkwargs = None):
		""" PEE path with time dependent policy functions; terminal steady state function"""
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1] = self.solve_PEE_T(self.db['θ'].values[-1], self.db['eps'].values[-1], self.db['ν'][-1], s, **noneInit(termkwargs, {}))
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

	def reportMain_PEE(self, sol):
		sol['B'] = pd.DataFrame(self.aux_B(sol['s[t-1]'].values, sol['h'].values, self.db['ν']).T, columns = self.db['i'], index = self.db['t'])
		sol['Γs[t-1]'] = pd.Series(self.aux_Γs(sol['B'].values.T, sol['τ'].values, self.db['θ'].values, self.db['eps'].values), index = self.db['t'])
		sol['B[t+1]'] = pd.DataFrame(self.aux_B(sol['s'].values, sol['h[t+1]'].values, self.leadSym(self.db['ν'])).T, columns = self.db['i'], index = self.db['t'])
		sol['τ[t+1]'] = self.leadSym(sol['τ'])
		sol['R'] = pd.Series(self.aux_R(sol['s[t-1]'].values, sol['h'].values, self.db['ν']), index = self.db['t'])
		return sol

	def reportMain_PEE_FH(self, sol):
		sol['B'] = pd.DataFrame(self.aux_B(sol['s[t-1]'].values, sol['h'].values, self.db['ν']).T, columns = self.db['i'], index = self.db['t'])
		sol['Γs[t-1]'] = pd.Series(self.aux_Γs(sol['B'].values.T, sol['τ'].values, self.db['θ'].values, self.db['eps'].values), index = self.db['t'])
		# sol['B[t+1]'] = pd.DataFrame(self.aux_B(sol['s'].values, sol['h[t+1]'].values, self.leadSym(self.db['ν'])).T, columns = self.db['i'], index = self.db['t'])
		sol['τ[t+1]'] = self.leadSym(sol['τ'])
		sol['R'] = pd.Series(self.aux_R(sol['s[t-1]'].values, sol['h'].values, self.db['ν']), index = self.db['t'])
		return sol

	##########	3.2. ESC:
	def solve_ESC_FH(self, s, gridOption = 'resample', s0 = None, returnPols = False, termkwargs = None, x0_t = False, tkwargs = None):
		policy = self.solve_ESC_policy_FH(s, gridOption=gridOption, termkwargs = termkwargs, x0_t = x0_t, tkwargs = tkwargs)
		if s0 is None:
			s0 = interpFixedPoint(policy[0]['s'], s) # interpolate approximate steady state level of savings
		sols  = self.solve_EE_ESC_FH(policy, s0)
		return (self.reportMain_ESC_FH(sols), policy) if returnPols else self.reportMain_ESC_FH(sols)

	def solve_ESC_policy_FH(self, s, gridOption = 'resample', termkwargs = None, x0_t = False, tkwargs = None):
		""" PEE path with time dependent policy functions; terminal steady state function"""
		sols = dict.fromkeys(self.db['t'])
		sols[self.T-1] = self.solve_ESC_T(self.db['ν'][-1], s, **noneInit(termkwargs, {}))
		if gridOption == 'resample':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.resampleSolution(self.f0['ESC_t'](self.db['ν'][t], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs, {})), s)
		elif gridOption == 'clean':
			for t in range(self.T-2,-1,-1):
				sols[t] = self.cleanSolution(self.f0['ESC_t'](self.db['ν'][t], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs, {})), s)
		else:
			sols[t] = self.f0['ESC_t'](self.db['ν'][t], self.db['ν'][t+1], sols[t+1], t = t if x0_t else None, **noneInit(tkwargs, {}))
		return sols

	def reportMain_ESC_FH(self, sol):
		sol['B'] = pd.DataFrame(self.aux_B(sol['s[t-1]'].values, sol['h'].values, self.db['ν']).T, columns = self.db['i'], index = self.db['t'])
		sol['Γs[t-1]'] = pd.Series(self.aux_Γs(sol['B'].values.T, sol['τ'].values, sol['θ'].values, sol['eps'].values), index = self.db['t'])
		# sol['B[t+1]'] = pd.DataFrame(self.aux_B(sol['s'].values, sol['h[t+1]'].values, self.leadSym(self.db['ν'])).T, columns = self.db['i'], index = self.db['t'])
		sol.update({f'{k}[t+1]': self.leadSym(sol[f'{k}']) for k in ('τ', 'θ','eps')})
		sol['R'] = pd.Series(self.aux_R(sol['s[t-1]'].values, sol['h'].values, self.db['ν']), index = self.db['t'])
		return sol

	#######################################################################
	##########			5.  Economic Equilibrium				###########
	#######################################################################

	########## 5.1. Economic Equilibrium given policies
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
	def EE_h_eq(self, Γs, s_, τ, τp, θp, epsp, ν):
		return self.aux_Θh(τ, Γs, τp, θp, epsp)*(s_/ν)**self.power_h
	def EE_s_eq(self, h, Γs):
		return h**((1+self.db['ξ'])/self.db['ξ'])*Γs
	def EE_Γs_eq(self, s, hp, νp, τp, θp, epsp):
		return self.aux_Γs(self.aux_B(s, hp, νp), τp, θp, epsp)
	def aux_Θh(self, τ, Γs, τp, θp, epsp):
		return ((1-self.db['α'])*(1-τ)/((1-self.db['αr']*self.auxPen(τp, epsp)*θp*Γs)))**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))
	def get_sLag(self, x, s0):
		return np.insert(self(x,'s[t-1]', ns = 'EE'), 0, s0) # insert s0 to the numpy array s[t-1]

	### Finite Horizon version
	def solve_EE_FH(self, τ, θ, eps, s0, x0_EE = None):
		τp, θp, epsp = self.leadSym(τ), self.leadSym(θ), self.leadSym(eps)
		sol = optimize.root(lambda x: self.EE_FH_eqs(x, τ, θ, eps, τp, θp, epsp, s0), noneInit(x0_EE, self.getx0('EE_FH')))
		assert sol['success'], f""" Could not identify economic equilibrium (self.solve_EE_FH) with parameter inputs: 
		τ: {τ}, θ: {θ}, ε: {eps}, s0: {s0}"""
		return self.report_EE_FH(sol, s0)

	def report_EE_FH(self, sol, s0):
		d = self.ns['EE_FH'].unloadSol(sol['x'])
		d['s[t-1]'] = pd.Series(self.get_sLag_FH(sol['x'], s0), index = self.db['t'])
		# d['Θs'] = d['s']/((d['s[t-1]']/self.db['ν'])**self.power_s)
		# d['Θh'] = d['h']/((d['s[t-1]']/self.db['ν'])**self.power_h)
		return d
	def EE_FH_eqs(self, x, τ, θ, eps, τp, θp, epsp, s0):
		return np.hstack([self.EE_FH_h_eq(self(x,'Γs', ns = 'EE_FH'), self.get_sLag_FH(x, s0), τ, τp, θp, epsp, self.db['ν'])-self(x,'h',ns='EE_FH'),
						  self.EE_s_eq(self(x,'h',ns='EE_FH')[:-1], self(x,'Γs', ns = 'EE_FH'))-self(x,'s',ns='EE_FH'),
						  self.EE_Γs_eq(self(x,'s',ns='EE_FH'), self(x,'h',ns='EE_FH')[1:], self.db['ν'][1:], τ[1:], θ[1:], eps[1:])-self(x,'Γs', ns = 'EE_FH')])
	def EE_FH_h_eq(self, Γs, s_, τ, τp, θp, epsp, ν):
		return np.hstack([self.aux_Θh(τ[:-1], Γs, τp[:-1], θp[:-1], epsp[:-1]), self.aux_Θh_T(τ[-1])])*(s_/ν)**self.power_h
	def aux_Θh_T(self, τ):
		return ((1-self.db['α'])*(1-τ))**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))
	def get_sLag_FH(self, x, s0):
		return np.insert(self(x,'s', ns = 'EE_FH'), 0, s0)

	########## 5.2. EE with policy functions for each t
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

	def solve_EE_FH_polFunc(self, gridsol, θ, eps, s0, x0_EE_FH_polFunc = None):
		θp, epsp = self.leadSym(θ), self.leadSym(eps)
		policyFunction = self.aux_vecPolFunction(gridsol)
		sol = optimize.root(lambda x: self.EE_FH_pol_eqs(x, policyFunction, θ, eps, θp, epsp, s0), noneInit(x0_EE_FH_polFunc, self.getx0('EE_FH_polFunc')))
		assert sol['success'], f""" Could not identify economic equilibrium (self.solve_EE_FH_polFunc) with parameter inputs: 
		θ: {θ}, ε: {eps}, s0: {s0}"""
		return self.report_EE_FH_polFunc(sol, policyFunction, s0)

	def EE_FH_pol_eqs(self, x, policyFunction, θ, eps, θp, epsp, s0):
		τ = policyFunction(self.get_sLag_FH(x,s0))
		τp = self.leadSym(τ)
		return self.EE_FH_eqs(x, τ, θ, eps, τp, θp, epsp, s0)

	def report_EE_FH_polFunc(self, sol, policyFunction, s0):
		d = self.ns['EE_FH'].unloadSol(sol['x'])
		d['s[t-1]'] = pd.Series(self.get_sLag_FH(sol['x'], s0), index = self.db['t'])
		d['τ'] = pd.Series(policyFunction(d['s[t-1]']), index = self.db['t'])
		return d

	########## 5.3. EE with policy functions for τ, θ, ε for each t
	def solve_EE_ESC_FH(self, gridsol, s0, x0_EE_ESC_FH = None):
		τf = self.aux_vecPolFunction(gridsol, y = 'τ')
		θf = self.aux_vecPolFunction(gridsol, y = 'θ')
		epsf = self.aux_vecPolFunction(gridsol, y = 'eps')
		sol = optimize.root(lambda x: self.EE_ESC_FH_eqs(x, τf, θf, epsf, s0), noneInit(x0_EE_ESC_FH, self.getx0('EE_ESC_FH')))
		assert sol['success'], f""" Could not identify economic equilibrium (self.solve_EE_ESC_FH)."""
		return self.report_EE_ESC_FH(sol, τf, θf, epsf, s0)

	def EE_ESC_FH_eqs(self, x, τf, θf, epsf, s0):
		s_ = self.get_sLag(x, s0)
		τ, θ, eps = τf(s_), θf(s_), epsf(s_)
		τp, θp, epsp = self.leadSym(τ), self.leadSym(θ), self.leadSym(eps)
		return self.EE_FH_eqs(x, τ, θ, eps, τp, θp, epsp, s0)

	def report_EE_ESC_FH(self, sol, τf, θf, epsf, s0):
		d = self.ns['EE_FH'].unloadSol(sol['x'])
		d['s[t-1]'] = pd.Series(self.get_sLag_FH(sol['x'], s0), index = self.db['t'])
		d['τ'] = pd.Series(τf(d['s[t-1]']), index = self.db['t'])
		d['θ'] = pd.Series(θf(d['s[t-1]']), index = self.db['t'])
		d['eps'] = pd.Series(epsf(d['s[t-1]']), index = self.db['t'])
		return d


	#######################################################################
	##########			6.  Finite Horizon Terminal State		###########
	#######################################################################

	########## 6.1: PEE 
	def solve_PEE_T(self, θ, eps, ν, sgrid, ite_PEE_T = True, **kwargs):
		sol = self.PEE_precomputations_T(sgrid)
		τ = np.empty(self.ngrid)
		x =  optimize.root(lambda τ: self.PEE_polObj_T(τ, θ, eps, ν, self.aux_soli(sol, 0)), self.getx0('PEE_T', i = 0))
		τ[0] = x['x']
		assert x['success'], f""" Could not identify PEE solution"""
		for i in range(1, self.ngrid):
			x = optimize.root(lambda τ: self.PEE_polObj_T(τ, θ, eps, ν, self.aux_soli(sol, i)), τ[i-1] if ite_PEE_T else self.getx0('PEE_T', i = i))
			τ[i] = x['x']
			assert x['success'], f""" Could not identify PEE solution"""
		return self.report_PEE_T(τ, θ, eps, ν, sol)

	def solve_PEEvec_T(self, θ, eps, ν, sgrid, **kwargs):
		""" Use ite_PEE_t = True to use sol_p['τ'] as initial value, 
			use 't' to indicate if the default initial value from self.x0 is a dictionary and it should rely on the t'th entry """
		sol = self.PEE_precomputations_T(sgrid)
		τ = optimize.root(lambda τ: self.PEE_polObjVec_T(τ, θ, eps, ν, sol), self.getx0('PEEvec_T'), **kwargs)
		assert τ['success'], f"""Could not identify PEE solution in terminal state (self.solve_PEEvec_T) with parameters:
		θ: {θ}, ε: {eps}, ν: {ν}"""
		return self.report_PEE_T(τ['x'], θ, eps, ν, sol)

	def PEE_polObj_T(self, τ, θ, eps, ν, sol):
		τBound = np.clip(τ, self.db['τl'], self.db['τu'])
		funcOfτ = self.funcOfτ_T(τBound, θ, eps, sol, ν)
		condition = (self.db['γ0']*self.ω20*self.aux_PEE_HtM_old_T(τBound, eps, ν, sol, funcOfτ)+np.matmul(self.ω2i * self.db['γi'], self.aux_PEE_retirees_T(τBound, θ, eps, ν, sol, funcOfτ))
			+ν*(self.db['γ0']*self.ω10*self.aux_PEE_HtM_young_T(τBound, eps, ν, sol, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_PEE_workers_scalar_T(sol, funcOfτ))))
		return self.adjustFocMultiplicative(condition, τ, l = self.db['τl'], u = self.db['τu'], kl = self.db['kτ_l'], ku = self.db['kτ_u'])

	def PEE_polObjVec_T(self, τ, θ, eps, ν, sol):
		τBound = np.clip(τ, self.db['τl'], self.db['τu'])
		funcOfτ = self.funcOfτ_T(τBound, θ, eps, sol, ν)
		condition = (self.db['γ0']*self.ω20*self.aux_PEE_HtM_old_T(τBound, eps, ν, sol, funcOfτ)+np.matmul(self.ω2i * self.db['γi'], self.aux_PEE_retirees_T(τBound, θ, eps, ν, sol, funcOfτ))
			+ν*(self.db['γ0']*self.ω10*self.aux_PEE_HtM_young_T(τBound, eps, ν, sol, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_PEE_workers_T(sol, funcOfτ))))
		return self.adjustFocMultiplicative(condition, τ, l = self.db['τl'], u = self.db['τu'], kl = self.db['kτ_l'], ku = self.db['kτ_u'])

	def PEE_precomputations_T(self, sgrid):
		sol = {'s' : np.zeros(self.ngrid), 's[t-1]': sgrid, 'dln(h)/dln(s)': np.zeros(self.ngrid)}
		return sol

	def funcOfτ_T(self, τ, θ, eps, sol, ν):
		funcOfτ = {'h': self.aux_h_T(τ, sol['s[t-1]'], ν), 'dln(h)/dτ': -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-τ))}
		funcOfτ['B'] = self.aux_B(sol['s[t-1]'], funcOfτ['h'], ν)
		funcOfτ['Γs'] = self.aux_Γs(funcOfτ['B'], τ, θ, eps)
		funcOfτ['si/s'] = self.savingsSpread(funcOfτ['B'], funcOfτ['Γs'], τ, θ, eps)
		return funcOfτ

	def report_PEE_T(self, τ, θ, eps, ν, sol):
		""" Return solution dictionary given vector of taxes"""
		sol['τ'] = np.clip(τ, self.db['τl'], self.db['τu'])
		sol['τ_notBound'] = τ
		sol['h'] = self.aux_h_T(sol['τ'], sol['s[t-1]'], ν)
		sol['B']  = self.aux_B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.aux_Γs(sol['B'], sol['τ'], θ, eps)
		sol['∂τ/∂s'] = np.gradient(sol['τ'], sol['s[t-1]'])
		sol['dln(h)/dln(s)'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s)'] = np.full(self.ngrid, self.power_h)
		sol['∂ln(h)/∂τ'] = -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-sol['τ']))
		return sol

	def aux_PEE_HtM_old_T(self, τ, eps, ν, sol, funcOfτ):
		return self.aux_PEE_HtM_old_(τ, eps, funcOfτ['dln(h)/dτ'], self.aux_c20_t(sol['s[t-1]'], funcOfτ['h'], ν, τ, eps))

	def aux_PEE_retirees_T(self, τ, θ, eps, ν, sol, funcOfτ):
		c2i_coeff = self.aux_c2i_coeff(funcOfτ['si/s'], τ, θ, eps)
		return self.aux_PEE_retirees_(θ, eps, funcOfτ['dln(h)/dτ'], self.aux_c2i_t(τ, eps, sol['s[t-1]'], funcOfτ['h'], ν, c2i_coeff), c2i_coeff)

	def aux_PEE_HtM_young_T(self, τ, eps, ν, sol, funcOfτ):
		return -(1+self.db['ξ'])*self.aux_c̃10_t(τ, sol['s[t-1]'], funcOfτ['h'], ν)**(1-1/self.db['ρ'])*(self.db['α']*funcOfτ['dln(h)/dτ']+(τ/(1-τ))*self.db['ξ']/(1+τ*self.db['ξ']))

	def aux_PEE_workers_T(self, sol, funcOfτ):
		return self.aux_c̃1i_T(funcOfτ['h'])**(1-1/self.db['ρ'])*(1+self.db['ξ'])*funcOfτ['dln(h)/dτ']/self.db['ξ']

	def aux_PEE_workers_scalar_T(self, sol, funcOfτ):
		return self.aux_c̃1i_scalar_T(funcOfτ['h'])**(1-1/self.db['ρ'])*(1+self.db['ξ'])*funcOfτ['dln(h)/dτ']/self.db['ξ']

	def aux_h_T(self, τ, s_, ν):
		return ((1-self.db['α'])*(1-τ))**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))*(s_/ν)**self.power_h

	def aux_c̃1i_T(self, h):
		return (h**((1+self.db['ξ'])/self.db['ξ'])/(1+self.db['ξ']))*self.aux_Prod.reshape(self.ni,1)

	def aux_c̃1i_scalar_T(self, h):
		return (h**((1+self.db['ξ'])/self.db['ξ'])/(1+self.db['ξ']))*self.aux_Prod

	########## 6.2: ESC Terminal Policy 
	def solve_ESC_T(self, ν, sgrid, ite_ESC_T = True, **kwargs):
		sol = self.PEE_precomputations_T(sgrid)
		esc = np.empty((self.ngrid, 3))
		x =  optimize.root(lambda x: self.ESC_polObj_T(x[0], x[1], x[2], ν, self.aux_soli(sol, 0)), self.getx0('ESC_T', i = 0))
		esc[0] = x['x']
		assert x['success'], f""" Could not identify ESC solution in T"""
		for i in range(1, self.ngrid):
			x = optimize.root(lambda x: self.ESC_polObj_T(x[0], x[1], x[2], ν, self.aux_soli(sol, i)), esc[i-1] if ite_ESC_T else self.getx0('ESC_T', i = i))
			esc[i] = x['x']
			# assert x['success'], f""" Could not identify ESC solution, iteration {i}"""
		return self.report_ESC_T(np.hstack([esc[:,0], esc[:,1], esc[:,2]]), ν, sol)

	def solve_ESCvec_T(self, ν, sgrid, **kwargs):
		sol = self.PEE_precomputations_T(sgrid)
		esc = optimize.root(lambda x: self.ESC_polObjVec_T(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, sol),  self.getx0('ESC_T'), **kwargs)
		assert esc['success'], f"""Could not identify ESC solution (self.solve_ESCvec_T) with parameters ν={ν}"""
		return self.report_ESC_T(esc['x'], ν, sol)


	def ESC_polObj_T(self, τ, θ, eps, ν, sol):
		τBound , θBound, epsBound = np.clip(τ, self.db['τl'], self.db['τu']), np.clip(τ, self.db['θl'], self.db['θu']), np.clip(τ, self.db['epsl'], self.db['epsu'])
		funcOfτ = self.funcOfτ_T(τBound, θBound, epsBound, sol, ν)
		young = ν*(self.db['γ0']*self.ω10*self.aux_PEE_HtM_young_T(τBound, epsBound, ν, sol, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_PEE_workers_scalar_T(sol, funcOfτ)))
		euler = self.aux_ESC_retirees_T(τBound, θBound, epsBound, ν, sol, funcOfτ)
		htm   = self.aux_ESC_HtM_old_T(τBound, epsBound, ν, sol, funcOfτ)
		return np.hstack([self.adjustFocMultiplicative(young+np.matmul(self.ω2i*self.db['γi'], euler['τ'])+self.db['γ0']*self.ω20*htm['τ'], τ, l = self.db['τl'], u = self.db['τu'], kl = self.db['kτ_l'], ku = self.db['kτ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['θ']), θ, l = self.db['θl'], u = self.db['θu'], kl = self.db['kθ_l'], ku = self.db['kθ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['eps'])+self.db['γ0']*self.ω20*htm['eps'], eps, l = self.db['epsl'], u = self.db['epsu'], kl = self.db['keps_l'], ku = self.db['keps_u'])])

	def ESC_polObjVec_T(self, τ, θ, eps, ν, sol):
		τBound , θBound, epsBound = np.clip(τ, self.db['τl'], self.db['τu']), np.clip(τ, self.db['θl'], self.db['θu']), np.clip(τ, self.db['epsl'], self.db['epsu'])
		funcOfτ = self.funcOfτ_T(τBound, θBound, epsBound, sol, ν)
		young = ν*(self.db['γ0']*self.ω10*self.aux_PEE_HtM_young_T(τBound, epsBound, ν, sol, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_PEE_workers_T(sol, funcOfτ)))
		euler = self.aux_ESC_retirees_T(τBound, θBound, epsBound, ν, sol, funcOfτ)
		htm   = self.aux_ESC_HtM_old_T(τBound, epsBound, ν, sol, funcOfτ)
		return np.hstack([self.adjustFocMultiplicative(young+np.matmul(self.ω2i*self.db['γi'], euler['τ'])+self.db['γ0']*self.ω20*htm['τ'], τ, l = self.db['τl'], u = self.db['τu'], kl = self.db['kτ_l'], ku = self.db['kτ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['θ']), θ, l = self.db['θl'], u = self.db['θu'], kl = self.db['kθ_l'], ku = self.db['kθ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['eps'])+self.db['γ0']*self.ω20*htm['eps'], eps, l = self.db['epsl'], u = self.db['epsu'], kl = self.db['keps_l'], ku = self.db['keps_u'])])

	def report_ESC_T(self, x, ν, sol):
		""" Return solution dictionary given vector of taxes"""
		sol['τ_notBound'] = self(x, 'τ')
		sol['θ_notBound'] = self(x,'θ')
		sol['eps_notBound'] = self(x, 'eps')
		sol['τ'], sol['θ'], sol['eps'] = np.clip(self(x, 'τ'), self.db['τl'], self.db['τu']), np.clip(self(x, 'θ'), self.db['θl'], self.db['θu']), np.clip(self(x,'eps'), self.db['epsl'], self.db['epsu'])
		sol['h'] = self.aux_h_T(sol['τ'], sol['s[t-1]'], ν)
		sol['B']  = self.aux_B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.aux_Γs(sol['B'], sol['τ'], sol['θ'], sol['eps'])
		sol.update({f'∂{k}/∂s': np.gradient(sol[f'{k}'], sol['s[t-1]']) for k in ('τ','θ','eps')})
		sol['dln(h)/dln(s)'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s)'] = np.full(self.ngrid, self.power_h)
		sol['∂ln(h)/∂τ'] = -self.db['ξ']/((1+self.db['α']*self.db['ξ'])*(1-sol['τ']))
		return sol

	def aux_ESC_retirees_T(self, τ, θ, eps, ν, sol, funcOfτ):
		c2i_coeff = self.aux_c2i_coeff(funcOfτ['si/s'], τ, θ, eps)
		c2i = self.aux_c2i_t(τ, eps, sol['s[t-1]'], funcOfτ['h'], ν, c2i_coeff)
		return {'τ': self.aux_PEE_retirees_(θ, eps, funcOfτ['dln(h)/dτ'], c2i, c2i_coeff),
				'θ': self.aux_ESC_retirees_θ(τ, θ, eps, c2i, c2i_coeff),
				'eps': self.aux_ESC_retirees_eps(τ, θ, eps, c2i, c2i_coeff)}

	def aux_ESC_HtM_old_T(self, τ, eps, ν, sol, funcOfτ):
		c20 = self.aux_c20_t(sol['s[t-1]'], funcOfτ['h'], ν, τ, eps)
		return {'τ': self.aux_PEE_HtM_old_T(τ, eps, ν, sol, funcOfτ),
				'eps': self.aux_ESC_HtM_old_eps(eps, c20)}
		

	#######################################################################
	##########			8.  Policy function period t			###########
	#######################################################################

	########## 8.1: PEE 
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
		x = optimize.root(lambda τ: self.PEE_polObj_t(τ, θ, eps, ν, θp, epsp, νp, self.aux_soli(sol, 0), self.aux_soli(sol_p,0)), sol_p['τ_notBound'][0] if ite_PEE_t else self.getx0('PEE_t',t=t,i=0), **kwargs)
		τ[0] = x['x']
		assert x['success'], f""" Couldn't identify with ν = {ν}, loop i=0 """
		for i in range(1, self.ngrid):
			x = optimize.root(lambda τ: self.PEE_polObj_t(τ, θ, eps, ν, θp, epsp, νp, self.aux_soli(sol, i), self.aux_soli(sol_p,i)), τ[i-1] if ite_PEE_t else self.getx0('PEE_t',t=t,i=i), **kwargs)
			τ[i] = x['x']
			assert x['success'], f""" Couldn't identify PEE iteration i = {i} out of {self.ngrid} """
		return self.report_PEE_t(τ, θ, eps, ν, sol, sol_p)

	def PEE_polObjVec_t(self, τ, θ, eps, ν, θp, epsp, νp, sol, sol_p):
		τBound = np.clip(τ, self.db['τl'], self.db['τu'])
		funcOfτ = self.funcOfτ(τBound, θ, eps, ν, sol, sol_p)
		condition = (self.db['γ0']*self.ω20*self.aux_PEE_HtM_old_t(τBound, eps, ν, sol, funcOfτ)+np.matmul(self.ω2i * self.db['γi'], self.aux_PEE_retirees_t(τBound, θ, eps, ν, sol, funcOfτ))
				 +ν*(self.db['γ0']*self.ω10*self.aux_PEE_HtM_young_t(τBound, epsp, ν, νp, sol, sol_p, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_PEE_workers_t(θp, epsp, sol, sol_p, funcOfτ))))
		return self.adjustFocMultiplicative(condition, τ, l = self.db['τl'], u = self.db['τu'], kl = self.db['kτ_l'], ku = self.db['kτ_u'])

	def PEE_polObj_t(self, τ, θ, eps, ν, θp, epsp, νp, sol, sol_p):
		τBound = np.clip(τ, self.db['τl'], self.db['τu'])
		funcOfτ = self.funcOfτ(τBound, θ, eps, ν, sol, sol_p)
		condition = (self.db['γ0']*self.ω20*self.aux_PEE_HtM_old_t(τBound, eps, ν, sol, funcOfτ)+np.matmul(self.ω2i * self.db['γi'], self.aux_PEE_retirees_t(τBound, θ, eps, ν, sol, funcOfτ))
				 +ν*(self.db['γ0']*self.ω10*self.aux_PEE_HtM_young_t(τBound, epsp, ν, νp, sol, sol_p, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_PEE_workers_scalar_t(θp, epsp, sol, sol_p, funcOfτ))))
		return self.adjustFocMultiplicative(condition, τ, l = self.db['τl'], u = self.db['τu'], kl = self.db['kτ_l'], ku = self.db['kτ_u'])

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

	def report_PEE_t(self, τ, θ, eps, ν, sol, sol_p):
		""" Return solution dictionary given vector of taxes"""
		sol['τ'] = np.clip(τ, self.db['τl'], self.db['τu'])
		sol['τ_notBound'] = τ
		sol['s[t-1]'] = sol['s_τ0']*(1-sol['τ'])**(-1/self.db['α'])
		sol['B']  = self.aux_B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.aux_Γs(sol['B'], sol['τ'], θ, eps)
		sol['∂τ/∂s'] = np.gradient(sol['τ'], sol['s[t-1]'])
		sol['dln(h)/dln(s)'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s)'] = self.db['α']*self.db['ξ']*(1+sol['Ψ']*(1-sol_p['∂ln(h)/∂ln(s)']))/((1+self.db['α']*self.db['ξ'])*sol['σ'])
		sol.update(self.aux_derivatives(sol['Ψ'], sol['σ'], sol_p['∂ln(h)/∂ln(s)'], sol['τ']))
		return sol

	def aux_PEE_logDev(self, sol, funcOfτ, k):
		if k == 's':
			return funcOfτ[f'∂ln({k})/∂τ']/(1-sol[f'{k}_strategy'])
		else:
			return funcOfτ[f'∂ln({k})/∂τ']+sol[f'{k}_strategy'] * funcOfτ['dln(s)/dτ']

	def aux_PEE_strategy(self, sol, sol_p, k):
		return sol[f'∂ln({k})/∂τ[t+1]'] * sol_p['∂τ/∂s'] * sol['s']

	########## 8.2: ESC
	def solve_ESCvec_t(self, ν, νp, sol_p, ite_ESCvec_t = True, t = None, **kwargs):
		sol = self.ESC_precomputations(ν, sol_p)
		esc = optimize.root(lambda x: self.ESC_polObjVec_t(self(x,'τ'), self(x,'θ'), self(x,'eps'), ν, νp, sol, sol_p), np.hstack([sol_p['τ_notBound'], sol_p['θ_notBound'], sol_p['eps_notBound']]) if ite_ESCvec_t else self.getx0('ESCvec_t',t=t), **kwargs)
		assert esc['success'], f"""Could not identify ESC solution (self.solve_ESCvec_t) with parameters:
		ν: {ν}, ν[t+1]: {νp}, and solution from t+1 with:
		τ[t+1]: {sol_p['τ']}, 
		θ[t+1]: {sol_p['θ']},
		ε[t+1]: {sol_p['eps']}"""
		return self.report_ESCB_t(esc['x'], ν, sol, sol_p)

	def solve_ESC_t(self, ν, νp, sol_p, ite_ESC_t = True, t = None, **kwargs):
		sol = self.ESC_precomputations(ν, sol_p)
		esc = np.empty((self.ngrid, 3))
		esc[0] = optimize.root(lambda x: self.ESC_polObj_t(x[0], x[1], x[2], ν, νp, self.aux_soli(sol, 0), self.aux_soli(sol_p,0)), [sol_p[k][0] for k in ['τ_notBound','θ_notBound','eps_notBound']] if ite_ESC_t else self.getx0('ESC_t',t=t,i=0), **kwargs)['x']
		for i in range(1, self.ngrid):
			esc[i] = optimize.root(lambda x: self.ESC_polObj_t(x[0], x[1], x[2], ν, νp, self.aux_soli(sol, i), self.aux_soli(sol_p,i)), esc[i-1] if ite_ESC_t else self.getx0('ESC_t',t=t,i=i), **kwargs)['x']
		return self.report_ESC_t(np.hstack([esc[:,0], esc[:,1], esc[:,2]]), ν, sol, sol_p)

	def report_ESC_t(self, x, ν, sol, sol_p):
		""" Return solution dictionary given vector of taxes"""
		sol['τ_notBound'] = self(x, 'τ')
		sol['θ_notBound'] = self(x,'θ')
		sol['eps_notBound'] = self(x, 'eps')
		sol['τ'], sol['θ'], sol['eps'] = np.clip(self(x, 'τ'), self.db['τl'], self.db['τu']), np.clip(self(x, 'θ'), self.db['θl'], self.db['θu']), np.clip(self(x,'eps'), self.db['epsl'], self.db['epsu'])
		sol['s[t-1]'] = sol['s_τ0']*(1-sol['τ'])**(-1/self.db['α'])
		sol['B']  = self.aux_B(sol['s[t-1]'], sol['h'], ν)
		sol['Γs'] = self.aux_Γs(sol['B'], sol['τ'], sol['θ'], sol['eps'])
		sol.update({f'∂{k}/∂s': np.gradient(sol[f'{k}'], sol['s[t-1]']) for k in ('τ','θ','eps')})
		sol['dln(h)/dln(s)'] = np.gradient(sol['h'], sol['s[t-1]']) * sol['s[t-1]']/sol['h']
		sol['∂ln(h)/∂ln(s)'] = self.db['α']*self.db['ξ']*(1+sol['Ψ']*(1-sol_p['∂ln(h)/∂ln(s)']))/((1+self.db['α']*self.db['ξ'])*sol['σ'])
		sol.update(self.aux_derivatives(sol['Ψ'], sol['σ'], sol_p['∂ln(h)/∂ln(s)'], sol['τ']))
		return sol

	def ESC_polObj_t(self, τ, θ, eps, ν, νp, sol, sol_p):
		""" Returns stacked FOCs for τ, θ, eps. θ, eps from sol_p are bounded versions (main solution), but inputs θ,eps are not yet bounded:"""
		τBound, θBound, epsBound = np.clip(τ, 1e-4, 1), np.clip(θ, 0, 1), np.clip(eps, 0,1)
		funcOfτ = self.ESC_funcOfτ(τBound, θBound, epsBound, ν, sol, sol_p)
		young = ν*(self.db['γ0']*self.ω10*self.aux_ESC_HtM_young_t(τBound, ν, νp, sol, sol_p, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_ESC_workers_scalar_t(sol, sol_p, funcOfτ)))
		euler = self.aux_ESC_retirees_t(τBound, θBound, epsBound, ν, sol, funcOfτ)
		htm   = self.aux_ESC_HtM_old_t(τBound, epsBound, ν, sol, funcOfτ)
		return np.hstack([self.adjustFocMultiplicative(young+np.matmul(self.ω2i*self.db['γi'], euler['τ'])+self.db['γ0']*self.ω20*htm['τ'], τ, l = 1e-4, kl = self.db['kτ_l'], ku = self.db['kτ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['θ']), θ, kl = self.db['kθ_l'], ku = self.db['kθ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['eps'])+self.db['γ0']*self.ω20*htm['eps'], eps, kl = self.db['keps_l'], ku = self.db['kθ_u'])])

	def ESC_polObjVec_t(self, τ, θ, eps, ν, νp, sol, sol_p):
		""" Returns stacked FOCs for τ, θ, eps. θ, eps from sol_p are bounded versions (main solution), but inputs θ,eps are not yet bounded:"""
		τBound, θBound, epsBound = np.clip(τ, 1e-4, 1), np.clip(θ, 0, 1), np.clip(eps, 0,1)
		funcOfτ = self.ESC_funcOfτ(τBound, θBound, epsBound, ν, sol, sol_p)
		young = ν*(self.db['γ0']*self.ω10*self.aux_ESC_HtM_young_t(τBound, ν, νp, sol, sol_p, funcOfτ)+np.matmul(self.ω1i*self.db['γi'], self.aux_ESC_workers_t(sol, sol_p, funcOfτ)))
		euler = self.aux_ESC_retirees_t(τBound, θBound, epsBound, ν, sol, funcOfτ)
		htm   = self.aux_ESC_HtM_old_t(τBound, epsBound, ν, sol, funcOfτ)
		return np.hstack([self.adjustFocMultiplicative(young+np.matmul(self.ω2i*self.db['γi'], euler['τ'])+self.db['γ0']*self.ω20*htm['τ'], τ, l = 1e-4, kl = self.db['kτ_l'], ku = self.db['kτ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['θ']), θ, kl = self.db['kθ_l'], ku = self.db['kθ_u']),
						  self.adjustFocMultiplicative(np.matmul(self.ω2i*self.db['γi'], euler['eps'])+self.db['γ0']*self.ω20*htm['eps'], eps, kl = self.db['keps_l'], ku = self.db['kθ_u'])])

	def ESC_precomputations(self, ν, sol_p):
		""" sol_p is the solution from t+1 """
		sol = {'s': sol_p['s[t-1]'],
					'h': (sol_p['s[t-1]']/sol_p['Γs'])**(self.db['ξ']/(1+self.db['ξ'])),
					'Ω': self.aux_Ω(sol_p['Γs'], sol_p['τ'], sol_p['θ'], sol_p['eps']),
					'Ψ': self.aux_Ψ(sol_p['B'], sol_p['τ'], sol_p['θ'], sol_p['eps'])}
		sol['s_τ0'] = ν*sol['h']**(1/self.power_h)*((1-self.db['αr']*sol_p['θ']*self.auxPen(sol_p['τ'], sol_p['eps'])*sol_p['Γs'])/((1-self.db['α'])))**(1/self.db['α'])
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

	def aux_ESC_strategy(self, sol, sol_p, k):
		return self.aux_PEE_strategy(sol, sol_p, k) + (sol[f'∂ln({k})/∂θ[t+1]']*sol_p['∂θ/∂s']+sol[f'∂ln({k})/∂eps[t+1]']*sol_p['∂eps/∂s'])*sol['s']

	#######################################################################
	##########		9.  Auxiliary functions for PEE/ESC			###########
	#######################################################################

	### 9.1. HTM RETIREES:
	def aux_ESC_HtM_old_t(self, τ, eps, ν, sol, funcOfτ):
		c20 = self.aux_c20_t(funcOfτ['s[t-1]'], sol['h'],ν,τ,eps)
		return {'τ': self.aux_PEE_HtM_old_(τ, eps, funcOfτ['dln(h)/dτ'], c20),
				'eps': self.aux_ESC_HtM_old_eps(eps, c20)}

	def aux_ESC_HtM_old_eps(self, eps, c20):
		return c20**(1-1/self.db['ρ'])*(1/eps-self.db['p̄']/(eps*self.db['p̄']+self.db['p']))

	def aux_PEE_HtM_old_t(self, τ, eps, ν, sol, funcOfτ):
		return self.aux_PEE_HtM_old_(τ, eps, funcOfτ['dln(h)/dτ'], self.aux_c20_t(funcOfτ['s[t-1]'], sol['h'], ν, τ, eps))

	def aux_PEE_HtM_old_(self, τ, eps, dlnh_dτ, c20):
		return c20**(1-1/self.db['ρ'])*(1/τ+(1-self.db['α'])*dlnh_dτ)

	def aux_c20_t(self, s, h, ν, τ, eps):
		return (1-self.db['α'])*ν*(s/ν)**(self.db['α'])*h**(1-self.db['α'])*eps*τ/(eps*self.db['p̄']+self.db['p'])

	### 9.2. EULER RETIREES
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

	### 9.3. HTM YOUNG
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

	### 9.4 EULER YOUNG
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

