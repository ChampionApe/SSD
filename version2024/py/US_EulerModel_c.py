import numpy as np, pandas as pd

class C:
	def __init__(self, m, ts = 'FH'):
		self.m = m
		self.db = m.db
		self.ts = ts
	@property
	def ω2i(self):
		return self.db['ω'] * self.db['pi'] * self.db['μi']
	@property
	def ω1i(self):
		return self.db['μi']
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
	def auxProd(self):
		return np.power(self.db['ηi'], 1+self.db['ξ'])/np.power(self.db['Xi'], self.db['ξ'])
	def auxΓB1(self, B):
		return np.matmul(self.db['γi'] * self.auxProd, B/(1+B))
	def auxΓB2(self,B):
		return np.matmul(self.db['γi'], 1/(1+B))
	def auxΓB3(self,B):
		return np.matmul(self.auxProd * self.db['γi'], B/((1+B)**2))
	def auxΓB4(self,B):
		return np.matmul(self.db['γi'], B/((1+B)**2))

	def aux_sτ0(self, h = None, ν = None, τp = None, κp = None, Γs = None):
		return ν*h**(1/self.power_h)*((1-self.db['αr']*κp*τp*Γs)/((1-self.db['α'])))**(1/self.db['α'])

	def adjustFocMultiplicative(self, z, x, l = 0, u = 1, kl = 10, ku = 10):
		""" Adjust FOC in a multiplicative way: z is the "original" marginal effect, x is the bounded variable """
		return z-abs(z)*(kl*(np.clip(x, None, l)-l)+ku*(np.clip(x,u,None)-u))

	def interiorFOC(self, z, x, var = 'τ'):
		return self.adjustFocMultiplicative(z, x, l = self.db[f'{var}_l'], u = self.db[f'{var}_u'], kl = self.db[f'k{var}_l'], ku = self.db[f'k{var}_u'])

	def R(self, s_ = None, h = None, ν = None, A = 1):
		return self.db['α'] * A * (ν*h/s_)**(1-self.db['α'])

	def B(self, s_ = None, h = None, ν = None, A = 1):
		return self.db['βi'].reshape(self.m.ni,1)**self.db['ρ'] * (self.R(s_ ,h, ν, A = A)/self.db['p'])**(self.db['ρ']-1)
	
	def Γs(self, B = None, τp = None, κp = None):
		""" τp, Bp, θp, epsp are vectors of the same length"""
		return (1/(1+self.db['ξ']))*self.auxΓB1(B)/(1+self.db['αr']*τp*(κp+self.auxΓB2(B)*(1-κp)))

	def savingsSpread(self, B = None, Γs = None, τp = None, κp = None):
		return self.auxProd.reshape(self.m.ni,1) * B / ((1+B)*(1+self.db['ξ'])*Γs)-self.db['αr']*τp*(κp*(self.auxProd.reshape(self.m.ni, 1)-1/(1+B))+1/(1+B))

	def steadyStateB_eq(self, B, Γs, τ, κ, ν):
		""" Requirement = 0 for B to be in steady state - this does not return ss level of B"""
		if self.db['ρ']<1:
			return self.db['βi'].reshape(self.m.ni,1)**(self.db['ρ']/(1-self.db['ρ']))*(1-self.db['α'])*(1-τ)-B**(1/(1-self.db['ρ']))*(ν/self.db['p'])*(self.db['α']-(1-self.db['α'])*κ*τ*Γs)/Γs
		elif self.db['ρ']>1:
			return B**(1/(self.db['ρ']-1))*(1-self.db['α'])*(1-τ)-self.db['βi'].reshape(self.m.ni,1)**(self.db['ρ']/(self.db['ρ']-1))*(ν/self.db['p'])*(self.db['α']-(1-self.db['α'])*κ*τ*Γs)/Γs

	def steadyStateΓs_eq(self, B, Γs, τ, κ, ν):
		""" Requirement = 0 for Γs to be in steady state - this does not return ss level of Γs """
		return self.auxΓB1(B)-Γs*(1+self.db['ξ'])*(1+self.db['αr']*τ*(κ+self.auxΓB2(B)*(1-κ)))

	def s_SS(self, Γs, τ, κ, ν):
		""" Return steady state level of savings"""
		return ( ((1-self.db['α'])*(1-τ)/(1-self.db['αr']*τ*κ*Γs))**(1+self.db['ξ'])*Γs**(1+self.db['α']*self.db['ξ'])/(ν**(self.db['α']*(1+self.db['ξ']))) )**(1/(1-self.db['α']))

	def dlnh_ds_SS(self, Ω, Ψ, τ, κ):
		""" get steady state level of ∂ln(h)/∂ln(s) given solution to solve_ss (ss) and parameters """
		a = Ψ*(1+τ*κ*Ω*(1+self.db['ξ'])/(1+self.db['ξ']*self.db['α']))
		b = -(1+Ψ*(1+self.db['α']*self.db['ξ']+τ*κ*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])))
		c = self.db['α']*self.db['ξ']*(1+Ψ)
		return (-b-np.sqrt(b**2-4*a*c))/(2*a)

	###### Log methods:
	def log_s_FH(self, sol, s0 = None):
		s = pd.Series(None, index = self.db['txE'], dtype = float)
		s.iloc[0] = sol['Θs'].iloc[0]*(s0/self.db['ν'][0])**(self.power_s)
		for t in self.db['txE'][1:]:
			s[t] = sol['Θs'].iloc[t]*(s[t-1]/self.db['ν'][t])**(self.power_s)
		return s
	def log_s_IH(self, sol, s0 = None):
		s = pd.Series(None, index = self.db['t'], dtype = float)
		s.iloc[0] = sol['Θs'].iloc[0]*(s0/self.db['ν'][0])**(self.power_s)
		for t in self.db['t'][1:]:
			s[t] = sol['Θs'].iloc[t]*(s[t-1]/self.db['ν'][t])**(self.power_s)
		return s
	def log_h_FH(self, sol):
		return sol['Θh']*(sol['s[t-1]']/self.db['ν'])**self.power_h

	@property
	def h_IH(self):
		return self.h_t
	def h_FH(self, τ = None, τp = None, κp = None, Γs = None, s_ = None, ν = None):
		return np.hstack([self.h_t(τ[:-1], τp[:-1], κp[:-1], Γs, s_[:-1], ν[:-1]),
						  self.h_T(τ[-1], s_[-1], ν[-1])])
	@property
	def Θh_IH(self):
		return self.Θh_t
	def Θh_FH(self, τ = None, τp = None, κp = None, Γs = None):
		return np.hstack([self.Θh_t(τ = τ[:-1], τp = τp[:-1], κp = κp[:-1], Γs = Γs), self.Θh_T(τ = τ[-1])])
	def Θh_t(self, τ = None, τp = None, κp = None, Γs = None):
		return ((1-self.db['α'])*(1-τ)/(1-self.db['αr']*τp*κp*Γs))**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))
	def Θh_T(self, τ = None):
		return ((1-self.db['α'])*(1-τ))**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))
	def h_t(self, τ = None, τp  = None, κp = None, Γs = None, s_ = None, ν = None):
		""" Note: Remove kwargs when it is done"""
		return self.Θh_t(τ = τ, τp = τp, κp=κp, Γs = Γs)*(s_/ν)**self.power_h
	def h_T(self, τ = None, s_ = None, ν = None):
		return self.Θh_T(τ = τ)*(s_/ν)**self.power_h

	def Γs_EE(self, s, hp, νp, τp, κp):
		return self.Γs(self.B(s, hp, νp), τp, κp)
	def s_t(self, h, Γs):
		return h**((1+self.db['ξ'])/self.db['ξ'])*Γs

	@property
	def ĉ1i_IH(self):
		return self.ĉ1i_t
	@property
	def ĉ1i_FH(self):
		""" This returns full length vector for simplicity - even though this is not defined for the terminal period """
		return self.ĉ1i_t	
	def ĉ1i_t(self, τp = None, κp = None, h = None, B = None, Γs = None):
		return h**((1+self.db['ξ'])/self.db['ξ'])*(1+B)**(1/(self.db['ρ']-1))*(self.auxProd.reshape(self.m.ni,1)/(1+self.db['ξ'])+self.db['αr']*τp*(1-κp)*Γs)
	def c̃1i_T(self, h = None):
		return (h**((1+self.db['ξ'])/self.db['ξ']))*self.auxProd.reshape(self.m.ni,1)/(1+self.db['ξ'])
		
	@property
	def c2i_IH(self):
		return self.c2i_t
	@property
	def c2i_FH(self):
		return self.c2i_t	
	def c2i_t(self, s_ = None, h = None, ν = None, c2i_coeff = None):
		return self.db['α']*(ν/self.db['p'])*(s_/ν)**self.db['α']*h**(1-self.db['α'])*c2i_coeff
	def aux_c2i_coeff(self, sSpread = None, τ = None, κ = None):
		return sSpread+self.db['αr']*τ*(1+κ*(self.auxReshape(self.auxProd, τ.ndim)-1))

	def Ω(self, Γs = None, τp = None, κp = None):
		return self.db['αr']*Γs/(1-self.db['αr']*τp*κp*Γs)
	def Ψ(self, Bp = None, τp = None, κp = None):
		return (1-self.db['α'])*(self.db['ρ']-1)*(self.auxΓB3(Bp)/self.auxΓB1(Bp)+self.db['αr']*τp*(1-κp)*self.auxΓB4(Bp)/(1+self.db['αr']*τp*(κp+self.auxΓB2(Bp)*(1-κp))))
	def σ(self, Ω = None, Ψ = None, dlnhp_dlns = None, τp = None, κp = None):
		return 1+(1+τp*κp*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*Ψ*(1-dlnhp_dlns)

	####### Derivatives	
	def EEDerivatives(self, Ψ = None, σ = None, dlnhp_dlns = None, τ = None):
		dlns_dτ  = -(1+self.db['ξ'])/((1+self.db['α']*self.db['ξ'])*(1-τ)*σ)
		dlnΓs_dτ = Ψ*(dlnhp_dlns-1)*dlns_dτ
		return {'∂ln(s)/∂τ': dlns_dτ,
				'∂ln(Γs)/∂τ': dlnΓs_dτ,
				'∂ln(h)/∂τ': self.db['ξ']*(dlns_dτ-dlnΓs_dτ)/(1+self.db['ξ'])}

	def EELaggedDerivatives_τ(self, Ω = None, Ψ = None, Bp = None, dlnhp_dτp = None, τp = None, κp = None):
		""" The derivative used here dlnhp_dτp = ∂ln(h[t+1])/∂τ[t+1]"""
		k1 = κp*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])
		k2 = self.db['αr']*(κp+self.auxΓB2(Bp)*(1-κp))
		k3 = k2/(1+k2*τp)
		dlns_dτp  = (1/(1+Ψ*(1+k1*τp)))*(k1+(1+k1*τp)*(Ψ*dlnhp_dτp-k3))
		dlnΓs_dτp = Ψ*(dlnhp_dτp-dlns_dτp)-k3
		return {'∂ln(s)/∂τ[t+1]': dlns_dτp,
				'∂ln(Γs)/∂τ[t+1]': dlnΓs_dτp,
				'∂ln(h)/∂τ[t+1]': self.db['ξ']*(dlns_dτp-dlnΓs_dτp)/(1+self.db['ξ'])}

	def EELaggedDerivatives_κ(self, Ω = None, Ψ = None, σ = None, Bp = None, dlnhp_dlns = None, τp = None, κp = None):
		k1 = ((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*τp*Ω
		k2 = self.db['αr']*τp*(1-self.auxΓB2(Bp))/(1+self.db['αr']*τp*(κp+self.auxΓB2(Bp)*(1-κp)))
		dlns_dκp = (k1-(1+k1*κp)*k2)/σ
		dlnΓs_dκp= Ψ*(dlnhp_dlns-1)*dlns_dκp-k2
		return {'∂ln(s)/∂κ[t+1]': dlns_dκp, 
				'∂ln(Γs)/∂κ[t+1]': dlnΓs_dκp, 
				'∂ln(h)/∂κ[t+1]': self.db['ξ']*(dlns_dκp-dlnΓs_dκp)/(1+self.db['ξ'])}


	####### Aux. log functions
	def log_dlnh_Dτ(self, τ):
		return -self.db['ξ']/((1-τ)*(1+self.db['α']*self.db['ξ']))

	def log_Γs(self, τp = None, κp = None):
		return self.auxΓB1(self.db['βi'].reshape(self.m.ni,1))/((1+self.db['ξ'])*(1+self.db['αr']*τp*(κp+self.auxΓB2(self.db['βi'].reshape(self.m.ni,1))*(1-κp))))

	def log_savingsSpread(self, τp = None, κp = None):
		return self.auxReshape(self.auxProd * self.db['βi'] / ((1+self.db['βi'])*(1+self.db['ξ'])), τp.ndim)/self.log_Γs(τp, κp)-self.db['αr']*τp*(κp*self.auxReshape(self.auxProd-1/(1+self.db['βi']), τp.ndim)+1/(1+self.auxReshape(self.db['βi'],τp.ndim)))

	def auxReshape(self, x, n):
		return x.reshape((self.m.ni,)+(1,)*max(n,1))

	def auxTile(self, x, n):
		return np.tile(x, (self.m.ni,)+(1,)*max(n,1))

	####### PEE - retirees
	@property
	def PEE2i_FH(self):
		return self.PEE2i_t
	@property
	def PEE2i_log_FH(self):
		return self.PEE2i_log_t

	@property
	def PEE2i_log_IH(self):
		return self.PEE2i_log_t
	@property
	def PEE2i_IH(self):
		return self.PEE2i_t

	def PEE2i_t(self, s_ = None, h = None, τ = None, κ = None, ν = None, dlnh_Dτ = None, sSpread = None):
		c2i_coeff = self.aux_c2i_coeff(sSpread, τ, κ)
		return self.aux_PEE2i(κ, dlnh_Dτ, self.c2i_t(s_, h, ν, c2i_coeff), c2i_coeff)
	def aux_PEE2i(self, κ = None, dlnh_Dτ = None, c2i = None, c2i_coeff = None):
		return c2i**(1-1/self.db['ρ'])*self.aux_PEE2i_log(κ = κ, dlnh_Dτ = dlnh_Dτ, c2i_coeff = c2i_coeff)

	def PEE2i_log_t(self, τ = None, κ = None, dlnh_Dτ = None):
		c2i_coeff = self.aux_c2i_coeff(self.log_savingsSpread(τ, κ), τ, κ)
		return self.aux_PEE2i_log(κ, dlnh_Dτ, c2i_coeff)
	def aux_PEE2i_log(self, κ = None, dlnh_Dτ = None, c2i_coeff = None):
		return ((1-self.db['α'])*dlnh_Dτ+self.db['αr']*(1+κ*(self.auxReshape(self.auxProd, dlnh_Dτ.ndim)-1))/c2i_coeff)

	####### ESC - Euler retirees
	@property
	def ESC2i_FH(self):
		return self.ESC2i_t
	@property
	def ESC2i_IH(self):
		return self.ESC2i_t	

	@property
	def ESC2i_log_FH(self):
		return self.ESC2i_log_t
	@property
	def ESC2i_log_IH(self):
		return self.ESC2i_log_t

	def ESC2i_t(self, s_ = None, h = None, τ = None , κ=None, ν = None, dlnh_Dτ = None, sSpread = None):
		c2i_coeff = self.aux_c2i_coeff(sSpread, τ, κ)
		c2i = self.c2i_t(s_, h, ν, c2i_coeff)
		return {'τ': self.aux_PEE2i(κ, dlnh_Dτ, c2i, c2i_coeff),
				'κ': self.aux_ESC2i_κ(τ, κ, c2i, c2i_coeff)}
	def aux_ESC2i_κ(self, τ = None, κ = None, c2i = None, c2i_coeff = None):
		return c2i**(1-1/self.db['ρ'])*self.aux_ESC2i_κ_log(τ, κ, c2i_coeff)

	def ESC2i_log_t(self, τ = None, κ= None, dlnh_Dτ = None):
		c2i_coeff = self.aux_c2i_coeff(self.log_savingsSpread(τ, κ), τ, κ)
		return {'τ': self.aux_PEE2i_log(κ, dlnh_Dτ, c2i_coeff),
				'κ': self.aux_ESC2i_κ_log(τ, κ, c2i_coeff)}
	def aux_ESC2i_κ_log(self, τ = None, κ = None, c2i_coeff=None):
		return self.db['αr']*τ*(self.auxReshape(self.auxProd, τ.ndim)-1)/c2i_coeff


	####### PEE - Euler workers
	@property
	def PEE1i_IH(self):
		return self.PEE1i_t
	@property
	def PEE1i_log_IH(self):
		return self.PEE1i_log_t

	def PEE1i_t(self, h = None, τp = None, κp = None, Γs = None, Bp = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None):
		x = self.db['αr']*(1-κp)*(1+self.db['ξ'])*Γs
		return self.ĉ1i_t(τp, κp, h, Bp, Γs)**(1-1/self.db['ρ'])*((
				((1+self.db['ξ'])/self.db['ξ'])*dlnh_Dτ + (Bp/(1+Bp))*(1-self.db['α'])*(dlnhp_Dlns-1)*dlns_Dτ
			+   x*(dτp_dτ+τp*dlnΓs_Dτ)/(self.auxProd.reshape(self.m.ni,1)+τp*x)
			))

	def PEE1i_T(self, h = None, dlnh_Dτ = None):
		return self.c̃1i_T(h)**(1-1/self.db['ρ'])*(1+self.db['ξ'])*dlnh_Dτ/self.db['ξ']

	def PEE1i_log_t(self, dlnh_Dτ=None):
		return dlnh_Dτ*(1+self.auxReshape(self.db['βi'], dlnh_Dτ.ndim) * self.power_s)*(1+self.db['ξ'])/self.db['ξ']
	def PEE1i_log_T(self, dlnh_Dτ=None):
		return self.auxTile(dlnh_Dτ*(1+self.db['ξ'])/self.db['ξ'], dlnh_Dτ.ndim)
		# np.tile((dlnh_Dτ*(1+self.db['ξ'])/self.db['ξ']), (self.m.ni,1))

	####### ESC - Euler workers
	@property
	def ESC1i_IH(self):
		return self.ESC1i_t

	def ESC1i_t(self, h = None, τp = None, κp = None, Γs = None, Bp = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, dκp_dτ = None):
		x = self.db['αr']*(1+self.db['ξ'])*Γs
		return self.ĉ1i_t(τp, κp, h, Bp, Γs)**(1-1/self.db['ρ']) * (
				((1+self.db['ξ'])/self.db['ξ'])*dlnh_Dτ + (Bp/(1+Bp))*(1-self.db['α'])*(dlnhp_Dlns-1)*dlns_Dτ
			+   (x/(self.auxProd.reshape(self.m.ni, 1)+x*τp*(1-κp)))*(dlnΓs_Dτ*τp*(1-κp)+dτp_dτ*(1-κp)-dκp_dτ*τp)
			)

	@property
	def ESC1i_T(self):
		return self.PEE1i_T
	@property
	def ESC1i_log_t(self):
		return self.PEE1i_log_t
	@property
	def ESC1i_log_T(self):
		return self.PEE1i_log_T

	####### PEE objectives
	@property
	def PEE_IH(self):
		return self.PEE_t

	def PEE_log_IH(self, τ = None, κ = None, ν = None):
		τBound = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		v1i = self.PEE1i_log_IH(dlnh_Dτ = self.log_dlnh_Dτ(τBound))
		v2i = self.PEE2i_log_IH(τBound, κ, self.log_dlnh_Dτ(τBound))
		return self.aux_PEE(v1i = v1i, v2i = v2i, ν = ν, τ = τ)

	def PEE_log_FH(self, τ = None, κ = None, ν = None):
		""" Takes either a 1d input (over time) or 2d input. In this case, first dimension is time, second is potential grid. """
		τBound = np.clip(τ, self.db['τ_l'], self.db['τ_u'])
		dlnh_Dτ = self.log_dlnh_Dτ(τBound)
		v1i = np.hstack([self.PEE1i_log_t(dlnh_Dτ = dlnh_Dτ[:-1]), self.PEE1i_log_T(dlnh_Dτ = dlnh_Dτ[-1:])])
		v2i = self.PEE2i_log_FH(τBound, κ, dlnh_Dτ)
		return self.aux_PEE_log(v1i = v1i, v2i = v2i, ν = ν, τ = τ)

	def PEE_T(self, τBound = None, τ = None, κ = None, ν = None, s_ = None, h = None, dlnh_Dτ = None, sSpread = None):
		v1i = self.PEE1i_T(h = h, dlnh_Dτ = dlnh_Dτ)
		v2i = self.PEE2i_t(s_ = s_, h = h, τ = τBound, κ = κ, ν = ν, dlnh_Dτ = dlnh_Dτ, sSpread = sSpread)
		return self.aux_PEE(v1i = v1i, v2i = v2i, ν = ν, τ = τ)

	def PEE_t(self, τBound = None, τ  = None, κ  = None, ν  = None, 
								   τp = None, κp = None, νp = None, 
					s_ = None, s = None, h = None, hp = None, Γs = None, Bp = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, sSpread = None):
		v1i = self.PEE1i_t(h = h, τp = τp, κp = κp, Γs = Γs, Bp = Bp, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ)
		v2i = self.PEE2i_t(s_= s_, h = h, τ = τBound, κ = κ, ν = ν, dlnh_Dτ = dlnh_Dτ, sSpread = sSpread)
		return self.aux_PEE(v1i = v1i, v2i = v2i, ν = ν, τ = τ)
		
	def aux_PEE_log(self, v1i = None, v2i = None, ν = None, τ = None):
		return self.interiorFOC(np.multiply(self.auxReshape(self.db['γi']*self.ω2i, τ.ndim), v2i).sum(axis=0)+ν*(np.multiply(self.auxReshape(self.db['γi']*self.ω1i, τ.ndim), v1i).sum(axis=0)), τ)

	def aux_PEE(self, v1i = None, v2i = None, ν = None, τ = None):
		return self.interiorFOC(np.matmul(self.db['γi']*self.ω2i, v2i)+ν*(np.matmul(self.db['γi']*self.ω1i, v1i)), τ)

	####### ESC objectives
	@property
	def ESC_IH(self):
		return self.ESC_t

	def ESC_log_IH(self, τ = None, κ = None, ν = None):
		τBound, κBound = np.clip(τ, self.db['τ_l'], self.db['τ_u']), np.clip(κ, self.db['κ_l'], self.db['κ_u'])
		v1i = self.ESC1i_log_t(dlnh_Dτ = self.log_dlnh_Dτ(τBound))
		v2i = self.ESC2i_log_t(τ = τBound, κ= κBound, dlnh_Dτ = self.log_dlnh_Dτ(τBound))
		return self.aux_ESC(v1i = v1i, v2i = v2i, ν = ν, τ = τ, κ = κ)

	def ESC_log_FH(self, τ = None, κ = None, ν = None):
		τBound, κBound = np.clip(τ, self.db['τ_l'], self.db['τ_u']), np.clip(κ, self.db['κ_l'], self.db['κ_u'])
		v1i = np.hstack([self.ESC1i_log_t(dlnh_Dτ = self.log_dlnh_Dτ(τBound[:-1])), self.ESC1i_log_T(dlnh_Dτ = self.log_dlnh_Dτ(τBound[-1:]))])
		v2i = self.ESC2i_log_t(τ = τBound, κ= κBound, dlnh_Dτ = self.log_dlnh_Dτ(τBound))
		return self.aux_ESC_log(v1i = v1i, v2i = v2i, ν = ν, τ = τ, κ = κ)

	def ESC_t(self, τBound = None, κBound = None, τ = None, κ = None, ν = None,
												  τp= None, κp= None, νp= None,
					s_ = None, s = None, h = None, hp= None, Γs = None, Bp = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, dκp_dτ = None, sSpread = None):
		v1i = self.ESC1i_t(h = h, τp = τp, κp = κp, Γs = Γs, Bp = Bp, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, dκp_dτ = dκp_dτ)
		v2i = self.ESC2i_t(s_ = s_, h = h, τ = τBound, κ = κBound, ν = ν, dlnh_Dτ = dlnh_Dτ, sSpread = sSpread)
		return self.aux_ESC(v1i = v1i, v2i = v2i, ν = ν, τ = τ, κ = κ)

	def ESC_T(self, τBound = None, κBound = None, τ = None, κ = None, ν = None, s_ = None, h = None, dlnh_Dτ = None, sSpread = None):
		v1i = self.ESC1i_T(h = h, dlnh_Dτ = dlnh_Dτ)
		v2i = self.ESC2i_t(s_ = s_, h = h, τ = τBound, κ = κBound, ν = ν, dlnh_Dτ = dlnh_Dτ, sSpread = sSpread)
		return self.aux_ESC(v1i = v1i, v2i = v2i, ν = ν, τ = τ, κ = κ)

	def aux_ESC_log(self, v1i = None, v2i = None, ν = None, τ = None, κ = None):
		return np.hstack([self.aux_PEE_log(v1i = v1i, v2i = v2i['τ'], ν = ν, τ = τ),
					self.interiorFOC(np.multiply(self.auxReshape(self.db['γi']*self.ω2i, κ.ndim), v2i['κ']).sum(axis=0), κ, var = 'κ')])

	def aux_ESC(self, v1i = None, v2i = None, ν = None, τ = None, κ = None):
		return np.hstack([self.aux_PEE(v1i = v1i, v2i = v2i['τ'], ν = ν, τ = τ),
						  self.interiorFOC(np.matmul(self.db['γi']*self.ω2i, v2i['κ']), κ, var = 'κ')])
