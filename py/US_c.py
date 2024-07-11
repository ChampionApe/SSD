import numpy as np

class C:
	def __init__(self, m, ts = 'FH'):
		self.m = m
		self.db = m.db
		self.ts = ts
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
	@property
	def auxProd(self):
		return np.power(self.db['ηi'], 1+self.db['ξ'])/np.power(self.db['Xi'], self.db['ξ'])
	def auxΓB1(self, B):
		return np.matmul(self.auxProd * self.db['γi'], B/(1+B))
	def auxΓB2(self,B):
		return np.matmul(self.db['γi'], 1/(1+B))
	def auxΓB3(self,B):
		return np.matmul(self.auxProd * self.db['γi'], B/((1+B)**2))
	def auxΓB4(self,B):
		return np.matmul(self.db['γi'], B/((1+B)**2))
	def auxPen(self, τ = None, eps = None):
		return self.db['p']*τ/(eps*self.db['p̄']+self.db['p'])
	def aux_sτ0(self, h = None, ν = None, θp =None, τp = None, epsp = None, Γs = None):
		return ν*h**(1/self.power_h)*((1-self.db['αr']*θp*self.auxPen(τp, epsp)*Γs)/((1-self.db['α'])))**(1/self.db['α'])

	def adjustFocMultiplicative(self, z, x, l = 0, u = 1, kl = 10, ku = 10):
		""" Adjust FOC in a multiplicative way: z is the "original" marginal effect, x is the bounded variable """
		return z-abs(z)*(kl*(np.clip(x, None, l)-l)+ku*(np.clip(x,u,None)-u))

	def interiorFOC(self, z, x, var = 'τ'):
		return self.adjustFocMultiplicative(z, x, l = self.db[f'{var}_l'], u = self.db[f'{var}_u'], kl = self.db[f'k{var}_l'], ku = self.db[f'k{var}_u'])

	def R(self, s_ = None, h = None, ν = None, A = 1):
		return self.db['α'] * A * (ν*h/s_)**(1-self.db['α'])

	def B(self, s_ = None, h = None, ν = None, A = 1):
		return self.db['βi'].reshape(self.m.ni,1)**self.db['ρ'] * (self.R(s_ ,h, ν, A = A)/self.db['p'])**(self.db['ρ']-1)
	
	def Γs(self, B = None, τp = None, θp = None, epsp = None):
		""" τp, Bp, θp, epsp are vectors of the same length"""
		return (1/(1+self.db['ξ']))*np.matmul(self.db['γi'] * self.auxProd, B/(1+B)) /(1+self.db['αr']*self.auxPen(τp, epsp)*(θp+(epsp+(1-θp)/(1-self.db['γ0']))*np.matmul(self.db['γi'], 1/(1+B))))

	def savingsSpread(self, B = None, Γs = None, τp = None, θp = None, epsp = None):
		return self.auxProd.reshape(self.m.ni,1) * B / ((1+B)*(1+self.db['ξ'])*Γs)-self.db['αr']*self.auxPen(τp, epsp)*(θp*self.auxProd.reshape(self.m.ni,1)+(epsp+(1-θp)/(1-self.db['γ0']))/(1+B))	

	def steadyStateB_eq(self, B, Γs, τ, θ, eps, ν):
		""" Requirement = 0 for B to be in steady state - this does not return ss level of B"""
		if self.db['ρ']<1:
			return self.db['βi'].reshape(self.m.ni,1)**(self.db['ρ']/(1-self.db['ρ']))*(1-self.db['α'])*(1-τ)-B**(1/(1-self.db['ρ']))*(ν/self.db['p'])*(self.db['α']-(1-self.db['α'])*self.auxPen(τ, eps)*θ*Γs)/Γs
		elif self.db['ρ']>1:
			return B**(1/(self.db['ρ']-1))*(1-self.db['α'])*(1-τ)-self.db['βi'].reshape(self.m.ni,1)**(self.db['ρ']/(self.db['ρ']-1))*(ν/self.db['p'])*(self.db['α']-(1-self.db['α'])*self.auxPen(τ, eps)*θ*Γs)/Γs

	def steadyStateΓs_eq(self, B, Γs, τ, θ, eps, ν):
		""" Requirement = 0 for Γs to be in steady state - this does not return ss level of Γs """
		return self.auxΓB1(B)-Γs*(1+self.db['ξ'])*(1+self.db['αr']*self.auxPen(τ,eps)*(θ+(eps+(1-θ)/(1-self.db['γ0']))*self.auxΓB2(B)))

	def s_SS(self, Γs, τ, θ, eps, ν):
		""" Return steady state level of savings"""
		return ( ((1-self.db['α'])*(1-τ)/(1-self.db['αr']*self.auxPen(τ, eps)*θ*Γs))**(1+self.db['ξ'])*Γs**(1+self.db['α']*self.db['ξ'])/(ν**(self.db['α']*(1+self.db['ξ']))) )**(1/(1-self.db['α']))

	def dlnh_ds_SS(self, Ω, Ψ, τ, θ, eps):
		""" get steady state level of ∂ln(h)/∂ln(s) given solution to solve_ss (ss) and parameters """
		a = Ψ*(1+τ*θ*Ω*(1+self.db['ξ'])/(1+self.db['ξ']*self.db['α']))
		b = -(1+Ψ*(1+self.db['α']*self.db['ξ']+τ*θ*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])))
		c = self.db['α']*self.db['ξ']*(1+Ψ)
		return (-b-np.sqrt(b**2-4*a*c))/(2*a)


	@property
	def h_IH(self):
		return self.h_t
	def h_FH(self, τ = None, τp = None, θp = None, epsp = None, Γs = None, s_ = None, ν = None):
		return np.hstack([self.h_t(τ[:-1], τp[:-1], θp[:-1], epsp[:-1], Γs, s_[:-1], ν[:-1]),
						  self.h_T(τ[-1], s_[-1], ν[-1])])
	def Θh_t(self, τ = None, τp = None, θp = None, epsp = None, Γs = None):
		return ((1-self.db['α'])*(1-τ)/(1-self.db['αr']*self.auxPen(τp, epsp)*θp*Γs))**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))
	def Θh_T(self, τ = None):
		return ((1-self.db['α'])*(1-τ))**(self.db['ξ']/(1+self.db['α']*self.db['ξ']))
	def h_t(self, τ = None, τp  = None, θp = None, epsp = None, Γs = None, s_ = None, ν = None):
		""" Note: Remove kwargs when it is done"""
		return self.Θh_t(τ = τ, τp = τp, θp = θp, epsp = epsp, Γs = Γs)*(s_/ν)**self.power_h
	def h_T(self, τ = None, s_ = None, ν = None):
		return self.Θh_T(τ = τ)*(s_/ν)**self.power_h

	def Γs_EE(self, s, hp, νp, τp, θp, epsp):
		return self.Γs(self.B(s, hp, νp), τp, θp, epsp)
	def s_t(self, h, Γs):
		return h**((1+self.db['ξ'])/self.db['ξ'])*Γs

	@property
	def ĉ1i_IH(self):
		return self.ĉ1i_t
	@property
	def ĉ1i_FH(self):
		""" This returns full length vector for simplicity - even though this is not defined for the terminal period """
		return self.ĉ1i_t	
	def ĉ1i_t(self, τp = None, θp = None, epsp = None, h = None, B = None, Γs = None):
		return h**((1+self.db['ξ'])/self.db['ξ'])*(1+B)**(1/(self.db['ρ']-1))*(self.auxProd.reshape(self.m.ni,1)/(1+self.db['ξ'])+self.db['αr']*self.auxPen(τp, epsp)*(epsp+(1-θp)/1-self.db['γ0'])*Γs)
	def c̃1i_T(self, h = None):
		return (h**((1+self.db['ξ'])/self.db['ξ']))*self.auxProd.reshape(self.m.ni,1)/(1+self.db['ξ'])
	
	@property
	def c̃10_IH(self):
		return self.c̃10_t
	@property
	def c̃10_FH(self):
		return self.c̃10_t
	def c̃10_t(self, τ = None, s_ = None, h = None, ν = None):
		""" This is also applies in the finite horizon """
		return (self.db['η0']**(1+self.db['ξ'])/self.db['X0']**self.db['ξ'])*((1-self.db['α'])*(s_/(ν*h))**self.db['α'])**(1+self.db['ξ'])*((1-τ)**self.db['ξ']-self.db['ξ']*(1-τ)**(1+self.db['ξ'])/(1+self.db['ξ']))
	@property
	def c2p0_IH(self):
		return self.c2p0_t
	@property
	def c2p0_FH(self):
		""" This returns full length vector for simplicity - even though this is not defined for the terminal period """
		return self.c2p0_t
	def c2p0_t(self, τp = None, epsp = None, s = None, hp = None, νp = None):
		return (epsp*τp/(epsp*self.db['p̄']+self.db['p']))*νp*(1-self.db['α'])*(s/νp)**self.db['α']*hp**(1-self.db['α'])
	
	@property
	def c2i_IH(self):
		return self.c2i_t
	@property
	def c2i_FH(self):
		return self.c2i_t	
	def c2i_t(self, τ = None, eps = None, s_ = None, h = None, ν = None, c2i_coeff = None):
		return self.db['α']*(ν/self.db['p'])*(s_/ν)**self.db['α']*h**(1-self.db['α'])*c2i_coeff
	def aux_c2i_coeff(self, sSpread = None, τ = None, θ = None, eps = None):
		return sSpread+self.db['αr']*self.auxPen(τ,eps)*(eps+θ*self.auxProd.reshape(self.m.ni,1)+(1-θ)/(1-self.db['γ0']))

	@property
	def c20_IH(self):
		return self.c20_t
	@property
	def c20_FH(self):
		return self.c20_t
	def c20_t(self, s_ = None, h = None, ν = None, τ = None, eps = None):
		return (1-self.db['α'])*ν*(s_/ν)**(self.db['α'])*h**(1-self.db['α'])*eps*τ/(eps*self.db['p̄']+self.db['p'])

	def Ω(self, Γs = None, τp = None, θp = None, epsp = None):
		k = Γs*self.db['αr']*self.db['p']/(epsp*self.db['p̄']+self.db['p'])
		return k/(1-τp*k*θp)
	def Ψ(self, Bp = None, τp = None, θp = None, epsp = None):
		return (1-self.db['α'])*(self.db['ρ']-1)*(self.auxΓB3(Bp)/self.auxΓB1(Bp)+self.db['αr']*self.auxPen(τp, epsp)*(epsp+(1-θp)/(1-self.db['γ0']))*self.auxΓB4(Bp)/(1+self.db['αr']*self.auxPen(τp, epsp)*(θp+(epsp+(1-θp)/(1-self.db['γ0'])))*self.auxΓB2(Bp)))
	def σ(self, Ω = None, Ψ = None, dlnhp_dlns = None, τp = None, θp = None):
		return 1+(1+τp*θp*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*Ψ*(1-dlnhp_dlns)

	####### Derivatives	
	def EEDerivatives(self, Ψ = None, σ = None, dlnhp_dlns = None, τ = None):
		dlns_dτ  = -(1+self.db['ξ'])/((1+self.db['α']*self.db['ξ'])*(1-τ)*σ)
		dlnΓs_dτ = Ψ*(dlnhp_dlns-1)*dlns_dτ
		return {'∂ln(s)/∂τ': dlns_dτ,
				'∂ln(Γs)/∂τ': dlnΓs_dτ,
				'∂ln(h)/∂τ': self.db['ξ']*(dlns_dτ-dlnΓs_dτ)/(1+self.db['ξ'])}

	def EELaggedDerivatives_τ(self, Ω = None, Ψ = None, Bp = None, dlnhp_dτp = None, τp = None, θp = None, epsp = None):
		""" The derivative used here dlnhp_dτp = ∂ln(h[t+1])/∂τ[t+1]"""
		k1 = θp*Ω*(1+self.db['ξ'])/(1+self.db['α']*self.db['ξ'])
		k2 = self.db['αr']*(θp+(epsp+(1-θp)/(1-self.db['γ0']))*self.auxΓB2(Bp))*self.db['p']/(epsp*self.db['p̄']+self.db['p'])
		k3 = k2/(1+k2*τp)
		dlns_dτp  = (1/(1+Ψ*(1+k1*τp)))*(k1+(1+k1*τp)*(Ψ*dlnhp_dτp-k3))
		dlnΓs_dτp = Ψ*(dlnhp_dτp-dlns_dτp)-k3
		return {'∂ln(s)/∂τ[t+1]': dlns_dτp,
				'∂ln(Γs)/∂τ[t+1]': dlnΓs_dτp,
				'∂ln(h)/∂τ[t+1]': self.db['ξ']*(dlns_dτp-dlnΓs_dτp)/(1+self.db['ξ'])}

	def EELaggedDerivatives_θ(self, Ω = None, Ψ = None, σ = None, Bp = None, dlnhp_dlns = None, τp = None, θp = None, epsp = None):
		k1 = ((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*τp*Ω
		k2 = self.db['αr']*self.auxPen(τp, epsp)
		k3 = k2*(1-self.auxΓB2(Bp)/(1-self.db['γ0']))/(1+k2*(θp+(epsp+(1-θp)/(1-self.db['γ0']))*self.auxΓB2(Bp)))
		dlns_dθp = (k1-(1+k1*θp)*k3)/σ
		dlnΓs_dθp= Ψ*(dlnhp_dlns-1)*dlns_dθp-k3
		return {'∂ln(s)/∂θ[t+1]': dlns_dθp, 
				'∂ln(Γs)/∂θ[t+1]': dlnΓs_dθp, 
				'∂ln(h)/∂θ[t+1]': self.db['ξ']*(dlns_dθp-dlnΓs_dθp)/(1+self.db['ξ'])}

	def EELaggedDerivatives_eps(self, Ω = None, Ψ = None, σ = None, Bp = None, dlnhp_dlns = None, τp = None, θp = None, epsp = None):
		k1 = ((1+self.db['ξ'])/(1+self.db['α']*self.db['ξ']))*τp*Ω*θp
		k2 = self.db['αr']*self.auxPen(τp, epsp)
		k3 = θp*self.db['p̄']+(self.db['p̄']*(1-θp)/(1-self.db['γ0'])-self.db['p'])*self.auxΓB2(Bp)
		k4 = θp+(epsp+(1-θp)/(1-self.db['γ0']))*self.auxΓB2(Bp)
		k5 = (k2/(epsp*self.db['p̄']+self.db['p']))*k3/(1+k2*k4)
		dlns_depsp = (1/σ)*((1+k1)*k5-k1*self.db['p̄']/(epsp*self.db['p̄']+self.db['p']))
		dlnΓs_depsp = Ψ*dlns_depsp*(dlnhp_dlns-1)+k5
		return {'∂ln(s)/∂eps[t+1]': dlns_depsp,
				'∂ln(Γs)/∂eps[t+1]': dlnΓs_depsp,
				'∂ln(h)/∂eps[t+1]': self.db['ξ']*(dlns_depsp-dlnΓs_depsp)/(1+self.db['ξ'])}

	####### PEE - HtM retirees
	@property
	def PEE20_FH(self):
		return self.PEE20_t

	@property
	def PEE20_IH(self):
		return self.PEE20_t

	def PEE20_t(self, s_ = None, h = None, τ = None, eps = None, ν = None, dlnh_Dτ = None):
		return self.aux_PEE20(τ, eps, dlnh_Dτ, c20 = self.c20_t(s_, h, ν, τ, eps))

	def aux_PEE20(self, τ = None, eps = None, dlnh_Dτ = None, c20 =None):
		return c20**(1-1/self.db['ρ'])*(1/τ+(1-self.db['α'])*dlnh_Dτ)

	####### ESC - HTM retirees
	def ESC20_t(self, s_ = None, h = None, τ = None, eps = None, ν = None, dlnh_Dτ = None):
		c20 = self.c20_t(s_, h, ν, τ, eps)
		return {'τ': self.aux_PEE20(τ, eps, dlnh_Dτ, c20),
				'eps': self.aux_ESC20_eps(eps = eps, c20 = c20)}

	def aux_ESC20_eps(self, eps = None, c20 = None):
		return c20**(1-1/self.db['ρ'])*(1/eps-self.db['p̄']/(eps*self.db['p̄']+self.db['p']))

	####### PEE - Euler retirees
	@property
	def PEE2i_FH(self):
		return self.PEE2i_t

	@property
	def PEE2i_IH(self):
		return self.PEE2i_t

	def PEE2i_t(self, s_ = None, h = None, τ = None, θ = None, eps = None, ν = None, dlnh_Dτ = None, sSpread = None):
		c2i_coeff = self.aux_c2i_coeff(sSpread, τ, θ, eps)
		return self.aux_PEE2i(θ, eps, dlnh_Dτ, self.c2i_t(τ, eps, s_, h, ν, c2i_coeff), c2i_coeff)

	def aux_PEE2i(self, θ = None, eps = None, dlnh_Dτ = None, c2i = None, c2i_coeff = None):
		return c2i**(1-1/self.db['ρ'])*((1-self.db['α'])*dlnh_Dτ+(self.db['αr']*(self.db['p']/(eps*self.db['p̄']+self.db['p']))*(eps+θ*self.auxProd.reshape(self.m.ni,1)+(1-θ)/(1-self.db['γ0'])))/c2i_coeff)

	####### ESC - Euler retirees
	@property
	def ESC2i_FH(self):
		return self.ESC2i_t

	@property
	def ESC2i_IH(self):
		return self.ESC2i_t	

	def ESC2i_t(self, s_ = None, h = None, τ = None , θ = None, eps = None, ν = None, dlnh_Dτ = None, sSpread = None):
		c2i_coeff = self.aux_c2i_coeff(sSpread, τ, θ, eps)
		c2i = self.c2i_t(τ, eps, s_, h, ν, c2i_coeff)
		return {'τ': self.aux_PEE2i(θ, eps, dlnh_Dτ, c2i, c2i_coeff),
				'θ': self.aux_ESC2i_θ(τ, θ, eps, c2i, c2i_coeff),
				'eps': self.aux_ESC2i_eps(τ, θ, eps, c2i, c2i_coeff)}

	def aux_ESC2i_eps(self, τ = None, θ = None, eps = None, c2i = None, c2i_coeff = None):
		return c2i**(1-1/self.db['ρ'])*(self.db['αr']*self.db['p']*τ*(self.db['p']-self.db['p̄']*(θ*self.auxProd.reshape(self.m.ni,1)+(1-θ)/(1-self.db['γ0'])))/(eps*self.db['p̄']+self.db['p'])**2)/c2i_coeff

	def aux_ESC2i_θ(self, τ = None, θ = None, eps = None, c2i = None, c2i_coeff = None):
		return c2i**(1-1/self.db['ρ'])*(self.db['αr']*self.db['p']*τ*(self.auxProd.reshape(self.m.ni,1)-1/(1-self.db['γ0']))/(eps*self.db['p̄']+self.db['p']))/c2i_coeff

	####### PEE - HtM workers
	@property
	def PEE10_IH(self):
		return self.PEE10_t

	def PEE10_T(self, s_ = None, h = None, τ = None, eps = None, ν = None, dlnh_Dτ = None):
		return -(1+self.db['ξ'])*self.c̃10_t(τ, s_, h, ν)**(1-1/self.db['ρ'])*(self.db['α']*dlnh_Dτ+(τ/(1-τ))*self.db['ξ']/(1+τ*self.db['ξ']))

	def PEE10_t(self, s_ = None, s = None, h = None, hp = None, τ = None, τp = None, epsp = None, ν = None, νp = None, dlnh_Dτ = None, dlns_Dτ = None, dτp_dτ = None, dlnhp_Dlns = None):
		return (-(1+self.db['ξ'])*self.c̃10_t(τ, s_, h, ν)**(1-1/self.db['ρ'])*(self.db['α']*dlnh_Dτ+(τ/(1-τ))*self.db['ξ']/(1+τ*self.db['ξ']))
			+   self.db['β0']*self.c2p0_t(τp, epsp, s, hp, νp)**(1-1/self.db['ρ'])*(
					dτp_dτ+(self.db['α']+(1-self.db['α'])*dlnhp_Dlns)*dlns_Dτ
			))

	####### ESC - HtM workers
	@property
	def ESC10_IH(self):
		return self.ESC10_t

	@property
	def ESC10_T(self):
		return self.PEE10_T

	def ESC10_t(self, s_ = None, s = None, h = None, hp = None, τ = None, τp = None, epsp = None, ν = None, νp = None, dlnh_Dτ = None, dlns_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, depsp_dτ = None):
		return (-(1+self.db['ξ'])*self.c̃10_t(τ, s_, h, ν)**(1-1/self.db['ρ'])*(self.db['α']*dlnh_Dτ+(τ/(1-τ))*self.db['ξ']/(1+τ*self.db['ξ']))
			+   self.db['β0']*self.c2p0_t(τp, epsp, s, hp, νp)**(1-1/self.db['ρ'])*(
					dτp_dτ+(self.db['α']+(1-self.db['α'])*dlnhp_Dlns)*dlns_Dτ+(depsp_dτ/epsp)*self.db['p']/(epsp*self.db['p̄']+self.db['p'])
			))

	####### PEE - Euler workers
	@property
	def PEE1i_IH(self):
		return self.PEE1i_t

	def PEE1i_t(self, h = None, τp = None, θp = None, epsp = None, Γs = None, Bp = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None):
		k = self.db['αr']*(1+self.db['ξ'])*Γs*(epsp+(1-θp)/(1-self.db['γ0']))*self.db['p']/(self.db['p̄']*epsp+self.db['p'])
		return self.ĉ1i_t(τp, θp, epsp, h, Bp, Γs)**(1-1/self.db['ρ'])*((
				((1+self.db['ξ'])/self.db['ξ'])*dlnh_Dτ + (Bp/(1+Bp))*(1-self.db['α'])*(dlnhp_Dlns-1)*dlns_Dτ
			+   k*(dτp_dτ+τp*dlnΓs_Dτ)/(self.auxProd.reshape(self.m.ni,1)+τp*k)
			))

	def PEE1i_T(self, h = None, dlnh_Dτ = None):
		return self.c̃1i_T(h)**(1-1/self.db['ρ'])*(1+self.db['ξ'])*dlnh_Dτ/self.db['ξ']

	####### ESC - Euler workers
	@property
	def ESC1i_IH(self):
		return self.ESC1i_t

	def ESC1i_t(self, h = None, τp = None, θp = None, epsp = None, Γs = None, Bp = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, dθp_dτ = None, depsp_dτ = None):
		k1 = self.db['αr']*(1+self.db['ξ'])*Γs*self.db['p']/(self.db['p̄']*epsp+self.db['p'])
		k2 = epsp+(1-θp)/(1-self.db['γ0'])
		return self.ĉ1i_t(τp, θp, epsp, h, Bp, Γs)**(1-1/self.db['ρ']) * (
				((1+self.db['ξ'])/self.db['ξ'])*dlnh_Dτ + (Bp/(1+Bp))*(1-self.db['α'])*(dlnhp_Dlns-1)*dlns_Dτ
			+   (k1/(self.auxProd.reshape(self.m.ni,1)+k1*τp*k2))*(
					dlnΓs_Dτ*τp*k2+dτp_dτ*k2-dθp_dτ*τp/(1-self.db['γ0'])+depsp_dτ*τp*(1-self.db['p̄']*k2/(epsp*self.db['p̄']+self.db['p']))
			))

	@property
	def ESC1i_T(self):
		return self.PEE1i_T

	####### PEE objectives
	@property
	def PEE_IH(self):
		return self.PEE_t

	def PEE_T(self, τBound = None, τ = None, θ = None, eps = None, ν = None, s_ = None, h = None, dlnh_Dτ = None, sSpread = None):
		v1i = self.PEE1i_T(h = h, dlnh_Dτ = dlnh_Dτ)
		v10 = self.PEE10_T(s_ = s_, h = h, τ = τBound, eps = eps, ν = ν, dlnh_Dτ = dlnh_Dτ)
		v2i = self.PEE2i_t(s_ = s_, h = h, τ = τBound, θ = θ, eps = eps, ν = ν, dlnh_Dτ = dlnh_Dτ, sSpread = sSpread)
		v20 = self.PEE20_t(s_ = s_, h = h, τ = τBound, eps = eps, ν = ν, dlnh_Dτ = dlnh_Dτ)
		return self.aux_PEE(v1i = v1i, v10 = v10, v2i = v2i, v20 = v20, ν = ν, τ = τ)

	def PEE_t(self, τBound = None, τ  = None, θ  = None, eps  = None, ν  = None, 
								   τp = None, θp = None, epsp = None, νp = None, 
					s_ = None, s = None, h = None, hp = None, Γs = None, Bp = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, sSpread = None):
		v1i = self.PEE1i_t(h = h, τp = τp, θp = θp, epsp = epsp, Γs = Γs, Bp = Bp, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ)
		v10 = self.PEE10_t(s_= s_, s = s, h = h, hp = hp, τ = τBound, τp = τp, epsp = epsp, ν = ν, νp = νp, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dτp_dτ = dτp_dτ, dlnhp_Dlns = dlnhp_Dlns)
		v2i = self.PEE2i_t(s_= s_, h = h, τ = τBound, θ = θ, eps = eps, ν = ν, dlnh_Dτ = dlnh_Dτ, sSpread = sSpread)
		v20 = self.PEE20_t(s_ = s_, h = h, τ = τBound, eps = eps, ν = ν, dlnh_Dτ = dlnh_Dτ)
		return self.aux_PEE(v1i = v1i, v10 = v10, v2i = v2i, v20 = v20, ν = ν, τ = τ)
		
	def aux_PEE(self, v1i = None, v10 = None, v2i = None, v20 = None, ν = None, τ = None):
		return self.interiorFOC(self.db['γ0']*self.ω20*v20+np.matmul(self.db['γi']*self.ω2i, v2i)+ν*(self.db['γ0']*self.ω10*v10+np.matmul(self.db['γi']*self.ω1i, v1i)), τ)

	####### ESC objectives
	@property
	def ESC_IH(self):
		return self.ESC_t

	def ESC_t(self, τBound = None, θBound = None, epsBound = None, τ = None, θ = None, eps = None, ν = None,
																   τp= None, θp= None, epsp= None, νp= None,
					s_ = None, s = None, h = None, hp= None, Γs = None, Bp = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, dθp_dτ = None, depsp_dτ = None, sSpread = None):
		v1i = self.ESC1i_t(h = h, τp = τp, θp = θp, epsp = epsp, Γs = Γs, Bp = Bp, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, dθp_dτ = dθp_dτ, depsp_dτ = depsp_dτ)
		v10 = self.ESC10_t(s_= s_, s = s, h = h, hp = hp, τ = τBound, τp = τp, epsp = epsp, ν = ν, νp = νp, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, depsp_dτ = depsp_dτ)
		v2i = self.ESC2i_t(s_ = s_, h = h, τ = τBound, θ = θBound, eps = epsBound, ν = ν, dlnh_Dτ = dlnh_Dτ, sSpread = sSpread)
		v20 = self.ESC20_t(s_ = s_, h = h, τ = τBound, eps = epsBound, ν = ν, dlnh_Dτ = dlnh_Dτ)
		return self.aux_ESC(v1i = v1i, v10 = v10, v2i = v2i, v20 = v20, ν = ν, τ = τ, θ = θ, eps = eps)

	def ESC_T(self, τBound = None, θBound = None, epsBound = None, τ = None, θ = None, eps = None, ν = None, 
					s_ = None, h = None, dlnh_Dτ = None, sSpread = None):
		v1i = self.ESC1i_T(h = h, dlnh_Dτ = dlnh_Dτ)
		v10 = self.ESC10_T(s_ = s_, h = h, τ = τBound, eps = epsBound, ν = ν, dlnh_Dτ = dlnh_Dτ)
		v2i = self.ESC2i_t(s_ = s_, h = h, τ = τBound, θ = θBound, eps = epsBound, ν = ν, dlnh_Dτ = dlnh_Dτ, sSpread = sSpread)
		v20 = self.ESC20_t(s_ = s_, h = h, τ = τBound, eps = epsBound, ν = ν, dlnh_Dτ = dlnh_Dτ)
		return self.aux_ESC(v1i = v1i, v10 = v10, v2i = v2i, v20 = v20, ν = ν, τ = τ, θ = θ, eps = eps)

	def aux_ESC(self, v1i = None, v10 = None, v2i = None, v20 = None, ν = None, τ = None, θ = None, eps = None):
		return np.hstack([self.aux_PEE(v1i = v1i, v10 = v10, v2i = v2i['τ'], v20 = v20['τ'], ν = ν, τ = τ),
						  self.interiorFOC(np.matmul(self.db['γi']*self.ω2i, v2i['θ']), θ, var = 'θ'),
						  self.interiorFOC(np.matmul(self.db['γi']*self.ω2i, v2i['eps'])+self.db['γ0']*self.ω20*v20['eps'], eps, var = 'eps')])