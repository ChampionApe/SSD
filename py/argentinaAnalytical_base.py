import numpy as np, pandas as pd
from pyDbs import is_iterable
def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

class _Base_A:
	def __init__(self, m):
		self.m = m
		self.db = m.db

	#######################################################################
	##########					0. Aux methods				 	###########
	#######################################################################
	def ω2i(self, t = None):
		return (self('pi[t-1]', t).mul(self('ω',t), axis = 0).mul(self('μi',t), axis = 0)).values
	def ω20(self, t = None):
		return self.get('p0[t-1]', t) * self.get('ω',t) * self.get('μ0',t)
	def ω1i(self, t = None):
		return self.get('μi',t)
	def ω10(self, t = None):
		return self.get('μ0',t)
	def power_s(self, t = None):
		return self.get('α',t)*(1+self.get('ξ',t))/(1+self.get('α',t)*self.get('ξ',t))
	def power_h(self, t = None):
		return self.get('α',t)*self.get('ξ',t)/(1+self.get('α',t)*self.get('ξ',t))
	def power_p(self, t = None):
		return self.power_s(t)**2
	def auxProd(self, t = None):
		return (self('ηi', t).pow(1+self('ξ',t), axis = 0)/self('Xi',t).pow(self('ξ',t), axis = 0)).values
	def auxProd0(self, t = None):
		return self.get('η0', t)**(1+self.get('ξ',t))/self.get('X0',t)**self.get('ξ',t)

	def auxInf0(self, t = None):
		return self.auxProd0(t)*((1-self.get('α',t))/self.get('Γh',t)**(self.get('α',t)))**((1+self.get('ξ',t))/(1+self.get('ξ',t)*self.get('α',t)))/(1+self.get('ξ',t))
	def auxForm1(self, t = None):
		return self.get('αr',t)*self.get('p',t)*self.get('θ[t+1]',t)/self.get('κ',t)

	def adjustFocMultiplicative(self, z, x, l = 0, u = 1, kl = 10, ku = 10):
		""" Adjust FOC in a multiplicative way: z is the "original" marginal effect, x is the bounded variable """
		return z-abs(z)*(kl*(np.clip(x, None, l)-l)+ku*(np.clip(x,u,None)-u))
	def interiorFOC(self, z, x, var = 'τ'):
		return self.adjustFocMultiplicative(z, x, l = self.db[f'{var}_l'], u = self.db[f'{var}_u'], kl = self.db[f'k{var}_l'], ku = self.db[f'k{var}_u'])

	#######################################################################
	##########					1. Simple defs				 	###########
	#######################################################################
	#### NOTE: NUMPY METHODS
	def R(self, s_ = None, h = None, t = None):
		return self.get('α',t) * (self.get('ν',t)*h/s_)**(1-self.get('α',t))
	def Γs(self, Bi = None, τp = None, t = None):
		return (1/(1+self.get('ξ',t)))*self.auxΓB1(Bi, t = t)/(1+self.get('αr',t)*(self.get('p',t)*τp/self.get('κ',t))*(self.get('θ[t+1]',t)+self.auxΓB2(Bi, t = t)*(1-self.get('θ[t+1]',t))))

	def Θh_t(self, τ = None, τp = None, Γs = None, t = None):
		return self.get('Γh',t)**((1+self.get('ξ',t))/(1+self.get('α',t)*self.get('ξ',t))) * ((1-self.get('α',t))*(1-τ)/(self.get('Γh',t)-self.auxForm1(t)*τp * Γs))**(self.get('ξ',t)/(1+self.get('α',t)*self.get('ξ',t)))
	def Θs_t(self, Θh = None, Γs = None, t = None):
		return (Θh/self.get('Γh',t))**((1+self.get('ξ',t))/self.get('ξ',t)) * Γs
	def h_t(self, s_ = None, τ = None, τp = None, Γs = None, t = None):
		return self.Θh_t(τ = τ, τp = τp, Γs = Γs, t = t)*(s_/self.get('ν',t))**self.power_h(t)
	def s_t(self, h = None, Γs = None, t = None):
		return (h/self.get('Γh',t))**((1+self.get('ξ',t))/self.get('ξ',t))*Γs

	def hFromΘh_t(self, s_ = None, Θh = None, t = None):
		return Θh * (s_/self.get('ν',t))**self.power_h(t)


class BaseScalar_A(_Base_A):
	def __init__(self, m, t = None):
		super().__init__(m)
		self.t = t
		self.t0 = self.db['t'][0]

	def __call__(self, k, t = None):
		return self.db[f'{k}'].xs(max(noneInit(t, self.t), 0))

	def get(self, k, t = None):
		s = self(k, t = t)
		return s.values if isinstance(s, (pd.Series, pd.DataFrame)) else s

	#######################################################################
	##########					0. Aux methods				 	###########
	#######################################################################
	def Γh(self, t = None):
		return (self('γi') * self.auxProd()).sum()
	def auxΓB1(self, Bi, t = None):
		return np.matmul((self('γi',t) * self.auxProd(t)).values, (Bi/(1+Bi)).T)
	def auxΓB2(self, Bi, t = None):
		return np.matmul(self('γi',t).values, 1/(1+Bi.T))
	def auxΓB3(self, Bi, t = None):
		return np.matmul((self('γi',t) * self.auxProd(t)).values, (Bi/(1+Bi)**2).T)
	def auxΓB4(self, Bi, t = None):
		return np.matmul(self('γi',t).values, (Bi/(1+Bi)**2).T)

	def aux_PEE(self, v1i = None, v10 = None, v2i = None, v20 = None, τ = None, t = None):
		return self.interiorFOC(np.matmul(v2i, self.get('γi[t-1]',t)*self.ω2i(t))+v20*self.get('γ0[t-1]',t)*self.ω20(t)+self.get('ν',t)*(np.matmul(v1i, self.get('γi',t)*self.ω1i(t))+v10*self.get('γ0',t)*self.ω10(t)), τ)

	# Aux parameters:
	def Ω(self, Γs = None, τp = None, t = None):
		return self.get('αr',t)*(self.get('p',t)*self.get('θ[t+1]',t)/self.get('κ',t))*Γs/(self.get('Γh',t)-(self.get('p',t)*self.get('θ[t+1]',t)/self.get('κ',t))*Γs*τp)
	def Ψ(self, Bip = None, τp = None, t = None):
		return (1-self.get('α',t))*(self.get('ρ',t)-1)*(self.auxΓB3(Bip, t)/self.auxΓB1(Bip,t)+self.get('αr',t)*(self.get('p',t)*τp/self.get('κ',t))*self.auxΓB4(Bip,t)/(1+self.get('αr',t)*(self.get('p',t)*τp/self.get('κ',t))*(self.get('θ[t+1]',t)+(1-self.get('θ[t+1]',t))*self.auxΓB2(Bip,t))))
	def σ(self, Ω = None, Ψ = None, dlnhp_dlns = None, τp = None, t = None):
		return 1+Ψ*(1+τp*Ω*(1+self.get('ξ',t))/(1+self.get('α',t)*self.get('ξ',t)))*(1-dlnhp_dlns)

	# Derivatives
	def EELaggedDerivatives(self, Ω = None, Ψ = None, Bip = None, dlnhp_dτp = None, τp = None, t = None):
		""" The derivative used here dlnhp_dτp = ∂ln(h[t+1])/∂τ[t+1]"""
		k1 = Ω * (1+self.get('ξ',t))/(1+self.get('α',t)*self.get('ξ',t))
		k2 = self.get('αr',t)*self.get('p',t)/self.get('κ',t) * (self.get('θ[t+1]',t)+(1-self.get('θ[t+1]',t))*self.auxΓB2(Bip,t))
		k3 = k2/(1+τp*k2)
		dlns_dτp  = (1/(1+Ψ*(1-self.power_h(t))*(1+k1*τp))) * (k1 + (1+k1*τp)*(Ψ*dlnhp_dτp-k3))
		k4 = dlnhp_dτp-dlns_dτp*(1-self.power_h(t))
		dlnΓs_dτp = Ψ*k4-k3
		return {'∂ln(s)/∂τ[t+1]': dlns_dτp,
				'∂ln(Γs)/∂τ[t+1]': dlnΓs_dτp,
				'∂ln(h)/∂τ[t+1]': self.get('ξ',t)*Ω*(τp*dlnΓs_dτp+1)/(1+self.get('α',t)*self.get('ξ',t))}

	def LOG_LaggedEEDerivatives(self, Ω = None, τp = None, Bip = None, t = None):
		k = (self.get('αr[t+1]',t)*self.get('p',t)/self.get('κ',t)) * (self.get('θ[t+1]',t)+(1-self.get('θ[t+1]',t))*self.auxΓB2(Bip, t))
		dlnΓs_dτp = k/(1+τp*k)
		dlnh_dτp  = self.get('ξ',t)*Ω*(τp*dlnΓs_dτp+1)/(1+self.get('α',t)*self.get('ξ',t))
		return {'∂ln(Γs)/∂τ[t+1]': dlnΓs_dτp, '∂ln(h)/∂τ[t+1]': dlnh_dτp, '∂ln(s)/∂τ[t+1]': ((1+self.get('ξ',t))/self.get('ξ',t))*dlnh_dτp+dlnΓs_dτp}

	def EEDerivatives(self, Ψ = None, σ = None, dlnhp_dlns = None, τ = None, τp = None, t = None):
		dlns_dτ  = -(1+self.get('ξ',t))/((1+self.get('α',t)*self.get('ξ',t))*(1-τ)*σ)
		dlnΓs_dτ = Ψ*(dlnhp_dlns-1)*dlns_dτ
		return {'∂ln(s)/∂τ': dlns_dτ, '∂ln(Γs)/∂τ': dlnΓs_dτ, '∂ln(h)/∂τ': self.get('ξ',t)*(dlns_dτ-dlnΓs_dτ)/(1+self.get('ξ',t))}

	def LOG_EEDerivatives(self, τ = None, t = None):
		dlns_dτ = -(1+self.get('ξ',t))/((1+self.get('α',t)*self.get('ξ',t))*(1-τ))
		return {'∂ln(s)/∂τ' : -(1+self.get('ξ',t))/((1+self.get('α',t)*self.get('ξ',t))*(1-τ)),
			 	'∂ln(h)/∂τ' : -self.get('ξ',t)/((1+self.get('α',t)*self.get('ξ',t))*(1-τ)),
			 	'∂ln(Γs)/∂τ': 0}

	def recursive_dlnh_dlns_(self, Ψ = None, σ = None, dlnhp_dlns = None, t = None):
		return self.get('α',t)*self.get('ξ',t)*(1+Ψ*(1-dlnhp_dlns))/((1+self.get('α',t)*self.get('ξ',t))*σ)

	#######################################################################
	##########					1. Simple defs				 	###########
	#######################################################################
	def Bi(self, s_ = None, h = None, t = None):
		return (self.get('βi',t)**self.get('ρ',t))*(self.R(s_ = s_, h = h, t = t)/self('p',t))**(self.get('ρ',t)-1)

	def si_s(self, Bi = None, Γs = None, τp = None, t = None):
		return Bi*self.auxProd(t) /((1+Bi)*(1+self('ξ',t))*Γs)-self('αr',t)*τp*(self('p',t)/self('κ',t))*((1-self('θ[t+1]',t))/(1+Bi)+self('θ[t+1]',t)*self.auxProd(t)/self('Γh',t))

	#######################################################################
	##########				2. Terminal state (FH)			 	###########
	#######################################################################
	def PEE_T(self, τBound = None, τ = None, s_ = None, h = None, Θh = None, dlnh_Dτ = None, si_s = None, t = None):
		""" First order condition for PEE equilibrium in terminal state T"""
		v1i = self.PEE1i_T(h = h, dlnh_Dτ = dlnh_Dτ, t = t)
		v2i = self.PEE2i(τ = τBound, s_ = s_, h = h, dlnh_Dτ = dlnh_Dτ, si_s = si_s, t = t)
		v20 = self.PEE20(τ = τBound, Θh = Θh, s_ = s_, dlnh_Dτ = dlnh_Dτ, t = t)
		return self.aux_PEE(v1i = v1i, v10 = 0, v2i = v2i, v20 = v20, τ = τ)

	def Θh_T(self, τ = None, t = None):
		return self.get('Γh',t)**(1/(1+self.get('ξ',t)*self.get('α',t))) * ((1-self.get('α',t))*(1-τ))**(self.get('ξ',t)/(1+self.get('α',t)*self.get('ξ',t)))
	def h_T(self, s_ = None, τ = None, t = None):
		return self.Θh_T(τ = τ, t = t)*(s_/self.get('ν',t))**self.power_h(t)
	def c1i_T(self, h = None, t = None):
		return self.auxProd(t)*(h/self('Γh',t))**((1+self('ξ',t))/self('ξ',t))
	def c̃1i_T(self, h = None, t = None):
		return self.c1i_T(h = h, t = t)/(1+self('ξ',t))
	def PEE1i_T(self, h = None, dlnh_Dτ = None, t = None):
		return self.c̃1i_T(h =  h, t = t)**(1-1/self('ρ',t))*self.LOG_PEE1i_T(dlnh_Dτ=dlnh_Dτ, t = t)
	def dlnh_Dτ_T(self, τ = None, t = None):
		return -self.get('ξ',t)/((1+self.get('ξ',t)*self.get('α',t))*(1-τ))

	#### LOG METHODS
	def LOG_PEE_T(self, τBound = None, τ = None, dlnh_Dτ = None, si_s = None, t = None):
		v1i = self.LOG_PEE1i_T(dlnh_Dτ = dlnh_Dτ, t = t)
		v2i = self.LOG_PEE2i(τ = τBound, dlnh_Dτ = dlnh_Dτ, si_s = si_s, t = t)
		v20 = self.LOG_PEE20(τ = τBound, dlnh_Dτ = dlnh_Dτ, t = t)
		return self.aux_PEE(v1i = v1i, v10 = 0, v2i = v2i, v20 = v20, τ = τ)
	def LOG_PEE1i_T(self, dlnh_Dτ = None, t= None):	
		return np.full(self.m.ni, dlnh_Dτ*(1+self.get('ξ',t))/self.get('ξ',t))

	#######################################################################
	##########				3. Steady state methods				###########
	#######################################################################
	def steadyStateEq_Bi(self, Bi, Γs, τ, t = None):
		if self.get('ρ',t)<1:
			return self.get('βi',t)**(self.get('ρ',t)/(1-self.get('ρ',t))) * (1-self.get('α',t))*(1-τ)-Bi**(1/(1-self.get('ρ',t)))*(self.get('ν',t)/self.get('p',t))*(self.get('α',t)*self.get('Γh',t)-(1-self.get('α',t))*self.get('p',t)*self.get('θ',t)*τ*Γs/self.get('κ',t))/Γs
		elif self.get('ρ',t)>1:
			return Bi**(1/(self.get('ρ',t)-1))*(1-self.get('α',t))*(1-τ)-self.get('βi',t)**(self.get('ρ',t)/(self.get('ρ',t)-1))*(self.get('ν',t)/self.get('p',t))*(self.get('α',t)*self.get('Γh',t)-(1-self.get('α',t))*self.get('p',t)*self.get('θ',t)*τ*Γs/self.get('κ',t))/Γs

	def steadyStateEq_Γs(self, Bi, Γs, τ, t = None):
		return self.auxΓB1(Bi, t)-Γs*(1+self.get('ξ',t))*(1+(self.get('αr',t)*self.get('p',t)/self.get('κ',t))*τ*(self.get('θ',t)+(1-self.get('θ',t))*self.auxΓB2(Bi,t)))

	def steadyStateEqs(self, Bi, Γs, τ, t = None):
		return np.hstack([self.steadyStateEq_Bi(Bi, Γs, τ, t = t).reshape(-1), self.steadyStateEq_Γs(Bi, Γs, τ, t = t)])

	def steadyState_s(self, Γs, τ, t = None):
		""" Return steady state level of savings"""
		return self.get('Γh',t)**(1+self.get('ξ',t))*( ((1-self.get('α',t))*(1-τ)/(self.get('Γh',t)-self.auxForm1(t)*τ*Γs))**(1+self.get('ξ',t))* Γs**(1+self.get('α',t)*self.get('ξ',t))/self.get('ν',t)**(self.get('α',t)*(1+self.get('ξ',t))))**(1/(1-self.get('α',t)))

	def steadyStateScalar_Bi(self, Γs, τ, t = None):
		""" Requires scalar inputs and τ∈[0, 1) """
		if self.get('ρ',t)<1:
			return self.get('βi',t)**(self.get('ρ',t))*((self.get('α',t)/self.get('p',t)) * (self.get('ν',t)/Γs)*(self.get('Γh',t)-self.get('αr',t)*self.get('p',t)*self.get('θ[t+1]',t)*τ*Γs/self.get('κ',t))/((1-self.get('α',t))*(1-τ)))**(self.get('ρ',t)-1)
		else:
			return self.get('βi',t)**(self.get('ρ',t))*((self.get('p',t)/self.get('α',t)) * (Γs/self.get('ν',t))*((1-self.get('α',t))*(1-τ))/(self.get('Γh',t)-self.get('αr',t)*self.get('p',t)*self.get('θ[t+1]',t)*τ*Γs/self.get('κ',t)))**(1-self.get('ρ',t))

	def steadyStateScalarEq(self, Γs, τ, t = None):
		""" This = 0 in steady state for Γs """
		return self.steadyStateEq_Γs(self.steadyStateScalar_Bi(Γs, τ, t = t), Γs, τ, t = t)

	def LOG_steadyState_Γs(self, τ, t = None):
		return self.Γs(Bi = self.get('βi',t), τp = τ, t = t)

	#######################################################################
	##########				4. Out of terminal state			###########
	#######################################################################
	def PEE_t(self, τBound = None, τ  = None, τp = None, s_ = None, s = None, h = None, Γs = None, Bip = None, si_s = None, Θh = None, Θhp = None, 
					dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dlnhp_Dτ = None, dτp_dτ = None, t = None):
		v1i = self.PEE1i_t(τp = τp, h = h, Γs = Γs, Bip = Bip, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, t = t)
		v10 = self.PEE10_t(s = s, τp = τp, Θhp = Θhp, dlns_Dτ = dlns_Dτ, dτp_dτ = dτp_dτ, dlnhp_Dτ = dlnhp_Dτ, t = t)
		v2i = self.PEE2i(τ = τBound, s_ = s_, h = h, dlnh_Dτ = dlnh_Dτ, si_s = si_s, t = t)
		v20 = self.PEE20(τ = τBound, Θh = Θh, s_ = s_, dlnh_Dτ = dlnh_Dτ, t = t)
		return self.aux_PEE(v1i = v1i, v10 = v10, v2i = v2i, v20 = v20, τ = τ)

	def LOG_PEE_t(self, τBound = None, τ  = None, τp = None, Γs = None, Bip = None, B0p = None, si_s = None, Θs = None,
					dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		v1i = self.LOG_PEE1i_t(τp = τp, Γs = Γs, Bip = Bip, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, t = t)
		v10 = self.LOG_PEE10_t(τp = τp, B0p = B0p, Θs = Θs, dlns_Dτ = dlns_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, t = t)
		v2i = self.LOG_PEE2i(τ = τBound, dlnh_Dτ = dlnh_Dτ, si_s = si_s, t = t)
		v20 = self.LOG_PEE20(τ = τBound, dlnh_Dτ = dlnh_Dτ, t = t)
		return self.aux_PEE(v1i = v1i, v10 = v10, v2i = v2i, v20 = v20, τ = τ)

	# Retired formal
	def PEE2i(self, τ = None, s_ = None, h = None, dlnh_Dτ = None, si_s = None, t = None):
		return self.c2i(τ = τ, s_ = s_, h = h, si_s = si_s, t = t)**(1-1/self.get('ρ[t-1]',t))*self.LOG_PEE2i(τ = τ, dlnh_Dτ=dlnh_Dτ, si_s= si_s, t = t)
	def aux_c2i_coeff(self, t = None):
		return self.get('αr',t)*(self.get('p[t-1]',t)/self.get('κ[t-1]',t))*(1+self.get('θ',t)*(self.auxProd(noneInit(t,self.t)-1)/self.get('Γh[t-1]',t)-1))
	def c2i(self, τ = None, s_ = None, h = None, si_s = None,  t = None):
		return self.get('α',t)*(self.get('ν',t)/self.get('p[t-1]',t))*h**(1-self.get('α',t))*(s_/self.get('ν',t))**self.get('α',t)*(si_s + τ*self.aux_c2i_coeff(t))

	def LOG_PEE2i(self, τ = None, dlnh_Dτ = None, si_s = None, t = None):
		return (1-self.get('α',t))*dlnh_Dτ+self.aux_c2i_coeff(t)/(si_s+τ * self.aux_c2i_coeff(t))

	# Retired informal
	def PEE20(self, τ = None, Θh = None, s_ = None, dlnh_Dτ = None, t = None): 
		return self.c̃20(τ = τ, Θh = Θh, s_ = s_, t = t)**(1-1/self.get('ρ[t-1]',t))*self.LOG_PEE20(τ = τ, Θh = Θh, dlnh_Dτ = dlnh_Dτ, t = t)
	def Θc̃20(self, τ = None, Θh = None, t = None):
		return self.get('χ[t-1]',t)**(1+self.get('ξ[t-1]',t))*self.auxInf0(noneInit(t,self.t)-1)+(1-self.get('α',t))*self.get('ν',t)*self.get('eps',t)*τ *Θh**(1-self.get('α',t)) /self.get('κ[t-1]',t)
	def c̃20(self, τ = None, Θh = None, s_ = None, t = None):
		return self.Θc̃20(τ = τ, Θh = Θh, t = t)*(s_/self.get('ν',t))**self.power_s(t)
	def LOG_PEE20(self, τ = None, Θh = None, dlnh_Dτ = None, t = None):
		return ((1-self.get('α',t))*self.get('ν',t)*self.get('eps',t)*Θh**(1-self.get('α',t)) /self.get('κ[t-1]',t)) * (1+(1-self.get('α',t)*τ*dlnh_Dτ))/self.Θc̃20(τ = τ, Θh = Θh, t = t)

	# working formal
	def PEE1i_t(self, τp = None, h = None, Γs = None, Bip = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		return self.ĉ1i_t(τp = τp, h = h, Bip = Bip, Γs = Γs, t=t)**(1-1/self.get('ρ',t))*self.dlnĉ1i_dτ(τp = τp, Γs = Γs, Bip = Bip, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, t = t)
	def dlnĉ1i_dτ(self, τp = None, Γs = None, Bip = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		x = self.get('αr',t)*self.get('p',t)*(1-self.get('θ[t+1]',t))*Γs/self.get('κ',t)
		return dlnh_Dτ*(1+self.get('ξ',t))/self.get('ξ',t)+(Bip/(1+Bip))*(1-self.get('α',t))*(dlnhp_Dlns-1)*dlns_Dτ+x*(dτp_dτ+τp*dlnΓs_Dτ)/(self.auxProd(t)/(1+self.get('ξ',t))+τp*x)
	def ĉ1i_t(self, τp = None, h = None, Bip = None, Γs = None, t = None):
		return (h/self.get('Γh',t))**((1+self.get('ξ',t))/self.get('ξ',t))*(1+Bip)**(1/(self.get('ρ',t)-1)) * (self.auxProd(t)/(1+self.get('ξ',t))+Γs*self.get('αr',t)*self.get('p',t)*τp*(1-self.get('θ[t+1]',t))/self.get('κ',t))

	def LOG_dlnĉ1i_dτ(self, τp = None, Γs = None, dlnh_Dτ = None, dlnΓs_Dτ = None, dτp_dτ=None, t = None):
		k = self.get('αr',t)*self.get('p',t)*(1-self.get('θ[t+1]',t))*Γs/self.get('κ',t)
		return dlnh_Dτ*(1+self.get('ξ',t))/self.get('ξ',t)+(k*(dτp_dτ+τp*dlnΓs_Dτ))/(self.auxProd(t)/(1+self.get('ξ',t))+(τp*k))
	def LOG_PEE1i_t(self, τp = None, Γs = None, Bip = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		return self.LOG_dlnĉ1i_dτ(τp = τp, Γs = Γs, dlnh_Dτ = dlnh_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dτp_dτ = dτp_dτ, t = t)*(1+Bip)+Bip*(1-self.get('α',t))*(dlnhp_Dlns-1)*dlns_Dτ

	# working, informal
	def PEE10_t(self, s = None, τp = None, Θhp = None, dlns_Dτ = None, dτp_dτ = None, dlnhp_Dτ = None, t = None):
		return self.get('β0',t)*self.c̃2p0(τp = τp, Θhp = Θhp, s = s, t = t)**(1-1/self.get('ρ',t))*self.dlnc̃2p0_dτ(τp = τp, Θhp = Θhp, dτp_dτ = dτp_dτ, dlns_Dτ = dlns_Dτ, dlnhp_Dτ=dlnhp_Dτ, t = t)
	def c̃2p0(self, τp = None, Θhp = None, s = None, t = None):
		return self.c̃20(τ = τp, Θh = Θhp, s_ = s, t = noneInit(t, self.t)+1)
	def dlnc̃2p0_dτ(self, τp = None, Θhp = None, dτp_dτ = None, dlns_Dτ = None, dlnhp_Dτ = None, t = None):
		tp = noneInit(t, self.t)+1
		return self.power_s(tp)*dlns_Dτ+((1-self.get('α[t+1]',t))*self.get('ν[t+1]',t)*self.get('eps[t+1]',t)*Θhp**(1-self.get('α[t+1]',t))/self.get('κ',t))*(dτp_dτ+(1-self.get('α[t+1]',t))*dlnhp_Dτ)/self.Θc̃20(τ = τp, Θh = Θhp, t = tp)
	def c̃10(self, s_= None, t = None):
		return self.auxInf0(t)*(s_/self.get('ν',t))**self.power_s(t)


	def LOG_backOutΘs(self, B0p = None,  τp = None, t = None):
		return B0p * self.auxInf0(t) / (self.auxInf1(t)*τp)
	def LOG_backOutτ(self, Θs = None, Γs = None, τp = None, t = None):
		return 1-(Θs/Γs)**((1+self.get('α',t)*self.get('ξ',t))/(1+self.get('ξ',t)))*(self.get('Γh',t)-self.get('αr',t)*(self.get('p',t)*self.get('θ[t+1]',t)*τp/self.get('κ',t))*Γs)/(self.get('Γh',t)**(1-self.get('α',t))*(1-self.get('α',t)))

	def backOutH(self, s = None, Γs = None, t = None):
		return (s/Γs)**(self.get('ξ',t)/(1+self.get('ξ',t)))*self.get('Γh',t)
	def backOutS_(self, h = None, τ = None, τp = None, Γs = None, t = None):
		return self.get('ν',t)*h**(1/self.power_h(t))*((self.get('Γh',t)-self.auxForm1(t)*τp*Γs)/((1-self.get('α',t))*(1-τ)))**(1/self.get('α',t))
	def backOutΘs(self, s_ = None, s = None, t = None):
		return s/((s_/self.get('ν',t))**self.power_s(t))
	def backOutτ(self, Γs = None, τp = None, Θs = None, t = None):
		return 1-(Θs/Γs)**((1+self.get('α',t)*self.get('ξ',t))/(1+self.get('ξ',t)))*(self.get('Γh',t)-self.get('αr',t)*self.get('p',t)*self.get('θ[t+1]',t)*τp*Γs/self.get('κ',t))/(self.get('Γh',t)**(1-self.get('α',t))*(1-self.get('α',t)))

	#######################################################################
	##########			5. Calibration specific methods			###########
	#######################################################################

	def calib_η0(self, τ = None, Θh = None, t = None):
		t = noneInit(t, self.db['t0'])
		return (self('zη0',t)/self('zx0',t)) * (1-self('α',t))*(1-τ)/(Θh**(self('α',t))* ((1-self('α',t))/self('Γh',t)**(self('α',t)))**(1/(1+self('ξ',t)*self.get('α',t))))
	def calib_X0(self, η0 = None, Θh = None, t = None):
		t = noneInit(t, self.db['t0'])
		return η0 * ((1-self('α',t))/self('Γh',t)**(self('α',t)))**(1/(1+self('ξ',t)*self('α',t))) / ((Θh*self('zx0',t))**(1/self('ξ',t)))
	# def calib_savingsRate(self, Θs = None, Θh = None, t = None):
	# 	t = noneInit(t, self.db['t0'])
	# 	return Θs/((1-self('α',t))*(Θh**(1-self('α',t))))
	def calib_savingsRate(self, s_ = None, s= None, h = None, t = None):
		t = noneInit(t, self.db['t0'])
		return s / ((s_/self.get('ν',t))**self.get('α',t) * h**(1-self.get('α',t)))

class BaseGrid_A(BaseScalar_A):
	def __init__(self, m, t = None):
		super().__init__(m,t=t)

	#######################################################################
	##########					1. Simple defs				 	###########
	#######################################################################
	def Bi(self, s_ = None, h = None, t = None):
		return ((self.R(s_ = s_, h = h, t = t)/self('p',t))**(self('ρ',t)-1))[:,None] * (self.get('βi',t)**self('ρ',t))[:,None].T

	def si_s(self, Bi = None, Γs = None, τp = None, t = None):
		return Bi*self.auxProd(t) /((1+Bi)*(1+self('ξ',t))*Γs[:,None])-(self('αr',t)*τp*self('p',t)/self('κ',t))[:,None]*((1-self('θ[t+1]',t))/(1+Bi)+self('θ[t+1]',t)*self.auxProd(t)/self.get('Γh',t))

	#######################################################################
	##########				2. Terminal state (FH)			 	###########
	#######################################################################
	def c1i_T(self, h = None, t = None):
		return ((h/self('Γh',t))**((1+self('ξ',t))/self('ξ',t)))[:,None]*self.auxProd(t)[:,None].T
	def PEE1i_T(self, h = None, dlnh_Dτ = None, t = None):
		return self.c̃1i_T(h =  h, t = t)**(1-1/self('ρ',t))*((1+self('ξ',t))/self('ξ',t))*dlnh_Dτ[:,None]
	def LOG_PEE1i_T(self, dlnh_Dτ = None, t= None):	
		return np.tile(dlnh_Dτ[:,None]*(1+self.get('ξ',t))/self.get('ξ',t), self.m.ni)


	#######################################################################
	##########				3. Steady state methods				###########
	#######################################################################
	def steadyStateEq_Bi(self, Bi, Γs, τ, t = None):
		if self.get('ρ',t)<1:
			return self.get('βi',t)[:,None].T**(self.get('ρ',t)/(1-self.get('ρ',t))) * (1-self.get('α',t))*(1-τ[:,None])-Bi**(1/(1-self.get('ρ',t)))*((self.get('ν',t)/self.get('p',t))*(self.get('α',t)*self.get('Γh',t)-(1-self.get('α',t))*self.get('p',t)*self.get('θ',t)*(τ*Γs/self.get('κ',t)))/Γs)[:,None]
		elif self.get('ρ',t)>1:
			return Bi**(1/(self.get('ρ',t)-1))*(1-self.get('α',t))*(1-τ[:,None])-self.get('βi',t)[:,None].T**(self.get('ρ',t)/(self.get('ρ',t)-1))*((self.get('ν',t)/self.get('p',t))*(self.get('α',t)*self.get('Γh',t)-(1-self.get('α',t))*self.get('p',t)*self.get('θ',t)*(τ*Γs/self.get('κ',t)))/Γs)[:,None]


	#######################################################################
	##########				4. Out of terminal state			###########
	#######################################################################
	def PEE2i(self, τ = None, s_ = None, h = None, dlnh_Dτ = None, si_s = None, t = None):
		return self.c2i(τ = τ, s_ = s_, h = h, si_s = si_s, t = t)**(1-1/self.get('ρ[t-1]',t))*self.LOG_PEE2i(τ = τ, dlnh_Dτ = dlnh_Dτ, si_s = si_s, t = t)
	def c2i(self, τ = None, s_ = None, h = None, si_s = None,  t = None):
		return (self.get('α[t-1]',t)*(self.get('ν',t)/self.get('p[t-1]',t))*h**(1-self.get('α',t))*(s_/self.get('ν',t))**self.get('α',t))[:,None]*(si_s + self.aux_c2i_coeff(t)*τ[:,None])
	def LOG_PEE2i(self, τ = None, dlnh_Dτ = None, si_s = None, t = None):
		return (1-self.get('α',t))*dlnh_Dτ[:,None]+self.aux_c2i_coeff(t)/(si_s+self.aux_c2i_coeff(t)*τ[:,None])


	def ĉ1i_t(self, τp = None, h = None, Bip = None, Γs = None, t = None):
		return ((h/self.get('Γh',t))**((1+self.get('ξ',t))/self.get('ξ',t)))[:,None]*(1+Bip)**(1/(1-self.get('ρ',t))) * ((Γs*self.get('αr',t)*self.get('p',t)*τp*(1-self.get('θ[t+1]',t))/self.get('κ',t))[:,None]+self.auxProd(t)/(1+self.get('ξ',t)))
	def dlnĉ1i_dτ(self, τp = None, Γs = None, Bip = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		x = (self.get('αr',t)*self.get('p',t)*(1-self.get('θ[t+1]',t))*Γs/self.get('κ',t))
		return dlnh_Dτ[:,None]*(1+self.get('ξ',t))/self.get('ξ',t)+(Bip/(1+Bip))*(1-self.get('α',t))*((dlnhp_Dlns-1)*dlns_Dτ)[:,None]+(x*(dτp_dτ+τp*dlnΓs_Dτ))[:,None]/(self.auxProd(t)/(1+self.get('ξ',t))+(τp*x)[:,None])

	def LOG_dlnĉ1i_dτ(self, τp = None, Γs = None, dlnh_Dτ = None, dlnΓs_Dτ = None, dτp_dτ=None, t = None):
		k = self.get('αr',t)*self.get('p',t)*(1-self.get('θ[t+1]',t))*Γs/self.get('κ',t)
		return dlnh_Dτ[:,None]*(1+self.get('ξ',t))/self.get('ξ',t)+(k*(dτp_dτ+τp*dlnΓs_Dτ))[:,None]/(self.auxProd(t)/(1+self.get('ξ',t))+(τp*k)[:,None])
	def LOG_PEE1i_t(self, τp = None, Γs = None, Bip = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		return self.LOG_dlnĉ1i_dτ(τp = τp, Γs = Γs, dlnh_Dτ = dlnh_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dτp_dτ = dτp_dτ, t = t)*(1+Bip)+Bip*(1-self.get('α',t))*((dlnhp_Dlns-1)*dlns_Dτ)[:,None]


class BaseTime_A(_Base_A):
	def __init__(self, m, ts = 'FH'):
		super().__init__(m)
		self.ts = ts
	def __call__(self, k, t = None):
		return self.db[k] if t is None else self.db[k].loc[t]
	def get(self, k, t = None):
		s = self(k, t = t)
		return s.values if isinstance(s, (pd.Series, pd.DataFrame)) else s

	#######################################################################
	##########					0. Aux methods				 	###########
	#######################################################################
	def Γh(self, t = None):
		return (self('γi',t) * self.auxProd(t)).sum(axis=1)
	def auxΓB1(self, Bi, t = None):
		return ((self('γi',t) * self.auxProd(t)).values * (Bi/(1+Bi))).sum(axis=1)
	def auxΓB2(self, Bi, t = None):
		return (self.get('γi',t) * (1/(1+Bi))).sum(axis=1)
	def auxΓB3(self, Bi, t = None):
		return ((self('γi',t) * self.auxProd(t)).values * (Bi/(1+Bi)**2)).sum(axis=1)
	def auxΓB4(self, Bi, t = None):
		return (self.get('γi',t)*Bi/(1+Bi)**2).sum(axis=1)

	#######################################################################
	##########					1. Simple defs				 	###########
	#######################################################################
	def Bi(self, s_ = None, h = None, t = None):
		return self('βi',t).pow(self('ρ',t), axis = 0).values * ((self.R(s_ = s_, h = h, t = t)/self.get('p',t))**(self.get('ρ',t)-1))[:,None]

	def si_s(self, Bi = None, Γs = None, τp = None, t = None):
		return Bi*self.auxProd(t) /((1+Bi)*(1+self.get('ξ',t)[:,None])*Γs[:,None])-(self.get('αr',t)*τp*self.get('p',t)/self.get('κ',t))[:,None]*((1-self.get('θ[t+1]',t)[:,None])/(1+Bi)+self.get('θ[t+1]',t)[:,None]*self.auxProd(t)/self.get('Γh',t)[:,None])

	#######################################################################
	##########				2. Terminal state (FH)			 	###########
	#######################################################################
	def Θh_T(self, τ = None, t = None):
		return self.get('Γh',t)**(1/(1+self.get('ξ',t)*self.get('α',t))) * ((1-self.get('α',t))*(1-τ))**(self.get('ξ',t)/(1+self.get('α',t)*self.get('ξ',t)))
	def h_T(self, s_ = None, τ = None, t = None):
		return self.Θh_T(τ = τ, t = t)*(s_/self.get('ν',t))**self.power_h(t)
	

	#######################################################################
	##########				4. Finite horizon methods			###########
	#######################################################################
	def FH_Θh(self, τ = None, τp = None, Γs = None):
		return np.hstack([self.Θh_t(τ = τ[:-1], τp = τp[:-1], Γs = Γs, t = self.db['txE']), self.Θh_T(τ = τ[-1], t = self.db['t'][-1])])
	def FH_h(self, s_ = None, τ = None, τp = None, Γs = None):
		return np.hstack([self.h_t(s_=s_[:-1], τ = τ[:-1], τp = τp[:-1], Γs = Γs, t = self.db['txE']), self.h_T(s_ = s_[-1], τ = τ[-1], t = self.db['t'][-1])])
	def FH_s(self, h = None, Γs = None):
		return self.s_t(h = h[:-1], Γs = Γs, t = self.db['txE'])
	def FH_Γs(self, s = None, hp = None, τp = None):
		return self.Γs(self.Bi(s_ = s, h = hp, t = self.db['txE']), τp = τp[:-1], t = self.db['txE'])
	def FH_BackOutΘs(self, s_ = None, s = None):
		return s/((s_[:-1]/self.get('ν',t = self.db['txE']))**self.power_s(self.db['txE']))
	def FH_BackOutΘh(self, s_ = None, h = None):
		return h/((s_/self('ν'))**self.power_h())

	#### LOG METHODS:
	def FH_LOG_Γs(self, τ = None, τp = None):
		return self.Γs(Bi = self.get('βi', t = self.db['txE']), τp = τp[:-1], t = self.db['txE'])
	@property
	def FH_LOG_Θh(self):
		return self.FH_Θh
	def FH_LOG_Θs(self, Θh = None, Γs = None):
		return self.Θs_t(Θh[:-1], Γs = Γs, t = self.db['txE'])
	def FH_LOG_s(self, Θs = None, s0 = None):
		s = np.empty(self.m.T-1)
		s[0] = Θs[0]*(s0/self.db['ν'][0])**(self.power_s(self.db['t'][0]))
		for t in range(1, self.m.T-1):
			s[t] = Θs[t]*(s[t-1]/self.db['ν'][t])**(self.power_s(self.db['t'][t]))
		return s
	def FH_LOG_h(self, Θh = None, s_ = None):
		return Θh * (s_/self.get('ν'))**self.power_h()
