import numpy as np, pandas as pd
from pyDbs import is_iterable
def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

class _Base:
	def __init__(self, m):
		self.m = m
		self.db = m.db
		self.t0 = self.db['t'][0]

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
	def auxProd_(self, t = None):
		return (self('ηi[t-1]',t).pow(1+self('ξ[t-1]',t), axis = 0)/self('Xi[t-1]',t).pow(self('ξ[t-1]',t), axis = 0)).values
	def auxProd0(self, t = None):
		return self.get('η0', t)**(1+self.get('ξ',t))/self.get('X0',t)**self.get('ξ',t)

	def auxInf0(self, t = None):
		return self.auxProd0(t)*((1-self.get('α',t))/self.get('Γh',t)**(self.get('α',t)))**((1+self.get('ξ',t))/(1+self.get('ξ',t)*self.get('α',t)))/(1+self.get('ξ',t))
	def auxInf1(self, t = None):
		return (self.get('αr[t+1]',t)/self.get('α0[t+1]',t))*self.get('p0',t)*self.get('eps[t+1]',t)/self.get('κ',t)
	def auxInf1_(self, t = None):
		return (self.get('αr',t)/self.get('α0',t))*self.get('p0[t-1]',t)*self.get('eps',t)/self.get('κ[t-1]',t)
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


class BaseScalar(_Base):
	def __init__(self, m, t = None):
		super().__init__(m)
		self.t = t

	def __call__(self, k, t = None):
		return self.db[f'{k}'].xs(max(noneInit(t, self.t), self.t0))

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
	def EELaggedDerivatives(self, Ω = None, Ψ = None, Bip = None, dlnhp_dτp = None, τp = None, B0p = None, Θs = None, t = None):
		""" The derivative used here dlnhp_dτp = ∂ln(h[t+1])/∂τ[t+1]"""
		k1 = Ω * (1+self.get('ξ',t))/(1+self.get('α',t)*self.get('ξ',t))
		k2 = (self.get('αr',t)*self.get('p',t)/self.get('κ',t)) * (self.get('θ[t+1]',t)+(1-self.get('θ[t+1]',t))*self.auxΓB2(Bip,t))
		k3 = k2/(1+τp*k2)
		dlns_dτp  = (1/(1+Ψ*(1-self.power_h(t))*(1+k1*τp))) * (k1 + (1+k1*τp)*(Ψ*dlnhp_dτp-k3))
		k4 = dlnhp_dτp-dlns_dτp*(1-self.power_h(t))
		dlnΓs_dτp = Ψ*k4-k3
		d = {'∂ln(s)/∂τ[t+1]': dlns_dτp,
			 '∂ln(Γs)/∂τ[t+1]': dlnΓs_dτp,
			 '∂ln(h)/∂τ[t+1]': self.get('ξ',t)*Ω*(τp*dlnΓs_dτp+1)/(1+self.get('α',t)*self.get('ξ',t))}
		if B0p is None:
			return d
		else:
			x1 = (B0p/(1+B0p))*self.auxInf0(t)/Θs
			x2 = (self.get('ρ',t)-1)*(1-self.get('α',t))*k4/(1+B0p)
			x3 = (B0p/(1+B0p))*self.auxInf1(t)
			d['∂(s0/s)/∂τ[t+1]'] = x1*(x2-dlns_dτp)+x3*(τp*x2+1)
			return d

	def LOG_LaggedEEDerivatives(self, Ω = None, τp = None, Bip = None, B0p = None, Θs = None, t = None):
		k = (self.get('αr[t+1]',t)*self.get('p',t)/self.get('κ',t)) * (self.get('θ[t+1]',t)+(1-self.get('θ[t+1]',t))*self.auxΓB2(Bip, t))
		dlnΓs_dτp = -k/(1+τp*k)
		dlnh_dτp  = self.get('ξ',t)*Ω*(τp*dlnΓs_dτp+1)/(1+self.get('α',t)*self.get('ξ',t))
		d = {'∂ln(Γs)/∂τ[t+1]': dlnΓs_dτp, '∂ln(h)/∂τ[t+1]': dlnh_dτp, '∂ln(s)/∂τ[t+1]': ((1+self.get('ξ',t))/self.get('ξ',t))*dlnh_dτp+dlnΓs_dτp}
		if B0p is None:
			return d
		else:
			d['∂(s0/s)/∂τ[t+1]'] = (B0p/(1+B0p))*(self.auxInf1(t)-d['∂ln(s)/∂τ[t+1]']*self.auxInf0(t)/Θs)
			return d

	def EEDerivatives(self, Ψ = None, σ = None, dlnhp_dlns = None, τ = None, τp = None, B0p = None, Θs = None, t = None):
		dlns_dτ  = -(1+self.get('ξ',t))/((1+self.get('α',t)*self.get('ξ',t))*(1-τ)*σ)
		dlnΓs_dτ = Ψ*(dlnhp_dlns-1)*dlns_dτ
		d = {'∂ln(s)/∂τ': dlns_dτ, '∂ln(Γs)/∂τ': dlnΓs_dτ, '∂ln(h)/∂τ': self.get('ξ',t)*(dlns_dτ-dlnΓs_dτ)/(1+self.get('ξ',t))}
		if B0p is None:
			return d
		else:
			x1 = self.auxInf0(t)/Θs
			x2 = (self.get('ρ',t)-1)*(1-self.get('α',t))*(dlnhp_dlns-1)/(1+B0p)
			d['∂(s0/s)/∂τ'] = dlns_dτ*(B0p/(1+B0p))*(x1*(x2-1)+x2*self.auxInf1(t)*τp)
			return d

	def LOG_EEDerivatives(self, τ = None, B0p = None, Θs = None, t = None):
		dlns_dτ = -(1+self.get('ξ',t))/((1+self.get('α',t)*self.get('ξ',t))*(1-τ))
		d = {'∂ln(s)/∂τ' : -(1+self.get('ξ',t))/((1+self.get('α',t)*self.get('ξ',t))*(1-τ)),
			 '∂ln(h)/∂τ' : -self.get('ξ',t)/((1+self.get('α',t)*self.get('ξ',t))*(1-τ)),
			 '∂ln(Γs)/∂τ': 0}
		if B0p is None:
			return d
		else:
			d['∂(s0/s)/∂τ'] = -d['∂ln(s)/∂τ'] * (B0p/(1+B0p)) * self.auxInf0(t)/Θs
			return d

	def recursive_dlnh_dlns_(self, Ψ = None, σ = None, dlnhp_dlns = None, t = None):
		return self.get('α',t)*self.get('ξ',t)*(1+Ψ*(1-dlnhp_dlns))/((1+self.get('α',t)*self.get('ξ',t))*σ)

	#######################################################################
	##########					1. Simple defs				 	###########
	#######################################################################
	def Bi(self, s_ = None, h = None, t = None):
		return (self.get('βi',t)**self.get('ρ',t))*(self.R(s_ = s_, h = h, t = t)/self('p',t))**(self.get('ρ',t)-1)
	def B0(self, s_ = None, h = None, t = None):
		return self('β0',t)**(self('ρ',t))*(self.R(s_=s_, h=h,t=t)/self('p0',t))**(self('ρ',t)-1)

	def si_s(self, Bi = None, Γs = None, τp = None, t = None):
		return Bi*self.auxProd(t) /((1+Bi)*(1+self('ξ',t))*Γs)-self('αr',t)*τp*(self('p',t)/self('κ',t))*((1-self('θ[t+1]',t))/(1+Bi)+self('θ[t+1]',t)*self.auxProd(t)/self('Γh',t))
	def s0_s(self, B0 = None, Θs = None, τp = None, t = None):
		return (1/(1+B0))*(B0*self.auxInf0(t)/Θs-τp*self.auxInf1(t))

	#######################################################################
	##########				2. Terminal state (FH)			 	###########
	#######################################################################
	def PEE_T(self, τBound = None, τ = None, s_ = None, h = None, dlnh_Dτ = None, si_s = None, s0_s = None, t = None):
		""" First order condition for PEE equilibrium in terminal state T"""
		v1i = self.PEE1i_T(h = h, dlnh_Dτ = dlnh_Dτ, t = t)
		v2i = self.PEE2i(τ = τBound, s_ = s_, h = h, dlnh_Dτ = dlnh_Dτ, si_s = si_s, t = t)
		v20 = self.PEE20(τ = τBound, s_ = s_, h = h, dlnh_Dτ = dlnh_Dτ, s0_s = s0_s, t = t)
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
	def LOG_PEE_T(self, τBound = None, τ = None, dlnh_Dτ = None, si_s = None, s0_s = None, t = None):
		v1i = self.LOG_PEE1i_T(dlnh_Dτ = dlnh_Dτ, t = t)
		v2i = self.LOG_PEE2i(τ = τBound, dlnh_Dτ = dlnh_Dτ, si_s = si_s, t = t)
		v20 = self.LOG_PEE20(τ = τBound, dlnh_Dτ = dlnh_Dτ, s0_s = s0_s, t = t)
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

	def PEE_t(self, τBound = None, τ  = None, τp = None, s_ = None, h = None, Γs = None, Bip = None, B0p = None, si_s = None, s0_s = None, Θs = None,
					dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		v1i = self.PEE1i_t(τp = τp, h = h, Γs = Γs, Bip = Bip, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, t = t)
		v10 = self.PEE10_t(τp = τp, s_ = s_, B0p = B0p, Θs = Θs, dlns_Dτ = dlns_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, t = t)
		v2i = self.PEE2i(τ = τBound, s_ = s_, h = h, dlnh_Dτ = dlnh_Dτ, si_s = si_s, t = t)
		v20 = self.PEE20(τ = τBound, s_ = s_, h = h, dlnh_Dτ = dlnh_Dτ, s0_s = s0_s, t = t)
		return self.aux_PEE(v1i = v1i, v10 = v10, v2i = v2i, v20 = v20, τ = τ)

	def LOG_PEE_t(self, τBound = None, τ  = None, τp = None, Γs = None, Bip = None, B0p = None, si_s = None, s0_s = None, Θs = None,
					dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		v1i = self.LOG_PEE1i_t(τp = τp, Γs = Γs, Bip = Bip, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, t = t)
		v10 = self.LOG_PEE10_t(τp = τp, B0p = B0p, Θs = Θs, dlns_Dτ = dlns_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, t = t)
		v2i = self.LOG_PEE2i(τ = τBound, dlnh_Dτ = dlnh_Dτ, si_s = si_s, t = t)
		v20 = self.LOG_PEE20(τ = τBound, dlnh_Dτ = dlnh_Dτ, s0_s = s0_s, t = t)
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
	def PEE20(self, τ = None, s_ = None, h = None, dlnh_Dτ = None, s0_s = None, t = None):
		return self.c20(τ = τ, s_ = s_, h = h, s0_s = s0_s, t = t)**(1-1/self.get('ρ[t-1]',t))*self.LOG_PEE20(τ = τ, dlnh_Dτ = dlnh_Dτ, s0_s = s0_s, t = t)
	def aux_c20_coeff(self, t = None):
		return (self.get('αr',t)/self.get('α0',t))*(self.get('p0[t-1]',t)*self.get('eps',t))/self.get('κ[t-1]',t)
	def c20(self, τ = None, s_ = None, h = None, s0_s = None, t = None):
		return self.get('α0',t)*self.get('α',t)*(self.get('ν',t)/self.get('p0[t-1]',t))*(s_/self.get('ν',t))**self.get('α',t)*h**(1-self.get('α',t))*(s0_s+τ*self.aux_c20_coeff(t))
	def LOG_PEE20(self, τ = None, dlnh_Dτ = None, s0_s = None, t = None):
		return (1-self.get('α',t))*dlnh_Dτ+self.aux_c20_coeff(t)/(s0_s+τ*self.aux_c20_coeff(t))


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
	def PEE10_t(self, τp = None, s_ = None, B0p = None, Θs = None, dlns_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		return self.ĉ10_t(τp = τp, s_ = s_, B0p = B0p, Θs = Θs, t= t)**(1-1/self.get('ρ',t))*self.dlnĉ10_dτ(τp = τp, B0p = B0p, Θs = Θs, dlns_Dτ = dlns_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, t = t)
	def ĉ10_t(self, τp = None, s_ = None, B0p = None, Θs = None, t = None):
		return (s_/self.get('ν',t))**self.power_s(t)*(1+B0p)**(1/(self.get('ρ',t)-1))*(self.auxInf0(t) + Θs*τp*self.auxInf1(t))
	def dlnĉ10_dτ(self, τp = None, B0p = None, Θs = None, dlns_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		return (B0p/(1+B0p))*(1-self.get('α',t))*(dlnhp_Dlns-1)*dlns_Dτ+Θs * self.auxInf1(t) *(dτp_dτ+τp*dlns_Dτ)/(self.auxInf0(t)+Θs*τp*self.auxInf1(t)) 

	def LOG_dlnĉ10_dτ(self, τp = None, Θs = None, dlns_Dτ = None, dτp_dτ = None, t = None):
		return self.auxInf1(t)*Θs *(dτp_dτ+τp*dlns_Dτ)/(self.auxInf0(t)+τp * self.auxInf1(t) * Θs)
	def LOG_PEE10_t(self, τp = None, B0p = None, Θs = None, dlns_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		return (1+B0p)*self.LOG_dlnĉ10_dτ(τp = τp, Θs = Θs, dlns_Dτ = dlns_Dτ, dτp_dτ = dτp_dτ, t = t)+B0p*(1-self.get('α',t))*(dlnhp_Dlns-1)*dlns_Dτ


	def LOG_backOutΘs(self, B0p = None, s0_s = None, τp = None, t = None):
		return B0p * self.auxInf0(t) / ((1+B0p)*s0_s+self.auxInf1(t)*τp)
	def LOG_backOutτ(self, Θs = None, Γs = None, τp = None, t = None):
		return 1-(Θs/Γs)**((1+self.get('α',t)*self.get('ξ',t))/(1+self.get('ξ',t)))*(self.get('Γh',t)-self.get('αr',t)*(self.get('p',t)*self.get('θ[t+1]',t)*τp/self.get('κ',t))*Γs)/(self.get('Γh',t)**(1-self.get('α',t))*(1-self.get('α',t)))

	def backOutH(self, s = None, Γs = None, t = None):
		return (s/Γs)**(self.get('ξ',t)/(1+self.get('ξ',t)))*self.get('Γh',t)
	def backOutS_(self, s = None, s0 = None, B0 = None, τp = None, t = None):
		return self.get('ν',t)*((s0*(1+B0)+self.auxInf1(t)*τp*s)/(B0*self.auxInf0(t)))**(1/self.power_s(t))
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
		return s / ((1-self.get('α',t))*(s_/self.get('ν',t))**self.get('α',t) * h**(1-self.get('α',t)))

class BaseGrid(BaseScalar):
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
		return ((h/self.get('Γh',t))**((1+self.get('ξ',t))/self.get('ξ',t)))[:,None]*(1+Bip)**(1/(self.get('ρ',t)-1)) * ((Γs*self.get('αr',t)*self.get('p',t)*τp*(1-self.get('θ[t+1]',t))/self.get('κ',t))[:,None]+self.auxProd(t)/(1+self.get('ξ',t)))

	def dlnĉ1i_dτ(self, τp = None, Γs = None, Bip = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		x = (self.get('αr',t)*self.get('p',t)*(1-self.get('θ[t+1]',t))*Γs/self.get('κ',t))
		return dlnh_Dτ[:,None]*(1+self.get('ξ',t))/self.get('ξ',t)+(Bip/(1+Bip))*(1-self.get('α',t))*((dlnhp_Dlns-1)*dlns_Dτ)[:,None]+(x*(dτp_dτ+τp*dlnΓs_Dτ))[:,None]/(self.auxProd(t)/(1+self.get('ξ',t))+(τp*x)[:,None])

	def LOG_dlnĉ1i_dτ(self, τp = None, Γs = None, dlnh_Dτ = None, dlnΓs_Dτ = None, dτp_dτ=None, t = None):
		k = self.get('αr',t)*self.get('p',t)*(1-self.get('θ[t+1]',t))*Γs/self.get('κ',t)
		return dlnh_Dτ[:,None]*(1+self.get('ξ',t))/self.get('ξ',t)+(k*(dτp_dτ+τp*dlnΓs_Dτ))[:,None]/(self.auxProd(t)/(1+self.get('ξ',t))+(τp*k)[:,None])
	def LOG_PEE1i_t(self, τp = None, Γs = None, Bip = None, dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		return self.LOG_dlnĉ1i_dτ(τp = τp, Γs = Γs, dlnh_Dτ = dlnh_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dτp_dτ = dτp_dτ, t = t)*(1+Bip)+Bip*(1-self.get('α',t))*((dlnhp_Dlns-1)*dlns_Dτ)[:,None]


	def PEE_t_s0grid(self, τBound = None, τ  = None, τp = None, s_ = None, h = None, Γs = None, Bip = None, B0p = None, si_s = None, s0_s = None, Θs = None,
						 dlnh_Dτ = None, dlns_Dτ = None, dlnΓs_Dτ = None, dlnhp_Dlns = None, dτp_dτ = None, t = None):
		""" Like PEE_t, but with s0_s being a separate grid """
		v1i = self.PEE1i_t(τp = τp, h = h, Γs = Γs, Bip = Bip, dlnh_Dτ = dlnh_Dτ, dlns_Dτ = dlns_Dτ, dlnΓs_Dτ = dlnΓs_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, t = t)
		v10 = self.PEE10_t(τp = τp, s_ = s_, B0p = B0p, Θs = Θs, dlns_Dτ = dlns_Dτ, dlnhp_Dlns = dlnhp_Dlns, dτp_dτ = dτp_dτ, t = t)
		v2i = self.PEE2i(τ = τBound, s_ = s_, h = h, dlnh_Dτ = dlnh_Dτ, si_s = si_s, t = t)
		v20 = self.PEE20_s0grid(τ = τBound, s_ = s_, h = h, dlnh_Dτ = dlnh_Dτ, s0_s = s0_s, t = t)
		return self.aux_PEE_s0grid(v1i = v1i, v10 = v10, v2i = v2i, v20 = v20, τ = τ)
	def PEE20_s0grid(self, τ = None, s_ = None, h = None, dlnh_Dτ = None, s0_s = None, t = None):
		return self.c20_s0grid(τ = τ, s_ = s_, h = h, s0_s = s0_s, t = t)**(1-1/self.get('ρ[t-1]',t))*((1-self.get('α',t))*dlnh_Dτ[:,None]+self.aux_c20_coeff(t)/(s0_s+τ[:,None]*self.aux_c20_coeff(t)))
	def c20_s0grid(self, τ = None, s_ = None, h = None, s0_s = None, t = None):
		return (self.get('α0',t)*self.get('α',t)*(self.get('ν',t)/self.get('p0[t-1]',t))*(s_/self.get('ν',t))**self.get('α',t)*h**(1-self.get('α',t)))[:,None]*(s0_s+τ[:,None]*self.aux_c20_coeff(t))
	def aux_PEE_s0grid(self, v1i = None, v10 = None, v2i = None, v20 = None, τ = None, t = None):
		return self.interiorFOC(np.matmul(v2i, self.get('γi[t-1]',t)*self.ω2i(t))[:,None]+v20*self.ω20(t)+self.get('ν',t)*(np.matmul(v1i, self.get('γi',t)*self.ω1i(t))+v10*self.ω10(t))[:,None], τ[:,None])

class BaseTime(_Base):
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

	def felicityFuncPd(self, x, t = None):
		""" Assumes x = pandas object with time in rows """
		return np.log(x) if self.db['ρ'].iloc[0] == 1 else x.pow(1-1/self('ρ',t),axis=0).div(1-1/self('ρ',t),axis=0)

	#######################################################################
	##########					1. Simple defs				 	###########
	#######################################################################
	def Bi(self, s_ = None, h = None, t = None):
		return self('βi',t).pow(self('ρ',t), axis = 0).values * ((self.R(s_ = s_, h = h, t = t)/self.get('p',t))**(self.get('ρ',t)-1))[:,None]
	def B0(self, s_ = None, h = None, t = None):
		return self.get('β0',t)**(self.get('ρ',t))*(self.R(s_=s_, h=h, t =t)/self.get('p0',t))**(self.get('ρ',t)-1)

	def si_s(self, Bi = None, Γs = None, τp = None, t = None):
		return Bi*self.auxProd(t) /((1+Bi)*(1+self.get('ξ',t)[:,None])*Γs[:,None])-(self.get('αr',t)*τp*self.get('p',t)/self.get('κ',t))[:,None]*((1-self.get('θ[t+1]',t)[:,None])/(1+Bi)+self.get('θ[t+1]',t)[:,None]*self.auxProd(t)/self.get('Γh',t)[:,None])
	def s0_s(self, B0 = None, Θs = None, τp = None, t = None):
		return (1/(1+B0))*(B0*self.auxInf0(t)/Θs-τp*self.auxInf1(t))
	#######################################################################
	##########				2. Terminal state (FH)			 	###########
	#######################################################################
	def Θh_T(self, τ = None, t = None):
		return self.get('Γh',t)**(1/(1+self.get('ξ',t)*self.get('α',t))) * ((1-self.get('α',t))*(1-τ))**(self.get('ξ',t)/(1+self.get('α',t)*self.get('ξ',t)))
	def h_T(self, s_ = None, τ = None, t = None):
		return self.Θh_T(τ = τ, t = t)*(s_/self.get('ν',t))**self.power_h(t)
	def Θc̃1i_T(self, Θh = None, t = None):
		return ((Θh/self.get('Γh',t))**((1+self.get('ξ',t))/self.get('ξ',t))/(1+self.get('ξ',t)))[:,None] * self.auxProd(t)
	def Θc̃10_T(self, t = None):
		return self.auxInf0(t)
	def util1i_T(self, c̃1i = None, Δy1i = 0, t = None):
		return self.felicityFuncPd(c̃1i+Δy1i, t = t)
	def util10_T(self, c̃10 = None, Δy10 = 0, t = None):
		return self.felicityFuncPd(c̃10+Δy10, t = t)

	#######################################################################
	##########				3. Non-terminal state f			 	###########
	#######################################################################
	def Θc̃1i_t(self, Θh = None, Γs = None, Bip = None, τp = None, t = None):
		return ((Θh/self.get('Γh',t))**((1+self.get('ξ',t))/self.get('ξ',t)))[:,None]*(self.auxProd(t)/(1+self.get('ξ',t)[:,None])+(Γs*self.get('αr',t)* self.get('p',t)*τp*(1-self.get('θ[t+1]',t))/self.get('κ',t))[:,None])/(1+Bip)
	def Θc2i(self, τ = None, Θh = None, si_s = None, t = None):
		return (self.get('α',t)*self.get('ν',t)*Θh**(1-self.get('α',t))/self.get('p[t-1]',t))[:,None] * (si_s + (self.get('αr',t)*self.get('p[t-1]',t)*τ/self.get('κ[t-1]',t))[:,None]*(1+self.get('θ',t)[:,None]*(self.auxProd_(t)/self.get('Γh[t-1]',t)[:,None]-1)))
	def Θc2pi_t(self, τp = None, Θhp = None, Θs = None, si_s = None, t = None):
		return ((Θs/self.get('ν[t+1]',t))**self.power_s(t)*self.get('α[t+1]',t)*self.get('ν[t+1]',t)*Θhp**(1-self.get('α[t+1]',t))/self.get('p',t))[:,None] * (si_s+(self.get('αr[t+1]',t)*self.get('p',t)*τp/self.get('κ',t))[:,None]*(1+self.get('θ[t+1]',t)[:,None]*(self.auxProd(t)/self.get('Γh',t)[:,None]-1)))
	def Θc̃10_t(self, B0p = None, τp = None, t = None):
		return (self.auxInf0(t)+self.auxInf1(t)*τp)/(1+B0p)
	def Θc20(self, τ = None, Θh = None, s0_s = None, t = None):
		return (self.get('α',t)*self.get('α0',t)*self.get('ν',t)/self.get('p0[t-1]',t))*Θh**(1-self.get('α',t))*(s0_s+self.auxInf1_(t)*τ)
	def Θc2p0(self, τp = None, Θhp = None, Θs = None, s0_s = None, t = None):
		return ((Θs/self.get('ν[t+1]',t))**self.power_s(t)*self.get('α[t+1]',t)*self.get('α0[t+1]',t)*self.get('ν[t+1]',t)/self.get('p0',t))*Θhp**(1-self.get('α[t+1]',t))*(s0_s+self.auxInf1(t)*τp)

	def util1i_t(self, c̃1i = None, c2pi = None, Δy1i = 0, Δo1i = 0, t = None):
		return self.felicityFuncPd(c̃1i+Δy1i, t = t)+self.get('βi',t)*self.felicityFuncPd(c2pi+Δo1i, t = t)
	def util10_t(self, c̃10 = None, c2p0 = None, Δy10 = 0, Δo10 = 0, t = None):
		return self.felicityFuncPd(c̃10+Δy10, t = t)+self.get('β0',t)*self.felicityFuncPd(c2p0+Δo10, t = t)
	def util2i(self, c2i = None, Δo2i = 0, t = None):
		return self.felicityFuncPd(c2i+Δo2i, t = t)
	def util20(self, c20 = None, Δo20 = 0, t = None):
		return self.felicityFuncPd(c20+Δo20, t = t)

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
	def FH_s0_s(self, s_ = None, s = None, hp = None, τp = None):
		return self.s0_s(B0 = self.B0(s_ = s, h = hp, t = self.db['txE']), Θs = self.FH_BackOutΘs(s_ = s_, s = s), τp = τp[:-1], t = self.db['txE'])
	def FH_BackOutΘh(self, s_ = None, h = None):
		return h/((s_/self('ν'))**self.power_h())

	# Coefficient methods - relies on dictionary of solution structure, returns pandas objects
	def FH_Θc̃1i(self, sd):
		return pd.DataFrame(np.vstack([self.Θc̃1i_t(Θh = sd['Θh'].values[:-1], Γs = sd['Γs'].values, Bip = sd['Bi'].values[1:,], τp = sd['τ[t+1]'].values[:-1], t = self.db['txE']), 
									   self.Θc̃1i_T(Θh = sd['Θh'].values[-1:], t = self.db['t'][-1:])]), index = self.db['t'], columns = self.db['i'])
	def FH_Θc2i(self, sd):
		return pd.DataFrame(self.Θc2i(τ = sd['τ'].values, Θh = sd['Θh'].values, si_s = sd['si/s[t-1]'].values), index = self.db['t'], columns = self.db['i'])
	def FH_Θhi(self, sd):
		return pd.DataFrame((self.get('ηi')/self.get('Xi'))**(self.get('ξ')[:,None]) / self.get('Γh')[:,None], index = self.db['t'], columns = self.db['i'])
	def FH_Θc2pi(self, sd):
		return pd.DataFrame(self.Θc2pi_t(τp = sd['τ'].values[1:], Θhp = sd['Θh'].values[1:], Θs = sd['Θs'].values, si_s = sd['si/s[t-1]'].values[1:], t = self.db['txE']), index = self.db['txE'], columns = self.db['i'])
	def FH_Θc̃10(self, sd):
		return pd.Series(np.hstack([self.Θc̃10_t(B0p = sd['B0'].values[1:], τp = sd['τ'].values[1:], t = self.db['txE']),
									self.Θc̃10_T(t = self.db['t'][-1:])]), index = self.db['t'])
	def FH_Θc20(self, sd):
		return pd.Series(self.Θc20(τ = sd['τ'].values, Θh = sd['Θh'].values, s0_s = sd['s0/s[t-1]'].values), index = self.db['t'])
	def FH_Θc2p0(self, sd):
		return pd.Series(self.Θc2p0(τp = sd['τ'].values[1:], Θhp = sd['Θh'].values[1:], Θs = sd['Θs'].values, s0_s = sd['s0/s[t-1]'].values[1:], t = self.db['txE']), index = self.db['txE'])
	# Reporting methods for "levels" - relies on dictionary solution structure, return pandas objects
	def FH_hi_h(self, sd):
		return pd.DataFrame(self.auxProd()/self.get('Γh')[:,None], index = self.db['t'], columns = self.db['i'])
	def FH_c̃1i(self, sd):
		return sd['Θc̃1i'].mul((sd['s[t-1]']/self.get('ν'))**self.power_s(),axis=0)
	def FH_c2i(self, sd):
		return sd['Θc2i'].mul((sd['s[t-1]']/self.get('ν'))**self.power_s(), axis = 0)
	def FH_c2pi(self, sd):
		return sd['Θc2pi'].mul((sd['s[t-1]'].iloc[:-1]/self.get('ν', t = self.db['txE']))**self.power_p(t = self.db['txE']), axis = 0)
	def FH_c̃10(self, sd):
		return sd['Θc̃10']*((sd['s[t-1]']/self.get('ν'))**self.power_s())
	def FH_c20(self, sd):
		return sd['Θc20']*((sd['s[t-1]']/self.get('ν'))**self.power_s())
	def FH_c2p0(self, sd):
		return sd['Θc2p0']*((sd['s[t-1]'].iloc[:-1]/self.get('ν',t=self.db['txE']))**self.power_p(t = self.db['txE']))

	# Reporting methods for utility
	def FH_util1i_(self, sd, Δy1i = None, Δo1i = None):
		Δy1i, Δo1i = noneInit(Δy1i, np.zeros((len(self.db['t']), self.m.ni))), noneInit(Δo1i, np.zeros((len(self.db['t']), self.m.ni)))
		return pd.concat([self.util1i_t(c̃1i = sd['c̃1i'].iloc[:-1], c2pi = sd['c2pi'], Δy1i = Δy1i[:-1], Δo1i = Δo1i[:-1], t = self.db['txE']),
						  self.util1i_T(c̃1i = sd['c̃1i'].iloc[-1:], Δy1i = Δy1i[-1:], t = self.db['t'][-1:])], axis = 0)
	def FH_util1i(self, sd, Δ1i = None, **kwargs):
		Δ1i = noneInit(Δ1i, np.zeros((len(self.db['t']), self.m.ni)))
		Bip = self.m.leadSym(sd['Bi'].values)
		return self.FH_util1i_(sd, Δy1i = Δ1i /(1+Bip), Δo1i  = Δ1i * self.m.leadSym(sd['R'].values/self.get('p[t-1]'))[:,None]*Bip/(1+Bip))

	def FH_util2i(self, sd, Δ2i = None, **kwargs):
		Δ2i = noneInit(Δ2i, np.zeros((len(self.db['t']), self.m.ni)))
		return self.util2i(c2i = sd['c2i'], Δo2i = Δ2i)
	def FH_util10_(self, sd, Δy10 = None, Δo10 = None):
		Δy10, Δo10 = noneInit(Δy10, np.zeros(len(self.db['t']))), noneInit(Δo10, np.zeros(len(self.db['t'])))
		return pd.concat([self.util10_t(c̃10 = sd['c̃10'].iloc[:-1], c2p0 = sd['c2p0'], Δy10 = Δy10[:-1], Δo10 = Δo10[:-1], t = self.db['txE']),
						  self.util10_T(c̃10 = sd['c̃10'].iloc[-1:], Δy10 = Δy10[-1:], t = self.db['t'][-1:])], axis = 0)
	def FH_util10(self, sd, Δ10 = None, **kwargs):
		Δ10 = noneInit(Δ10, np.zeros(len(self.db['t'])))
		B0p = self.m.leadSym(sd['B0'].values)
		return self.FH_util10_(sd, Δy10 = Δ10 /(1+B0p), Δo10  = Δ10 * self.m.leadSym(sd['R'].values/self.get('p0[t-1]'))*B0p/(1+B0p))

	def FH_util20(self, sd, Δ20 = 0, **kwargs):
		Δ20 = noneInit(Δ20, np.zeros(len(self.db['t'])))
		return self.util20(c20 = sd['c20'], Δo20 = Δ20)
	def FH_utilPol_(self, sd, Δy1i = None, Δo1i = None, Δy10 = None, Δo10 = None, Δo2i = None, Δo20 = None):
		return (self('ν')*((self.FH_util1i(sd, Δy1i = Δy1i, Δo1i = Δo1i) * self.ω1i() * self('γi')).sum(axis=1)+self.FH_util10(sd, Δy10 = Δy10, Δo10 = Δo10))
						  +(self.FH_util2i(sd, Δo2i = Δo2i) * self.ω2i()).sum(axis=1) + self.FH_util20(sd, Δo20 = Δo20))
	def FH_utilPol(self, sd, ΔPol = None, **kwargs):
		ΔPol = noneInit(ΔPol, np.zeros(len(self.db['t'])))
		return (self('ν')*((self.FH_util1i(sd, Δ1i = ΔPol[:,None]) * self.ω1i() * self('γi')).sum(axis=1)+self.FH_util10(sd, Δ10 = ΔPol))
						  +(self.FH_util2i(sd, Δ2i = ΔPol[:,None]) * self.ω2i()).sum(axis=1) + self.FH_util20(sd, Δ20 = ΔPol))


	#### LOG METHODS:
	def FH_LOG_Γs(self, τp = None):
		return self.Γs(Bi = self.get('βi', t = self.db['txE']), τp = τp[:-1], t = self.db['txE'])
	@property
	def FH_LOG_Θh(self):
		return self.FH_Θh
	def FH_LOG_Θs(self, Θh = None, Γs = None):
		return self.Θs_t(Θh[:-1], Γs = Γs, t = self.db['txE'])
	def FH_LOG_s(self, Θs = None, s0 = None):
		s = np.empty(self.m.T-1)
		s[0] = Θs[0]*(s0/self.get('ν',self.t0))**(self.power_s(self.t0))
		for t in range(1, self.m.T-1):
			s[t] = Θs[t]*(s[t-1]/self.get('ν',self.db['t'][t]))**(self.power_s(self.db['t'][t]))
		return s
	def FH_LOG_h(self, Θh = None, s_ = None):
		return Θh * (s_/self.get('ν'))**self.power_h()
	def FH_LOG_s0_s(self, Θs = None, τp = None):
		return self.s0_s(B0 = self.get('β0', t = self.db['txE']), Θs = Θs, τp = τp[:-1], t = self.db['txE'])


