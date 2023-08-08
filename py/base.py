import numpy as np
from scipy import optimize

def noneInit(x, fallBackValue):
	return fallBackValue if x is None else x

##########################################################################
##################	 			1. Base 				##################
##########################################################################

# Technology:
def aggregateLabor(γVector, ηVector, hiVector):
	return sum(ηVector * γVector * hiVector)

def aggregateSavings(γVector, siVector):
	return sum(γVector * siVector)

def interestRate(α, A, ν, s, h):
	return α*A*(ν*h/s)**(1-α)

def wageRate(α, A, ν, s, h):
	return (1-α)*A*(s/(ν*h))**(1-α)

# Auxiliary functions:
def auxΓh(γVector, ηVector, XVector, ξ):
	return sum(γVector * np.power(ηVector, 1+ξ) / np.power(XVector, ξ))

def auxΓβ1(βVector, γVector, ηVector, XVector, ξ):
	return sum( (βVector/(1+βVector)) * γVector * np.power(ηVector, 1+ξ)/np.power(XVector, ξ))

def auxΓβ2(βVector, γVector, ηVector, XVector, ξ):
	return sum( (1/(1+βVector))  * γVector * np.power(ηVector, 1+ξ)/np.power(XVector, ξ))

def auxΓβ3(βVector, γVector):
	return sum( γVector / (1+βVector))

##########################################################################
##################	 2. Economic Equilibrium Functions 	##################
##########################################################################

def auxIncome(Υ, α, A, ξ, Γh, τ):
	return (1-α)*(1-τ)*A*(Υ**(ξ*(1-α)/(1+ξ)))/(Γh**(α))

def auxSavings(Υ, α, A, ξ, Γh, τ):
	return auxIncome(Υ, α, A, ξ, Γh, τ)-(ξ/(1+ξ))*Υ

def auxΥ(α, A, ε, θ, γu, Γh, Θh, Θs, τ, τp):
	return ((1-α)*(1-τ)*A*(Θh**(1-α)) + ((1-α)/α)*(θ * τp)/(1-γu+γu*(1-ε)*(1-θ))*Θs)/Γh

def auxΘh(ξ, Γh, Υ):
	return Γh * (Υ**(ξ/(1+ξ)))

def auxΘs(α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Υ, Γh, τ, τp):
	return auxSavings(Υ, α, A, ξ, Γh, τ)*auxΓβ1(βVector, γVector, ηVector, XVector, ξ)/(1+((1-α)/α)*(τp/(1-γu+γu*(1-ε)*(1-θ)))*((θ/Γh)*auxΓβ2(βVector, γVector, ηVector, XVector, ξ)+(1-θ)*auxΓβ3(βVector, γVector)))

def savingsSpread(α, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, τp):
	""" Return vector of Θsi/Θs """
	x1 = (βVector * np.power(ηVector, 1+ξ)/(np.power(XVector, ξ)*(1+βVector))) / auxΓβ1(βVector, γVector, ηVector, XVector, ξ)
	x2 = (θ/Γh)*auxΓβ2(βVector, γVector, ηVector, XVector, ξ)+(1-θ)*auxΓβ3(βVector, γVector)
	x3 = ((1-α)/α)*τp/(1-γu+γu*(1-ε)*(1-θ))
	x4 = (θ*np.power(ηVector, 1+ξ)/(Γh*np.power(XVector, ξ))+1-θ)/(1+βVector)
	return x1*(1+x2*x3)-x3*x4

def economicEquilibriumEqs(x, α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, τ, τp):
	""" Core equations for economic equilibrium in given year given parameters. x is a vector of Υ, Θh, Θs"""
	return np.array([auxΥ(α, A, ε, θ, γu, Γh, x[1], x[2], τ, τp)-x[0], 
					 auxΘh(ξ, Γh, x[0])-x[1],
					 auxΘs(α, A, ϵ, θ, γu, βVector, γVector, ηVector, XVector, ξ, x[0], Γh, τ, τp)-x[2]])

def solveCoreEE(α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, τ, τp, x0 = None):
	sol, _, ier, msg = optimize.fsolve(lambda x: economicEquilibriumEqs(x, α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, τ, τp), 
	noneInit(x0, [0.5, 0.5, 0.5]), full_output = True)
	if ier == 1:
		return sol
	else:
		print(f"solveCoreEE couldn't identify an equilibrium - fsolve returns {msg}")

def auxLnDevΥ(dlnΘh, dlnΘs, Υ, Θh, Θs, α, A, ε, θ, γu, Γh,τ,τp):
	""" Define ∂ln(Υ)/∂τ"""
	return (1/(Υ*Γh))*((1-α)*A*Θh**(1-α)*((1-τ)*(1-α)*dlnΘh-1)+((1-α)/α)*θ*τp * Θs * dlnΘs /(1-γu+γu*(1-ε)*(1-θ)))

def auxLnDevΘh(dlnΥ, ξ):
	""" Define ∂ln(Θh)/∂τ"""
	return (ξ/(1+ξ))*dlnΥ

def auxLnDevΘs(dlnΥ, Υ, α, A, ξ, Γh, τ):
	""" Define ∂ln(Θh)/∂τ"""
	return (dlnΥ*((ξ*(1-α)/(1+ξ))*(1-α)*(1-τ)*A*(Υ**(ξ*(1-α)/(1+ξ)))/(Γh**α)-ξ*Υ/(1+ξ))-(1-α)*A*(Υ**(ξ*(1-α)/(1+ξ)))/(Γh**α)) / auxSavings(Υ, α, A, ξ, Γh, τ)

def economicEquilibriumLogDevEqs(x, Υ, Θh, Θs, α, A, ε, θ, γu, ξ, Γh, τ, τp):
	""" System of equations that identify log-derivatives for three core auxiliary variables. x is a vector of ∂ln(Υ)/∂τ, ∂ln(Θh)/∂τ, ∂ln(Θs)/∂τ"""
	return np.array([auxLnDevΥ(x[1], x[2], Υ, Θh, Θs, α, A, ε, θ, γu, Γh, τ, τp)-x[0],
					 auxLnDevΘh(x[0], ξ)-x[1],
					 auxLnDevΘs(x[0], Υ, α, A, ξ, Γh, τ)-x[2]])

def solveCoreLogDevEE(Υ, Θh, Θs, α, A, ε, θ, γu, ξ, Γh, τ, τp, x0 = None):
	sol, _, ier, msg = optimize.fsolve(lambda x: economicEquilibriumLogDevEqs(x, Υ, Θh, Θs, α, A, ε, θ, γu, ξ, Γh, τ, τp),
	noneInit(x0, [1,1,1]), full_output = True)
	if ier == 1:
		return sol
	else:
		print(f"solveCoreLogDevEE did not solve - fsolve returns {msg}")

### 2.1: Full system of coefficients
def auxΘhi(Υ, ηVector, XVector, ξ):
	return (Υ**(ξ/(1+ξ)))*np.power(ηVector, 1+ξ)/np.power(XVector, ξ)

def auxΘsi(Υ, Θs, α, A, ε, θ, γu, βVector, ηVector, XVector, ξ, Γh, τ, τp):
	return (1/(1+βVector))*((βVector*np.power(ηVector, 1+ξ)/np.power(XVector, ξ)) * auxSavings(Υ, α, A, ξ, Γh, τ)-((1-α)/α)*auxPensionRate(ε, θ, γu, ηVector, XVector, ξ, Γh)*τp*Θs)

def auxΘc1i(Υ, Θs, α, A, ε, θ, γu, βVector, ηVector, XVector, ξ, Γh, τ, τp):
	return (1/(1+βVector))*((np.power(ηVector, 1+ξ)/np.power(XVector, ξ))*(auxIncome(Υ, α, A, ξ, Γh, τ)+βVector*ξ*Υ/(1+ξ)) +((1-α)/α)*auxPensionRate(ε, θ, γu, ηVector, XVector, ξ, Γh)*τp*Θs)

def auxΘc2i(Θh, α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, ν, τ):
	return α*A*ν*(Θh**(1-α))*(savingsSpread(α, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, τ)+((1-α)/α)*auxPensionRate(ε, θ, γu, ηVector, XVector, ξ, Γh)*τ)

def auxΘc2pi(Θhp, Θs, α, Ap, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, νp, τp):
	return α * Ap * νp * (Θhp**(1-α)) * (Θs/νp)**(α*(1+ξ)/(1+α*ξ)) * (savingsSpread(α, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, τp)+((1-α)/α)*auxPensionRate(ε, θ, γu, ηVector, XVector, ξ, Γh)*τp)

def auxΘ̃c1i(Υ, Θs, α, A, ε, θ, γu, βVector, ηVector, XVector, ξ, Γh, τ, τp):
	return (1/(1+βVector))*((np.power(ηVector, 1+ξ)/np.power(XVector, ξ)) * auxSavings(Υ, α, A, ξ, Γh, τ)+((1-α)/α)*auxPensionRate(ε, θ, γu, ηVector, XVector, ξ, Γh)*τp*Θs)

def auxc1u(Θh, α, A, χ1):
	return χ1*A*(Θh**(1-α))

def auxc2u(Θh, α, A, ε, θ, γu, χ2, ν, τ):
	return α*A*ν*(Θh**(1-α))*(χ2/(α*ν)+τ*((1-α)/α)*(1-ε)*(1-θ)/(1-γu+γu*(1-ε)*(1-θ)))

def auxc2pu(Θhp, Θs, α, Ap, ε, θ, γu, ξ, χ2, νp, τp):
	return α * Ap * νp * (Θhp**(1-α))*(Θs/νp)**(α*(1+ξ)/(1+α*ξ))*(χ2/(α*νp)+τp*((1-α)/α)*(1-ε)*(1-θ)/(1-γu+γu*(1-ε)*(1-θ)))

##########################################################################
##################	 		3. PEE Functions 			##################
##########################################################################

# 3.1. Marginal effect of changing τ on indirect utility of various agents:
def auxPensionRate(ε, θ, γu, ηVector, XVector, ξ, Γh):
	""" Auxiliary vector that is frequently used """
	return (Γh*(1-θ)+θ*np.power(ηVector, 1+ξ)/np.power(XVector, ξ))/(Γh*(1-γu+γu*(1-ε)*(1-θ)))

def polSupportRetirees(dlnΘh, α, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, χ2, ω, Γh, ν, τ):
	return ω*(γu*polSupportUnemployedRetiree(dlnΘh, α, ε, θ, γu, χ2, ν, τ)+(1-γu)*sum(γVector*polSupportRetireeVector(dlnΘh, α, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, τ)))

def polSupportUnemployedRetiree(dlnΘh, α, ε, θ, γu, χ2, ν, τ):
	return (1-α)*(dlnΘh+(1-ε)*(1-θ)/(χ2 * (1-γu+γu*(1-ε)*(1-θ))/ν+(1-α)*(1-ε)*(1-θ)*τ) )

def polSupportRetireeVector(dlnΘh, α, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, τ):
	x = auxPensionRate(ε, θ, γu, ηVector, XVector, ξ, Γh)
	return (1-α)*(dlnΘh+x/(α*savingsSpread(α, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, τ)+(1-α)*x*τ))

def polSupportYoung(dlnΘh, dlnΘs, α, γu, βVector, γVector, βu, ξ, ν):
	return ν*(γu*polSupportUnemployedYoung(dlnΘh, dlnΘs, α, βu, ξ)+(1-γu)*sum(γVector*polSupportWorkerVector(dlnΘs, α, βVector, ξ)))

def polSupportUnemployedYoung(dlnΘh, dlnΘs, α, βu, ξ):
	return (1-α)*dlnΘh+βu*α*dlnΘs*(1+ξ)/(1+α*ξ)

def polSupportWorkerVector(dlnΘs, α, βVector, ξ):
	return dlnΘs*(1+βVector*α*(1+ξ)/(1+α*ξ))

def PEECondition(dlnΘh, dlnΘs, α, ε, θ, γu, βVector, γVector, ηVector, XVector, βu, ξ, χ2, ω, Γh, ν, τ):
	return (polSupportYoung(dlnΘh, dlnΘs, α, γu, βVector, γVector, βu, ξ, ν)
		   +polSupportRetirees(dlnΘh, α, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, χ2, ω, Γh, ν, τ))

def corePEEEqs(x, α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, βu, ξ, χ2, ω, Γh, ν, τp):
	""" x = τ, Υ, Θh, Θs, ∂ln(Υ)/∂τ, ∂ln(Θh)/∂τ, ∂ln(Θs)/∂τ """
	return np.hstack([PEECondition(x[5], x[6], α, ε, θ, γu, βVector, γVector, ηVector, XVector, βu, ξ, χ2, ω, Γh, ν, x[0]),
					  economicEquilibriumEqs(x[1:4], α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, x[0], τp),
					  economicEquilibriumLogDevEqs(x[4:], x[1], x[2], x[3], α, A, ε, θ, γu, ξ, Γh, x[0], τp)])

def solveCorePEE(α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, βu, ξ, χ2, ω, Γh, ν, τp, x0 = None):
	""" Return vector of variables x = τ, Υ, Θh, Θs, ∂ln(Υ)/∂τ, ∂ln(Θh)/∂τ, ∂ln(Θs)/∂τ """
	sol, _, ier, msg = optimize.fsolve(lambda x: corePEEEqs(x, α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, βu, ξ, χ2, ω, Γh, ν, τp),
		noneInit(x0, [.1, 0.5, 0.5, 0.5, 1, 1, 1]), full_output = True)
	if ier == 1:
		return sol
	else:
		print(f"solveCorePEE did not solve - fsolve returns {msg}")

def solveCorePEE_stst(α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, βu, ξ, χ2, ω, Γh, ν, x0=None):
	sol, _, ier, msg = optimize.fsolve(lambda x: corePEEEqs(x, α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, βu, ξ, χ2, ω, Γh, ν, x[0]),
		noneInit(x0, [.1, 0.5, 0.5, 0.5, 1, 1, 1]), full_output = True)
	if ier == 1:
		return sol
	else:
		print(f"solveCorePEE_stst did not solve - fsolve returns {msg}")


#### 3.2. Levels of utility given s_{t-1}:
def addLevelToUtil(x, ν, par, s_):
	return x if s_ is None else x+par*np.log(s_/ν)

def indirectUtilityRetireeVector(Θh, α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, ν, τ , s_ = None):
	return addLevelToUtil(np.log(auxΘc2i(Θh, α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, ν, τ)), 
							ν, α*(1+ξ)/(1+α*ξ), s_)
	
def indirectUtilityUnemployedRetiree(Θh, α, A, ε, θ, γu, ξ, χ2, ν, τ, s_ = None):
	return addLevelToUtil(np.log(auxc2u(Θh, α, A, ε, θ, γu, χ2, ν, τ)),
							ν, α*(1+ξ)/(1+α*ξ), s_)

def indirectUtilityWorkerVector(Υ, Θhp, Θs, α, A, Ap, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, ν, νp, τ, τp, s_= None):
	return (addLevelToUtil(np.log(auxΘ̃c1i(Υ, Θs, α, A, ε, θ, γu, βVector, ηVector, XVector, ξ, Γh, τ, τp)),
							ν, α*(1+ξ)/(1+α*ξ), s_)
	+βVector * addLevelToUtil(np.log(auxΘc2pi(Θhp, Θs, α, Ap, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Γh, νp, τp)), 
							ν, (α*(1+ξ)/(1+α*ξ))**2, s_)
			)

def indirectUtilityUemployedYoung(Θh, Θhp, Θs, α, A, Ap, ε, θ, γu, χ1, χ2, βu, ξ, ν, νp, τp, s_=None):
	return (addLevelToUtil(np.log(auxc1u(Θh, α, A, χ1)), 
							ν, α*(1+ξ)/(1+α*ξ), s_)
	+βu * addLevelToUtil(np.log(auxc2pu(Θhp, Θs, α, Ap, ε, θ, γu, ξ, χ2, νp, τp)),
							ν, (α*(1+ξ)/(1+α*ξ))**2, s_)
			)