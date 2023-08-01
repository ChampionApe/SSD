import numpy as np

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

# EquilibriumFunctions:
def auxΥ(α, A, ε, θ, γu, Γh, Θh, Θs, τ, τp):
	return ((1-α)*(1-τ)*A*(Θh**(1-α)) + ((1-α)/α)*(θ * τp)/(1-γu+γu*(1-ε)*(1-θ))*Θs)/Γh

def Θh(ξ, Γh, Υ):
	return Γh * (Υ**(ξ/(1+ξ)))

def ΘhiVector(ηVector, XVector, ξ, Υ):
	return np.power(ηVector/XVector, ξ) * (Υ**(ξ/(1+ξ)))

def Θs(α, A, ε, θ, γu, βVector, γVector, ηVector, XVector, ξ, Υ, Γh, τ, τp):
	return ((1-α)*A*(1-τ)*(Υ**(ξ*(1-α)/(1+ξ)))/(Γh**α)-(ξ/(1+ξ))*Υ)*auxΓβ1(βVector, γVector, ηVector, XVector, ξ)/(1+((1-α)/α)*(τp/(1-γu+γu*(1-ε)*(1-θ)))*((θ/Γh)*auxΓβ2(βVector, γVector, ηVector, XVector, ξ)+(1-θ)*auxΓβ3(βVector, γVector)))

