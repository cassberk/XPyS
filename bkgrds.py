import logging
import numpy as np

logger = logging.getLogger(__name__)

# class background_sub(object):
# functions = ('linear','shirley','Tougaard_fit','Tougaard')

def linear(energy, intensity):
	background = np.linspace(intensity[0], intensity[-1], len(energy))
	return background



def shirley(energy, intensity, orbital, tol=1e-5, maxit=20):
	"""Calculates shirley background."""
	if energy[0] < energy[-1]:
		is_reversed = True
		energy = energy[::-1]
		intensity = intensity[::-1]
	else:
		is_reversed = False

	background = np.ones(energy.shape) * intensity[-1]
	integral = np.zeros(energy.shape)
	spacing = (energy[-1] - energy[0]) / (len(energy) - 1)
#
#    subtracted = intensity - background
#    ysum = subtracted.sum() - np.cumsum(subtracted)
#    for i in range(len(energy)):
#        integral[i] = spacing * (ysum[i] - 0.5
#                                 * (subtracted[i] + subtracted[-1]))

	iteration = 0
	while iteration < maxit:
		subtracted = intensity - background
		integral = spacing * (subtracted.sum() - np.cumsum(subtracted))
		bnew = ((intensity[0] - intensity[-1])
				* integral / integral[0] + intensity[-1])
		if np.linalg.norm((bnew - background) / intensity[0]) < tol:
			background = bnew.copy()
			break
		else:
			background = bnew.copy()
		iteration += 1
	if iteration >= maxit:
		logger.warning(orbital+" shirley: Max iterations exceeded before convergence.")

	if is_reversed:
		return background[::-1]
	return background


def K2(p, E):
	B,C = p
	K_ = B * E / ((C + E**2)**2)
	K_ = K_*(K_>0)
	return K_

def K3(p, E):

	B, C, D = p
	K_ = B * E / ((C + E**2)**2 + D*E**2)
	K_ = K_*(K_>0)
	return K_


def Tougaard(pars, spectrum,E):
	
	parsval = pars.valuesdict()
	B = parsval['B']
	C = parsval['C']
	D = parsval['D']

	
	dE = np.abs(E[1] - E[0])
#     spectrum = spectrum - min(spectrum)
	spectrum = spectrum - spectrum[-1]

	#set highest energy to zero, respect physical model behind Tougaard 
	
#     tkern = K2((B,C),np.arange(len(spectrum)))
	tkern = K2((B,C),np.arange(0,len(E),dE))

	bg = dE*np.convolve(spectrum,tkern[::-1], 'full')

	bg2=bg[bg.size - spectrum.size:]

	sub = spectrum - bg2

	return bg2,sub

def Tougaard_fit(pars,spectrum,E, fit_inds = (0,-1)):

	bkgrd, sub_spectra = Tougaard(pars,spectrum,E)
	spec_adj = spectrum - spectrum[-1]
	
	return bkgrd[fit_inds[0]:fit_inds[1]] - spec_adj[fit_inds[0]:fit_inds[1]]