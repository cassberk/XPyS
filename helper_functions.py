import numpy as np
import pandas as pd
import pickle
import gc
import h5py
def index_of(arr, val):
    """Return index of array nearest to a value."""
    if val < min(arr):
        return 0
    return np.abs(arr-val).argmin()

def guess_from_data(x,y, peakpos, lims=2, ampscale=1.0, sigscale=1.0):
    """Estimate amp, cen, sigma for a peak, create params."""
    if x is None:
        return 1.0, 0.0, 1.0
    
    Ei = [np.argmin(abs(peakpos-lims-x)), np.argmin(abs(peakpos+lims-x))]
    
    if np.argmin(abs(peakpos-lims-x))==np.argmin(abs(peakpos+lims-x)):
        return 0, 0
#     locmax = argrelextrema(y[min(Ei):max(Ei)],np.greater)
    locmax = max(y[min(Ei):max(Ei)])
    locmin = min(y[min(Ei):max(Ei)])

    cen = x[index_of(y,locmax)]
    amp = (locmax - locmin)*ampscale
    
    halfmax_vals = np.where(y > (locmax+locmin)/2.0)[0]

    return amp, cen

def closehdf5():
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was already closed

