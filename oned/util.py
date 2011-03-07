import onedspec
import numpy as np
import scipy
import scipy.optimize

def fitprof(spectrum, line, peak_pm=1.5):
    """
    
    Fits a profile to a given spectrum at the line point. The peak value must be
    found in order to fit the peak correctly, so a +/- region is allowed to find
    the location of the peak.
    
    
    Inputs
    ------
    
    spectrum    :   Spectrum to fit a line profile to. Must be a :class:`onedspec`
                    class object.
                    
    line        :   Floating point location of the line (Angstroms)
    
    peak_pm     :   The peak of the profile will be found in the region from
                    (line - peak_pm) to (line + peak_pm). This can be set to
                    zero if the peak of the profile is already known.
                    
                    
    Outputs
    -------
    
    peak        :   Peak value of the profile found
    
    position    :   Position of the profile centroid (Angstroms)
    
    sigma       :   Gaussian sigma found for the best-fitting profile.
    
    """

    if not isinstance(spectrum, onedspec):
        return ValueError('Spectrum must be a onedspec class object')
        
    try:
        line = float(line)
        
    except TypeError:
        raise TypeError('Line profile position must be a floating point.')
        
        
    
    peak_spectrum = spectrum[line - peak_pm:peak_pm + line]
    
    peak = max(peak_spectrum.flux)
    position = peak_spectrum.wavelength[peak_spectrum.flux.searchsorted(peak)]
    
    index = spectrum.wavelength.searchsorted(position)
    
    
    p, m = 0, 0
    num = len(spectrum.flux)
    
    while (num > index + p + 1) and (spectrum.flux[index + p] > spectrum.flux[index + p + 1]):
        p += 1
        
    while (index - m - 1 > 0) and (spectrum.flux[index - m] > spectrum.flux[index - m - 1]):
        m += 1

    fitfunc = lambda p, x: p[0] * scipy.exp(-(x - p[1])**2 / (2.0 * p[2]**2))
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    
    p0 = scipy.c_[peak, position, 5]
    p1, success = scipy.optimize.leastsq(errfunc, p0.copy()[0], args=(spectrum[index - m:p + index].wavelength, spectrum[index - m:p + index].flux))
    
    return p1    
    


