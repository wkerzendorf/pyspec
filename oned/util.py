from base import onedspec

import numpy as np
import scipy
import scipy.optimize
from scipy import ndimage


def fit_arclines(arc_spectrum, arc_lines='auto', sigma_clip=3.0, peak_tol=1.5,):
    """
    Fits profiles to multiple arc lines in a given spectrum. These can be defined
    as a list of arc line locations, or they can be automagically found.
    
    
    Inputs
    ------
    
    arc_spectrum    :   Observed arc frame. Must be a :class:`onedspec` class
                        object.
                        
    arc_lines       :   List of arc line peaks. If 'auto' is specified then the
                        arc line peaks will automatically be found and fitted to.
                        
    peak_tol        :   If a list of arc line peak locations were specified, then
                        this provides a tolerance (in Angstroms) of where to look
                        for the peak. If the peak location is not found within
                        the window of [location - peak_tol, location + peak_tol]
                        then the solution becomes unbounded and that arc line will
                        not be fitted to.
                        

    """
    
    if not isinstance(arc_spectrum, onedspec):
        raise TypeError('Arc spectrum supplied must be a onedspec object class')
        
    # Fit all the profile lines
    if not arc_lines == 'auto' and type(arc_lines) not in [list, tuple]:
        raise TypeError('Arc line locations must be given in the form of a list\
                        or a tuple.')
        
        
    if arc_lines == 'auto':
        # Auto-magically find all the peaks and put them into a list
        
        # todo - there is probably a MUCH better way of doing this
        grad = np.gradient(arc_spectrum.flux)
        sigma = (arc_spectrum.flux - np.mean(arc_spectrum.flux))/np.std(arc_spectrum.flux)
        
        arc_line_peaks = []
        
        for i, value in enumerate(grad):
            if grad[i-1] > 0 and grad[i+1] < 0 and sigma[i] > peak_tol: # todo - allow sigma to be set
                
                if len(arc_line_peaks) > 0:
                    if min(abs(np.array(arc_line_peaks) - arc_spectrum.wave[i])) > peak_tol:
                        arc_line_peaks.append(arc_spectrum.wave[i])  
                else: arc_line_peaks.append(arc_spectrum.wave[i])
                

        
        
    else: arc_line_peaks = arc_lines
    
    profiles = fitprofs(arc_spectrum, arc_line_peaks, peak_tol) 
    fitted_arc = onedspec(arc_spectrum.wave, np.zeros(len(arc_spectrum.wave)), type='waveflux')
    
    fitfunc = lambda p, x: p[0] * scipy.exp(-(x - p[1])**2 / (2.0 * p[2]**2))
    for profile in profiles:
        
        fitted_arcline = onedspec(fitted_arc.wave, fitfunc(profile, fitted_arc.wave), type='waveflux')
        #todo - doing fitted_arc += gave an unsupported operand error
        fitted_arc += fitted_arcline
        

    return (profiles, fitted_arc)
    
    
    
    
def continuum(synthetic_spectrum, arc_spectrum, **kwargs):
    
    """
    Returns a spectrum covering the wavelength region in union to the arc_spectrum
    and the synthetic_spectrum.
    
    """
    
    
    # Initialise with default keyword arguments if they are not specified
    
    # 0.96 in flux and 0.5 A was used as per Kirby et al 200?
    default_kwargs = { 'arc_lines' : 'auto', 'method' : 'Kirby_2009', 'continuum_min_flux' : 0.96, 'continuum_min_diff' : 0.5 }
    
    for key, value in default_kwargs.iteritems():
        if not kwargs.has_key(key): kwargs[key] = value
    
    
    if method == 'Kirby_2009':
        # Sanity checks - these could easily be put into a loop but since the
        # normalisation will run many hundreds of times, it's probably best not to
        # duplicate memory blocks unnecessarily.
        
        if not isinstance(synthetic_spectrum, onedspec):
            raise TypeError('Synthetic spectrum supplied must be a onedspec object class.')
        
        if not isinstance(arc_spectrum, onedspec):
            raise TypeError('Arc spectrum supplied must be a onedspec object class.')
        
        float_kwargs = ['continuum_min_flux', 'continuum_min_diff']
        for float_kwarg in float_kwargs:
                
            try:
                kwargs[float_kwarg] = float(kwargs[float_kwarg])
            
            except ValueError:
                raise ValueError('Keyword argument "%s" must be a floating point type.' % (float_kwarg,))
                
        
        
 
        # Fit a polynomial to the sigma line
        
        # Weighting
        #w = 1/sigma**2 * scipy.exp(-diff(synthetic_spectrum.wavelength)**2 / (2.*(10)**2))
    
        # Regions defined as continuum
        # Minimum flux
        m1 = scipy.where(synthetic.flux > kwargs['continuum_min_flux'], 1, 0)
        
        # Minimum wavelength space
        m2 = np.zeros(len(m1))
        m2[0:-1] = scipy.where(np.diff(synthetic.wave) < kwargs['continuum_min_diff'], 1, 0)
        m2[-1] = 0
        
        m = m1 * np.transpose(m2)
        
        return onedspec(synthetic.wave, m, type='waveflux')
        
        
        in_continuum, region_start, wavelength_points = (False, 0, len(m))
        continuum_regions = []
        for i, is_continuum in enumerate(m):
            
            if is_continuum and not in_continuum: # Continuum region starts
                region_start, in_continuum = synthetic.wave[i], True
                
            elif in_continuum and (not is_continuum or (wavelength_points == i + 1)): # Continuum region ends
                region_end, in_continuum = synthetic.wave[i-1], False
                continuum_regions.append((region_start, region_end))
                
  
  
        #continuum = sum(mi * w * flux) / sum(w * flux?)
    else:
        raise NotImplementedError('Normalisation method "%s" has not been implemented yet.' % (kwargs['method'], ))
        
        
    return (continuum, continuum_regions, sigma)
    

def fitprofs(spectrum, lines, peak_tol=1.5):
    """
    
    Fits a profile to a given spectrum at the line point. The peak value must be
    found in order to fit the peak correctly, so a +/- region is allowed to find
    the location of the peak.
    
    
    Inputs
    ------
    
    spectrum    :   Spectrum to fit a line profile to. Must be a :class:`onedspec`
                    class object.
                    
    line        :   Floating point location of the line (Angstroms)
    
    peak_tol    :   The peak of the profile will be found in the region from
                    (line - peak_tol) to (line + peak_tol). This can be set to
                    zero if the peak of the profile is already known.
                    
                    
    Outputs
    -------
    
    peak        :   Peak value of the profile found
    
    position    :   Position of the profile centroid (Angstroms)
    
    sigma       :   Gaussian sigma found for the best-fitting profile.
    
    """
    
    #todo - allow profile types to be voight/lorentzian?

    if not isinstance(spectrum, onedspec):
        return ValueError('Spectrum must be a onedspec class object')
    
        
    
    if type(lines) in [list, tuple, np.array]:
        
        try:
            lines = map(float, lines)
        
        except TypeError:
            raise TypeError('Line profile position ("%s") must be a floating point.' % line)
    
    
    else:
        try:
            lines = [float(lines)]
        
        except TypeError:
            raise TypeError('Line profile position ("%s") must be a floating point.' % line)

        
    try:
        peak_tol = float(peak_tol)
    
    except TypeError:
        raise TypeError('Position error must be a floating point.')
        
    
    profiles = []
    
    for line in lines:

        peak_spectrum = spectrum[line - peak_tol:peak_tol + line]
        
        peak = max(peak_spectrum.flux)
        index = scipy.where(peak_spectrum.flux==peak)[0][0]
        position = peak_spectrum.wave[index]
        
        index = spectrum.wave.searchsorted(position)
        
        
        p, m = 0, 0
        num = len(spectrum.flux)
        
        while (num > index + p + 1) and (spectrum.flux[index + p] > spectrum.flux[index + p + 1]):
            p += 1
            
        while (index - m - 1 > 0) and (spectrum.flux[index - m] > spectrum.flux[index - m - 1]):
            m += 1
    
        fitfunc = lambda p, x: p[0] * scipy.exp(-(x - p[1])**2 / (2.0 * p[2]**2))
        errfunc = lambda p, x, y: fitfunc(p, x) - y
        
        p0 = scipy.c_[peak, position, 5]
        p1, success = scipy.optimize.leastsq(errfunc, p0.copy()[0], args=(spectrum[index - m:p + index].wave, spectrum[index - m:p + index].flux))
        
        profiles.append(tuple(p1))
        
    if len(profiles) == 1:
        return profiles[0]
    
    return profiles


def normalise(spectrum, function='biezer', order=3, low_reject=3., high_reject=1., niterate=2, grow=1., continuum_regions=None, weights=None, **kwargs):

    """
    
    A one dimensional spectrum is fit to the continuum of the spectra provided. The
    resulting spectrum is the original spectrum which has been flux normalised. The
    fitted function may be a legendre polynomial, chebyshev polynomial, spline, or
    a beizer spline.
    
    Inputs
    ------
    
    spectrum    :   Spectrum to be continuum normalised. This must be a onedspec
                    class.
                    
    function    :   The function type to fit to the continuum. The available options
                    are 'legendre' polynomial, 'chebyshev' polynomial, a 'spline' or
                    a 'biezer' spline.
                    
    order       :   The order of the fitting polynomial or spline.
    
    low_reject  :   Rejection limit below the continuum fit in units of the residual
                    sigma.
                    
    high_reject :   Rejection limit above the continuum fit in units of the residual
                    sigma.
    
    niterate    :   Maximum number of rejection iterations.
    
    grow        :   There are two types of input for this argument. If an integer is
                    supplied, then when a pixel range is rejected pixels within this
                    distance are also rejected. If a float is specified then a line
                    will try to be fitted to the rejection region. If the region is
                    well characterised by a fitted profile, then the number of sigma
                    specified by the grow float on either side will be rejected. If
                    no profile is fitted, then the number of pixels on either side
                    will be rejected.
    
    """

    # Input checks
    
    functions = ['legendre', 'chebyshev', 'spline', 'biezer']
    
    if function not in functions:
        raise ValueError('Unknown continuum function type specified (%s). Available function types are: %s' % (function, ', '.join(functions), ))
    
    try: order = int(order)
    except ValueError: raise TypeError('Invalid input for profile order; \'%s\'. Order must be an integer-type.' % (order, ))
    
    try: low_reject = float(low_reject)
    except ValueError: raise TypeError('Invalid input for lower rejection limit; \'%s\'. Rejection limit must be a float-type.' % (low_reject, ))
    
    try: high_reject = float(high_reject)
    except ValueError: raise TypeError('Invalid input for higher rejection limit; \'%s\'. Rejection limit must be a float-type.' % (high_reject, ))
    
    try: niterate = int(niterate)
    except ValueError: raise TypeError('Invalid input for maximum interation number; \'%s\'. Maximum iteration number must be an integer-type.' % (niterate, ))
    
    if type(grow) not in [float, int]:
        raise TypeError('Invalid input for grow number; \'%s\'. Pixels to grow must be either an int- or float-type.' % (grow, ))
    
    if continuum_regions != None and len(continuum_regions) > 0:
        if type(continuum_regions) not in (list, tuple, np.array):
            raise TypeError('Invalid type for continuum_regions provided. This must be a list-type of tuples demonstrating the start and end of continuum regions e.g. [(x1, x2), (x3, x4), ..., (xN, xN+1)]')
        else:
            try:
                continuum_regions = [map(float, (a, b)) for (a, b) in continuum_regions]
                
            except TypeError, ValueError:
                raise TypeError('Invalid type for continuum_regions provided. This must be a list-type of tuples demonstrating the start and end of continuum regions, which must be float-types.')
    
        # Continuum regions specified are good, let's continue
        
        positions, values = [], []
        for (start, end) in continuum_regions:
            continuum_region = spectrum[start:end]
            
            # todo this could be faster/smarter/harder/better
            
            [positions.append(position) for position in continuum_region.wave]
            [values.append(value) for value in continuum_region.flux]
        
    else: positions, values = spectrum.wave, spectrum.flux
    
    first_pass = True
    sample = len(spectrum.wave) + 1
    
    while (niterate > 0) and (sample > len(positions)):
        
        sample = len(positions)
    
        if function == 'legendre':
            
            coeffs = scipy.polyfit(positions, values, order)
            continuum = scipy.polyval(coeffs, spectrum.wave)
           
        elif function == 'biezer':
            
            if kwargs.has_key('knots'):
                knots = kwargs['knots']
            else:
                if kwargs.has_key('bp'): bp = kwargs['bp']
                else: bp = 50.
                
                knots = np.arange((int(positions[0] / bp) + 1) * bp, int(positions[-1] / bp) * bp, bp)
            
            if first_pass:
                positions = list(positions)
                values = list(values)
                for knot in knots:
                    try:
                        idx = positions.index(knot)
                    except:
                        pass
                    else:
                        del positions[idx], values[idx]
                
                first_pass = False
            
            spline = scipy.interpolate.splrep(positions, values, t=knots)
            continuum = scipy.interpolate.splev(spectrum.wave, spline)
            
        elif function == 'spline':
            
            spline = scipy.interpolate.splrep(positions, values, task=0, s=0.5)
            continuum = scipy.interpolate.splev(spectrum.wave, spline)
              
        else:
            raise NotImplementedError('This type of continuum profile fitting hasn\'t been implemented yet')

            
        residual = spectrum.flux - continuum
        sigma = np.std(residual)
        
        sigma_residual = residual/sigma

        allowed_low = scipy.where(low_reject > sigma_residual, 1, 0)
        allowed_high = scipy.where(sigma_residual > -high_reject, 1, 0)
        
        allowed = allowed_low * allowed_high.T
        
        i, n = 0, len(allowed)
        while n > i:
            if not allowed[i]:
                rs, re = np.max([0, i-grow]), np.min([n-1, i+grow+1])
                allowed[rs:re] = np.zeros(re-rs)
                i += grow
            i += 1
    
    
        positions, values = zip(*spectrum.data[scipy.nonzero(allowed)])
        niterate -= 1

    continuum_regions = []
    in_continuum, region_start, wavelength_points = (False, 0, len(continuum))
    
    for i, is_continuum in enumerate(allowed):
        
        if is_continuum and not in_continuum: # Continuum region starts
            region_start, in_continuum = spectrum.wave[i], True
            
        elif in_continuum and (not is_continuum or (wavelength_points == i + 1)): # Continuum region ends
            region_end, in_continuum = spectrum.wave[i-1], False
            continuum_regions.append((region_start, region_end))
            
    if in_continuum: # Continuum region is on the reddest edge
        continuum_regions.append((region_start, spectrum.wave[-1]))
    
    
    if function in ('biezer', 'spline'):
        return (onedspec(spectrum.wave, spectrum.flux / continuum, type='waveflux'), continuum, continuum_regions, spline)
    else:
        return (onedspec(spectrum.wave, spectrum.flux / continuum, type='waveflux'), continuum, continuum_regions, coeffs)
        
        

def cross_correlate(spectrum, template):
    """
    Cross correlates a spectrum with a template spectrum
    
    
    Inputs
    ------
    
    spectrum    :   Spectrum to crosscorrelate against template. Must be a :class:`onedspec`
                    class object.
                    
    template    :   Template to crosscorrelate against spectrum. Must be a :class:`onedspec`
                    class object.
    
    peak_tol    :   The peak of the profile will be found in the region from
                    (line - peak_tol) to (line + peak_tol). This can be set to
                    zero if the peak of the profile is already known.
                    
                    
    Outputs
    -------
    
    peak        :   Peak value of the profile found
    
    position    :   Position of the profile centroid (Angstroms)
    
    sigma       :   Gaussian sigma found for the best-fitting profile.
    
    """
    newWave = 3e5*(spectrum.wave - np.mean(spectrum.wave)) / np.mean(spectrum.wave)
    crossCorrelation = ndimage.correlate1d(spectrum.flux, template.interpolate(spectrum.wave).flux, mode='wrap')
    
    peak = newWave[np.argmax(crossCorrelation)]
    crossCorrSpectrum = onedspec(newWave, crossCorrelation, type='waveflux')
    
    return fitprofs(crossCorrSpectrum, peak)[1]
    
    

