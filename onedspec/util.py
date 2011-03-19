#from onedspec import onedspec
import numpy as np
import scipy
import scipy.optimize

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
                    if min(abs(np.array(arc_line_peaks) - arc_spectrum.wavelength[i])) > peak_tol:
                        arc_line_peaks.append(arc_spectrum.wavelength[i])  
                else: arc_line_peaks.append(arc_spectrum.wavelength[i])
                

        
        
    else: arc_line_peaks = arc_lines
    
    profiles = fitprofs(arc_spectrum, arc_line_peaks, peak_tol) 
    fitted_arc = onedspec(arc_spectrum.wavelength, np.zeros(len(arc_spectrum.wavelength)), type='waveflux')
    
    fitfunc = lambda p, x: p[0] * scipy.exp(-(x - p[1])**2 / (2.0 * p[2]**2))
    for profile in profiles:
        
        fitted_arcline = onedspec(fitted_arc.x, fitfunc(profile, fitted_arc.x), type='waveflux')
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
        m2[0:-1] = scipy.where(np.diff(synthetic.wavelength) < kwargs['continuum_min_diff'], 1, 0)
        m2[-1] = 0
        
        m = m1 * np.transpose(m2)
        
        
        in_continuum, region_start, wavelength_points = (False, 0, len(m))
        continuum_regions = []
        for i, is_continuum in enumerate(m):
            
            if is_continuum and not in_continuum: # Continuum region starts
                region_start, in_continuum = synthetic.wavelength[i], True
                
            elif in_continuum and (not is_continuum or (wavelength_points == i + 1)): # Continuum region ends
                region_end, in_continuum = synthetic.wavelength[i-1], False
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
        position = peak_spectrum.wavelength[index]
        
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
        
        profiles.append(tuple(p1))
        
    if len(profiles) == 1:
        return profiles[0]
    
    return profiles


