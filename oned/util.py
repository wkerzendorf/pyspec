from base import onedspec

import numpy as np
import scipy
import scipy.optimize
from scipy import ndimage, interpolate, optimize

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
    fitted_arc = onedspec(arc_spectrum.wave, np.zeros(len(arc_spectrum.wave)), mode='waveflux')
    
    fitfunc = lambda p, x: p[0] * scipy.exp(-(x - p[1])**2 / (2.0 * p[2]**2))
    for profile in profiles:
        
        fitted_arcline = onedspec(fitted_arc.wave, fitfunc(profile, fitted_arc.wave), mode='waveflux')
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
        
        return onedspec(synthetic.wave, m, mode='waveflux')
        
        
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
    

def fitprofs(spectrum, lines, peak_tol=1.5, base=None):
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
                    
                    
    base        : where to start the base of the gaussian fit (e.g. 1 for normalized . 
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
        
        peak = np.max(peak_spectrum.flux)
        index = scipy.where(peak_spectrum.flux==peak)[0][0]
        position = peak_spectrum.wave[index]
        
        
        
        
        if base is None:
            base = np.median(spectrum.flux)
        
        if width is None:
            width=5
    
        fitfunc = lambda p, x: p[0]+ p[1] * np.exp(-(x - p[2])**2 / (2.0 * p[3]**2))
        errfunc = lambda p, x, y: fitfunc(p, x) - y
        
        p0 = scipy.c_[base, peak, position, width]
        p1, success = scipy.optimize.leastsq(errfunc, p0.copy()[0], args=(peak_spectrum.wave, peak_spectrum.flux))
        
        profiles.append(tuple(p1))
        
    if len(profiles) == 1:
        return profiles[0]
    
    return profiles


def _running_std(values, neighbours=50):
    
    stds = []
    num_values = len(values)
    for i in xrange(num_values):
        stds.append(np.std(values[np.max([0, i - neighbours]):np.min([num_values, i + neighbours])]))

    return np.array(stds)
    
    

def normalise(spectrum, function='spline',
              order=2, low_reject=1., high_reject=5.,
              niterate=4, grow=4., continuum_regions=None,
              weights=None, **kwargs):

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
    
    functions = ['legendre', 'spline']
    
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
    
    try: grow = int(grow)
    except ValueError: raise TypeError('Invalid input for grow number; \'%s\'. Pixels to grow must be either an int- or float-type.' % (grow, ))
    
    if continuum_regions != None and len(continuum_regions) > 0:
        if type(continuum_regions) not in (list, tuple, np.array):
            raise TypeError('Invalid type for continuum_regions provided. This must be a list-type of tuples demonstrating the start and end of continuum regions e.g. [(x1, x2), (x3, x4), ..., (xN, xN+1)]')
        else:
            try:
                continuum_regions = [map(float, (a, b)) for (a, b) in continuum_regions]
                
            except:
                raise TypeError('Invalid type for continuum_regions provided. This must be a list-type of tuples demonstrating the start and end of continuum regions, which must be float-types.')
    
        # Continuum regions specified are good, let's continue
        
        positions, values = [], []
        for (start, end) in continuum_regions:
            continuum_region = spectrum[start:end]
            
            # todo this could be faster/smarter/harder/better
            
            [positions.append(position) for position in continuum_region.wave]
            [values.append(value) for value in continuum_region.flux]
        
    else: positions, values = spectrum.wave, spectrum.flux
    
    if weights == None: weights = np.ones(len(positions))
    
    # Keyword argument checks
    
    if kwargs.has_key('point_spacing'):
        try:
            point_spacing = float(kwargs['point_spacing'])
        except:
            raise ValueError('Invalid keyword argument for point_spacing provided. This must be a floating point-type value.')
        
    else: point_spacing = 25
    
    if function == 'spline':
        if kwargs.has_key('knot_spacing'):
            try:
                knot_spacing = float(kwargs['knot_spacing'])
            except:
                raise ValueError('Invalid keyword argument for knot_spacing provided. This must be a floating point-type value.')
        else: knot_spacing = 150
        
        if kwargs.has_key('static_std') and not kwargs['static_std'] and kwargs.has_key('std_neighbours'):
            try:
                std_neighbours = int(kwargs['std_neighbours'])
            except:
                raise ValueError('Invalid keyword argument for std_neighbours provided. This must be an integer-type value.')
            
        else: std_neighbours = 150
    
    if order == 0:
        # Check to see if a region was set by continuum_regions, which would have
        # been passed into positions, values
        try:
            _ = values
        except NameError:
            # No problem, default to all values of flux
            values = spectrum.flux
        
        finally:
            continuum = [np.mean(values)] * len(spectrum.wave)
        
        niterate = 0
        spline = []
        allowed = continuum_regions
        
    
    while niterate > 0:
        
        assert len(positions) > 0
    
        if function == 'legendre':
            
            coeffs = scipy.polyfit(positions, values, order)
            continuum = scipy.polyval(coeffs, spectrum.wave)
           
        elif function == 'spline':
            
            edge = ((positions[-1] - positions[0]) % knot_spacing) / 2
            knots = np.arange(positions[0] + edge, positions[-1], knot_spacing)
            
            for i, knot in enumerate(knots):
                if knot in positions:
                    knots[i] += np.diff(positions)[0]/2.
                    
            #print knots
            spline = scipy.interpolate.splrep(positions, values, w=weights, k=order, t=knots)
            continuum = scipy.interpolate.splev(spectrum.wave, spline)
            
              
        else:
            raise NotImplementedError('This type of continuum profile fitting hasn\'t been implemented yet')

    
        residual = spectrum.flux - continuum
        
        if kwargs.has_key('static_std') and not kwargs['static_std']: sigma = _running_std(residual, neighbours=std_neighbours)
        else: sigma = np.std(residual)
        
        residual_sigma = residual/sigma
    
    
        allowed_low = scipy.where(residual_sigma > -low_reject, 1, 0)
    
        j, num = (0, len(allowed_low))
        in_excluded, region_start = (False, 0)
        
        while (num > j):
            
            excluded = allowed_low[j]
            
            if not excluded and not in_excluded:
                region_start, in_excluded = (j, True)
                
            elif excluded and in_excluded:
                region_end, in_excluded = (j - 1, False)
                
                if type(grow) == float:

                    subset = spectrum[region_start:region_end + 1]
                    peak = np.min(subset.flux)
                    position = subset.wave[scipy.argmin(subset.flux)]
                    
                    index = spectrum.wave.searchsorted(position)
                
                    p, m, n = (0, 0, len(spectrum.wave))
                
                    while (n > index + p + 1) and (spectrum.flux[index + p] < spectrum.flux[index + p + 1]):
                        p += 1
                        
                    while (index - m - 1 > 0) and (spectrum.flux[index - m] < spectrum.flux[index - m - 1]):
                        m += 1
                        
                    # If the line extends past subset region, then expand it and try to
                    # fit a line, otherwise we will just grow outwards
                    
                    fitfunc = lambda p, x: continuum[spectrum.wave.searchsorted(x[0]):spectrum.wave.searchsorted(x[-1]) + 1] - p[0] * scipy.exp(-(x - p[1])**2 / (2. * p[2]**2))
                    errfunc = lambda p, x, y: fitfunc(p, x) - y
            
                    p0 = scipy.c_[peak, position, 2]
                    
                    o = np.max([p, m])
                    
                    
                    a1 = np.max([0, index - o])
                    b1 = np.min([o + index, n])
                    #print a1, b1
                    
                    try:
                        p1, success = scipy.optimize.leastsq(errfunc, p0.copy()[0], args=(spectrum[a1:b1].wave, spectrum[a1:b1].flux))
                    
                    except:
                        allowed_low[region_start - grow:region_end + grow] = 0
                        j += grow
                      
                    else:
                        if p1[2] > 0 and 2 > p1[0]/peak and p1[0]/peak > 0:
                            
                            idx_a = spectrum.wave.searchsorted(position - grow*p1[2])
                            idx_b = spectrum.wave.searchsorted(position + grow*p1[2])
                            
                            a = np.min([idx_a, region_start])
                            b = np.max([idx_b, region_end])
                            
                            #print niterate, p1, success, [spectrum.wave[_] for _ in [region_start, region_end, idx_a, idx_b, a, b]]
                            
                            allowed_low[a:b] = np.zeros(b-a)
                            j = b 
                    
                        else:
                            allowed_low[region_start - grow:region_end + grow] = 0
                            j += grow
                else:
                
                    allowed_low[region_start - grow:region_end + grow] = 0
                    j += grow
            
            j += 1
            
        
        allowed_high = scipy.where(high_reject > residual_sigma, 1, 0)
        # todo put in code for allowed_high checking for cosmic rays or skylines
        
        allowed = allowed_low * allowed_high.T
    
        values = []
        weights = []
        positions = []
        continuum_regions = []
    
        region_start, in_continuum = (0, False)
        
        for j, is_continuum in enumerate(allowed):
            
            if is_continuum and not in_continuum: region_start, in_continuum = (j, True)
            
            elif in_continuum and (not is_continuum or j + 1 == len(allowed)):
                
                region_end, in_continuum = (j - 1, False)
                if is_continuum: j += 1 # correction for last continuum region        
                
                if (spectrum.wave[region_end] - spectrum.wave[region_start]) > 1.:
                    continuum_regions.append((region_start, region_end))
                    
                    if spectrum.wave[region_end] - spectrum.wave[region_start] > point_spacing * 3:
                        regions = []
                        
                        edge = ((spectrum.wave[region_end] - spectrum.wave[region_start]) % point_spacing) / 2
                        points = np.arange(spectrum.wave[region_start] + edge, spectrum.wave[region_end] - edge, point_spacing)
                        
                        for point in points:
                            regions.append((point - point_spacing / 2., point + point_spacing / 2.))
                            
                    else:
                        regions = [(region_start, region_end)]
                        
                        
                    for (region_start, region_end) in regions:
                        region = spectrum[region_start:region_end]
                        
                        positions.append(np.median(region.wave))
                        values.append(np.max(region.flux) - 0.5 * (np.max(region.flux) - np.median(region.flux)))
                        weights.append(np.sum((continuum[region.wave.searchsorted(region_start):region.wave.searchsorted(region_end) + 1] - region.flux)**2 * (region_end - region_start)))
                

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
    
    
    # Put continuum as onedspec object
    continuum = onedspec(spectrum.wave, continuum, mode='waveflux')
    
    
    if function in ('biezer', 'spline'):
        return (spectrum / continuum, continuum, continuum_regions, spline)
    else:
        return (spectrum / continuum, continuum, continuum_regions, coeffs)
        
def continuum2(spectrum, low_rej=2., high_rej=3., function='legendre', maxiter=3, order=5, mode='normal'):
    def fitLegendre(x, y, mode='fit'):
        p = np.polynomial.Legendre.fit(x, y, order)
        if mode=='fit':
            return p(spectrum.wave)
        if mode=='func':
            return p

    def fitChebyshev(x, y, mode='fit'):
        p = np.polynomial.Chebyshev.fit(x, y, order)
        if mode=='fit':
            return p(spectrum.wave)
        if mode=='func':
            return p
    if function=='legendre':
        fitfunc = fitLegendre
        
    if function=='chebyshev':
        fitfunc = fitChebyshev

    contFlux = fitfunc(spectrum.wave, spectrum.flux)
    mask = spectrum.dq
    high_rej_mask = np.zeros(spectrum.wave.shape).astype(bool)
    low_rej_mask = np.zeros(spectrum.wave.shape).astype(bool)
    for i in range(maxiter):
        residual = (spectrum.flux - contFlux)
        residual_sigma = residual / np.std(residual[mask])
        high_rej_mask = np.logical_or(residual_sigma > high_rej, high_rej_mask)
        low_rej_mask = np.logical_or(residual_sigma < -low_rej, low_rej_mask)
        mask = np.logical_and(mask, np.logical_not(np.logical_or(high_rej_mask, low_rej_mask)))
        contFlux = fitfunc(spectrum.wave[mask], spectrum.flux[mask])
    if mode == 'normal':
        return onedspec(spectrum.wave, contFlux, mode='waveflux')
        
    elif mode == 'rms':
        residual = (spectrum.flux[mask] - contFlux[mask])
        return onedspec(spectrum.wave, contFlux, mode='waveflux'), np.std(residual)
        
    elif mode == 'full':
        residual = (spectrum.flux[mask] - contFlux[mask])
        return (onedspec(spectrum.wave, contFlux, mode='waveflux'),
                onedspec(spectrum.wave, residual_sigma, mode='waveflux'),
                fitfunc(spectrum.wave[mask], spectrum.flux[mask], mode='func'))
    
def resolving_power(spectrum, mode='normal'):
    """Calculates the resolving power (R) of a given spectrum
    
    
    Modes available:
    ----------------
    
    normal  -   Returns the median, deviation, minimum and maximum resolving
                power across the wavelength provided.
    full    -   Returns the resolving power R as a function of lambda across
                the wavelength provided.
    """
    
    if not isinstance(spectrum, onedspec):
        raise TypeError('Spectrum provided must be a pyspec object')
        
    available = 'normal full'.split()
    if mode not in available:
        raise ValueError('Mode provided for resolving power is not available. Available modes are: %s' % (', '.join(available, )))
        
    delta = np.diff(spectrum.wave)
    wave = spectrum.wave[0:-1] # Err on the minimum R-edge
    R = wave/delta
    
    if mode == 'full': return R
    else:
        return (np.median(R), np.std(R), np.min(R), np.max(R))
    

def cross_correlate(spectrum, template, mode='shift'):
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
    crossCorrelation -= np.median(crossCorrelation)
    
    peakPos = newWave[np.argmax(crossCorrelation)]
    peak = np.max(crossCorrelation)
    
    crossCorrSpectrum = onedspec(newWave, crossCorrelation, mode='waveflux')
    if mode == 'spectrum':
        return crossCorrSpectrum
    elif mode == 'shift':
        fitfunc = lambda p, x: p[0] * np.exp(-(x - p[1])**2 / (2.0 * p[2]**2))
        errfunc = lambda p, x, y: fitfunc(p, x) - y
        
        return optimize.leastsq(errfunc, (peak, peakPos, 1), args=(crossCorrSpectrum.wave, crossCorrSpectrum.flux))[0]
    else:
        raise NotImplementedError('Mode %s is not supported' % mode)


