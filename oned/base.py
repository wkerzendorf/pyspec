import os
import numpy as np
from scipy import interpolate, ndimage
import logging
import sqlite3
import cPickle as pickle
import pdb
import pyfits

debug = True
num_precission = 1e-6
c = 299792.458
def spec_operation(func):
    def convert_operands(self, operand):
        """
        Decorator to automatically prepare the given operand for operation.
        Automatically checks for spectra and scalar units
        """
        
        if isinstance(operand, self.__class__):
            if self.op_mode != operand.op_mode:
                raise ValueError("Both spectra need to have the same op_mode.\n"
                                 "%s != %s" % (self.op_mode, operand.op_mode))
            
            else:

                
                resSelf = np.mean(np.diff(self.wave))
                resOperand = np.mean(np.diff(operand.wave))

                #Checking if spectral wave grids are the same
                #that makes operations easy as they can just act on the np arrays
                if  self.wave.shape[0] == operand.wave.shape[0] and \
                    np.mean(np.abs(self.wave - operand.wave)) < num_precission:
                    
                    return func(self, operand.flux)
                    
                #Checking if the spectral wave grids are the same on an overlaping grid
                elif self.wave.shape[0] == \
                    operand.wave.shape[0] and \
                    np.mean(np.abs(self.wave - operand.wave)) < num_precission:
                    
                    return func(self, operand.flux)
                
                #Checking which resolution is lower and interpolating onto that grid
                elif self.op_mode == "on_resolution":
                    
                    coMin = np.max((operand.wave.min(), self.wave.min()))
                    coMax = np.min((operand.wave.max(), self.wave.max()))

                    operand_bound = operand[coMin:coMax]
                    self_bound = self[coMin:coMax]

                    # Check the edges
                    union_end = False
                    union_start = False
                    
                    if operand_bound.wave[0] > self_bound.wave[0]:

                        operand_bound.data = operand_bound.data[1::]
                        union_start = operand.interpolate(self_bound.wave[0]).data
                        
                    elif self_bound.wave[0] > operand_bound.wave[0]:
                        union_start = self.interpolate(operand_bound.wave[0]).data
                    
                    if self.wave[-1] > operand_bound.wave[-1]:
                        self_bound.data = self_bound.data[:-1:]
                        union_end = self.interpolate(operand_bound.wave[-1]).data
                        
                    elif operand.wave[-1] > self_bound.wave[-1]:
                        union_end = operand.interpolate(self_bound.wave[-1]).data

                    
                    if resSelf < resOperand:
                        u_x = self_bound
                        
                        if type(union_end) != type(bool()): u_x.data = np.append(u_x.data, [union_end], axis=0)
                        if type(union_start) != type(bool()): u_x.data = np.insert(u_x.data, 0, union_start, axis=0)
                            
                        u_y = operand.interpolate(u_x.wave).flux
                        
                    else:
                        
                        u_x = self.interpolate(operand_bound.wave)
                        
                        if type(union_start) != type(bool()): u_x.data = np.insert(u_x.data, 0, union_start, axis=0)
                        if type(union_end) != type(bool()): u_x.data = np.append(u_x.data, [union_end], axis=0)
                        
                        u_y = operand.interpolate(u_x.wave).flux
                        
                    return func(u_x, u_y)
                else:
                    raise NotImplementedError("Operation mode %s not implemented")
                
                
        elif np.isscalar(operand):
            return func(self, operand)
        
        else:
            raise ValueError("unsupported operand type(s) for operation: %s and %s" %
                             (type(self), type(operand)))
    return convert_operands

class onedspec(object):

    """
    
    1-D spectrum class.
    This is an object class for one dimensional spectra.
    
    Parameters:
    ===========
    
    spectra = onedspec(spectrum)
    
    
    Examples:
    =========
    
    spectra = onedspec('myspectrum.txt')
    
    spectra = onedspec('night1.fits')
    

    """
    
    def __repr__(self):
        return r'<onedspec object over [%3.5f to %3.5f] Angstroms with [flux_min, flux_max] = [%2.1f, %2.1f] values>' % (self.wave.min(), self.wave.max(), self.flux.min(), self.flux.max(),)
    
    @classmethod
    def from_ascii(cls, filename, **kwargs):
        """use the same kwargs with loadtxt"""
        data = np.loadtxt(filename, **kwargs)

        return cls(data[:,:2], type='ndarray')

    @classmethod
    def from_fits(cls, filename, **kwargs):
        fitsFile = pyfits.open(filename, **kwargs)
        header = fitsFile[0].header
        
        if not (header.has_key('CRVAL1') and header.has_key('CDELT1') and header.has_key('CRPIX1') and header.has_key('NAXIS1')):
            raise ValueError('Could not find spectrum WCS keywords: CRVAL1, CDELT1, CRPIX1 and NAXIS1).\n'
                             'onedspec can\'t create a spectrum from this fitsfile')
        wave = np.arange(header['CRVAL1'], header['CRVAL1'] + (header['NAXIS1'])*header['CDELT1'], header['CDELT1'])
        flux = fitsFile[0].data.reshape([item for item in fitsFile[0].data.shape if item!=1])
        return cls(wave, flux, type='waveflux')
        
    def __init__(self, *args, **kwargs):
        
        if kwargs.has_key('type'):
            if kwargs['type'] == 'ndarray':

                self.data = args[0]
            elif kwargs['type'] ==  'waveflux':

                self.data = np.vstack((args[0], args[1])).transpose()
        else:        
            #---- auto check-----
            # If only one argument is passed, it is likely a text file or an arra
            if isinstance(args[0], np.ndarray):
                        self.data = args[0]
            elif len(args) == 1:
                
                # If it is a string we assume a text file
                if type(args[0]) == str:
                
                    extension = args[0].split('.')[-1].lower()
                    
                    if extension == 'fits':
    
                        try:                    
                            import pyfits
                        
                        except ImportError:
                            raise ImportError('FITS extension recognised and pyfits could not be imported')
                        
                        else:
                            
                            fits_file = pyfits.open(args[0])
                            
                            # Assume the data is stored in the first item of the HDU list
                            # for the moment.
                            
                            # todo - make this smarter
                            onedspec_list = []
                            spectrum_data = fits_file[0].data
                            
                            for item in spectrum_data:
                                onedspec_list.append(self.__class__(np.arange(0, item.shape[0]), item))
                                
                            return onedspec_list
                                
                    
                    else:
                    
                        # Assume text file
                        
                        self.data = np.loadtxt(args[0], unpack=True, **kwargs)
                        #self.wavelength, self.flux = self.data
                        
                elif type(args[0]) == onedspec:
                    self = args[0]
                    
                else:
                    data = args[0]
            
            
            elif len(args) == 2:
                data = zip(*args)
                
                
            else:
                raise ValueError('unknown spectrum input provided')
            
        
        
        self.op_mode = 'on_resolution'
        return None
        
    def getWavelength(self):
        return self.data[:, 0]
    def setWavelength(self, x):
        self.data[:, 0]=x
    def getFlux(self):
        return self.data[:, 1]
    def setFlux(self, y):
        self.data[:, 1]=y
    def setXY(self, val):
        raise NotImplementedError('XY can\'t be set')
    def getXY(self):
        return self.data[:, 0], self.data[:, 1]
    
    wave = property(getWavelength, setWavelength)
    flux = property(getFlux, setFlux)
    
    
    xy=property(getXY, setXY)

        
    def __getitem__(self, index):

        if isinstance(index, float):
            
            new_index = self.wave.searchsorted(index)
            return self.flux[new_index]
            
        elif isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step

            if isinstance(index.start, float):
                start = self.wave.searchsorted(index.start)
                    
            if isinstance(index.stop, float):
                stop = self.wave.searchsorted(index.stop)
                if len(self.wave) > stop: stop += 1
            
            return self.__class__(self.data[slice(start, stop)], type='ndarray')
            
        else:
            return self.data[index]
            
    def interpolate(self, wl_reference, mode='linear'):
        """
        Interpolate your spectrum on the reference wavelength grid.
        
        """

        if mode == 'linear':
            f = interpolate.interp1d(self.wave, self.flux, kind='linear', copy=False)
            if isinstance(wl_reference, float):
                return self.__class__(np.array([wl_reference, f(wl_reference)]), type='ndarray')
            else:
                return self.__class__(np.array(zip(wl_reference, f(wl_reference))), type='ndarray')
        else:
            return NotImplementedError('Non-linear interpolation not implemented yet.')


    @spec_operation
    def __add__(self, operand):
        
        """Adds two spectra together, or adds finite real numbers across an entire spectrum."""
        
        return self.__class__(self.wave, self.flux + operand, type='waveflux')
        

    @spec_operation
    def __sub__(self, operand):
        
        """Adds two spectra together, or adds finite real numbers across an entire spectrum."""
        
        return self.__class__(self.wave, self.flux - operand, type='waveflux')
        

    @spec_operation
    def __mul__(self, operand):
        
        """Adds two spectra together, or adds finite real numbers across an entire spectrum."""
        
        return self.__class__(self.wave, self.flux * operand, type='waveflux')
        

    @spec_operation
    def __div__(self, operand):
        
        """Adds two spectra together, or adds finite real numbers across an entire spectrum."""
        
        return self.__class__(self.wave, self.flux / operand, type='waveflux')
        
    @spec_operation
    def __pow__(self, operand):
        
        """Performs power operations on spectra."""
        
        return self.__class__(self.wave, self.flux ** operand, type='waveflux')
        

    def __len__(self):
        return len(self.wave)


    # Mirror functions
    
    def __radd__(self, spectrum, **kwargs):
        return self.__add__(spectrum, **kwargs)
        
    def __rsub__(self, spectrum, **kwargs):
        return self.__sub__(spectrum, **kwargs)
        
    def __rmul__(self, spectrum, **kwargs):
        return self.__mul__(spectrum, **kwargs)
            
    def __rdiv__(self, spectrum, **kwargs):
        return self.__div__(spectrum, **kwargs)
    
    def __rpow__(self, spectrum, **kwargs):
        return self.__pow__(spectrum, **kwargs)
    
    def add_noise(self, reqS2N, assumS2N=np.inf):
        #Adding noise to the spectrum. you can give it required noise and assumedNoise
        #remember 1/inf =0 for synthetic spectra
        noiseKernel = np.sqrt( (1/float(reqS2N))**2 - (1/assumS2N)**2)
        
        def makeNoise(item):
            return np.random.normal(item,noiseKernel*item)
            
            
        makeNoiseArray = np.vectorize(makeNoise)
        newy = makeNoiseArray(self.flux)
        return self.__class__(self.wave, newy, type='waveflux')


    def shift_velocity(self,v=None,z=None):
        #shift the spectrum given a velocity or a redshift. velocity is assumed to be in km/s
        if v==None and z==None:
            raise ValueError('Please provide either v or z to shift it')
        #Velocity in km/s
        c=3e5
        if v != None:
            return self.__class__(self.wave * np.sqrt((1+v/c)/(1-v/c)), self.flux, type='waveflux')
            
        elif z!=None:
            return self.__class__(self.wave*(1+z) , self.flux, type='waveflux')
    def gaussian_smooth(self, kernel, **kwargs):
        """
        
        Convolves the spectra using a Gaussian kernel with a standard deviation
        of <kernel> pixels.
        
        """
        
        self.flux = ndimage.gaussian_filter1d(self.flux, kernel, **kwargs)
        
        return self
    def convolve_rotation(self, vrot, beta=0.4, smallDelta=None):
        """
            Convolves the spectrum with a rotational kernel
            vrot is given in km/s
            beta is a limb-darkening factor (default=0.4)
            smallDelta is the wavlength delta that will be used when interpolating
        """
        if smallDelta == None:
            smallDelta = np.diff(self.wave).min()
        
        maxWave = self.wave.max()
        minWave = self.wave.min()
        
        bound = maxWave * (vrot / c)
        
        rotKernelX = (np.arange(maxWave - bound, maxWave + bound, smallDelta) - maxWave) / bound
        rotKernelX[rotKernelX**2>1] = 1.
        rotKernel = ((2/np.pi) * np.sqrt(1-rotKernelX**2) + (beta/2) * (1-rotKernelX**2)) / (1+(2*beta)/3)
        
        rotKernel /= np.sum(rotKernel)
        logWave =  10**np.arange(np.log10(minWave), np.log10(maxWave), np.log10(maxWave/(maxWave-smallDelta)))
        
        f = interpolate.splrep(self.wave, self.flux, k=1)
        
        logSpec = interpolate.splev(logWave, f)
        
        rotLogSpec = ndimage.convolve1d(logSpec, rotKernel, mode='nearest')
        
        f = interpolate.splrep(logWave, rotLogSpec, k=1)
        
        return self.__class__(self.wave, interpolate.splev(self.wave, f), type='waveflux')
        
    def convolve_profile(self, R, initialR = np.inf, smallDelta=None):
        """
            Smooth to given resolution
            * R = resolution to smooth to
            * initial R =
            smallDelta is the wavlength delta that will be used when interpolating
        """
        if smallDelta == None:
            smallDelta = np.diff(self.wave).min()
        
        maxWave = self.wave.max()
        minWave = self.wave.min()
        smoothFWHM = np.sqrt((maxWave/R)**2 - (maxWave/initialR)**2)
        smoothSigma = (smoothFWHM/(2*np.sqrt(2*np.log(2))))/smallDelta
        
        logWave =  10**np.arange(np.log10(minWave), np.log10(maxWave), np.log10(maxWave/(maxWave-smallDelta)))
        
        f = interpolate.splrep(self.wave, self.flux, k=1)
        
        logSpec = interpolate.splev(logWave, f)
        smoothLogSpec = ndimage.gaussian_filter1d(logSpec, smoothSigma)
        
        f = interpolate.splrep(logWave, smoothLogSpec, k=1)
        
        return self.__class__(self.wave, interpolate.splev(self.wave, f), type='waveflux')
        
        
    def to_ascii(self, filename):
        pass
    
    def to_fits(self, filename):
        crval1 = self.wave.min()
        crpix1 = 1
        #checking for uniformness
        cdelt1 = np.mean(np.diff(self.wave))
        testWave = np.arange(crval1, self.wave.max()+cdelt1, cdelt1, dtype=self.wave.dtype)
        if np.max(testWave-self.wave) > num_precission:
            raise ValueError("Spectrum not on a uniform grid (error %s), cannot save to fits" % np.max(testWave-self.wave))
        
        primaryHDU = pyfits.PrimaryHDU(self.flux)
        
        primaryHDU.header.update('CRVAL1', crval1)
        primaryHDU.header.update('CRPIX1', crpix1)
        primaryHDU.header.update('CDELT1', cdelt1)
        primaryHDU.writeto(filename, clobber=True)
    
    
    def __conform__(self, protocol):
        """
            Function that will automatically return an sqlite binary.
            This makes it easy to store it in sqlite databases.
            Most of the time this is called in the background
        """

        if protocol is sqlite3.PrepareProtocol:
            return sqlite3.Binary(self.data.tostring())
        else:
            raise NotImplementedError('Conforming to protocol %s has not been implemented yet' % protocol)
        
            