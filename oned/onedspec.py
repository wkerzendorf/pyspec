import os
import numpy as np
from scipy import interpolate
import logging
import bz2
import sqlite3
import cPickle as pickle


debug = True

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
        return r'<onedspec object over [%3.1f to %3.1f] Angstroms with [min, max] = [%2.1f, %2.1f] values>' % (self.wavelength[0], self.wavelength[-1], np.min(self.flux), np.max(self.flux),)
    
    @classmethod
    def from_ascii(cls, filename, **kwargs):
        """use the same kwargs with loadtxt"""
        data = np.loadtxt(filename, **kwargs)
        print "tmp"
        return cls(data)

    @classmethod
    def from_fits(cls):
        raise NotImplementedError('Reading from Fits is not implemented YET!!')
        
    def __init__(self, *args, **kwargs):
        if kwargs.has_key('type'):
            if kwargs['type'] == 'ndarray':
                self.data = args[0]
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
    
    x = property(getWavelength, setWavelength)
    y = property(getFlux, setFlux)
    
    wavelength = x
    flux = y
    
    xy=property(getXY, setXY)

        
    def __getitem__(self, index):
        
        if isinstance(index, float):
            
            new_index = self.wavelength.searchsorted(index)

            return self.flux.__getitem__(new_index)
            
        elif type(index) == slice:
            start, stop, step = index.start, index.stop, index.step

            if isinstance(index.start, float):
                    start = self.wavelength.searchsorted(index.start)
                    
            if isinstance(index.stop, float):
                    stop = self.wavelength.searchsorted(index.stop)
                    
            return [self.wavelength[slice(start,stop,step)], self.flux[slice(start,stop,step)]]
            
        else:
            return self.flux.__getitem__(index)
            
    
    def _map(self, spectrum, **kwargs):
        
        """
        
        Maps a spectra onto a common, union wavelength space with the spectrum
        provided.
        
        """
        
        # Are they on the same lambda space?
        try:
            difference = np.abs(self.wavelength - spectrum.wavelength)
        
        except ValueError:
            # Shape mismatch, they have different mappings
            
            
            self_resolution, spectrum_resolution = [np.median(np.diff(item.wavelength)) for item in [self, spectrum]]
            
            if self_resolution > spectrum_resolution:
                if debug: logging.info('onedspec._map: from self (%s) to spectrum (%s)' % (self, spectrum,))
                f = interpolate.interp1d(self.wavelength, self.flux, kind='linear', copy=False, **kwargs)
                
                return self.__class__(spectrum.wavelength, f(spectrum.wavelength))
            
            else:
                if debug: logging.info('onedspec._map: from spectrum (%s) to self (%s)' % (spectrum, self,))
                f = interpolate.interp1d(spectrum.wavelength, spectrum.flux, kind='linear', copy=False, **kwargs)
                
                return self.__class__(self.wavelength, f(self.wavelength))
        
        else:
            if np.max(difference) == 0.0:
                # Same wavelength map, no interpolation required
                
                return self
        
        
    def __add__(self, spectrum, **kwargs):
        """
        
        Adds two spectra together, or adds finite real numbers across an entire
        spectrum.
        
        """
        
        if type(spectrum) == self.__class__:
            
            # Ensure the two spectra are mapped onto the same wavelength space
            spectrum = spectrum._map(self, **kwargs)
            self = self._map(spectrum, **kwargs)
            
            return self.__class__(self.wavelength, self.flux + spectrum.flux)
                
        
        elif np.alltrue(np.isfinite(spectrum)):
            return self.__class__(self.wavelength, self.flux + spectrum)
        
        else: raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" % (type(self), type(spectrum),))
                         
    
    def __sub__(self, spectrum, **kwargs):
        
        """
        
        Subtracts two spectra together, or subtracts finite real numbers across
        an entire spectrum.
        
        """
        
        if type(spectrum) == self.__class__:
            
            spectrum = spectrum._map(self, **kwargs)
            self = self._map(spectrum, **kwargs)
            
            return self.__class__(self.wavelength, self.flux - spectrum.flux)
            
        elif np.alltrue(np.isfinite(spectrum)):
            return self.__class__(self.wavelength, self.flux - spectrum)
            
        else: raise TypeError("unsupported operand type(s) for -: '%s' and '%s'" % (type(self), type(spectrum),))
        

    def __mul__(self, spectrum, **kwargs):
        """
        
        Multiplies two spectra together, or multiplies finite real numbers across
        an entire spectrum.
        
        """

        if type(spectrum) == self.__class__:
            
            spectrum = spectrum._map(self, **kwargs)
            self = self._map(spectrum, **kwargs)
            
            return self.__class__(self.wavelength, self.flux * spectrum.flux)
            
        elif np.alltrue(np.isfinite(spectrum)):
            return self.__class__(self.wavelength, self.flux * spectrum)
            
        else: raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" % (type(self), type(spectrum),))
    
        
    def __div__(self, spectrum, **kwargs):
        """
        
        Divides two spectra together, or divides finite real numbers across an
        entire spectrum.
        
        """
        
        if type(spectrum) == self.__class__:
            
            spectrum = spectrum._map(self, **kwargs)
            self = self._map(spectrum, **kwargs)
            
            return self.__class__(self.wavelength, self.flux / spectrum.flux)
            
        elif np.alltrue(np.isfinite(spectrum)):
            return self.__class__(self.wavelength, self.flux / spectrum)
            
        else: raise TypeError("unsupported operand type(s) for /: '%s' and '%s'" % (type(self), type(spectrum),))
            
      
    # Mirror functions
    
    def __radd__(self, spectrum, **kwargs):
        return self.__add__(spectrum, **kwargs)
        
    def __rsub__(self, spectrum, **kwargs):
        return self.__sub__(spectrum, **kwargs)
        
    def __rmul__(self, spectrum, **kwargs):
        return self.__mul__(spectrum, **kwargs)
            
    def __rdiv__(self, spectrum, **kwargs):
        return self.__div__(spectrum, **kwargs)
    


    def gaussian_smooth(self, kernel, **kwargs):
        """
        
        Convolves the spectra using a Gaussian kernel with a standard deviation
        of <kernel> pixels.
        
        """
        
        self.flux = ndimage.gaussian_filter1d(self.flux, kernel, **kwargs)
        
        return self
    
    def __conform__(self):
        """
            Function that will automatically return an sqlite binary.
            This makes it easy to store it in sqlite databases.
            Most of the time this is called in the background
        """
        if protocol is sqlite3.PrepareProtocol:
            pickleSpec = pickle.dumps(self)
            zSpec = bz2.compress(pickleSpec)
            return sqlite3.Binary(zSpec)
        
            