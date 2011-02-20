import os
import numpy as np
from scipy import interpolate
import logging
import bz2
import sqlite3
import cPickle as pickle
import pdb

debug = True
num_precission = 1e-14

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
                coMin = np.max((operand.wavelength.min(), self.wavelength.min()))
                coMax = np.min((operand.wavelength.max(), self.wavelength.max()))
                
                resSelf = np.mean(np.diff(self.wavelength))
                resOperand = np.mean(np.diff(operand.wavelength))
                #Checking if spectral wavelength grids are the same
                #that makes operations easy as they can just act on the np arrays
                if  self.wavelength.shape[0] == operand.wavelength.shape[0] and \
                    np.mean(np.abs(self.wavelength - operand.wavelength)) < num_precission:
                    return func(self, operand.y)
                    
                #Checking if the spectral wavelength grids are the same on an overlaping grid
                elif self[coMin:coMax].wavelength.shape[0] == \
                    operand[coMin:coMax].wavelength.shape[0] and \
                    np.mean(np.abs(self[coMin:coMax].wavelength - operand[coMin:coMax].wavelength)) < num_precission:
                    return func(self[coMin:coMax], operand[coMin:coMax].y)
                    
                #Checking which resolution is lower and interpolating onto that grid
                elif self.op_mode == "on_resolution":
                    if resSelf > resOperand:
                        return func(self, operand.interpolate(self.wavelength).y)
                    else:
                        return func(self.interpolate(operand.wavelength), operand.y)
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
        return r'<onedspec object over [%3.1f to %3.1f] Angstroms with [flux_min, flux_max] = [%2.1f, %2.1f] values>' % (self.wavelength[0], self.wavelength[-1], np.min(self.flux), np.max(self.flux),)
    
    @classmethod
    def from_ascii(cls, filename, **kwargs):
        """use the same kwargs with loadtxt"""
        data = np.loadtxt(filename, **kwargs)

        return cls(data[:,:2], type='ndarray')

    @classmethod
    def from_fits(cls):
        raise NotImplementedError('Reading from Fits is not implemented YET!!')
        
    def __init__(self, *args, **kwargs):
        
        if kwargs.has_key('type'):
            if kwargs['type'] == 'ndarray':
                self.data = args[0]
                
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
    
    x = property(getWavelength, setWavelength)
    y = property(getFlux, setFlux)
    
    wavelength = x
    flux = y
    
    xy=property(getXY, setXY)

        
    def __getitem__(self, index):
        
        if isinstance(index, float):
            
            new_index = self.wavelength.searchsorted(index)

            return self.data[index]
            
        elif isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step

            if isinstance(index.start, float):
                    start = self.wavelength.searchsorted(index.start)
                    
            if isinstance(index.stop, float):
                    stop = self.wavelength.searchsorted(index.stop)
                    
            #return [self.wavelength[slice(start,stop,step)], self.flux[slice(start,stop,step)]]
            return onedspec(self.data[slice(start,stop)], type='ndarray')
            
        else:
            return self.data[index]
            
    def interpolate(self, wl_reference, mode='linear'):
        """
        Interpolate your spectrum on the reference wavelength grid.
        
        """
        if mode == 'linear':
            f = interpolate.interp1d(self.wavelength, self.flux, kind='linear', copy=False)
            return self.__class__(np.array(zip(wl_reference, f(wl_reference))), type='ndarray')
        else:
            return NotImplementedError()


    @spec_operation
    def __add__(self, operand):
        """
        
        Adds two spectra together, or adds finite real numbers across an entire
        spectrum.
        
        """
        
        return self.__class__(np.array(zip(self.wavelength, self.flux + operand)), type='ndarray')
        

    @spec_operation
    def __sub__(self, operand):
        """
        
        Adds two spectra together, or adds finite real numbers across an entire
        spectrum.
        
        """
        
        return self.__class__(np.array(zip(self.wavelength, self.flux + operand)), type='ndarray')
        

    @spec_operation
    def __mul__(self, operand):
        """
        
        Adds two spectra together, or adds finite real numbers across an entire
        spectrum.
        
        """
        
        return self.__class__(np.array(zip(self.wavelength, self.flux + operand)), type='ndarray')
        

    @spec_operation
    def __div__(self, operand):
        """
        
        Adds two spectra together, or adds finite real numbers across an entire
        spectrum.
        
        """
        
        return self.__class__(np.array(zip(self.wavelength, self.flux + operand)), type='ndarray')
        

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
            return sqlite3.Binary(self.data.tostring())
        
            