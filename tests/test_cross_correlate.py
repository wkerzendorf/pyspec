from nose.tools import with_setup
import unittest

import os
import numpy as np

from pyraf import iraf
from ... import pyspec

# Setup IRAF
iraf.noao()
iraf.rv()

# todo - remove this when we can provide fibre extensions when opening fits files in pyspec
import pyfits


# Get fibre lists to correlate against



template_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/T04500G00M15V002K2SNW.fits')
assert os.path.exists(template_filename)

# Observed file
image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/M4.fits')
assert os.path.exists(image_filename)

# Load the template file
template = pyspec.oned.base.onedspec.from_fits(template_filename)

# Load the observed file
image = pyfits.open(image_filename)

# Normalise template
normalised = pyspec.oned.normalise(template, function='legendre')
normalised_template = normalised[0]




def test_all():

    fibre_list = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/fibres.list')
    assert os.path.exists(fibre_list)
    
    all_fibres = np.loadtxt(fibre_list, usecols=(0, 2), dtype=str)

    for fibre_number, fibre_type in all_fibres:
        if fibre_type == 'P' and fibre_number:
            yield fxcor_comparison, int(fibre_number)
    
    
        
    
       
def tearDown():
    os.system('rm -f %s' % (os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/fxcor.*'), ))



@with_setup(setup=None,teardown=tearDown)
def fxcor_comparison(fibre):

     wavelength_map = list(pyspec.oned.onedspec.from_fits(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/M4.fits')).wave)
     
     wavelength = []
     wavelength.extend(wavelength_map)
     data = list(image[0].data[fibre - 1, :])
     
     i = 0
     for (w, f) in zip(wavelength_map, data):
         if not np.isfinite(f):
             del data[i]
             del wavelength[i]
             i -= 1
         i += 1
     
     assert len(data) == len(wavelength)
     
     observed = pyspec.oned.onedspec(wavelength, data, mode='waveflux')
     
     assert type(observed) == pyspec.oned.onedspec
     
     normalised = pyspec.oned.normalise(observed, function='legendre', niterate=3, order=3)
     normalised_observed = normalised[0]
     
     assert type(normalised_observed) == pyspec.oned.onedspec
     
     sample = [8400.0, 8800.0]
     
     # Perform spectrum type cross-correlation
     cross_correlated = pyspec.oned.util.cross_correlate(normalised_observed[sample[0]:sample[1]], normalised_template[sample[0]:sample[1]], mode='spectrum')
     
     assert type(cross_correlated) == pyspec.oned.onedspec
     
     # Perform shift type cross-correlation
     peak, position, sigma = pyspec.oned.util.cross_correlate(normalised_observed[sample[0]:sample[1]], normalised_template[sample[0]:sample[1]], mode='shift')
     
     # Assertions for the returned data types
     assert type(peak) in [float, np.float, np.float32, np.float64]
     assert type(position) in [float, np.float, np.float32, np.float64]
     assert type(sigma) in [float, np.float, np.float32, np.float64]
     
     
     iraf.hedit(images="%s[0][*,%i]" % (image_filename, fibre,), fields='CUNIT1', value='Angstroms', add=1, verify=0, show=1, update=1, Stdout=1)
     iraf.rv.fxcor(objects="%s[0][*,%i]" % (image_filename, fibre,) , templates=template_filename, rebin='template',
                   osample=':'.join(map(str, sample)), rsample=':'.join(map(str, sample)),
                   output=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/fxcor'), verbose='long',interactive=0, Stdout=1)
     
     output = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/fxcor.txt'), comments='#', usecols=(7,8,9,11,13))
     hght, fwhm, tdr, vrel, verr = output
     
     percentage_diff = 100 * ((peak - vrel)/vrel - 1.)
     
     print "fxcor: %2.3f +/- %2.2f" % (vrel, verr, )
     print "cross: %2.3f +/- %2.2f" % (peak, sigma, )
     print "percentage diff: %2.2f%% (|limit| < 5.00%%)" % (percentage_diff, )
     
     # I'll allow a 5% discrepency
     assert np.abs(percentage_diff) < 5.
    



        