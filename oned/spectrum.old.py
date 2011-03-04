from numpy import array, ndarray,copy,argsort, linspace, argmin, argmax, mean, std,loadtxt,savetxt
from numpy import polyfit, polyval, diff,arange,sqrt,trapz
from scipy import interpolate, optimize, ndimage
import numpy
import lines
import convolve
import sqlite3
import zlib
import cPickle
import pdb
#from copy import deepcopy
#Todo:do the slice step thing: spectrum[300.:500.:1.]should work
class spectrum(object):
	def __init__(self, *args,**kwargs):
	    if len(args)==1:
		if isinstance(args[0],str):
			data=loadtxt(args[0],**kwargs)
		else:
			data=args[0]
	
	    elif len(args)>=2:
	        data=zip(*args)
	    else:
	        raise Exception('Did not give enough or too many arguments in constructor (number of arguments: %s)'%len(args))
	    data=array(data)
	    data=data[argsort(data[:,0])]
	    self.data=data
	    self.debugInfo=False
	    #getting approximate pixelscale
	    self.pixScale=numpy.mean(numpy.diff(self.x))
	def getX(self):
	    #print "getting x"
	    return self.data[:, 0]
	def setX(self, x):
	    #print "setting x"
	    self.data[:, 0]=x
	def getY(self):
	    return self.data[:, 1]
	def setY(self, y):
	    self.data[:, 1]=y
	def setXY(self, val):
	    print "XY can't be set"
	def getXY(self):
	    return self.data[:, 0], self.data[:, 1]
	x=property(getX, setX)
	y=property(getY, setY)
	xy=property(getXY, setXY)
	def __getitem__(self, index):
	    if isinstance(index, float):
	        new_index=self.data[:, 0].searchsorted(index)
	        return self.data.__getitem__(new_index)
	    elif type(index)==slice:
		start=index.start
		stop=index.stop
		step=index.step
		if isinstance(index.start,float):
			start=self.data[:, 0].searchsorted(index.start)
		if isinstance(index.stop,float):
			stop=self.data[:, 0].searchsorted(index.stop)
		return spectrum(self.data[slice(start,stop,step)])
	    else:
	        return self.data.__getitem__(index)
	def __setitem__(self, index,value):
	    if isinstance(index, float):
	        new_index=self.data[:, 0].searchsorted(index)
	        self.data.__setitem__(new_index,value)
		return None
	    elif type(index)==slice:
		start=index.start
		stop=index.stop
		step=index.step
		if isinstance(index.start,float):
			start=self.data[:, 0].searchsorted(index.start)
		if isinstance(index.stop,float):
			stop=self.data[:, 0].searchsorted(index.stop)
		if isinstance(value,spectrum):#Making sure to set the data value of a spectrum
			self.data.__setitem__(slice(start,stop,step),value.data)
		else:
			self.data.__setitem__(slice(start,stop,step),value)
		return None
	    else:
	        return self.data.__getitem__(index)
	def __add__(self, spec):
		if type(spec)==spectrum:
			newSpec=spec.interpolate(xref=self.x, order=1)
			return spectrum(self.x, self.y+newSpec.y)
		elif numpy.isfinite(spec):
			return spectrum(self.x, self.y+spec)
		else: raise Exception("Can't add spectrum and %s"%type(spec))
	def __sub__(self, spec):
		#import numpy as numpy
		if type(spec)==spectrum:
			new_spec=spec.interpolate(xref=self.x, order=1)
			return spectrum(self.x, self.y-new_spec.y)
		elif numpy.isfinite(spec):
			return spectrum(self.x, self.y-spec)
		else: raise Exception("Can't subtract spectrum and %s"%type(spec))
		
	def __div__(self, spec):
		if type(spec)==spectrum:
			new_spec=spec.interpolate(xref=self.x, order=1)
			return spectrum(self.x, self.y/new_spec.y)
		elif numpy.isfinite(spec):
			return spectrum(self.x, self.y/spec)
		else: raise Exception("Can't divide spectrum and %s"%type(spec))
	def __mul__(self, spec):
		if type(spec)==spectrum:
			new_spec=spec.interpolate(xref=self.x, order=1)
			return spectrum(self.x, self.y*new_spec.y)
		elif numpy.isfinite(spec):
			return spectrum(self.x, self.y+spec)
		else: raise Exception("Can't multiply spectrum and %s"%type(spec))
		
	def __conform__(self, protocol):
	#an adapter to store a spectrum in sqlite
		if protocol is sqlite3.PrepareProtocol:
			pickleSpec = cPickle.dumps(self)
			zSpec = zlib.compress(pickleSpec)
			return sqlite3.Binary(zSpec)

	def interpolate(self, xref=None, nop=1000, order=3):
	    if xref==None:
	        newWl=linspace(self.data[0, 0], self.data[-1, 0],nop)
	        newIntens=interpolate.splev(newWl, interpolate.splrep(self.x, self.y, k=order))
	        return spectrum(zip(newWl, newIntens))
	    else:
	        newIntens=interpolate.splev(xref, interpolate.splrep(self.x, self.y, k=3))
	        return spectrum(xref, newIntens)
		
	def smoothGauss(self, kernelsize, mode='wlScale'):
		#Two different modes, using wlScale or pxScale
		x=self.x
		y=self.y
		if   mode=='pxScale':
			return spectrum(x, ndimage.gaussian_filter1d(y, kernelsize))
		elif mode=='wlScale':
			return spectrum(x, ndimage.gaussian_filter1d(y, kernelsize/self.pixScale))
		#Implement useful exceptions
	def smoothg(self, kernelsize, mode='wlScale'):
		return self.smoothGauss(kernelsize, mode=mode)
	def smoothMax(self, kernelsize, mode='wlScale'):
		x=self.x
		y=self.y
		if   mode=='pxScale':
			return spectrum(x, ndimage.maximum_filter1d(y, kernelsize))
		elif mode=='wlScale':
			return spectrum(x, ndimage.maximum_filter1d(y, kernelsize/self.pixScale))

	def convRot(self,rotvel,beta=0.4):
		#Beta is the limdarkening coefficient
		newIntens=convolve.convolve(self,convolve.rotKernel,rotvel=rotvel,beta=beta)
		return spectrum(self.x,newIntens)
	def convProf(self,R):
		#R is the resolution to convolve to 
		newIntens=convolve.convolve(self,convolve.profKernel,R=R)
		return spectrum(self.x,newIntens)
	def intTrapz(self,dx=1.0):
		#Integrate the spectrum with trapezoid rule
		return trapz(self.y,self.x,dx=dx)
	def bin(self,binsize,mode='mean'):
		newx=[mean(self.x[i:i+binsize]) for i in arange(0,len(self.x),step=binsize)]
		if mode=='mean':
			newy=[mean(self.y[i:i+binsize]) for i in arange(0,len(self.x),step=binsize)]
		else:
			raise Exception('Other Modes not implemented yet')
		return spectrum(newx,newy)
	def binLog(self):
		newLogBins=numpy.logspace(numpy.log10(self.x.min()),numpy.log10(self.x.max()),self.x.ptp())
		return newLogBins
		pass
	def  min(self):
	    return self[argmin(self.y)]
	def  max(self):
	    return self[argmax(self.y)]
	def addNoise(self,reqS2N,assumS2N=numpy.inf):
		#Adding noise to the spectrum. you can give it required noise and assumedNoise
		#remember 1/inf =0 for synthetic spectra
		noiseKernel=numpy.sqrt((1/float(reqS2N))**2-(1/assumS2N)**2)
		#pdb.set_trace()
		def makeNoise(item):
			return numpy.random.normal(item,noiseKernel*item)
		makeNoiseArray=numpy.vectorize(makeNoise)
		newy=makeNoiseArray(self.y)
		return spectrum(self.x,newy)
	def fitContinuum(self,weights=None,func='poly3',iter=3, sigmas=3,lsigma=None,hsigma=None):
		if lsigma==None and hsigma==None:
			lsigma=sigmas
			hsigma=sigmas
		x=self.x
		y=self.y
		sel=arange(len(y))
#		if weights!=None
		if func.startswith('poly'):
			order=int(func.strip('poly'))
			for i in range(iter):
				if self.debugInfo:
					print "Starting %s iteration (%s points used in fit)"%(i+1,len(y))
				x, y=x[sel], y[sel]
				ids=arange(len(y))
				params=polyfit(x, y, order)
				yfit=polyval(params,x)
				delta=y-yfit
				#FIXME READUP on sigma reject
				#print delta[delta>0]
				if hsigma!=None: hrej=hsigma*std(delta)
				#print delta[delta<0]
				if lsigma!=None: lrej=-lsigma*std(delta)
				
				if hsigma==None: hsel=delta>0
				else: hsel=(delta<hrej)*(delta>0)
				
				if lsigma==None: lsel=delta<0
				else: lsel=(delta>lrej)*(delta<0)
				
				#print lsel.shape
				sel=ids[(lsel+hsel)]
				#pdb.set_trace()
				#print sel.size
			
				#return x[sel],y[sel]
			return spectrum(self.x,polyval(params,self.x))
	def shiftVelocity(self,v=None,z=None):
		if v==None and z==None:
			raise Exception('Please provide either v or z to shift it')
		#Velocity in km/s
		c=3e5
		if v!=None: return spectrum(self.x*sqrt((1+v/c)/(1-v/c)),self.y)
		elif z!=None: return spectrum(self.x*(1+z),self.y)
		
	def markLines(self,lineList,xlim=None,plotSpec=True,ax=None,sampleSize=10):
		import pylab
		if plotSpec: 
			if xlim==None: pylab.plot(*self.xy)
			if xlim!=None: pylab.plot(*self[slice(*map(float,xlim))].xy)
		if type(lineList)==str:
			lineList=loadtxt(lineList,dtype=str)
		noLines=len(lineList)
		j=0.1
		if ax==None:
			ax=pylab.gca()
		
		for i,item in enumerate(lineList):
			if float(i)/noLines>j:
				print "Marked %d %%"%(j*100)
				j+=0.1
			wl=float(item[0])
			if xlim!=None and (wl<xlim[0] or wl>xlim[1]):
				continue
			if wl<self.x.min() or wl>self.x.max():
				continue
			desc=' '.join(map(str,item[1:]))
			i=self.x.searchsorted(wl)
			y=mean(self[slice(int(i-sampleSize),(i+sampleSize))][:,1])
			lineScale=0.9
			ax.vlines(wl,y*lineScale,y)
			ax.text(wl,y*lineScale,desc,horizontalalignment='center',verticalalignment='top',rotation=270)
#			pylab.annotate(desc,(wl,y),(wl,y-0.1),arrowprops=dict(arrowstyle="->"))
	def write2ascii(self,fname,fmt='%.18e'):
		savetxt(fname,self.data,fmt=fmt)
	def write2fits(self,fname):
		print "Not implemented yet"
#		savetxt('tmp.fitsdat',self.data,fmt=fmt)
#		iraf.oned()
#		iraf.wspectext('tmpfits.dat',fname)
	def fwhm(self, pos=None, contin=None, iter=3, sigmas=3, thresh=10000):
		if pos==None:
		    pos=self.min()[0]
		if contin==None:
		    contin=self.fitContinuum()[pos][1]
		hm=(self[pos][1]+contin)/2
		index=self.x.searchsorted(pos)
		i=1
		found=array([False, False])
		hmpoint=[None, None]
		while found.all()==False:
		    
		    if self.y[index+i]>hm and found[1]==False:
		        found[1]=True
		        cur_index=index+i
		        m2=(diff(self.y[cur_index-1:cur_index+1])/diff(self.x[cur_index-1:cur_index+1]))[0]
		        b2=self.y[cur_index]-m2*self.x[cur_index]
		    if self.y[index-i]>hm and found[0]==False: 
		        found[0]=True
		        cur_index=index-i
		        m1=(diff(self.y[cur_index-1:cur_index+1])/diff(self.x[cur_index-1:cur_index+1]))[0]
		        b1=self.y[cur_index]-m1*self.x[cur_index]
		    if i>thresh: raise Exception('Could not find a point thats above the half maximum! ')
		    i+=1
		#print 'Half-maximum at %s'%hm
		x1=(hm-b1)/m1
		x2=(hm-b2)/m2
		#print x1, x2
		#print 'm%s b%s'%(m1, b1)
		return abs(x2-x1)
	def fitLineg(self, pos,mode='fit'):
		pos=float(pos)
		pos=self.findMin(pos)[0]
		fwhm=self.fwhm(pos)
		fitfunc=lambda p, x: lines.gLine(x, p=p)
		errfunc=lambda p, x, y: fitfunc(p, x)-y
		cont=self.fitContinuum()[self[pos][0]][1]
		p0=[cont,self[pos][1]-cont, pos,fwhm]
		p1,success=optimize.leastsq(errfunc, p0, args=(self.x,self.y))
		print "Success %s"%success
		if mode=='fit':    return spectrum(self.x,fitfunc(p1,self.x))
		elif mode=='param':     return p1
	def findMin(self, pos,mode='data'):
   		"""mode can be data for actual finding the data minimum or gauss for gaussian minimum"""
		def finder(i): 
		    y=self.y[i]
		    #print "i %s i+1 %s i-1 %s"%(y, self.y[i+1], self.y[i-1])
		    if self.y[i+1]>y<self.y[i-1]:
		        #print "Done returning %s"%i
		        return i
		    elif self.y[i+1]<y>self.y[i-1]:
		        raise Exception('Started on maximum, unstable solution')
		    elif self.y[i+1]<y:
		        return finder(i+1)
		    elif self.y[i-1]<y:
		        return finder(i-1)
		if mode=='data':return  self.data[finder(self.x.searchsorted(pos))]
		elif mode=='gauss': return self.fitLineg(pos,mode='param')[[2,1]]
	def pos2ind(self, pos):
		return self.x.searchsorted(pos)
	def __repr__(self):
		return self.data.__repr__()

"""	       for i in range(iter):
	               delta=abs(y-mean(y))
	               sel=[i for i, idelta in enumerate(delta) if idelta<sigmas*std(y)]
	               
	       if mode=='sigmaclip':
	           return spectrum(zip(x, y))
	       if func.startswith('poly'):
	           order=int(func.strip('poly'))
	           sel=range(len(self.y))
	           params=polyfit(x, y, order)
	           return spectrum(zip(self.x, polyval(params, self.x)))
	       if func.startswith('spline'):
	           order=int(func.strip('spline'))
	           sel=range(len(self.y))
	           params=interpolate.splrep(x, y, k=order)
	           return spectrum(self.x, interpolate.splev(self.x, params))"""
