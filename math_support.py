# this is the file with most of the mathematical and statistical functions
# this is not important to most people 
import numpy as np
import pandas as pd

from scipy import sparse     
from numpy.matlib import repmat as repmat
from scipy.special import expit                   # to deal with error overflow in exp
from numpy.linalg import norm
from scipy import stats #                       for fitting straightline /baseline 

from scipy.signal import get_window
from scipy.fftpack import fft, fftfreq, fftshift, ifft
from scipy.signal import find_peaks

from scipy.optimize import curve_fit
import scipy
import matplotlib.pyplot as plt

from math import inf

import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.colors as colors 
from mpl_toolkits.axes_grid1 import make_axes_locatable
        

# the wavenumbers are in x/cm^-1
# so the can be express in wavelength by 
#so, 2000 cm^-1 is 5 micro = 5000 nm 
# (1/2000)*(100)*1e6 = 5.0
####################################################
# recalculate numbers 
###################################################
#INPUT:
# m = #waves
# n = refraction index
# wavenumber 1  [cm^-1]
# wavenumber 2  [cm^-1]
#example calculate_thickness(2, 1.546,2121.073,1991.505) = 50
#calculate_thickness(2, 1.546,2121.073,1991.505)  =50 micro meter
#OUTPUT:
#t = thickness [micro m]
def calculate_thickness(m,n,l1, l2):
  return m/(2.*n*np.abs(l1-l2))*(1e6/100)

class MyNormalize(colors.Normalize):
              def __call__(self, value, clip=None):
                # function to normalize any input between vmin and vmax linearly to [0,1]
                n = lambda x: (x-self.vmin)/(self.vmax-self.vmin)
                # nonlinear function between [0,1] and [0,1]
                f = lambda x,a: (2*x)**a*(2*x<1)/2. +(2-(2*(1-1*x))**a)*(2*x>=1)/2.
                return np.ma.masked_array(f(n(value),0.5))

# the goals of the part is the find the part responsible for the fringes from FFT
# eliminate the fringes from this usineg the Solheim algorithm. 
# to calculate frequencies. We want each point in our frequecy array by spaced by delta f = sample frequency/N
# 1. get length of the data N 
# 2. generate an index  array n
# 3. get the sampling rate (mean step size between consecutive points)  sr
# 4. get the sampling frequency T = N / sr
# 5. get the frequency array by dividing the index-array by T, freq= n/T
def fourier_fit(xss, f1):
      factor = 5
      n = len(xss)
      T = (max(xss) - min(xss))/n
      NQ=round(n/2)
      xf = fftfreq(factor*n, T)
      #yf = fft(f1* get_window('bartlett', int(round(n))),n=factor*n)
      #yf = fft(f1* get_window('blackmanharris', int(round(n))),n=factor*n)
      yf = fft(f1* get_window('hann', int(round(n))),n=factor*n)
      
      # there are two version for normalisation. Divide by n or NQ (it differs by a factor of 2) 
      #yplot = 2.0*np.abs(yf)/NQ
      yplot = 2.0*np.abs(yf)/NQ
      xplot = np.abs(xf)

      # get peaks 
      h = np.median(yplot[1:NQ])
      
      #peaks, _ = find_peaks(yplot[:NQ], height=1.50*h)
      peaks, _ = find_peaks(yplot[1:NQ], height=0.10*h)
      
      # now sort by magnitude
      index = np.argsort(yplot[peaks])[::-1]
      
      #print(" Amplitude [FFT]:", yplot[peaks[index[0]]]) 
      #print (" Frequency [FFT]:", xplot[peaks[index[0]]]*2*np.pi)
      #print(" Wavelength [FFT]", 2.0*np.pi/xplot[peaks[index[0]]])
  
      
      A = yplot[peaks[index[0]]]
      w = xplot[peaks[index[0]]]*2*np.pi

      c = np.mean(f1)
      
      p = np.angle(yf[peaks[index[0]]]) + np.pi
      
      #print (" Phase-Values(FFT)", p)
      z = [A, w,p,c]
      return z, xplot, yplot



def CooksDistance(X,y,imfo):
   import textwrap
   print('       ----- Cooks distance (begin) -------')
   n = len(y)
   # by hand using LA
   b = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
   p = len(b)
   hat = X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T))
   diagonal = np.diag(hat)
   I = np.ones(len(diagonal))
   residuals = np.dot(X,b) - y
   s2 = np.dot(residuals,residuals)/(len(y)-p)
   D = (residuals**2)*diagonal/(p*s2*(I-diagonal)**2)
   # Create CooksDistance plot
   x_val = np.arange(len(y))
   fig =  plt.figure(figsize=(10, 10))
   plt.hlines(y=4/n,xmin=0, xmax=n, color='red')
   plt.vlines(x=x_val,ymin=np.zeros(len(y)), ymax=D)
   fig.savefig(imfo+'/CooksDistancePlot.png', bbox_inches='tight')
   #plt.show()
   plt.close()
   # identify which points might need to be thrown away
   index = np.where(D < 4/n)
   #print("       Index of points to include:", index)
   prefix = '         Index of points to include:'
   message = np.array2string(np.asarray(index))
   message.replace('[','')
   message.replace('\n','')
   message.replace(']','')
   preferredWidth = 70
   wrapper = textwrap.TextWrapper(initial_indent=prefix , width=preferredWidth, subsequent_indent=' '*len(prefix))
   
   print(wrapper.fill(message))
   print('       ----- Cooks distance (end)-------')
   return index
   
   
    
def extrapolate(px,py,img3,xsize, ysize, index,imfo):
        print('    ------------- EXTRAPOLATION (begin)----')
        data = np.c_[px,py,img3[index]]  
        # regular grid covering the domain of the data
        #mn = np.min(data, axis=0)
        # regular grid covering the domain of the data
        mx = np.max(data, axis=0)
        mn = np.array([0,0,0])

        X,Y = np.meshgrid(np.linspace(mn[0], mx[0], xsize), np.linspace(mn[1], mx[1], ysize))
        XX = X.flatten()
        YY = Y.flatten()
        

        # best-fit linear plane (1st-order)
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
        Cooks_index = CooksDistance(A,data[:,2],imfo)
        
        # perform fit again with the data that passed the cooks-distance criteria
        data = np.c_[px[Cooks_index],py[Cooks_index],img3[index][Cooks_index]]  
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = scipy.linalg.lstsq(A, img3[index][Cooks_index])    # coefficients
      
        #A = np.c_[data[Cooks_index,0], data[Cooks_index,1], np.ones(len(Cooks_index))]
        #print("shape new A",np.shape(A))
        #print("shape new y",np.shape(img3[index][Cooks_index]))
        
        #b = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),img3[index][Cooks_index])
      #  print(C)
        
        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]

        fig1 =  plt.figure(figsize=(10, 10))
        ax = fig1.add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        #ax[r,c].scatter(py, px, c=original_thickness[index], alpha=1.0, marker='.', cmap=plt.cm.Oranges)
        ax.scatter(data[:,0], data[:,1],data[:,2], c=data[:,2], s=50, marker='.')
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_zlabel('Z')
        #ax.axis('equal')
        ax.axis('tight')
        fig1.savefig(imfo+'/Extra_Thickness_map_LS_FIT_3D.png', bbox_inches='tight')
        plt.close()
        
        #print(np.shape(Z))
        #print(np.shape(X))
        #print(np.shape(Y))
        # Evaluate the R-score for the linear fit (now this doesnt look good yet.)
        Rscore = calc_R(img3[index][Cooks_index],Z[index][Cooks_index])
        print("       New Rscore", Rscore)

        fig =  plt.figure(figsize=(10, 10))
        plt.hist(img3[index][Cooks_index]-Z[index][Cooks_index])
        fig.savefig(imfo+'/ErrorHist.png', bbox_inches='tight')
        plt.close()
        
        
        vmin_,vmax_ = np.min(Z),np.max(Z)
        norm=  MyNormalize(vmin=vmin_, vmax=vmax_)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        
        im3 = plt.imshow(Z, norm=norm,cmap=plt.cm.viridis,origin='lower',zorder=1,extent=[0, xsize, 0,ysize])
        plt.scatter(px[Cooks_index], py[Cooks_index], c=data[:,2], alpha=0.2, marker='.',zorder=2)
        #ax.scatter(data[:,0], data[:,1],data[:,2], c=data[:,2], s=50, marker='.')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')
        textstr2 = '\n'.join((r'$R-Score: %.3f$' % (Rscore, ) , r'$Accepted: %s$' % (Rscore > 0.85, )))
        ax.text(0.05, 0.05, textstr2 , fontweight="bold", transform=ax.transAxes,bbox=dict(facecolor='white', alpha=0.5,boxstyle='round,pad=1'))   
        fig.savefig(imfo+'/Extra_Thickness_map_extrapolate.png', bbox_inches='tight')
        #plt.show()  
        plt.close()
        
        print('    ------------- EXTRAPOLATION (end)----' )
        return Z
        

# get R-score for all data 
# Fraction of Variance Unexplained (https://medium.com/microsoftazure/how-to-better-evaluate-the-goodness-of-fit-of-regressions-990dbf1c0091)
def calc_Rscore(y_data,y_fitted):
     num = np.shape(y_fitted)[1] # number of samples 
     n = np.shape(y_fitted)[0]
    # print(np.shape(y_data))
     SSres = np.sum((y_data - y_fitted)**2, axis=0)/n
     SStot = np.sum((y_data - np.mean(y_data, axis=0))**2, axis=0)/n
     R2 = 1.0- SSres/SStot 
     index = np.asarray(R2 > 0.90)
     return R2, index

# Fraction of Variance Unexplained (https://medium.com/microsoftazure/how-to-better-evaluate-the-goodness-of-fit-of-regressions-990dbf1c0091)
def calc_R(y_data, y_fitted):
      n = len(y_fitted)
      SSres = np.sum((y_data - y_fitted)**2)/n
      #print(SSres)
      SStot = np.sum((y_data - np.mean(y_data))**2)/n
      R2 = 1.0- SSres/SStot 
      return R2

####################################################
#solver
###################################################
def solver1(M,v1):
 from scipy.linalg import cho_solve, cho_factor
 from scipy.linalg import svd
 from numpy import diag
 from numpy import dot
 
 
 #method 1: # 2
 #method 4: # 2
 #method 5: # 0
 #method 6: # 2
 
 method = 1

 try: 
   # regualr lin-alg solver
  if method ==1: 
       z1 = np.linalg.solve(M,v1)
       #or 
     #  z1 = np.linalg.inv(M).dot(v1)
       # check to see if it worked
    #    M.dot(z1)
  if method ==2: 
       c, low = cho_factor(M)
       z1 = cho_solve((c, low), v1)
 
  if method == 3:       
       U, s, VT = svd(M)
       # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
       c = np.dot(U.T,v1)
       # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w (trivial inversion of a diagonal matrix)
       w = np.dot(np.diag(1/s),c)
       # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
       z1 = np.dot(VT.conj().T,w)
 
  if method == 4:
    
       z1 = np.linalg.lstsq(M,v1)[0]

  if method == 5:
       X=M 
       XT = np.matrix.transpose(X)
       XT_X = np.matmul(XT, X)
       XT_y = np.matmul(XT, v1)
       z1 = np.matmul(np.linalg.inv(XT_X), XT_y)
  if method == 6:  
       z1 = numpy.linalg.pinv(M).dot(v1)
 except:
   z1 = np.linalg.lstsq(M,v1)[0]
   
 return z1       
 
       
  



####################################################
# develop a method that does the same
###################################################
#x =xs
#y =ys
def fit_sin(x,y):
  
  succes = True  
  num_samples = len(x)
  diff_x = np.zeros(num_samples)
  diff_y= np.zeros(num_samples)
  S = np.zeros(num_samples) 
  SS = np.zeros(num_samples)
  diff_x[1:] = x[1:]-x[:-1]
  diff_y[1:]= y[1:]+y[:-1]
  temp = np.ones(num_samples**2).reshape(num_samples,num_samples)  
  temp2 =np.triu(temp, k=0)
  temp2 = np.flipud(temp2)
  temp2 = np.fliplr(temp2)
  S[1:] = 0.5*(diff_x[1:]*diff_y[1:]) 
  S = np.inner(S, temp2) 
  SS[1:] = + 0.5*(S[1:] + S[:-1] )*diff_x[1:]  
  SS = np.inner(SS, temp2) 
  m1 = [np.sum(SS**2),np.sum((x**2)*SS),np.sum(x*SS),np.sum(SS) ]
  m2 = [np.sum((x**2)*SS),np.sum(x**4),np.sum(x**3),np.sum(x**2) ]
  m3 = [np.sum(x*SS),np.sum(x**3),np.sum(x**2),np.sum(x) ]
  m4 = [np.sum(SS),np.sum(x**2),np.sum(x),num_samples ]
  M = np.array([m1, m2, m3, m4])
  v1 = np.array([np.sum(y*SS),np.sum(y*(x**2)),np.sum(y*x),np.sum(y) ])

  z1 = solver1(M,v1)
  if z1[0] > 0:
     z1 = np.linalg.pinv(M).dot(v1)
     
  A,B,C,D=z1
  if (A < 0):
     omega1 = np.sqrt(-A)
  else:
     omega1 = np.sqrt(A)
     succes = False  
  a1 = 2.*B/omega1**2
  b1 = (B*x[0]**2 + C*x[0]+D-a1)*np.sin(omega1*x[0])+(1./omega1)*(C + 2.*B*x[0])*np.cos(omega1*x[0])
  c1 = (B*x[0]**2 + C*x[0]+D-a1)*np.cos(omega1*x[0])-(1./omega1)*(C + 2.*B*x[0])*np.sin(omega1*x[0])
  rho1 = np.hypot(c1, b1)
  phi1 = np.arctan2(c1, b1)
  
  epsilon = 1e-5
  if (b1 < 0):
     phi1 = phi1 + np.pi
  if (b1**2  <  epsilon):
     if (c1 > 0):
     
          phi1 = 0.5*np.pi
     else:       
          phi1 = -0.5*np.pi
   
 # print("Found omega 1, a,b,c, rho, phi:",omega1, a1,b1,c1, rho1, phi1)
  f = y - a1
  dis = rho1**2 - f**2
  try:
    m = (dis >= 0)
    Phi = np.arctan(np.divide(f, np.sqrt(dis, where=m), where=m), where=m)
  except:
    succes = False 
    print("method failed")
  np.copysign(0.5*np.pi, f, where=~m, out=Phi)
  kk = np.round((omega1 * x + phi1) / np.pi)
  theta = (-1)**kk * Phi + np.pi * kk
  
  #try:
  #  m = (dis >= 0)
   # Phi = np.arctan(np.divide(f, np.sqrt(dis, where=m), where=m), where=m)
  #except:
  #  succes = False 
   # print("method failed")
  #np.copysign(0.5*np.pi, f, where=~m, out=Phi)
  #kk = np.round((omega1 * x + phi1) / np.pi)
  #m = np.where(dis >= epsilon)  
  
   #  Phi = np.arctan(f / np.sqrt(dis) )
    # theta = (-1)**kk * Phi + np.pi * kk
  #else:
   #  if (y[-1] > a1):
    #     theta = (-1)**kk*(0.5*np.pi) + np.pi * kk
     #else:    
      #   theta = (-1)**kk*(-0.5*np.pi) + np.pi * kk
  
  m1 = [np.sum(x**2),np.sum(x)] 
  m2 = [np.sum(x),num_samples] 
  M = np.array([m1, m2])
  v1 = np.array([np.sum(theta*x),np.sum(theta) ])
  
  z1 = solver1(M,v1)

  #z2 = np.inner(np.linalg.inv(M),np.array(v1))
  #z3 = np.inner(v1,np.linalg.inv(M))
  omega2, phi2 = z1 
  a2 = a1
  rho2 = rho1
  b2 = rho2*np.cos(phi2)
  c2 = rho2*np.sin(phi2)
  #print("Found omega2, a,b,c, rho, phi:",omega2, a2,b2,c2, rho2, phi2)
  m1 = [num_samples, np.sum(np.sin(omega2*x)),np.sum(np.cos(omega2*x))] 
  m2 = [np.sum(np.sin(omega2*x)),np.sum(np.sin(omega2*x)**2),np.sum(np.sin(omega2*x)*np.cos(omega2*x))] 
  m3 = [np.sum(np.cos(omega2*x)), np.sum(np.sin(omega2*x)*np.cos(omega2*x)),np.sum(np.cos(omega2*x)**2)] 
  M = np.array([m1, m2, m3])
  v1 = np.array([np.sum(y),np.sum(y*np.sin(omega2*x)),np.sum(y*np.cos(omega2*x)) ])
  
  z1 = solver1(M,v1)
  
  #z2 = np.inner(np.linalg.inv(M),np.array(v1))
  #z3 = np.inner(v1,np.linalg.inv(M))
  a3, b3,c3 = z1
  rho3 = np.hypot(c3, b3)
  phi3 = np.arctan2(c3, b3)
  #print("Found omega, a,b,c, rho, phi:",omega2, a3,b3,c3, rho3, phi3)
  
  #values = np.asarray([omega2, a3,b3,c3, rho3, phi3])
  
    
  values = np.asarray([omega2, a3,b3,c3, rho3, phi3])
 
  
  #index = [values[(0,1,2,3)] <0 ]
  
  #values[values < 0] = -1*values[values < 0]
  
  #values[5]= values[5]  - np.pi*np.floor(values[5]/ (2*np.pi))
  
  return values, succes  
  



##################################################
# Function to fit 1 set of sinodal data
###############################################
def sinfunc(t, A, w, p, c): 
   return A * np.sin(w*t + p) + c


###############################################
def fit_sin2_multi(tt, yy):    
 #   '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy, axis=0))
    guess_freq = abs(ff[np.argmax(Fyy[1:,], axis=0)+1])*2.*np.pi   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy, axis=0) * 2.**0.5
    guess_offset = np.mean(yy, axis=0)
    num = np.shape(yy)[1]
    i = np.arange(num)
    # this line is a problem if NO peaks are found we could try something like this: #list(map(lambda r:min(find_peaks(yy[:,r].T)[0],default=-1),j))
    try: 
        peaks = pd.DataFrame(map(lambda r:find_peaks(r),yy.T) )
        j = np.arange(num)
        #print(peaks)
        #peak_loc1 = list(map(lambda x: peaks[x][0],j))
        peak_loc1 =list(map(lambda x: peaks[0][x][0],j))
        wavelength = 2.0*np.pi/guess_freq
        temp = x[peak_loc1] - 0.25*wavelength
    except:
        temp = np.zeros(np.shape(yy)[1])+1e-6
        
    guess = np.array([guess_amp, guess_freq, temp, guess_offset])
  #  np.shape(guess)
   # np.shape(yy)
    #print(guess)
    lower_bound= 1e-7
 #   popt = list(map(lambda x:scipy.optimize.curve_fit(sinfunc, tt, yy[:,x],p0=guess[:,x],bounds=([0.0,0,0], [11., 11., 11.,11]),maxfev=200000),i))
    popt = list(map(lambda x:scipy.optimize.curve_fit(sinfunc, tt, yy[:,x],p0=guess[:,x],bounds=((lower_bound,lower_bound,lower_bound,-1), (inf,inf,inf,inf)), maxfev=200000),i))
 #   popt = list(map(lambda x:scipy.optimize.curve_fit(sinfunc, tt, yy[:,x],p0=guess[:,x],maxfev=200000),i))
    coeff = np.asarray(list(map(lambda x: popt[x][0],i)) )
    pcov =  np.asarray(list(map(lambda x: popt[x][1],i))) 
    
    # calculate the variance from the co-variance matrices
    var =  np.asarray(list(map(lambda x: np.sqrt(np.diag(pcov[x,:,:])),i))) 
    # remove negative values
    index = np.where(coeff[:,0] < 0)
    coeff[index,0] = coeff[index,0]*(-1)
    coeff[index,2] = coeff[index,2]+np.pi
    
    
    
    #index = np.where(coeff[:,2] < 0)
    #coeff[index,2] = coeff[index,2]*(-1)
    
    #index = np.where(coeff < 0)
    #coeff[index] = coeff[index] *(-1)
    
    #index = np.where(coeff[:,2] > 2*np.pi)
    #coeff[index,2] = coeff[index,2] - 2*np.pi 
    #index = np.where(coeff[:,2] < 0)
    #coeff[index,2] = coeff[index,2]*(-1)
    
    
    #coeff[index,2] = coeff[index,2] + 2.0*np.pi
    #index = np.where(coeff[:,2] > 2.0*np.pi)
    #coeff[index,2]= coeff[index,2] - 2.*np.pi*np.floor(coeff[index,2]/ (2.*np.pi))
    
    #var = list(map(lambda x: popt[x][1],i)) 
    guess_df = pd.DataFrame(np.transpose(guess), columns=['a','b','c','k'] )  
    df2 = pd.DataFrame(coeff, columns=['a','b','c','k'] ) 
    # reduc to value between -pi and pi
#    df2.c = df2.c - np.pi*np.floor(df2.c/ np.pi)
    # for positive values 
    #df2.a[df2.a < 0] = -1*df2.a[df2.a < 0]
    #df2.c[df2.c < 0] = df2.c[df2.c < 0] + np.pi
  #  print("THEIR FIT")
   # print("  {0}".format(df2)) 
    return df2, var, guess_df

#x = spec_x[interval]
#y = spec_y[interval,:]

#x = xss
#y = spec_y[interval,:]-y_fit[:,:]

def fit_sin_multi_new(x,y):
   print("---------------------------------------------------------------")
   epsilon = 1e-10
   x_new = x
   num_samples,N = np.shape(y)
   d = 0.5 * np.diff(x)
   x = np.transpose(np.asarray([x]*N))
   ins = np.zeros(N)
   S = np.cumsum((y[1:,:] + y[:-1,:]) * np.transpose([d]*N), axis=0)
   S = np.insert(S, 0,ins,0)
   SS = np.cumsum((S[1:,:] + S[:-1,:]) * np.transpose([d]*N), axis=0)
   SS = np.insert(SS, 0,ins,0)
   
   m1 = [np.sum(SS**2, axis=0),np.sum((x**2)*SS, axis=0),np.sum(x*SS, axis=0),np.sum(SS, axis=0) ]
   m2 = [np.sum((x**2)*SS,axis=0),np.sum(x**4, axis=0),np.sum(x**3, axis=0),np.sum(x**2, axis=0) ]
   m3 = [np.sum(x*SS, axis=0),np.sum(x**3, axis=0),np.sum(x**2, axis=0),np.sum(x, axis=0) ]
   m4 = [np.sum(SS, axis=0),np.sum(x**2, axis=0),np.sum(x,axis=0),num_samples*np.ones(N) ]
   M = np.array([m1, m2, m3, m4])
   
   v1 = np.array([np.sum(y*SS,axis=0),np.sum(y*(x**2),axis=0),np.sum(y*x,axis=0),np.sum(y,axis=0) ])
   j = np.arange(N)  
   
   # determine if matrix is singular or not 
   #determinants= np.asarray(list(map(lambda x:np.linalg.det(M[:,:,x]),j)))
   #dets = np.asarray(np.where(determinants >  0)[0])
   #not_dets = np.asarray(np.where(determinants <= 0)[0])
   
   # find solution for positive definite matrices
   
   z1 = np.asarray(list(map(lambda x: np.linalg.solve(M[:,:,x],v1[:,x]),j)))

   # see if there are solution with z1 > 0
   checklist = np.asarray(np.where(z1[:,0] > 0 )[0])
   checklist2 = np.asarray(np.where(z1[:,0] < 0)[0])
      
   print(checklist)
   print(pd.DataFrame(z1[:,0]))
   if (np.shape(checklist)[0] >0) :
      print("Found faulty solutions, adding to z2 list ")
      #z1 =np.delete(z1,checklist, axis=0 )   
      #not_dets = np.sort(np.concatenate([np.ravel(not_dets),np.ravel(checklist) ] ))
      z2 = np.asarray(list(map(lambda x: np.linalg.pinv(M[:,:,x]).dot(v1[:,x]),j[checklist])))
      z1[checklist,:] = z2
      
   print(pd.DataFrame(z1[:,0]))
     
    # see if there are solution with z1 > 0
   checklist = np.asarray(np.where(z1[:,0] > 0 )[0])
   checklist2 = np.asarray(np.where(z1[:,0] < 0)[0])
   if (np.shape(checklist)[0] >0) :
     # print("Found faulty solutions, adding to z2 list ")
    #  z1 =np.delete(z1,checklist, axis=0 )   
    #  not_dets = np.sort(np.concatenate([np.ravel(not_dets),np.ravel(checklist) ] ))
      z2 = np.asarray(list(map(lambda x: np.linalg.lstsq(M[:,:,x],v1[:,x])[0],j[checklist])))
      z1[checklist,:] = z2
      
   
   A,B,C,D= z1[:,0], z1[:,1],z1[:,2], z1[:,3]
   
   omega1 =  np.zeros(len(A))
   omega1[np.where(A < 0)] = np.sqrt(-A[np.where(A < 0)])
   omega1[np.where(A > 0)] = np.sqrt(A[np.where(A > 0)])
   
   
   a1 = 2.0*B/omega1**2
   b1 = (B*x_new[0]**2 + C*x_new[0]+D-a1)*np.sin(omega1*x_new[0])+(1./omega1)*(C + 2.0*B*x_new[0])*np.cos(omega1*x_new[0])
   c1 = (B*x_new[0]**2 + C*x_new[0]+D-a1)*np.cos(omega1*x_new[0])-(1./omega1)*(C + 2.0*B*x_new[0])*np.sin(omega1*x_new[0])
   rho1 = np.hypot(c1, b1)
   phi1 = np.arctan2(c1, b1)
  
   a2 =a1
   phi1[np.where(b1 < 0)] = phi1[np.where(b1 < 0)] + np.pi
   phi1[np.where((np.abs(b1) < epsilon ) & (c1 > 0))] = 0.5*np.pi
   phi1[np.where((np.abs(b1) < epsilon ) & (c1 < 0))] = -0.5*np.pi
  
   # reviewed this based on equations 
   f = (y - [a1])
   dis = rho1**2 - f**2
   m = (dis >= 0)
   
   kk = np.round((omega1 * x + phi1) / np.pi)
   Phi = np.zeros_like(y)
   Phi[m] = np.arctan(f[m] / np.sqrt(dis[m]) )
   
   theta = np.zeros_like(y)
   theta[m] = (-1)**kk[m]*Phi[m] + np.pi * kk[m]
   m = (dis < 0) & (y> a1) 
   theta[m] = (-1)**kk[m]*(0.5*np.pi) + np.pi * kk[m]
   m = (dis < 0) & (y < a1) 
   theta[m] = (-1)**kk[m]*(-0.5*np.pi) + np.pi * kk[m]
    
   
   m1 = [np.sum(x**2, axis=0),np.sum(x, axis=0)] 
   m2 = [np.sum(x, axis=0),num_samples*np.ones(N)] 
   M = np.array([m1, m2])
   
   v1 = np.array([np.sum(theta*x, axis=0),np.sum(theta, axis=0) ])
   
   z1 = np.asarray(list(map(lambda x: np.linalg.solve(M[:,:,x],v1[:,x]),j)))
   
   
   omega2, phi2 = z1[:,0],z1[:,1] 
   a2 = a1
   rho2 = rho1
   b2 = rho2*np.cos(phi2)
   c2 = rho2*np.sin(phi2)
   
   m1 = [num_samples*np.ones(N), np.sum(np.sin(omega2*x), axis=0),np.sum(np.cos(omega2*x), axis=0)] 
   m2 = [np.sum(np.sin(omega2*x), axis=0), np.sum(np.sin(omega2*x)**2, axis=0),np.sum(np.sin(omega2*x)*np.cos(omega2*x), axis=0)] 
   m3 = [np.sum(np.cos(omega2*x), axis=0), np.sum(np.sin(omega2*x)*np.cos(omega2*x), axis=0),np.sum(np.cos(omega2*x)**2, axis=0)] 
   M = np.array([m1, m2, m3])
   v1 = np.array([np.sum(y, axis=0),np.sum(y*np.sin(omega2*x), axis=0),np.sum(y*np.cos(omega2*x), axis=0) ])
   #z1 = np.linalg.solve(M,v1)
   #z1 = np.asarray(list(map(lambda x: np.linalg.solve(M[:,:,x],v1[:,x]),j)))
   #print(M[:,:,5])
   #determinants= np.asarray(list(map(lambda x:np.linalg.det(M[:,:,x]),j)))  
   #print(determinants)

   #l = np.shape(lost)[0]
   z1 = np.asarray(list(map(lambda x: np.linalg.solve(M[:,:,x],v1[:,x]),j)))
   #z1 = np.asarray(list(map(lambda x: np.linalg.pinv(M[:,:,x]).dot(v1[:,x]),j)))
  
   #z1 = np.asarray(list(map(lambda x: np.linalg.lstsq(M[:,:,x],v1[:,x])[0],j[determinants >  0])))
   #print(np.shape(z1),np.shape(z2), np.shape(z3))
   
   #determinants= np.asarray(list(map(lambda x:np.linalg.det(M[:,:,x]),j)))
   #cond= np.asarray(list(map(lambda x:np.linalg.cond(M[:,:,x]),j)))
   
   #dets = np.asarray(np.where(determinants >  0)[0])
   #not_dets = np.asarray(np.where(determinants <= 0)[0])
   #z1 = np.asarray(list(map(lambda x: scipy.linalg.solve(M[:,:,x],v1[:,x]),j)))
   #dets = np.asarray(np.where(determinants >  0)[0])
   #not_dets = np.asarray(np.where(determinants <= 0)[0])
   #z1 = np.asarray(list(map(lambda x: np.linalg.solve(M[:,:,x],v1[:,x]),j[determinants >  0])))
   #z2 = np.asarray(list(map(lambda x: np.linalg.lstsq(M[:,:,x],v1[:,x])[0],j[determinants <=  0])))
   #z2 = np.asarray(list(map(lambda x: np.linalg.pinv(M[:,:,x]).dot(v1[:,x]),j[determinants <=  0])))
   
    #z1 = numpy.linalg.pinv(M).dot(v1)
    
   #if (np.shape(z2)[0] > 0):
    # z = np.concatenate([z1, z2])
   
     #all_dets = np.concatenate([  np.ravel(dets), np.ravel(not_dets)] )
     #z1 = z[np.argsort(all_dets),:]
      
   #print(np.shape(z1))
   
   #print(lost)
   #print(not_dets)
   #print(np.where(np.isnan(determinants)))
   #stop
   #print(np.shape(z2))
   
   
   #rint("part3")
  #z2 = np.inner(np.linalg.inv(M),np.array(v1))
  #z3 = np.inner(v1,np.linalg.inv(M))
   a3, b3,c3 = z1[:,0],z1[:,1],z1[:,2]
   #print(np.where(a3 <0))
   
   rho3 = np.hypot(c3, b3)
   phi3 = np.arctan2(c3, b3)
   omega3 = omega2
   
   
   #phi3[np.where(b3 < 0)] = phi3[np.where(b3 < 0)] + np.pi
   
   
#   print("Found omega, a,b,c, rho, phi:",omega1, a1,b1,c1, rho1, phi1)
 #  stop
   #values = [omega2, a3,b3,c3, rho3, phi3]
   #print(np.shape(values))
   #print(np.shape(rho3))
   #print(np.shape(omega3))
   #print(np.shape(phi3))
   #print(np.shape(phi3))
   #print(np.shape(a3))

   
   #new_values2 = np.asarray([ values[:,4],values[:,0],values[:,5],values[:,1]])
   #df4 = pd.DataFrame(np.transpose(new_values2)  , columns=['a','b','c','k'] )  
   
   #for i in range(0,N): 
   #       print
   #print(omega2)
   #stop
   df4 = pd.DataFrame(np.array([rho3,omega3,phi3,a3]).transpose()  , columns=['a','b','c','k'] )          
   print(df4)        
   return(df4)  
  


#x = xss
#y = spec_y[interval,:]-y_fit[:,:]

def fit_sin_multi(x,y):
   x_new = x
   num_samples,N = np.shape(y)
   d = 0.5 * np.diff(x)
   x = np.transpose(np.asarray([x]*N))
   ins = np.zeros(N)
   S = np.cumsum((y[1:,:] + y[:-1,:]) * np.transpose([d]*N), axis=0)
   S = np.insert(S, 0,ins,0)
   SS = np.cumsum((S[1:,:] + S[:-1,:]) * np.transpose([d]*N), axis=0)
   SS = np.insert(SS, 0,ins,0)
   
   m1 = [np.sum(SS**2, axis=0),np.sum((x**2)*SS, axis=0),np.sum(x*SS, axis=0),np.sum(SS, axis=0) ]
   m2 = [np.sum((x**2)*SS,axis=0),np.sum(x**4, axis=0),np.sum(x**3, axis=0),np.sum(x**2, axis=0) ]
   m3 = [np.sum(x*SS, axis=0),np.sum(x**3, axis=0),np.sum(x**2, axis=0),np.sum(x, axis=0) ]
   m4 = [np.sum(SS, axis=0),np.sum(x**2, axis=0),np.sum(x,axis=0),num_samples*np.ones(N) ]
   M = np.array([m1, m2, m3, m4])
   
#   np.apply_along_axis(nplinalg.det, M,axis=2)
   
 #  np.apply_over_axes(np.linalg.det, M,[0,1])
   
   v1 = np.array([np.sum(y*SS,axis=0),np.sum(y*(x**2),axis=0),np.sum(y*x,axis=0),np.sum(y,axis=0) ])
   j = np.arange(N)  
   
   # determine if matrix is singular or not 
   determinants= np.asarray(list(map(lambda x:np.linalg.det(M[:,:,x]),j)))  
   dets = np.asarray(np.where(determinants >  0)[0])
   not_dets = np.asarray(np.where(determinants <= 0)[0])
   z1 = np.asarray(list(map(lambda x: np.linalg.solve(M[:,:,x],v1[:,x]),j[determinants >  0])))
   z2 = np.asarray(list(map(lambda x: np.linalg.lstsq(M[:,:,x],v1[:,x])[0],j[determinants <=  0])))
   if (np.shape(z2)[0] > 0):
     z = np.concatenate([z1, z2])
   
     all_dets = np.concatenate([  np.ravel(dets), np.ravel(not_dets)] )
     z1 = z[np.argsort(all_dets),:]
   
   #test = np.arange(5)
   #test2 = np.transpose(np.asarray([test]*6))
   #jj = np.arange(5) 
   
   #index = np.asarray(list(map(lambda x: test2[x,0] % 2 ,jj)))
   #det = np.asarray(np.where(index))
   #det_not = np.array(np.where(index==False))
   
  # z1 = test2[det,:][0]
   #z2 = test2[det_not,:][0]
   
   #z = np.concatenate([z1, z2])
   #dets = np.concatenate([  np.ravel(det), np.ravel(det_not)] )
   #z = z[np.argsort(dets),:]
   
   
    
   A,B,C,D= z1[:,0], z1[:,1],z1[:,2], z1[:,3] 
   omega1 = np.sqrt(-A)
   a1 = 2.*B/omega1**2
   b1 = (B*x_new[0]**2 + C*x_new[0]+D-a1)*np.sin(omega1*x_new[0])+(1./omega1)*(C + 2.*B*x_new[0])*np.cos(omega1*x_new[0])
   c1 = (B*x_new[0]**2 + C*x_new[0]+D-a1)*np.cos(omega1*x_new[0])-(1./omega1)*(C + 2.*B*x_new[0])*np.sin(omega1*x_new[0])
   rho1 = np.hypot(c1, b1)
   phi1 = np.arctan2(c1, b1)
#
   np.shape(a1)
   np.shape(b1)
   np.shape(c1)
   np.shape(omega1)
   np.shape(rho1)
   np.shape(phi1)
   #print("part2")
   f = y - [a1]
   dis = rho1**2 - f**2
   m = (dis >= 0)
   Phi = np.arctan(np.divide(f, np.sqrt(dis, where=m), where=m), where=m)
   np.copysign(0.5*np.pi, f, where=~m, out=Phi)
   #
   kk = np.round((omega1 * x + phi1) / np.pi)
   theta = (-1)**kk * Phi + np.pi * kk
   m1 = [np.sum(x**2, axis=0),np.sum(x, axis=0)] 
   m2 = [np.sum(x, axis=0),num_samples*np.ones(N)] 
   M = np.array([m1, m2])
   
   v1 = np.array([np.sum(theta*x, axis=0),np.sum(theta, axis=0) ])
   z1 = np.asarray(list(map(lambda x: np.linalg.solve(M[:,:,x],v1[:,x]),j)))
    #z2 = np.inner(np.linalg.inv(M),np.array(v1))
    #z3 = np.inner(v1,np.linalg.inv(M))
   omega2, phi2 = z1[:,0],z1[:,1] 
   a2 = a1
   rho2 = rho1
   b2 = rho2*np.cos(phi2)
   c2 = rho2*np.sin(phi2)
    #print("Found omega2, a,b,c, rho, phi:",omega2, a2,b2,c2, rho2, phi2)
   m1 = [num_samples*np.ones(N), np.sum(np.sin(omega2*x), axis=0),np.sum(np.cos(omega2*x), axis=0)] 
   m2 = [np.sum(np.sin(omega2*x), axis=0), np.sum(np.sin(omega2*x)**2, axis=0),np.sum(np.sin(omega2*x)*np.cos(omega2*x), axis=0)] 
   m3 = [np.sum(np.cos(omega2*x), axis=0), np.sum(np.sin(omega2*x)*np.cos(omega2*x), axis=0),np.sum(np.cos(omega2*x)**2, axis=0)] 
   M = np.array([m1, m2, m3])
   v1 = np.array([np.sum(y, axis=0),np.sum(y*np.sin(omega2*x), axis=0),np.sum(y*np.cos(omega2*x), axis=0) ])
   #z1 = np.linalg.solve(M,v1)
   #z1 = np.asarray(list(map(lambda x: np.linalg.solve(M[:,:,x],v1[:,x]),j)))
   #print(M[:,:,5])
   determinants= np.asarray(list(map(lambda x:np.linalg.det(M[:,:,x]),j)))  
   #print(determinants)
   dets = np.asarray(np.where(determinants >  0)[0])
   lost = np.asarray(np.where(np.isnan(determinants) )[0])
   not_dets1 = np.asarray(np.where(determinants <= 0)[0])
   l = np.shape(lost)[0]
   z1 = np.asarray(list(map(lambda x: np.linalg.solve(M[:,:,x],v1[:,x]),j[determinants >  0])))
   z2 = np.asarray(list(map(lambda x: np.linalg.lstsq(M[:,:,x],v1[:,x])[0],j[determinants <=  0])))
   z3 = np.zeros(l*3).reshape(l,3)
   #print(np.shape(z1),np.shape(z2), np.shape(z3))
   print(np.shape(lost))

   if ((np.shape(z2)[0] > 0)):
     z = np.concatenate([z1, z2])
     dets = np.concatenate([  np.ravel(dets), np.ravel(not_dets) ] )
     z1 = z[np.argsort(dets),:]
   
   if (l > 0):
    # print("ok")
     #print(np.shape(z1))
     z = np.concatenate([z1, z3])
     all_dets = np.concatenate([  np.ravel(dets), np.ravel(lost) ] )
     z1 = z[np.argsort(all_dets),:]
     #print(np.shape(z1))
      
   
   
   #print(lost)
   #print(not_dets)
   #print(np.where(np.isnan(determinants)))
   #stop
   #print(np.shape(z2))
   
   
   #rint("part3")
  #z2 = np.inner(np.linalg.inv(M),np.array(v1))
  #z3 = np.inner(v1,np.linalg.inv(M))
   a3, b3,c3 = z1[:,0],z1[:,1],z1[:,2]
   rho3 = np.hypot(c3, b3)
   phi3 = np.arctan2(c3, b3)
#   print("Found omega, a,b,c, rho, phi:",omega2, a3,b3,c3, rho3, phi3)

   #values = [omega2, a3,b3,c3, rho3, phi3]
   #print(np.shape(values))
   print(np.shape(rho3))
   print(np.shape(omega2))
   print(np.shape(phi2))
   print(np.shape(phi3))
   print(np.shape(a3))
   
   
   
   #new_values2 = np.asarray([ values[:,4],values[:,0],values[:,5],values[:,1]])
   #df4 = pd.DataFrame(np.transpose(new_values2)  , columns=['a','b','c','k'] )  
   
   #for i in range(0,N): 
   #       print
   #print(omega2)
   #stop
   df4 = pd.DataFrame(np.array([rho3,omega2,phi3,a3]).transpose()  , columns=['a','b','c','k'] )          
   print(df4)        
   return(df4)  
  
  
# The root mean square is defined by:
#\sigma = \sqrt{\frac{1}{n} \sum_{k=1}^n \left( f(x_k) - y_k \right)^2}  
def RMS(y,y_model):
    sigma = np.sqrt((1.0/len(y))*np.sum((y-y_model)**2)) 
    return sigma
    
    
def Pearsons(y,y_model):
    sigma = np.sum((y-y_model)**2/y_model)
    return sigma
    
def linear_baseline_fit(spec_x,spec_y,interval, widest_interval):
    # prepare arrays for fit             
    X,Y = np.asarray([spec_x[interval],np.ones(len(spec_x[interval]))]),np.transpose(spec_y[interval,:])
    # get beta for linear model, Y = beta_0 + beta_1*X 
    beta4 = np.matmul(np.linalg.pinv(np.matmul(X,X.T)),np.matmul(X,Y.T))

  #  print("y",np.shape(Y))
   # print("spec y",np.shape(spec_y))
  #  print("beta",np.shape(beta4))
  #  print("spec X",np.shape(X))
    
    # get the X for the entire interval (not just the segments)
    X =  np.asarray([spec_x[widest_interval],np.ones(len(spec_x[widest_interval]))])
    # get the fit of Y for the entire interval (not just the segments)
    y_fit = np.matmul(beta4.T,X)
    
   # print("X",np.shape(X))
  #  print(max(spec_x[widest_interval]),min(spec_x[widest_interval]))
    
  #  print("fit y",np.shape(y_fit))
    
    return y_fit
    
###########################################################
# FIND CURVED BASELINE 
#############################################################
# lam: larger values allows stronger deviation from original curve 
#   1e5 follows curve,
#   1e10 is straight line)
# ratio:
#            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
def curved_baseline_fit(spec_x,spec_y,widest_interval, dep):
     if dep == "Host":
         rep = 20
         ratio = 1e-5
         lam = 1e7
         niter= 5
         
         final_ratio = 1e-6
         final_lam = 1e7
         final_niter= 10

     if dep == "Trace":
         rep = 10
         ratio = 0.01
         lam = 1e6
         niter= 10
         
         final_ratio = 0.001
         final_lam = 1e6
         final_niter= 10
        
     if dep == "Fringes":
         rep = 5
         ratio = 0.5
         lam = 1e5
         niter= 100
         
         final_ratio = 0.5
         final_lam = 1e5
         final_niter= 100
     
     y =spec_y[widest_interval,:]
     
     for i in range(1,rep):
         y_fit = np.transpose(baseline_arPLS_vector(np.transpose(y), ratio=ratio, lam=lam, niter=niter, full_output=False))
         y = np.min([y,y_fit], axis=0)
         y[y < 0] = 0.0
         if np.array_equal(y_fit,y):
             break
     y_fit = np.transpose(baseline_arPLS_vector(np.transpose(y), ratio=final_ratio, lam=final_lam, niter=final_niter, full_output=False))
     return y_fit


#    Inputs:
#        y:
#            input data (i.e. chromatogram of spectrum)
#        lam:
#            parameter that can be adjusted by user. The larger lambda is,
#            the smoother the resulting background, z
#        ratio:
#            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
#        itermax:
#            number of iterations to perform
#def baseline_arPLS_vector(y, lam=1e4, ratio=0.05, itermax=100):
def baseline_arPLS_vector(y,ratio=1e-6, lam=100, niter=10, full_output=False):
   #test = mid_interval[interval]
   #print(np.shape(test))
   y = np.transpose(y)
   S = np.shape(y)
   L, N = S[0], S[1] 
   print("The implementation of ARPLS with find {} fits. Each of length {}".format(N,L))

   #print(S)
   #print(len(interval))
   D = sparse.csc_matrix(np.diff(np.eye(L), 2))
   Ds  = D.dot(D.transpose())
  
   Ds = np.asarray(repmat(Ds,1,N))   # shape (0xN) each matrix shape(L x L-2)
   w = np.ones(L*N).reshape(L,N)
   H = lam * Ds
   #print(S)
   #print(np.shape(w))
   #print(np.shape(interval))
   #print(np.shape(mid_interval))
   
   
   #w[test,:] = 0
   
   W =   np.asarray(list(map(lambda n: sparse.spdiags(n, 0, L, L) , w.T)))
   #print(np.shape(W))
   #print(np.shape(w.T))
   #D = np.diag(w.T) 
   
   crit = 1
   count = 0
   while crit > ratio:
   
        #z = linalg.spsolve(W + H, W * y)
        Z = np.asarray(W + lam * Ds)
        z =   np.asarray(list(map(lambda x,y: sparse.linalg.spsolve(x, y), Z[0], np.transpose(w*y))))
        
       # print(np.shape(z),np.shape(y),np.shape(z))
        d = y - np.transpose(z)
        m = np.mean(d*(d<0), axis=0)
        s = np.std(d*(d<0), axis=0)
       
        w_new = 1 / (1 + expit(2 * (d - (2*s - m))/s))
        
        crit = norm(w_new - w) / norm(w)

        w = w_new
        
        # set the diagonals by creating a new sparse matrix
        W =    np.asarray(list(map(lambda n: sparse.spdiags(n, 0, L, L) , w_new.T)))
  
        count += 1

        if count > niter:
            print('Maximum number of iterations exceeded')
            break

   if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
   else:
        return z

##################################################
# TURN A LIST OF TUPLES INTO A SQUARE ARRAY
###############################################
# y is spectral data
# dic_loc is dictionairy with the locations (list of list) 
# dic is dictionairy with the values (list of list) 
# interval is the interval of interest
def turn_into_square(y, dic_loc,dic, interval, spec_x):
     debug= False
     num =     np.shape(y)[1]
     # create float array for the values 
     peak_array = np.zeros(y.size, dtype=float).reshape(np.shape(y))

     # create float array for the values 
     peak_bool = np.zeros(y.size, dtype=bool).reshape(np.shape(y))

     mini = np.min(np.where(interval))

     # get number of peaks 
     number_of_peaks = np.asarray(list(map(lambda x: len(x), dic_loc.iloc[:])))
     
     if debug:
       print("------------------turn into square ----------------------------")
       print(np.shape(y))
       print(number_of_peaks)
    
     # get the coordinates of the peaks (location of spectra and location within spectra )
     index_peak_1 = np.concatenate(np.asarray(dic_loc.iloc[:])) + mini
     index_peak_2 = np.repeat(np.where(number_of_peaks >= 0),number_of_peaks[number_of_peaks >= 0])

     #peak_array = map(lambda peak_bool
     peak_bool[index_peak_1,index_peak_2] = True
     peak_array[index_peak_1,index_peak_2] = y[index_peak_1,index_peak_2]
    
     if debug:
       for i in range(0,num):
         print("dic-locations", dic_loc[i]+mini)
         print("peak array-locations",np.where(peak_array[:,i] > 0)) 
       for i in range(0,num):
         print("dic", dic[i])
         l = number_of_peaks[i]
         print(peak_array[:,i][peak_bool[:,i] >0]) 
     return peak_array, number_of_peaks ,index_peak_1 ,index_peak_2
