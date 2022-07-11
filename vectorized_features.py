import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import math_support                               # my function using all the math and stats 

from scipy.signal import find_peaks, peak_prominences              #  for locating the water-peaks 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.matlib import repmat as repmat
from scipy.fftpack import fft, fftfreq, fftshift, ifft
from scipy.optimize import curve_fit as scipy_optimize_curve_fit


#   - Implement Profiles for different type of:
#                      - Host Minerals(Olivine, Quartz, Clinopyroxene, Plagioclase) 
#                      - Melt compositions (Basalt, Andesite, Rhyolite)
#                      - Trace Chemicals (H2O, CO2, OH, etc)
#                      - Wavelength information for every component    

    # trace specify Type Either: Host, Melt or Trace 
    # traces for each elementL - contains information on chemical element  
    # traces                   - contains wavenumber interval for peak (Upper and Lower)    
    # traces                   - define either straight line or bend baseline fit: Straight or Bend
    # traces                   - lower interval for Straight baseline (Upper and Lower) 
    # traces                   - upper interval for Straight baseline (Upper and Lower) 
    # traces                   - minimal peak height, must always be higher than fringes amplitude


# recalc units (wave number are cm^-1)
      # the wavenumbers are in x/cm^-1
      # so the can be express in wavelength by 
      #so, 2000 cm^-1 is 5 micro = 5000 nm 
      # (1/2000)*(1/100)*1e6 = 5.0 micron 
      #0.00050
      # water peak at 3550 cm-1 is found at 
      # (1/3550)*(1/100)*1e6 = 2800 nm approx 2.734 micron
      
      #calculate_thickness(2, 1.546,2121.073,1991.505)  =50 micro meter
      

#########################################################
# This is where we detect the fringes
# 1. check if fringes exist
# 2. see if we can fit fringes (old school way, yes, we can about 50% of the time
# 3. see if we can find fringes (FFT method) [no, parameters obtain from FFT are not good enough]
# 4. see if we can fit fringes with the linear algebra method  [no, parameters obtain from LA method not good enough]
# 5. Known issues: Occasional horrible fit, or bad error estimate
###########################################################
def detect_fringes(*args,**kwargs):
    testing = False
    print("------------ STARTING DETECT FRINGES--------------------------------")
    spec_x = args[0]
    spec_y = args[1]
    now, nos = np.shape(spec_y)
    folder = kwargs.get("folder")

    # using two intervals: 
    # wide inteval: for the baseline fit
    # narrow interval: for the actual wave-function fit 
    wide_interval = np.where((spec_x < 2350)  & (spec_x > 2000))[0] 
    interval = np.where((spec_x < 2250) & (spec_x > 2050))[0] 

    extra = "Fringes"
    
    # fit the curved baseline
    y_fit = math_support.curved_baseline_fit(spec_x,spec_y,wide_interval,extra)
    
    # now reduce wideinterval
    new_index = np.where((spec_x[wide_interval] < 2250) & (spec_x[wide_interval] > 2050))[0] 
    y_fit = y_fit[new_index]
    
    # select suitable interval 
    xs =  spec_x[interval]
    ys =  spec_y[interval,0]
    
    # convert from [/cm] to wavelengths [micro meter] 
    xss = (1./100.)*1e6/xs
    
    # fit the adapted vectorized version method from stack overflow.
    df3,var, guess_df = math_support.fit_sin2_multi(xss,spec_y[interval,:]-y_fit[:,:])
    
    # estimate the errors on the period 
    new_errors = np.asarray(list(map(lambda xx: np.random.normal(loc=0, scale=var[xx,1], size=1) ,np.arange(nos) ))).T
    
    # recreate fit 
    sup_ = np.outer(xss,df3.b.values+new_errors)+df3.c.values
    sup2_ = np.sin(sup_)
    y3 = df3.a.values*sup2_ + df3.k.values  
    
    ##################################################
    # Test quality of fit using single function to fit 1 set of sinodal data
    ###############################################
    if testing == True:
        def sinfunc(t, A, w, p, c): 
            return A * np.sin(w*t + p) + c
        for i in range(225):
          guess = guess_df.values[i,:]
          popt,pcov = scipy_optimize_curve_fit(sinfunc, xss, spec_y[interval,i]-y_fit[:,i],p0=guess,maxfev=200000,method='trf')
          
          fig, ax = plt.subplots()
          plt.plot(spec_x[interval], spec_y[interval,i]-y_fit[:,i],label="data")
          plt.plot(spec_x[interval], y3[:,i],label="vector")
          
          A, w, p, c = popt
          
          errors =  np.sqrt(np.diag(pcov))
          #print('i=',i)
          #print(popt)
          #print(errors)
          # sanity check
                    # santi check 
          #if i in np.array([  5,   7,   8,   9,  13,  14,  20,  22,  23,  24,  28,  29,  37,  38,  39,  43,  49,  51,  56,  58,  72,  78,  82,  94, 109, 129,169, 187, 214, 222, 223]):
           #   p2p = w*np.pi/1e4
              #print("peak 2 peak distance (new)", p2p) 
              
            #  peaks, _ = find_peaks(y3[:,i],height=0)
            #  print("peak 2 peak distance", peaks)
             # for j in peaks:
              #   this_x = spec_x[interval][j]
              #   print(this_x)
               #  plt.vlines(x = this_x, ymin = 0, ymax = np.max(y3[:,i]),colors = 'purple')
              
              #l = len(peaks)-1
              #refraction = 1.6445166299999998
            #  refraction=  1.706244789243518
            #  dist = max(spec_x[interval][peaks])-min(spec_x[interval][peaks])
              #print("peak 2 peak distance (old)",l/dist)
             # print("thickness is",(0.5*l/(refraction*dist))*1e4)
              #plt.plot(spec_x[interval][peaks],y3[:,i][peaks],'*')
              
              
              
          
          
          test = sinfunc(xss, A, w, p, c)
          plt.plot(spec_x[interval], test+0.005,label="single")
          ax.invert_xaxis()
          ax.legend()
          fig.savefig(''.join([folder,'/Graph_of_fringes_plus_fit_plus_test',str(i) ,'.png']), bbox_inches='tight')
          plt.close(fig) 
          

    ################################################################################
    # Create output and calculate R-score 
    ################################################################################
    # for fit:  a* np.sin(b*t + c) + k
    # get the Peak-to-Peak distance 
    # I dont know if I should multiply by pi
    peak2peak = (df3.b.values+new_errors)*np.pi
    
    # get R-score
    R2_values, fringes_bool = math_support.calc_Rscore(spec_y[interval,:]-y_fit[:,:],y3) 
    
    # curve correct for interval
    curve_corrected_section = spec_y[interval,:]-y_fit[:,:]
    fit_section = y3
    print("------------ FINISHED FRINGES--------------------------------")
    return R2_values, fringes_bool,peak2peak, curve_corrected_section,fit_section


#########################################################
# This is where we detect the fringes
# 1. check if fringes exist
# 2. see if we can fit fringes (old school way, yes, we can about 50% of the time
# 3. see if we can find fringes (FFT method) [no, parameters obtain from FFT are not good enough]
# 4. see if we can fit fringes with the linear algebra method  [no, parameters obtain from LA method not good enough]
###########################################################
def detect_fourier(*args,**kwargs):
    print("--------------------------------------------")
    spec_x = args[0]
    spec_y = args[1]
    num = np.shape(spec_y)[1]
  
    # determine noise level from early interval < 3800
    noise_index = np.where(spec_x > 3800)
    noise = np.std(spec_y[noise_index,:], axis=1)[0]
    
    wide_interval = np.where((spec_x < 2350)  & (spec_x > 2000))[0] 
    
    interval = np.where((spec_x < 2250) & (spec_x > 2050))[0] 

    extra = "Fringes"
    
    y_fit = np.transpose(math_support.curved_baseline_fit(spec_x,spec_y,wide_interval,extra))
    
    # now reduce wideinterval
    new_index = np.where((spec_x[wide_interval] < 2250) & (spec_x[wide_interval] > 2050))[0] 
    y_fit = y_fit[new_index]
    
    widest_interval = np.where((spec_x < 3500)  & (spec_x > 2100))[0] 
    y_fit_all = np.transpose(math_support.curved_baseline_fit(spec_x,spec_y,widest_interval,extra))
    
    #_, spectra_arPLS, info = baseline_arPLS(spec_y, lam=1e5, niter=100, full_output=True)
    # now trying to find a fitting wavefunction
    
    print("interval", np.shape(interval))
    print("y_fit", np.shape(y_fit))
    
    # select suitable interval 
    xs =  spec_x[interval]
    ys =  spec_y[interval,0]
    
    # convert from [/cm] to wavelengths [micro meter] 
    xss = (1/100)*1e6/xs
    
    # fit the adapted vectorized version method from stack overflow
    df3,var, guess_df = math_support.fit_sin2_multi(xss,spec_y[interval,:]-y_fit[:,:])
    sup_ = np.outer(xss,df3.b.values)+df3.c.values
    sup2_ = np.sin(sup_)
    y3 = df3.a.values*sup2_ + df3.k.values  
    
    # get R-score
    R2_values, fringes_bool = math_support.calc_Rscore(spec_y[interval,:]-y_fit[:,:],y3)
    
    for j in range(num):
      print("****************************",j)
      dpi = 60
      peaks1, _ = find_peaks(y3[:,j])
      
      fig, axs = plt.subplots(2,3,dpi=200)
      axs[0,0].plot(spec_x, spec_y[:,j], linewidth=1);
      axs[0,0].plot(spec_x[interval], y_fit[:,j],linestyle='dotted', linewidth=1,label='Baseline');
      
      axs[0,1].plot(spec_x[interval], spec_y[interval,j]-y_fit[:,j],'.', linewidth=1,label='Corrected Data ');
      axs[0,1].plot(spec_x[interval], y3[:,j],linestyle='dashed', linewidth=1,label='y3');
      axs[0,1].plot(spec_x[interval][peaks1], y3[:,j][peaks1],'x', linewidth=1,label='peaks');
      
      textstr2 = '\n'.join((r'$R-Score: %.3f$' % (R2_values[j], ) , r'$Accepted: %s$' % (fringes_bool[j], )))
      #print(R2, R2_values[j])
      
      axs[0,1].text(0.05, 0.05, textstr2 , fontweight="bold", transform=axs[0,1].transAxes)   
    
      #axs.legend(loc='upper right')
      #axs.invert_xaxis()
     #ax.set_ylim(-0.1, np.max(spectra_arPLS)*1.1+0.05)
       # only interest interval from 3800-1300
      #ax.plot([1950, 2250],[0.0,0.0],linewidth=1,color='black')
      #axs.legend(loc='upper left')
     #add_interval(fig, ax)
      #plt.xlabel("Wavenumber $cm^{-1}$")
      ##plt.ylabel("Absorbance")
      
      # generate FFT from data AND fit t4 
      f1 = spec_y[interval,j]-y_fit[:,j]
      f2 = y3[:,j]
      x2 =spec_x[interval]
    
      # plot wide corrected interval 
      axs[1,0].plot(spec_x[widest_interval],spec_y[widest_interval,j] - y_fit_all[:,j],linestyle='dotted', linewidth=1,label='Baseline');

      # FIT FROM FFT /get fourier info to generate plot (fits from FFT parameters are not good)
      z,xplot,yplot =  math_support.fourier_fit(xss, f1)
      A, w,p, c = z[0], z[1], z[2], z[3]
      #temp = math_support.sinfunc(xss, A, w, p, c
      #axs[0,1].plot(spec_x[interval], temp, linewidth=1,label='FFT fit Data')
      
      axs[0,2].stem(xplot,yplot,use_line_collection=True)
      axs[0,2].set_xlim(0, 15)
      
      
      # print the peak-peak distance based on data, fit and FFT
      if len(peaks1) > 0:
        t =   math_support.calculate_thickness(len(peaks1)-1, 1.546,max(spec_x[interval][peaks1]),min(spec_x[interval][peaks1]))
      # 2 methods to calculate peak 2 peak distance: 
      #  - use the peak locations from the function generate using the fitting functions 
      #  - use the angle from the fit to calculate this quantity 
        print(" Thickness 2 [micro meter] ", t)
        print(" Peak-Peak Distance [micro meter]         {0}".format((1e6/100)*(len(xs[peaks1])-1) /(max(xs[peaks1]) - min(xs[peaks1]))))
        print(" Peak-Peak-Distance [micro meter] (Angle) {0} ".format(df3.b.values[j]*np.pi))
        print(" Peak-Peak-Distance [micro meter] (FFT)   {0}   ".format(w*np.pi))
        print(" Amplitude(Max-Min)       {0}".format(max(y3[:,j])-min(y3[:,j])))
        print(" Amplitude                {0}".format(2.0*df3.a.values[j]))
        print(" Amplitude [FFT]          {0}".format(2.0*A))
        print(" Noise          {0}".format(noise[j]))
      else:
        print("No peaks")
      #plt.show()
      #plt.close
      
      widest_interval = np.where((spec_x < 3500)  & (spec_x > 1450))[0] 
      y_fit_all = np.transpose(math_support.curved_baseline_fit(spec_x,spec_y,widest_interval,extra))
      xs2 =  spec_x[widest_interval]
      xss2 = (1/100)*1e6/xs2
      sub_x = xss2
      sub_y =  spec_y[widest_interval,j] - y_fit_all[:,j]
      
      #FIT FROM FFT /get fourier info to generate plot
      z,xplot,yplot =  math_support.fourier_fit(sub_x, sub_y)
      A, w,p, c = z[0], z[1], z[2], z[3]
      print(" Peak-Peak-Distance [micro meter] (all FFT )   {0}   ".format(2*np.pi*w))
      axs[1,1].stem(xplot,yplot,use_line_collection=True)
      axs[1,1].set_xlim(0, 20)
      #plt.xlim(0, 20)
      
      axs[1,2].plot(spec_x[widest_interval],spec_y[widest_interval,j],linestyle='dotted', linewidth=1,label='Data');
      axs[1,2].plot(spec_x[widest_interval],y_fit_all[:,j],linestyle='dotted', linewidth=1,label='Baseline');

      plt.show()
      
      #plt.show(block=False)
      #plt.pause(1.0)
      plt.close()
      
      ###########################################
      # FFT ON SLIDING WINDOWS 
      ########################################
      sliding_windows = True
      if sliding_windows:
        # get a sliding windows for 200 points 
        interval_size = len(widest_interval)
        print("Widest interval", len(widest_interval))
        window_size = 200
        step_size = 100
      
        number_of_step = round((interval_size- window_size)/step_size)

        store_FFT_wavelengths = np.zeros(number_of_step)
        store_FFT_wavelengths2 = np.zeros(number_of_step)
      
        # print the limits of the wave-length
        #minis = (1/min(spec_x[widest_interval]))*(1/100)*1e6
        #maxis = (1/max(spec_x[widest_interval]))*(1/100)*1e6
        #print("Wavelengt limits [$\mu$m]", minis, " " ,maxis)
      
        # data manual obtained from Pollock, 1973, Optical properties of some terrestrial rocks and glasses
        lambda_ = np.array([2.00 ,2.50, 3.00, 4.00,5.00,6.00, 6.50,7.0])
        n_     =  np.array([1.54, 1.53, 1.52, 1.50,1.46,1.40, 1.36,1.30])
        w1 = 1
        for i in range(number_of_step):
           sub_interval =  np.arange(200)+i*step_size
           sub_x = xss2[sub_interval]
           #print(np.shape(sub_x))
           #print(np.shape(y_fit_all[sub_interval,j]))
           sub_y =  spec_y[widest_interval,j][sub_interval] - y_fit_all[sub_interval,j]
           #y_fit_all[sub_interval,j]
           #print(i*step_size)
           #print(np.shape(sub_x))
           #print(np.shape(sub_y))
                 # plot wide corrected interval 
           #FIT FROM FFT /get fourier info to generate plot
           z,xplot,yplot =  math_support.fourier_fit(sub_x, sub_y)
           A, w,p, c = z[0], z[1], z[2], z[3]
           temp = math_support.sinfunc(sub_x, A, w, p, c)
           
           # find closest wavelengt for interval
           wl = np.mean(sub_x)
           index = np.abs(lambda_ - wl) == min(np.abs(lambda_ - wl))
           refrac = n_[index][0]
           #print(" w [micro meter] (sliding window FFT )   {0}   ".format(w))
           
           print(" w [micro meter] (sliding window FFT )   {0}   ".format(w1/w))
           w1 =w
           tt = sub_x[1:]-sub_x[:-1]
           
           plt.stem(xplot,yplot,use_line_collection=True)
           plt.xlim(0, 20)
           plt.ylim(0,0.01)
           
           textstr2 = '\n'.join((r'$Interval: %.3f:%.3f$' % (min(sub_x),max(sub_x)) ,r'$%.3f $' % (max(sub_x), )))
           plt.text(0.85, 0.85, textstr2 , fontweight="bold", transform=axs[1,0].transAxes)  
           
           #get peaks 
           plt.show(block=False)
           plt.pause(0.5)
           plt.close()
           #plt.plot(sub_x, sub_y,'.')
           #plt.plot(sub_x, temp,'.')
           
           #plt.show()
          # plt.close('all')
           store_FFT_wavelengths[i] = w*2.0*np.pi 
           store_FFT_wavelengths2[i] = np.mean(sub_x)
    # return the R-score and the booleans. 
    
    # curve correct for interval
    curve_corrected_section = spec_y[interval,:]-y_fit[:,:]
    fit_section = y3
    return R2_values, fringes_bool,curve_correct_section,fit_section



#########################################################
# This is where the criteria for the traces are defined
# try to determine boolean status: suitable, melt, ..etc.
# 1. Multiple high peaks 
# 2  Peaks within the correct interval
###########################################################
# Trace_criteria(spec_y,peak_vals, interval,number_of_peaks)
def Trace_criteria(*args,**kwargs):
    spec_y = args[0]
    peak_vals = args[1]
    peak_interval  = args[2]              # peak interval/mid_interval
    number_of_peaks = args[3]
    widest_interval  = args[4]
    p2p =args[5]
    spec_x = args[6]
    below_baseline = args[7]
    booleans = args[8]
    y_fit= args[9]
    debug_mode = kwargs.get("debug_mode")    
    mini = np.min(np.where(widest_interval))
    
    
    # set local variables 
    min_relative_peak_value = 0.02

    num = np.shape(spec_y)[1]            # number of spectra
    length = np.shape(spec_y)[0]        # length of spectra

    # set the 2D index for the peak locations 
    peak_boolean = [peak_vals > 0][0]
    
  
    # to check if the detect peaks fall within the interval defined in the profile  
    # we take the logical AND of the profiles peaks wavelengths and the detect wavelengths 
    mid_interval = np.logical_or.reduce(peak_interval, axis=0)
    temp = [mid_interval]*num
    temp2 = np.transpose(temp) & peak_boolean
    
    
    help_index1, help_index2= np.where(peak_boolean)
    
    # need to find the wavelength asociated with peaks 
    # if peak locations are in the right interval and high enough  
    relative_peak_val = spec_y[widest_interval,:] - y_fit
    relative_peak_val[~temp2[widest_interval,:]] = 0    
    relative_peak_val[np.where(relative_peak_val <  min_relative_peak_value)] = 0
    
    # get the highest of the host peaks 
    booleans['Trace'] = np.max(relative_peak_val > 0, axis=0)
    booleans["Cont"] = booleans["Melt"] & booleans["Host"]
    
    # get which peaks are in the mid_interval 
    help_index1 = np.where(temp2 > 0)
    if debug_mode:
        print("--------------------------------------------")
        print("Running Trace Criteria")
        print(np.shape(peak_boolean))
        print(help_index1, help_index2)
        print(peak_vals[help_index1])
    
        # print values with loop to check if I get the correct values 
        for i in range(0,num):
          help_index1 = np.where(peak_boolean[:,i])
          for j in range(0, np.shape(help_index1)[1]):
            if (spec_x[help_index1[0][j]] in spec_x[mid_interval]) & (relative_peak_val[help_index1[0][j]-mini,i] > 0.1): 
               #print("i: {0} x {1}".format(i,spec_y[help_index1[0][j],i]))
             
              print("i: {0} x {1}".format(i,relative_peak_val[help_index1[0][j]-mini,i]))
    return booleans



#########################################################
# This is where the criteria for the host are defined
# try to determine boolean status: suitable, melt, ..etc.
# 1. Multiple high peaks (within  3 interval
# 1. a. Are the peaks at the right locations
# 1. b. Are the peaks high enough  
###########################################################
# Melt_criteria(spec_y,peak_vals, interval,number_of_peaks)
def Host_criteria(*args,**kwargs):
    print("--------------------------------------------")
    spec_y = args[0]
    peak_vals = args[1]
    peak_interval  = args[2]              # peak interval/mid_interval
    number_of_peaks = args[3]
    widest_interval  = args[4]
    p2p =args[5]
    spec_x = args[6]
    below_baseline = args[7]
    booleans = args[8]
    y_fit= args[9]
    debug_mode = kwargs.get("debug_mode")    

    mini = np.min(np.where(widest_interval))
    
    # set local variables 
    min_relative_peak_value = 0.05

    num = np.shape(spec_y)[1]            # number of spectra
    length = np.shape(spec_y)[0]        # length of spectra

    # set the 2D index for the peak locations 
    peak_boolean = [peak_vals > 0][0]
    
    # get the maxium values and their associated indices 
    #max_index = np.argmax(spec_y[widest_interval], axis=0)
    #max_val = np.amax(spec_y[widest_interval], axis=0)

    # reduce the peak_interval array to one flat array
    # this array contains all wavelenghts associated with the host-peaks discribed in the profile
    mid_interval = np.logical_or.reduce(peak_interval, axis=0)
  
    # to check if the detect peaks fall within the interval defined in the profile  
    # we take the logical AND of the profiles peaks wavelengths and the detect wavelengths 
    temp = [mid_interval]*num
    temp2 = np.transpose(temp) & peak_boolean
  
    #print("interval",spec_x[mid_interval])
    #print(np.shape(temp2))
  
    # get which peaks are in the mid_interval 
   # help_index1 = np.where(temp2 > 0)
    #print(peak_vals[help_index1])
    
    # print values with loop to check if I get the correct values 
    #for i in range(0,num):
     #  help_index1 = np.where(peak_boolean[:,i])
      # for j in range(0, np.shape(help_index1)[1]):
       #   if spec_x[help_index1[0][j]] in spec_x[mid_interval]:
        #     print("i: {0} x {1}".format(i,spec_y[help_index1[0][j],i]))
       
    
    
    
    #print(spec_x[help_index1])
    #print(spec_y[temp2])
    
    #peak_vals = spec_y.copy() 
    #peak_vals[ ~temp2] = 0
    #print(peak_vals[temp2])
    #print(spec_y[temp2])

    # take a subset of the relative peaks that are within the peak intervals
    # 1. take the subset to calculate the relative peak height
    # 2. set all peaks outside the correct wavelenght range to zero
    # 3. set all peak height lower than threshold to zero
    relative_peak_val = spec_y[widest_interval,:] - y_fit
    relative_peak_val[~temp2[widest_interval,:]] = 0    
    #else:
    print(np.shape(y_fit))
    print(np.shape(temp2))
    print(np.shape(relative_peak_val))
    print(np.shape(widest_interval))
     # relative_peak_val = spec_y[widest_interval] - y_fit
    #  relative_peak_val[~temp2[widest_interval]] = 0
    relative_peak_val[np.where(relative_peak_val <  min_relative_peak_value)] = 0
    
    # get the highest of the host peaks 
    booleans['Host'] = np.max(relative_peak_val > 0, axis=0)
    booleans["Cont"] = booleans["Melt"] & booleans["Host"]
    
    #print(booleans["Melt"])
    #print(booleans["Suitable"])
    #print(booleans["Host"])
    #print(booleans["Host"][0])
    
    # get which peaks are in the mid_interval 
    help_index1 = np.where(temp2 > 0)
    print(peak_vals[help_index1])
    
    # print values with loop to check if I get the correct values 
    if debug_mode:
      for i in range(0,num):
         help_index1 = np.where(peak_boolean[:,i])
         for j in range(0, np.shape(help_index1)[1]):
            if (spec_x[help_index1[0][j]] in spec_x[mid_interval]) & (relative_peak_val[help_index1[0][j]-mini,i] > 0.1): 
             #print("i: {0} x {1}".format(i,spec_y[help_index1[0][j],i]))
             
             print("i: {0} x {1}".format(i,relative_peak_val[help_index1[0][j]-mini,i]))
             
             #relative_peak_val

       
         print("Host value i:  {0}, bool: {1}".format(i,booleans['Host'][i]))      
    
    #z = spec_x[np.where(spec_x[peak_boolean[:,0] ] in spec_x[mid_interval] )]  
    #print(z)
    #print(spec_x[peak_boolean[:,0]] in spec_x[mid_interval])
    
    #print(" * * * ", spec_x[peak_boolean[:,1]==True ] in spec_x[mid_interval])
    #for i in range(num):
     #   z = np.apply_along_axis(np.where,0,spec_x[peak_boolean[:,i]==True ] in spec_x[mid_interval])
      #  print(z)
     
     #   help_index1 = np.where(peak_boolean[:,i]) 
      #  print(spec_x[help_index1])
       # print(spec_y[help_index1,i])

     #  print(np.where(spec_x[help_index1,i] in spec_x[mid_interval]))
    #   print("True",spec_x[peak_boolean[:,i]])
    #   print(np.apply_along_axis(np.where,0,spec_x[peak_boolean[:,i]] in spec_x[mid_interval]))
       #print(np.where(spec_x[peak_boolean[:,i]] in spec_x[peak_interval]))
    # now find a way to check if they are in the right interval 
    # Oli intervals = np.array([[1650,1675], [1765, 1785], [2000,2020]])


    
    # make an array with stored peak location set every non-peak location to zero 
    #peak_vals = spec_y[widest_interval,:] 
    #peak_vals[ ~peak_boolean[widest_interval,:]] =0
    #print(peak_vals[peak_boolean[widest_interval,:]])
    #get the max for each spectrum
    #max_peaks_host = np.max(peak_vals, axis=0)
    #print(max_peaks_host)    
    

    return booleans


##########################################################
# This is where the criteria for the melts are defined
# try to determine boolean status: suitable, melt, ..etc.
# 1. Multiple high peaks (outside main interval (3510-3590)) = unsuitable
# 1. if there are multiple peaks large peaks in melt-interval
# 2.
# now determine werther this sample is suitable based on water line based on a few criteria
# 1. the curve is always above the fitting line in interval (water/no water)
# 2. curve has multiple extremly high peaks  (suitable/unsuitable)
# 3. curve has multiple small peak below treshold  (water/no watter)
# 4. curve has maximum outside proper range (water/no water)
###########################################################
# Melt_criteria(spec_y,peak_vals, interval,number_of_peaks)
def Melt_criteria(*args,**kwargs):
    print("--------------------------------------------")
    spec_y = args[0]
    peak_vals = args[1]
    interval  = args[2]              # peak interval/mid_interval
    number_of_peaks = args[3]
    widest_interval  = args[4]
    p2p =args[5]
    spec_x = args[6]
    below_baseline = args[7]
    booleans = args[8]
    debug_mode = kwargs.get("debug_mode")    
    # set local variables 
    min_relative_peak_value = 0.2
    num = np.shape(spec_y)[1]            # number of spectra
    length = np.shape(spec_y)[0]        # length of spectra
    index = np.arange(length)
    #booleans = np.zeros(num, dtype=[('Melt',bool),('Suitable',bool)])
  
    # set the 2D index for the peak locations 
    peak_boolean = [peak_vals > 0][0]
    
    # get the maxium values and their associated indices 
    max_index = np.argmax(spec_y[widest_interval], axis=0)
    max_val = np.amax(spec_y[widest_interval], axis=0)

    # make an array with stored peak location set every non-peak location to zero 
    peak_vals = spec_y[widest_interval,:] 
    peak_vals[ ~peak_boolean[widest_interval,:]] =0

    # print the peak values (allows to find the highest peak in every interval in case of multiple peak)
#    print("-- Peaks --:",spec_y[widest_interval,:][peak_boolean[widest_interval,:]])
#    print("-- Maxi Peaks --:", np.amax(peak_vals, axis=0))
    print("-- Location Peaks --:", np.argmax(peak_vals, axis=0))
    print("-- Maximum Peaks to Peak distance --:", p2p)
    print("nunber of peaks", number_of_peaks)
    print(np.where(number_of_peaks ==1))
    
    
    # if number of peaks in the melt-interval is zero there is no basalt 
    booleans['Suitable'][np.where(number_of_peaks == 0)] = True
    booleans['Melt'][np.where(number_of_peaks == 0)] = False
    
    # if number of peaks in the melt-interval is 1, there is basalt and the sample is considered suitable 
    # distance to near peak 
    booleans['Suitable'][np.where(number_of_peaks == 1)] = True
    booleans['Melt'][np.where(number_of_peaks == 1)] = True
  
    # if number of peaks in the melt-interval is 1, but the peak-to-peak distance is too large 
    #print("Type p2p:",type(p2p))
    #print("Type p2p:",type(number_of_peaks))

    #print("Type p2p:",p2p)
    #print("Type p2p:",number_of_peaks)

    
    booleans['Suitable'][np.where((number_of_peaks == 1) & (p2p > 50))] = False
  
    # if there are multiple peaks,all in close range to each other  
    booleans['Suitable'][np.where((number_of_peaks > 1) & (p2p < 50))] = True
    booleans['Melt'][np.where((number_of_peaks > 1) & (p2p < 50))]  = True
  
  
    # if the peaks/maximum is outside the range associated with the melt
    #print(spec_x[index[interval]])
    #print(max_index)
    if num > 1: 
      booleans['Melt'][  ((max_index < min(index[interval])) | (max_index > max(index[interval])))] = False
    
    #booleans['Melt'][np.where(   )  ]
     #   if sum(y_int) < sum(flat_baseline):
      #  booleans['Suitable'] = False
       # booleans['Basalt'] = None
        #print("    Curves lies under curve")
   # if booleans['Suitable']== False:     
    #    print(booleans['Suitable'])
     #   stop
    return booleans
    
 #  profiles = np.zeros(6, dtype=[('Type',str),('Name',str),('Peak',float), ('Peak Upper',float),('Peak Lower', float),('Baseline', str),('Left Interval Lower',float),('Left Interval Upper',float),('Right Interval Lower',float),('Right Interval Upper',float)])
  # profiles[0] = ({'Type':'Melt','Name':'Basalt','Peak':0.2,'Peak Upper':0,'Peak Lower':0,'Baseline':'Straight','Left Interval Lower':3725,'Left Interval Upper':3850,'Right Interval Lower':2450,'Right Interval Upper':2550 })
def set_profiles():
   profiles = []
   # determine if any peaks are detect and if peaks are detect they are high enough:
   # Water wavelength / Basalt peak 3510 - 3590  ( better to use a wider version )
   # Oli intervals = np.array([[1650,1675], [1765, 1785], [2000,2020]])
   # CO2 wavelength 1515 +/ 20 
   # CO2 wavelength 1430 +/ 20 
   # Every row is profiles of a peak.  

   profiles = [
   {'Type':'Melt','Name':'Basalt','Peak':0.2,'Peak Upper':3650,'Peak Lower':3450,'Baseline':'Straight','Left Interval Lower':3725,'Left Interval Upper':3850,'Right Interval Lower':2450,'Right Interval Upper':2550 },
   {'Type':'Host','Name':'Olivine','Peak':0.2,'Peak Upper':1675,'Peak Lower':1650,'Baseline':'Bend','Left Interval Lower':2050,'Left Interval Upper':1600,'Right Interval Lower':2050,'Right Interval Upper':1600  },
   {'Type':'Host','Name':'Olivine','Peak':0.2,'Peak Upper':1785,'Peak Lower':1765,'Baseline':'Bend','Left Interval Lower':2050,'Left Interval Upper':1600,'Right Interval Lower':2050,'Right Interval Upper':1600  },
   {'Type':'Host','Name':'Olivine','Peak':0.2,'Peak Upper':2020,'Peak Lower':2000,'Baseline':'Bend','Left Interval Lower':2050,'Left Interval Upper':1600,'Right Interval Lower':2050,'Right Interval Upper':1600  },
   {'Type':'Trace','Name':'CO2','Peak':0.2, 'Peak Upper':1450,'Peak Lower':1410,'Baseline':'Bend','Left Interval Lower':1600,'Left Interval Upper':1350,'Right Interval Lower':1600,'Right Interval Upper':1350 },
   {'Type':'Trace','Name':'CO2','Peak':0.2, 'Peak Upper':1535,'Peak Lower':1495,'Baseline':'Bend','Left Interval Lower':1600,'Left Interval Upper':1350,'Right Interval Lower':1600,'Right Interval Upper':1350 }
   ]
   df = pd.DataFrame(profiles)
   return df
   
def detect_trace(*args,**kwargs):
    spec_x = args[0]
    spec_y = args[1]
    traces  = args[2]
    imfo = args[3]
    extra = args[4]
    xsize = kwargs.get("xsize")
    ysize  = kwargs.get("ysize")

    debug_mode = kwargs.get("debug_mode")
    error = kwargs.get("error")
    
    # A boolean array to track the locations of the peaks 
    peak_boolean = np.asarray(args[5])
    
    # The boolean array to track the presense of fringes, water, olivine and volatiles 
    booleans = kwargs.get("booleans")
    
    # get the numer of points = number of samples 
    nos = num = np.shape(spec_y)[1]
    
    # import the profiles with the
    profiles = set_profiles()
  
    # minimal relative peak value (distance from top of peak to baseline) 
    min_relative_peak_value = 0.1 + error
  
    #  distinguish between Melt, Host and Trace  
    if extra == "Melt":
         trace =  traces[traces.Type == "Melt"]
    if extra == "Host":
         trace =  traces[traces.Type == "Host"]
    if extra == "Trace":
         trace =  traces[traces.Type == "Trace"]
    
    if debug_mode:
         print("\nThe number of spectra is: {0}".format(num))
         print("Shape of the data: {0}".format(np.shape(spec_y)))
         print("Melt, Host or Trace: {0}".format(extra))
         print("Profiles:")
    
    
    # determine if profile is for a Straight baseline or Bend baseline (mixed baselines dont work)
    try:
       if np.unique(trace.Baseline) == "Straight":  
          trace =  trace[trace.Baseline == "Straight"]
       if np.unique(trace.Baseline) == "Bend":   
          trace =  trace[trace.Baseline == "Bend"]
    except:
       print("Error in function: detect_trace.(1)")
       print("The baselines defined in set_profiles are probabably inconsistently defined.")
    
    
    # now the issue is how to treat type with multiple peak locations (assume the peak locations are different, but the widesst interval is the same)
    # now fuse the intervals to one array  
    # get the index of rows in subset 
    
    
    #get the row numbers of peak intervals in the profiles 
    number = np.asarray(trace.index[:])    
    
    # create for each line in profiles an array with booleans for each  
    if len(number) > 1:
         mid_interval = []
         for i in number:
             mid_interval.append(np.any([(spec_x > trace["Peak Lower"].loc[i]) & (spec_x < trace["Peak Upper"].loc[i])], axis=0)) 
         mid_interval = np.asarray(mid_interval) 
    else: 
          i=0     
          mid_interval = np.any([(spec_x > trace["Peak Lower"].loc[i]) & (spec_x < trace["Peak Upper"].loc[i])], axis=0)
    #print(trace)
    #print("Mid interval", np.shape(mid_interval))  
    
    # the intervals asociated with the flat areas at left and right side of the peaks
    interval_left = np.any([(spec_x > trace["Left Interval Lower"].loc[i]) & (spec_x < trace["Left Interval Upper"].loc[i])] , axis=0)
    interval_right = np.any([(spec_x > trace["Right Interval Lower"].loc[i]) & (spec_x < trace["Right Interval Upper"].loc[i])], axis=0)
         
    # this is a union of both flat intervals 
    interval = interval_left | interval_right
       
    segement_borders = np.asarray((trace["Left Interval Lower"].loc[i],trace["Left Interval Upper"].loc[i],trace["Right Interval Lower"].loc[i],trace["Right Interval Upper"].loc[i])).flatten()
    widest_interval = np.any([(spec_x > min(segement_borders) ) & (spec_x < max(segement_borders)) ] , axis=0)
          
    # get the size of the interval
    interval_size = len(spec_y[widest_interval])
    
    #number = trace.Baseline[trace.Baseline=="Straight"].shape[0]
    if trace.Baseline.loc[i] == "Straight":
               # get all the fitted lines / fit the linear relation 
              y_fit = np.transpose(math_support.linear_baseline_fit(spec_x,spec_y,interval,widest_interval))
              # make an array with stored RELATIVE peak location set every non-peak location to zero 
              relative_peak_val = spec_y[widest_interval,:] - y_fit
    if trace.Baseline.loc[i] == "Bend":
               # fit the curve relation 
              # y_fit = np.transpose(math_support.curved_baseline_fit(spec_x,spec_y,widest_interval, mid_interval, extra))[widest_interval,:]
               y_fit = math_support.curved_baseline_fit(spec_x,spec_y,widest_interval, extra)
               relative_peak_val = spec_y[widest_interval,:] - y_fit
    
    
    # determine the peaks and their locations and store into dataframe (because list of arrays are awkward to work with )
          #peaks = pd.DataFrame(map(lambda x:find_peaks(x , height=0.1,distance=5.0, prominence=0.01, width=5.5),spec_y[widest_interval].T),columns=['Peaks','Rest'])    
    #      peaks = pd.DataFrame(map(lambda x:find_peaks(x,height=0.1,width=(20,None),prominence=(0.01,1),spec_y[widest_interval].T),columns=['Peaks','Rest'])  
    if extra == "Melt":
         peaks = pd.DataFrame(map(lambda x:find_peaks(x,height=0.1,distance=5.0, prominence=0.01, width=4.0),spec_y[widest_interval].T),columns=['Peaks','Rest'])    
    
    if extra == "Host":
         peaks = pd.DataFrame(map(lambda x:find_peaks(x,height=0.01,distance=1.0, prominence=0.005),spec_y[widest_interval].T),columns=['Peaks','Rest'])    
    
    if extra == "Trace":
         peaks = pd.DataFrame(map(lambda x:find_peaks(x,height=(0.01,2.5),distance=15.0, prominence=0.02, width=4.0),spec_y[widest_interval].T),columns=['Peaks','Rest'])    
    
             #peaks, properties =  find_peaks(diff, height=(amplitude,2.5), width=4.0, prominence=0.02, distance=15)
             #peak_wavelengths = spec_x[interval][peaks]
    
    # split the second column into multiple columns according to the find_peaks dictionary 
    temp = peaks['Rest'].apply(pd.Series)
    result = pd.concat([peaks['Peaks'], temp], axis=1, join='inner')
    
    #print(np.shape(np.array(result['peak_heights']).flatten))
          
          # now the output can be accessed by locations for example 
          #print("---10---:", result.iloc[10])
          #print("---11--:",result['Peaks'].iloc[11])
    
    # get an index array of the peak values and peak locations  
    # this basically identifies the peak locations array on an array of the size/format that matches the original data. This can be very convenient   
    peaks_array,number_of_peaks,index_peak_1,index_peak_2  = math_support.turn_into_square(spec_y,result['Peaks'], result['peak_heights'],widest_interval, spec_x) 
    peak_boolean = peaks_array > 0
    printing = False
    if printing == True:
        print("Shape Boolean: ",np.shape(peak_boolean))
        for i in range(0,num):
             print(peaks_array[:,i][peak_boolean[:,i] >0]) 
             #print(peaks['Peaks'].iloc[i])
             print(result['peak_heights'].iloc[i])
    
    #determine the maximum peak to peak distance for consecutive elements add the -1 element to indicate emtpy arrays and avoid problems with empty arrays
    max_peak_to_peak_distance = np.asarray(list(map(lambda x:np.max(np.concatenate((x[1:]-x[:-1], [-1]))) ,result['Peaks'])))
          #peak_to_peak_distance = np.asarray(map(lambda x:x[1:]-x[:-1] ,result['Peaks']))
          
          # get the number of peaks for each spectrum 
          
          # get the coordinates of the peaks (location of spectra and location within spectra )
          #index_peak_1 = np.concatenate(np.asarray(result['Peaks'].iloc[:]))
          #index_peak_2 = np.repeat(np.where(number_of_peaks > 0),number_of_peaks[number_of_peaks > 0])
    
          # assign true values to the location of the peaks (for some reason I need 3 steps,I can't figure out how to do it all at once.)
     #     sub = peak_boolean[widest_interval,:]
      #    sub[index_peak_1,index_peak_2] = True
       #   peak_boolean[widest_interval,:] = sub
    # to confirm internal consistency    
    #sums = map(lambda x: sum(x), result['peak_heights'])
    #print("peak locations", np.sum(peaks_array[np.where(peaks_array > 0)] ),sum(sums))   
    #print("peak locations", np.sum(peaks_array[np.where(peaks_array > 0)] ),sum(sums))  
    #print("peak locations", np.apply_along_axis(np.where, 1, (peaks_array > 0)))

    #for i in range(0,num):
     #    print(peaks['Peaks'].iloc[i])
         #print(spec_x[peaks['Peaks'].iloc[i]])
        #print(spec_x)
         
  
    # make an array with stored peak location set every non-peak location to zero 
    peak_vals = spec_y[widest_interval,:] 
    peak_vals[ ~peak_boolean[widest_interval,:]] =0
    
    # test if the curve is below the baseline
    below_baseline = np.array(np.sum(spec_y[widest_interval,:], axis=0) < np.sum(y_fit, axis=0))    
    if debug_mode:
        print("Number of peaks per spectrum: {0}".format(number_of_peaks))
        print("-- Peaks -- \n{0}".format(result['peak_heights']))
  
    # check the criteria for each type of peaks (Melt, Host or CO2)
    if extra == "Melt":
              booleans = Melt_criteria(spec_y,peaks_array, mid_interval,number_of_peaks,widest_interval,max_peak_to_peak_distance, spec_x,below_baseline, booleans)
    if extra == "Host":
              booleans = Host_criteria(spec_y,peaks_array, mid_interval,number_of_peaks,widest_interval,max_peak_to_peak_distance, spec_x,below_baseline, booleans, y_fit)
    if extra == "Trace":
              booleans = Trace_criteria(spec_y,peaks_array, mid_interval,number_of_peaks,widest_interval,max_peak_to_peak_distance, spec_x,below_baseline, booleans,y_fit)
    #print(booleans)
    #print("SHAPE PEAK VALS", np.shape(peak_vals))
    #print("SHAPE PEAK ARRAY", np.shape(peaks_array))
    #print("SHAPE BOOLEANS ARRAY", np.shape(booleans))
    #print("shape widdest interval",np.shape(widest_interval))
    #make a quick map 
     # shape peak array is ()
    # SHAPE PEAK VALS (1089, 225)
    #SHAPE PEAK ARRAY (2449, 225)
     #SHAPE BOOLEANS ARRAY (225,)
     
    nop = np.asarray(number_of_peaks)

    if (num > 19):
        print("PEAK VALS", peak_vals[peak_vals[:,20] > 0,20])
    
    if (num ==1 ):
        print("PEAK VALS", peak_vals[peak_vals[:,0] > 0,0])
    
    # create a map from peak_vals 
    #print(trace['Peak Upper'],trace['Peak Lower'])
    #print("mid interval",len(np.array(np.shape(mid_interval))))
    #print("mid interval",np.shape(mid_interval))
    #print("widest ",np.shape(widest_interval))
    # loop of layers ()
    
    # NEED TO GENERATE MAPS/ARRAYS
    # water:1 map
    # olivine: 3 maps
    # CO2: 2 maps 
    if extra == "Melt":
       water_map = np.zeros(nos)
       temp_index = mid_interval  & widest_interval
       water_map = np.max(peaks_array[temp_index,:], axis=0)
       maps = water_map 
    if extra == "Host":
       olivine_map = np.zeros(3*nos).reshape(3,nos)
       for k in range(len(number)):
            temp_index = mid_interval[k,:]  & widest_interval
            olivine_map[k,:] = np.max(peaks_array[temp_index,:], axis=0)
            maps = olivine_map
    if extra == "Trace": 
        CO2_map = np.zeros(2*nos).reshape(2,nos)
        for k in range(len(number)):
           temp_index = mid_interval[k,:]  & widest_interval
           CO2_map[k,:] = np.max(peaks_array[temp_index,:], axis=0)
           maps = CO2_map
    
    for k in range(len(number)):
        if len(np.array(np.shape(mid_interval))) == 1:
            temp_index = mid_interval & widest_interval 
        else:   
            temp_index = mid_interval[k,:]  & widest_interval

        if (nos > 1):    
          fig, ax = plt.subplots()
          img = np.max(peaks_array[temp_index,:], axis=0)
          img1 = np.array(img).reshape(xsize,ysize)
          #from matplotlib import colormap as cm
          #cmap=mpl.cm.Blues
          ax.imshow(img1, cmap=plt.cm.Blues, aspect='auto')
          name = ''.join([imfo,"/temp_" , str(number[k]) , ".png" ])
          fig.savefig(name, bbox_inches='tight')
          plt.close()

    # NEED TO GENERATE MAPS (arrays)
    # Water: 1 map
    # Olivine: 3 maps
    # CO2: 2 maps 
    # 1. get profile intervals
    # 2  see if there are peaks in the interval and take the max in this interval
    
    
#    print(np.argmax(relative_peak_val, axis=0))
    # if there are multiple interval 
#    img3 = np.reshape(result['peak_heights'],(ysize,xsize))
 #   fig, ax = plt.subplots()
  #  ax.imshow(peak_vals, cmap=plt.cm.blue, aspect='auto')
   # fig.savefig('Temp_Water_Map.png', bbox_inches='tight')
  #  plt.close()
  #mini = np.min(np.where(widest_interval))
    
    plotting = False    
    if plotting == True:
         #   mini = np.min(np.where(interval))
         mini = np.min(np.where(widest_interval))
         
         relative_peak_val = spec_y[widest_interval,:] - y_fit
             
         num = np.shape(spec_y)[1]            # number of spectra

         mid_interval = np.logical_or.reduce(mid_interval, axis=0)
         temp = [mid_interval]*num
         temp2 = np.transpose(temp) & peak_boolean
         peak_boolean[~temp2]  = False
         
         # take the subset of the relative peaks that are within the peak intervals 
         relative_peak_val[~temp2[widest_interval,:]] = 0
         
         for j in range(0,num-1):
               # print("\n ")
                fig, ax = plt.subplots()
                plt.plot(spec_x,spec_y[:,j])
                plt.plot(spec_x[widest_interval],y_fit[:,j])                
                
                index = peak_boolean[:,j]
                
                plt.plot(spec_x[index],spec_y[index,j], "xk",color="black")
                
                help_index1 = np.where(peak_boolean[:,j]) 
                help_index2 = np.where(peak_boolean[:,j]) 
                
                #plt.vlines(spec_x[help_index1],y_fit[help_index2,j],y_fit[help_index2,j]+ relative_peak_val[help_index2,j])
                #plt.vlines(spec_x[help_index1],y_fit[help_index1,j],y_fit[help_index1,j]+ relative_peak_val[help_index1-mini,j])
                
                plt.vlines(spec_x[help_index1],y_fit[help_index1-mini,j],y_fit[help_index1-mini,j]+ relative_peak_val[help_index1-mini,j])
                
                
                #plt.vlines(spec_x[help_index1],0,1)
               # if len(help_index1) != 0:
                #   plt.plot(spec_x[help_index1],spec_y[help_index1,j], "xk",color="black")
 
                #plt.plot(spec_x[index],spec_y[index,j], "xk",color="black")
       
                #plt.plot(spec_x[widest_interval],y_fit[:,j])
      #         
               # index = result['Peaks'].iloc[j]
                #plt.plot(spec_x[widest_interval][index],spec_y[widest_interval][index,j], "xk",color="black")
                #plt.vlines(spec_x[peak_boolean[widest_interval,j]],0, spec_y[peak_boolean[widest_interval,j],j])
                
      #          if number_of_peaks[j] > 0:
       #             print(number_of_peaks[j])
        #            print("location: ",spec_x[widest_interval][result['Peaks'].iloc[j]])
         #           print("height: ",result['peak_heights'].iloc[j])
                 #   print("prominences: ",result['prominences'].iloc[j])
                #    print("widths: ",result['widths'].iloc[j])
                #    print("width_heights: ",result['width_heights'].iloc[j])


                
          #      plt.vlines(spec_x[widest_interval,j],0, 1)
              #  help_index1 = np.where(peak_boolean[:,j]) 
               # help_index2 = np.where(peak_boolean[widest_interval,j]) 
                
              #  plt.vlines(spec_x[help_index1],y_fit[help_index2,j],y_fit[help_index2,j]+ relative_peak_val[help_index2,j])
               # if len(help_index1) != 0:
                #   plt.plot(spec_x[help_index1],spec_y[help_index1,j], "xk",color="black")
               
               # plt.show()
               
    #(2449, 225)
    #(225,)
    #(2449,)
    #(2449,)           
    #print(np.shape(peak_boolean))
    #print(np.shape(booleans))
    #print(np.shape(widest_interval))
    #print(np.shape(mid_interval))
    # check the criteria for each type of peaks (Melt, Host or CO2)
    
    if extra == "Melt":
           maps = maps
    if extra == "Host":
           maps = np.max(maps,axis=0)
    if extra == "Trace":
           no_two_peak_index = ((maps[0,:] == 0) | (maps[1,:] == 0))
           two_peak_index = ((maps[0,:] > 0) & (maps[1,:] > 0))
           maps = np.mean(maps,axis=0)
           maps[no_two_peak_index] = 0
           if ((xsize > 1) & (ysize > 1)):
              f,a = plt.subplots(figsize=(10,6))
              a.imshow(maps.reshape(xsize,ysize))
              a.set_title("")
              name = ''.join([imfo,"/CO2_map.png" ])
              f.savefig(name, bbox_inches='tight')
              #plt.show()
              plt.close()
           
    return peak_boolean, y_fit, booleans, widest_interval, mid_interval, maps

  
#################################################################################
# calculate the thickness for the pure basalt, pure olovine and contaminaited
################################################################################
def calculate_thickness_vectorized(*args,**kwargs):
  ClassObject = args[0]
  chemistry = args[1]
  imfo = args[2]
  xsize = args[3]
  ysize = args[4]
  output=  kwargs.get("output")
  extrapolate = kwargs.get("Extrapolate")
  print("--------------------------------------------")
  print("   *** Calculate thickness for pure basalt or pure olivine  ***")
  print("imfo", imfo)
  print("extra", extrapolate)
  #print(np.shape(water_map))
  #thickness = np.NaN
  #calculate_thickness(ClassObject.peak2peak.get(), )
  #print(Oli_refrac)
  #print(Melt_refrac)
#  print(ClassObject.__dict__.items())
  #CALLABLES = types.FunctionType, types.MethodType
  #print([key for key, value in ClassObject.__dict__.items() if not isinstance(value, CALLABLES)])
  thickness = np.asarray((xsize*ysize)*[np.NaN])
  if output:
      print("    Peak to peak {0}".format(ClassObject.peak2peak_map))
      print("    Fringes {0}".format(ClassObject.booleans['Fringes']))
      print("    Contamination {0}".format(ClassObject.booleans['Cont']))
  
  
  oli_index =  np.asarray([(ClassObject.booleans['Cont'] == False) & (ClassObject.booleans['Host']) & (ClassObject.booleans['Fringes'])][0])
  basalt_index = np.asarray([(ClassObject.booleans['Cont'] == False) & (ClassObject.booleans['Melt']) & (ClassObject.booleans['Fringes'])][0])

  print("oliindex",oli_index)
  print("basalt index", basalt_index)
  #print(np.shape(oli_index))
  #print(np.shape(basalt_index))
  #print(np.shape(ClassObject.peak2peak_map))
  #print(thickness[oli_index])
  #print(np.asarray(ClassObject.peak2peak_map)[oli_index])
  
  # calculate the thicknesses the host and the melt 
  if np.sum(oli_index) > 0:
      print(chemistry['Refract_Host'])
      print(np.shape(np.asarray(ClassObject.peak2peak_map)))
      thickness[oli_index] = (0.5/chemistry['Refract_Host'])*np.asarray(ClassObject.peak2peak_map)[0][oli_index]
  if np.sum(basalt_index) > 0:
      thickness[basalt_index] = (0.5/chemistry['Refract_Melt'])**np.asarray(ClassObject.peak2peak_map)[0][basalt_index]
  
  print("Oli",chemistry['Refract_Host'])
  print("basalt",chemistry['Refract_Melt'])

  if ((extrapolate) & (np.sum(basalt_index + oli_index) > 3)): 
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        vmin_,vmax_ = np.min(thickness),np.max(thickness)
        # all thicknesses
        img3 = np.reshape(thickness,(ysize,xsize))   
        im3 = ax1.imshow(img3, cmap=plt.cm.copper, aspect='auto')
        ax1.invert_yaxis()
        # only the olivine
        temp = np.array(len(thickness)*[np.nan])
        temp[oli_index] = thickness[oli_index]
        img3 = np.reshape(temp,(ysize,xsize))  
        im3 = ax2.imshow(img3, cmap=plt.cm.copper, aspect='auto')
        ax2.invert_yaxis()
   
        # only the basalt
        temp = np.array(len(thickness)*[np.nan])
        temp[basalt_index] = thickness[basalt_index]
        img3 = np.reshape(temp,(ysize,xsize))  
        im3 = ax3.imshow(img3, cmap=plt.cm.copper, aspect='auto')
        ax3.invert_yaxis()
   
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes ('right', size='5%', pad=0.1)
        fig.colorbar(im3, cax=cax, orientation='vertical')
        fig.savefig(imfo+'/Thickness_map.png', bbox_inches='tight')
        plt.close()

        # prepare data for extrapolation (exclude Not-A-Number values)
        index = np.where(~np.isnan(img3) )
        px =index[0] 
        py =index[1] 

        print("------------ EXTRAPOLATE------------------------------------")
        thickness = math_support.extrapolate(px,py,img3,xsize, ysize,index,imfo)
        succes = True
  else:
        succes = False
  return thickness,succes
  
#################################################################################
# calculate the thickness for the pure basalt, pure olovine and contaminaited
################################################################################
def calculate_thickness(peak2peak_map, water_map, olivine_map, boolean_map,Oli_refrac, Melt_refrac, imfo,xsize, ysize, output="ON"):
  print("   *** Calculate thickness for pure basalt or pure olivine  ***")
  thickness = np.NaN
  if output:
      print("    Peak to peak {0}".format(peak2peak_map))
      print("    Fringes {0}".format(boolean_map['Fringes']))
      print("    Contamination {0}".format(boolean_map['Cont']))
  
  if ((boolean_map['Suitable']) & (boolean_map['Cont'] == False) & (boolean_map['Fringes'])):
     if ((boolean_map['Basalt']) & (boolean_map['Oli']== False)):
         n = Melt_refrac    
         print(" Sample is suitable and uncontaminated basalt,n= {0}".format(n))
     if ((boolean_map['Oli']) & (boolean_map['Basalt']== False)):
         n = Oli_refrac
         print(" Sample is suitable and uncontaminated olivine,n= {0}".format(n))
  
     thickness = (0.5/n)*peak2peak_map
  else:
      print(" This sample is either unsuitable, contaminated or does show any fringes. ")
      thickness = np.NaN
  #   if ((boolean_map['Basalt']) & (boolean_map['Oli']) & (boolean_map['Fringes'])):
   #      print(" This sample is contaminated. ")
    #     print(" Overestimate the thickness by using highest breaking index. ")
     #    n = chemistry['Oli_refrac']
      #   t_cont = (0.5/1.546)*peak2peak_map
       #  A_cont = water_map
        # A_oli = olivine_map
        # t_oli = (0.5/chemistry['Oli_refrac'])*peak2peak_map
        # thickness = t_cont - (A_oli/A_cont)*t_oli
         
  print(" Thickness {0}".format(thickness))
  return(thickness)
  
  
