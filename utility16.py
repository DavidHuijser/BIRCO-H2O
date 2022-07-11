##########################################################################
# UTILTY contains function for LOADING and SAVING data files 
# - load_chemical_composition
# - validate
###########################################################################
from os.path import basename as os_pathbasename
import tkinter as tk 
import numpy as np
import os
import copy
#import pandas as pd
#from PyAstronomy import pyasl    # needed for finding error on spectrum
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib.colorbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure 
matplotlib.use('Agg')
from tkinter import messagebox
#matplotlib.use('TkAgg')

import pandas as pd
import random
import math_support              #my own module
import vectorized_features      #my own module
import generate_graphs              #my own module
import pickle

# save data to file 
def save_data(data,folder):
    save_data_name = ''.join([folder,'/Saved_Data.dat'])
    with open(save_data_name, "wb") as fp:   #Pickling
        pickle.dump(data, fp)

#######################################
# VALIDATE LOADED FILES
####################################
def validate(self):
        scans = self.scans_var.get()
        height = self.height_var.get()
        width =  self.width_var.get()
        thickness =  self.thickness_var.get()
      
        name_spectra = os_pathbasename(self.spectra_filename.get())
        name_chem = os_pathbasename(self.chem_filename.get())
        
        loaded_chem  =  self.loaded_chem.get()
        loaded_spectra =  self.loaded_spectra.get()
        
        extrapolate_bool = self.extrapolate.get()
        H2O_map_bool = self.H2O_map_bool.get()
        CO2_map_bool=  self.CO2_map_bool.get()
        Graph_bool=  self.Graph_bool.get()

        if self.debug_bool.get() == True:        
            print("-----------------------------------------------")
            print("Height ", height)
            print("Scans ", scans)
            print("Width ", width)
            print("Thicknes ", thickness)
        
            print("Loaded Data: " , loaded_spectra)
            print("Loaded Chem: " , loaded_chem)
        
            print("Flags:")
            print("Extrapolate ",self.extrapolate.get())
            print("H2O Map ", H2O_map_bool)
            print("CO2 Map ",CO2_map_bool)
            print("Graphs ",Graph_bool)
                    #self.lightsource = tk.BooleanVar()   #[10] 0 = synchotron, 1 = Globar
            if self.lightsource.get(): 
                print("Light Source Type: Globar")
            else:
                print("Light Source Type: Synchrotron")
           
            print("Number of samples", self.nos)
            print("Number of wavelengths", self.now)
            
        # Add conditions for correct entry: 
        # 1. dimension check
        #nos = len(self.data.iloc[0,:].values)-1    # number of samples

       # Check if there is only numericals in the  spectral data 
        sx = pd.DataFrame(self.spec_x)
        sy = pd.DataFrame(self.spec_y)
        temp0 =sx[~sx.applymap(np.isreal).all(1)]
        temp1 =sy[~sy.applymap(np.isreal).all(1)]
      
        if  (len(temp0) !=0) or (len(temp1) !=0):
             print("ERROR: There seem to be non-numerical values in the dataset.")    
        else:     
          # check if ascendent sorted:
          lst = np.asarray(self.df.values[:,0])
          if not (all(lst[:-1] <= lst[1:]) or all(lst[:-1] >= lst[1:])):
               print("ERROR: Seems like your X values are not properly ordered") 
               print(self.df.values[:,0])


        if  self.nos == height*width:
              print("all loaded")
              self.Validate_bool.set(True)
              print(self.Validate_bool.get())
              self.e1.config(state='disable')
              self.e2.config(state='disable')
              self.e3.config(state='disable')
              #self.load_button4.config(state='disable') 
              #self.load_button5.config(state='disable') 
              self.e6.config(state='disable')
              self.e7.config(state='disable')
              self.e8.config(state='disable')
  
              #check dimensions 
              
              
              
              #arr = self.data.values

              #spec_x = self.data.iloc[:,0].values
  
              # get dimensions (number of wavelengths per spectra)
              #now = len(arr)
  
              x_size = int(width)
              y_size = int(height)
              print('*   Each spectrum got {} wavelengths '.format(self.now))  
              print('*   The data consist of {} spectra/datapoints'.format(self.nos))  
              print('*   The dimensions are width {} and height {}.\n'.format(x_size,y_size))  


              if self.nos != 1:
              # fast/simple  method to generate image. 
                specs = self.spec_y
                a = np.reshape(specs,(self.now,y_size,x_size))
                img3 = np.mean(a, axis=0) 
  
            
                plt.figure()
                fig, ax = plt.subplots()
                ax.imshow(img3, cmap=plt.cm.copper)
                ax.invert_yaxis()
                filename = ''.join([self.report_folder, '/Map.png'])
                matplotlib.pyplot.imsave(filename, img3)
                canvas = FigureCanvasTkAgg(fig, self.canvas)
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                canvas.draw()
              #fig.savefig('Map.png', dpi=300, bbox_inches='tight')
                plt.close()
              else:
                fig, ax = plt.subplots()
                
                ax.plot(self.spec_x, self.spec_y)
                ax.invert_xaxis()
                filename = ''.join([self.report_folder, '/NoMap.png'])
                #matplotlib.pyplot.imsave(filename, img3)
                canvas = FigureCanvasTkAgg(fig, self.canvas)
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                canvas.draw()
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
        else: 
              string = np.shape(self.df)
              messagebox.showerror("Error 2", "Incorrect dimension entered.The true shape of the datafile is "+string)
              print(np.shape(self.df))
        return True



#######################################
# LOAD THE DATA FROM THE FILE
####################################
def load_data(*args,**kwargs):
  df = args[0]
  debug_mode,folder = kwargs.get("debug_mode"), kwargs.get("folder")
  error_mode =0
  try:
    arr = np.asarray(df.values)
    spec_x,spec_y = arr[:,0],arr[:,1:]
    now, nos = np.shape(spec_y)  # (2249, 1539)
    print("\n*** Load Data (utility16.py) *** ")
    print('*   Each spectrum got {} wavelengths '.format(now))  
    print('*   The data consist of {} spectra/datapoints'.format(nos))  
    print(np.shape(spec_y))
    fig, axes = plt.subplots()
        # We need to draw the canvas before we start animating...
    if nos != 1:
           i = random.randint(0,nos)
           axes.plot(spec_x,np.asarray(spec_y[:,i]))
    else: 
           axes.plot(spec_x,spec_y)
    filename = ''.join([folder, '/temp.png'])
    axes.invert_xaxis()
    plt.savefig(filename)    
    plt.close()
  except: 
      error_mode = 1
  return error_mode, spec_x, spec_y,nos, now    
  # Spectra (Mapped)  
  if (x_size !=1 and y_size !=1):  
    try:
       arr = np.asarray(df.values)
       spec_x,spec_y = arr[:,0],arr[:,1:]
       now, nos = np.shape(spec_y)  # (2249, 1539)
      
      

       if nos != x_size*y_size:
           error_mode = 1
           print("There seem to be a problem with the entered dimensions.")     
       if debug_mode:
         print("\n*** Load Mapped Data (utility16.py) *** ")
         print('*   Each spectrum got {} wavelengths '.format(now))  
         print('*   The data consist of {} spectra/datapoints'.format(nos))  
         print('*   The dimensions are width {} and height {}.\n'.format(x_size,y_size))  
    except:
         print("\n*** Error in loading Mapped Data (utility16.py) *** ")
         spec_x, spec_y = None,None
         error_mode = 1
  
  # Spectra (single)
  else:
     try:
       print("single spectra")
       print(np.shape(df.values))
      #if True:  
       arr = np.asarray(df.values)
       spec_x,spec_y = arr[:,0],arr[:,1]
       spec_y = np.transpose([spec_y])
       now, nos = len(spec_y),1  # (2249, 1539)
       print(now,nos)
       if debug_mode:
           print("\n*** Load Single Spectra Data (utility16.py) *** ")
           print('*   Each spectrum got {} wavelengths '.format(now))  
           print('*   The data consist of {} spectra/datapoints'.format(nos))  
           print('*   The dimensions are width {} and height {}.\n'.format(x_size,y_size))  
     except:
        spec_x, spec_y = None,None
        error_mode = 1
        print("*** Error in loading Single Spectra Data (utility16.py) *** \n")
  
  if error_mode != 1:
        print("*** Loaded data succesfull (utility16.py) *** \n")
        #number = int(round(0.5*nos))
 #      ax = plt.subplots()
        #print(np.shape(spec_x),np.shape(spec_y) )
        #print("folder",folder)
        
        fig, axes = plt.subplots()
        # We need to draw the canvas before we start animating...
        if nos != 1:
           i = random.randint(0,nos)
           axes.plot(spec_x,np.asarray(spec_y[:,i]))
        else: 
           axes.plot(spec_x,spec_y)
        filename = ''.join([folder, '/temp.png'])
        axes.invert_xaxis()
        plt.savefig(filename)    
        plt.close()
  return error_mode, spec_x, spec_y,nos, now




##############################################################################################
# Read the chemical composition in the melt and the olivine according to the EMPA spreadsheet
# Calculate the Average CO2 and the average Olovine Index in two parts
# - MELT (part1)
# - HOST (part2)
################################################################################################
def load_chemical_composition(*args, **kwargs):
    df = args[0]
    debug_mode = kwargs.get("debug_mode")
    chemical_names = np.array(['SiO2_melt','TiO2_melt','Al2O3_melt','Fe2O3_melt','FeO_melt','MnO_melt','MgO_melt','CaO_melt','Na2O_melt','K2O_melt','SiO2_host','TiO2_host','Al2O3_host','FeO*_host','MnO_host','MgO_host','CaO_host','Na2O_host','NiO_host','Cr2O3_host'])
  
    # 3 version  
    #chemical_names_error =  [s + '_err' for s in chemical_names]
    chemical_names_error =  [''.join([s , '_err']) for s in chemical_names]
    #chemical_names_error =list(map(lambda ls: "".join([ls,'_err']),chemical_names))
    # print is on or off
    if debug_mode:
         print("\n----- obtaining chemistry details (utility16.py) ----------- ")
         print(chemical_names)
         print(chemical_names_error)
    
    ############################################################################
    # get concentrations for output 
    ############################################################################
    
    #if True:
    try: 
      
       # first located the header by finding sample name and rename column-names
       xxx = np.asarray(df)
       column_names_index = np.squeeze(np.where(xxx[:,0]=='Sample Name'))
       df.columns = xxx[column_names_index,:]
       #z = np.asarray([np.str(x).strip(' ') for x in xxx[column_names_index,:]])
       z = np.asarray(list(map(lambda x: np.str(x).strip(' '),xxx[column_names_index,:])))
       
       chemistry = {}
       chemistry_error = {}
    
       l = np.squeeze(np.shape(chemical_names))
       
       for i in range(0,l-1):
           index = np.where(chemical_names[i] == z)[0][0]
           chemistry[chemical_names[i]] = np.mean(np.double(xxx[column_names_index+1:,index]))
       
           error_index = np.where(chemical_names_error[i] == z)[0][0]
           chemistry_error[chemical_names_error[i]] = np.mean(np.double(xxx[column_names_index+1:,error_index]))
      
       ###################3
       #print(chemistry)
       #print(chemistry_error)    
       Succes = True 
       print(chemistry)
       print("\n----- Chemistry details loaded succesfully ----------- ")
   # else:   
    except:
       Succes = False
       chemistry = None     
       print("\n----- There seem to be incorrect values in your composition file ----------- ")
    return Succes, chemistry, chemistry_error 
  
       
def calc_refraction_index(*args, **kwargs):    
    chemistry = copy.deepcopy(args[0])
    chemistry_error = args[1]
    debug_mode = kwargs.get("debug_mode")
    conversion_factors         = [0.7147,0.7419 ,0.603, 1.0,0.7773, 0.7745]
    conversion_part =            [0     ,0      ,1     ,0 ,    1,       1]
    chemical_host_names= np.array(['SiO2_melt','TiO2_melt','Al2O3_melt','Fe2O3_melt','FeO_melt','MnO_melt','MgO_melt','CaO_melt','Na2O_melt','K2O_melt'])
    chemical_weights = np.array([0.460, 1.158, 0.581,1.090, 0.897,0.903, 0.767,0.795,0.505, 0.495]) 
    # add error to all chemical components   
    for key,value in enumerate(chemistry): 
        chemistry[value] =  chemistry[value] + chemistry_error[value+'_err']
    
    # create seperate dictionary for derived quantities from input     
    #chemistry_der = {}
    chemistry_der = chemistry
    

    try:
      # MELT
       chemistry_der['Ca_mol'] = chemistry['CaO_melt']*conversion_factors[0] /40.078  
       chemistry_der['Na_mol'] = chemistry['Na2O_melt']*conversion_factors[1] /22.9898
       chemistry_der['Na'] = chemistry_der['Na_mol']/(chemistry_der['Na_mol']+chemistry_der['Ca_mol'])
       CO2_mean = 451.0 - (342.0*chemistry_der['Na']) #molar absorptino coefficient for
    
       # HOST 
       chemistry_der['Mg_mol'] = chemistry['MgO_host']*conversion_factors[2] /24.305  
       #chemistry['FeO_mol'] = chemistry['FeO*_host']*conversion_factors[3] / 22.989
       chemistry_der['Fe_mol'] = chemistry['FeO*_host']*conversion_factors[4] /55.845  
       chemistry_der['Mn_mol'] = chemistry['MnO_host']*conversion_factors[5] /54.938
       chemistry_der['FO'] =   100.*chemistry_der['Mg_mol']/(chemistry_der['Mg_mol'] + chemistry_der['Fe_mol']  + chemistry_der['Mn_mol'] )     
    
       #################################################################################################
       # determine properties of alpha, beta and gamma lines graph from (Deer, Howie,Zussman 1996) and draw a number 
       # from the uniform distribution between alpha and gamma
       ###################################################################################################
       y_alpha = 1.635 + 0.001925*(100.-chemistry_der['FO']) 
       y_beta = 1.6485  + 0.00215*(100.-chemistry_der['FO'])
       y_gamma = 1.670 + 0.00205*(100.-chemistry_der['FO'])
       
       Oli_refrac = np.random.uniform(y_alpha,y_gamma)
       #Oli_refrac_without_error = -0.0022*chemistry['FO']+1.8655 

       # calculate refraction index of melt
       l = np.shape(chemical_host_names)[0]
       Melt_refrac = 1.000000000   
       for i in range(0,l):
            Melt_refrac = Melt_refrac + chemical_weights[i]*chemistry[chemical_host_names[i]]/100.
    
       if debug_mode:
          print("CO2 epsilon: {0}".format( CO2_mean))
          print("Average Fo: {0}  ".format(chemistry_der['FO']))
          print("Olivine Refraction: {0}  ".format(Oli_refrac))
          print("Melt Refraction: {0}  ".format(Melt_refrac))
          #without error this should be   Melt_refrac= 1.62391385 
          #without error this should be   Oli_refrac= 1.69919670571
          print("\n") 
          Succes = True
          print("\n----- chemistry composition succesfully loaded (utility16.py) ----------- ")
    except:
    #else:  
        Succes = False
        print("\n----- There seem to be incorrect values in your composition file ----------- ")
    
    # THIS IS ONLY FOR TESTING REMOVE THIS LATER 
    Melt_refrac= 1.62391385 
    Oli_refrac= 1.69919670571    
    #print(chemistry_der)  
    
    #chemistry = dict(CO2=CO2_mean, FO=chemistry_der['FO'], Refract_Melt=Melt_refrac, Refract_Host=Oli_refrac)
    #chemistry = copy.deepcopy(chemistry_der)
    chemistry_der['CO2'] = CO2_mean
    chemistry_der['Refract_Melt'] = Melt_refrac
    chemistry_der['Refract_Host'] = Oli_refrac
    #print(chemistry_der)   
    #print("**********",chemistry_der)
    return chemistry_der,Succes


####################################################
# create a new directory/folder for the reports 
#################################################
def get_report_folder():
      import os
      dirpath = os. getcwd()
      new = dirpath + '/report'
      i = 0
      while 1:
          isExist = os.path.exists(new)
          if not isExist:
            os.makedirs(new)
            break
          else:
            i = i+1
            new = ''.join([dirpath,'/report-',str(i)])    
            #os.makedirs(new)
            #break
      return new     
        


# wrapper
def calc_spectrum_error(*args, **kwargs): 
     y,x,x_size,y_size,LST = args[0],args[1],args[2],args[3], args[4]
     folder = kwargs.get("folder")
     N = kwargs.get("scans")
     error = calc_spectrum_error_true(y, x, x_size, y_size, folder,N, LST)
     return error



##############################################################################
# Calculated and visualize error (this will be run only once)
##################################################################################
def calc_spectrum_error_true(y, x, x_size, y_size, folder,Number_of_scans, LST):
        from scipy.stats import norm
        from PyAstronomy import pyasl    # needed for finding error on spectrum
        print("\n>>>>>>>>>> Calculating Error from the spectrum <<<<<<<<<")
        # we can severly reduce the spectrum to be restricted to certain wavenumbers from 1300 to 3800  
        index = x > 3800
        #arr = data.values
        #x = data.iloc[:,0].values
        #y = data.iloc[:,1:].values
        avg_ = np.mean(y, axis=0)
        max_ = np.max(y, axis=0)
        # get dimensions (number of wavelengths per spectra)
        # now = number of wavelengths
        # nos = number of samples
        #if  ((x_size != 1) and  (y_size !=1)): 
        now, nos = np.shape(y)  # (2249, 1539)
        #print("shape y",np.shape(y))   
        #else: 
         # now = np.shape(y)
          #nos = 1 
          #y = np.transpose([y])
          
        ############################################
        # Creating Fit to Data in interval (x > 3800)
        ##########################################
        fit = np.transpose(math_support.baseline_arPLS_vector(np.transpose(y[index,:]),ratio=0.001, lam=1e4, niter=100, full_output=False))

        # get the sigmas off all distributions. 
        ############################################
        # Calculate Beta Sigma  (mu and std)
        ###########################################
        N= 10
        j=10
        beq = pyasl.BSEqSamp()
        i = np.arange(np.shape(y)[1])
        ss = copy.deepcopy(np.asarray(list(map(lambda xx: beq.betaSigma(y[x > 1300,xx], N, j, returnMAD=False),i ))))
        error_betaSigma = ss[:,0]
        error_betaSigma_std = ss[:,1]
        #print(np.shape(std_error))
        
        # ############################################################
        # Calculate Mu and Std from error of subset (differece between fit and  data)
        ##############################################################
        ss2= np.asarray(list(map(lambda xx:norm.fit(y[index,xx]-fit[:,xx]),i )))
        mean_interval = ss2[:,0]
        error_interval = ss2[:,1]
        
        if (nos > 1):
          # get in a random sample of 4 from the entire data set for output for control 
          i = np.arange(np.shape(y[index,:])[1])
          t = np.asarray(random.sample(list(i), 4))
          # create a multiplot of graph of data in interval x< 3800 plus a fit 
          fig, a = plt.subplots(2,2)
          a = a.ravel()
          for idx,ax in enumerate(a):
            y_temp = y[index, t[idx]]
         
            ax.plot(x[index],y_temp,'.') 
            ax.plot(x[index],fit[:,t[idx]],'.',color='red') 
          fig.savefig(''.join([folder,'/Graph_of_data_of_subset_of_4_interval_plus_fit.png']), bbox_inches='tight')
          plt.close()
        
        
          # create a histogram error of data to fit in interval x< 3800
          # - Gausian error distribution using fit to interval (x>3800) 
          # - Gaussian error distribution using BetaSigma using interval(x>1300) 
          # - Gaussian error distribution using weighted average, factor and N  
          fig, a = plt.subplots(2,2)
          a = a.ravel()
          for idx,ax in enumerate(a):

            # Error of data to fit
            ax.hist(y[index,t[idx]]-fit[:,t[idx]], bins=25, density=True, alpha=0.6, color='green',range=[-0.011, 0.011])
            xmin, xmax = ax.get_xlim()
            
            # Fit normal distribution to histogram 
            x_ = np.linspace(-0.03, 0.03, 100)
            y_ = norm.pdf(x_, mean_interval[t[idx]], error_interval[t[idx]])
            l0 =ax.plot(x_, y_, linewidth=2, ls='-',color='red', label='Interval Error Distribution')[0]
            
            # draw distribution based on BetaSigma
            y_ = norm.pdf(x_, 0,error_betaSigma[t[idx]])
            l1 =ax.plot(x_, y_, linewidth=2, ls='--',color='yellow', label='BetaSigma Error Estimate')[0]
            
            # The  CORRECTED distribution with the correction factor seems suspicously narrow with factor therefore we stick to 1 
            std  = 0.018*error_interval[t[idx]]+ 0.18*error_betaSigma[t[idx]]
            factor = 1.0
            
            y_ = norm.pdf(x_, 0, factor*std)
            l2 =ax.plot(x_, y_, linewidth=2, ls='-',color='blue', label='Corrected Error Estimate')[0]
            #print(np.sqrt(Number_of_scans)*std)
          print(t)    
          lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]

          fig.legend(lines_labels[0][0], lines_labels[0][1])
        
          fig.savefig(''.join([folder,'/Error_histogram_subset_of_4.png']), bbox_inches='tight')
          plt.close()
          # Plot the histogram.
          fig, a = plt.subplots(2,2)
          a = a.ravel()
          for idx,ax in enumerate(a):
            subset =  (1300 < x) &  (x < 3800)
            ax.plot(x[subset], y[subset,t[idx]], linewidth=1)
            
            
  
            # draw corrected distribution 
            std  = 0.0185*error_interval[t[idx]]+ 0.182*error_betaSigma[t[idx]]
            factor = 1.0
          
            #draw errors len(subset)
            errors = np.random.normal(loc=0, scale=std, size=len(x[subset]))
            ax.plot(x[subset], y[subset,t[idx]]+errors, color='red', linewidth=1)
            ax.invert_xaxis()
        
          fig.savefig(''.join([folder,'/Graph_of_data_of_subset_of_4.png']), bbox_inches='tight')
          plt.close()
        
          error = error_interval # error on interval
          error2 = error_betaSigma # error from beta sigma 

          # using standard error on interval 
          vmin_,vmax_ = np.min([error,error2]),np.max([error,error2])
          norm=  math_support.MyNormalize(vmin=vmin_, vmax=vmax_)
        
          fig, (ax1,ax2) = plt.subplots(1,2)
          img3 = np.reshape(error,(y_size,x_size))   
          im1 = ax1.imshow(img3,norm=norm,aspect='auto')
          ax1.invert_yaxis()
          divider = make_axes_locatable(ax1)
          cax = divider.append_axes('right', size='5%', pad=0.25)
          fig.colorbar(im1, cax=cax, orientation='vertical')
   
          ax1.title.set_text('Error estimate (Intervals) ')
          # using standard error on interval 
          img3 = np.reshape(error2,(y_size,x_size))
          im3 = ax2.imshow(img3,norm=norm, aspect='auto')
          ax2.invert_yaxis()
          ax2.title.set_text('Error estimate (BetaSigma)')
        
          divider = make_axes_locatable(ax2)
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im3, cax=cax, orientation='vertical')
          fig.savefig(''.join([folder,'/Map_error2.png']), bbox_inches='tight')

          fig, (ax1,ax2,ax3) = plt.subplots(1,3)
          img3 = np.reshape(avg_,(y_size,x_size))   
          ax1.imshow(img3, cmap=plt.cm.copper, aspect='auto')
          ax1.invert_yaxis()
   
          img3 = np.reshape(max_,(y_size,x_size))
          ax2.imshow(img3, cmap=plt.cm.copper, aspect='auto')
          ax2.invert_yaxis()
   
          vmin_,vmax_ = np.min(error),np.max(error)
        
          norm=  math_support.MyNormalize(vmin=vmin_, vmax=vmax_)
          img3 = np.reshape(error,(y_size,x_size))
          im3 =  ax3.imshow(img3,norm=norm, aspect='auto')
          ax3.invert_yaxis()
    
          divider = make_axes_locatable(ax3)
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im3, cax=cax, orientation='vertical')

          fig.savefig(''.join([folder, '/Map_error.png']), bbox_inches='tight')
          plt.show()
        
        ####################################################################
        # the error distribution to be returned
        ##################################################################3
        if LST == True:
           print("light source is globar")
           error  = (0.8429*error_betaSigma+ 0.0563*error_interval)
        else:
           error  = (0.08897*error_betaSigma+ 0.7512*error_interval)
           print("light source is synchroton")
        print("\n>>>>>>>>>> DONE - Calculating Error from the spectrum <<<<<<<<<")
        return error



#class MyClass:
    #def __init__(data,xsize,ysize,ID_array):
    #    self.data = data
     #   self.xsize = xsize
      #  self.ysize = ysize
        #self.ID_array = ID_array

# now = number of wavelengths
# nos = number of specimen 
class MyClass:
       def __init__(self, xsize,ysize,now,nos):        
          self.xsize = xsize
          self.ysize = ysize
          self.now = now
          self.nos = nos
          self.length = xsize*ysize
          self.water_map = np.zeros(self.length)
          self.CO2_map = np.zeros(self.length)
          self.olivine_map = np.zeros(self.length)
          self.Rscores = np.zeros(self.length)     # R scores for the wave-fitting
          self.thickness_map = np.zeros(self.length)

          self.peak2peak_map = np.zeros(self.length)
          #boolean_map = np.zeros(length, dtype=[('Basalt',bool),('Oli',bool),('Cont',bool),  ('Fringes', bool),('Suitable',bool),('CO2',bool)])
          #self.amplitude_map = np.zeros(self.length)
          self.booleans = np.zeros(nos, dtype=[('Melt',bool),('Suitable',bool),('Host',bool),('Cont',bool),('Trace',bool), ('Fringes', bool)])
    
          # array to store slopes of the distance-wavelength correlation
          #self.slopes = np.zeros(self.length)
          
          # A boolean array to track the locations of the peaks 
          #peak_boolean = np.zeros(np.shape(spec_y)[1]*len(spec_x), dtype=bool).reshape(len(spec_x),np.shape(spec_y)[1])
          self.peak_boolean = np.zeros(now*nos, dtype=bool).reshape(now,nos)     
          self.CO2 = 0
          self.FO = 0
          self.Refract_Host = 0   #olivine
          self.Refract_Melt = 0   #basaltic glass 
          self.RScore_extrapolate = 0

       #def set_booleans(self,x):
        #          booleans = x
       #def set_Rscores(self,x):
       def printing(self):
          print("xsize", self.xsize)
          print("ysize", self.ysize)
          #print("booleans(suitable)", self.booleans['Suitable'])
          #print("boolean(fringes)", self.booleans['Fringes'])
          #print("boolean(melt)", self.booleans['Melt'])
          #print("booleans(host)", self.booleans['Host'])
          #print("booleans(cont)", self.booleans['Cont'])
          #print("booleans(CO2)", self.booleans['Trace'])
          #print("Peak Boolean", self.peak_boolean)
          #print("Peak-2-Peak Map", self.peak2peak_map)
          #print("Watermap", self.water_map)
          print("CO2", self.CO2)
          print("FO", self.FO)
          print("Refrac_Host", self.Refract_Host)
          print("Refrac_Melt", self.Refract_Melt)
          
          #print("booleans", self.booleans)
          variables1 = list(self.__dict__.keys()) 
          #variables2 = list(chemistry.__dict__.keys()) 
          print(variables1)
          #print(self.chemistry)

           #self.xsize = xsize
         # self.ysize = ysize
      #    self.now = now
      #    self.nos = nos
       #   self.length = xsize*ysize
        #  self.water_map = np.zeros(self.length)
        #  self.CO2_map = np.zeros(self.length)
        #  self.Rscores = np.zeros(self.length)
        #  self.thickness_map = np.zeros(self.length)
        #  self.olivine_map = np.zeros(3*self.length).reshape(3,self.length)
        #  self.peak2peak_map = np.zeros(self.length)
          #boolean_map = np.zeros(length, dtype=[('Basalt',bool),('Oli',bool),('Cont',bool),  ('Fringes', bool),('Suitable',bool),('CO2',bool)])
         # self.amplitude_map = np.zeros(self.length)
    
          #self.booleans = np.zeros(nos, dtype=[('Melt',bool),('Suitable',bool),('Host',bool),('Cont',bool),('Trace',bool), ('Fringes', bool)])
        #          Rscores = x            
          # there are 4 potential models/fits that could be save
          # 1. fringes
          # 2. water
          # 3. olovine
          # 4. CO2 
          
        
##############################################################################################
# Order the data in a suitable format
################################################################################################
# This is where we call the different identifitcation tools
# ID: Host                - HostMinerals(Olivine, Quartz, Clinopyroxene, Plagioclase) 
# ID: minerals/melts      - Melt compositions (Basalt, Andesite, Rhyolite)
# ID: traces:             - Trace Chemicals (H2O, CO2, OH, etc)
# ID: fringes:            - Wavelength information for every component
# This is only viable in a vectorized version. 
#water_map, olivine_map, co2_map, thickness_map, boolean_map  = utility12._features(spec_x, spec_y,baseline_corrected_data,chemistry, ID_array, imfo, xsize=27, ysize=57, output="ON") 
def detect_features_vectorized(*args,**kwargs):
    print("----- Start one loop of detect features ")
    spec_x = args[0]
    spec_y = args[1]
    chemistry  = args[2]
    imfo = args[3]
    error_distribution = args[4]
    xsize = kwargs.get("xsize")
    ysize =kwargs.get("ysize")
    debug_mode = kwargs.get("debug_mode")
    test = kwargs.get("test")
    GenImg = kwargs.get("GenImg")
    UseFringes = kwargs.get("UseFringes")
    Extrapolate = kwargs.get("Extrapolate")

    times = []
   
    if debug_mode:
        print("\n ***Detect Features ***")
        #print("  ID length: {0}. Reduce Data Frame length: {1}. Image size: {2}.".format(len(ID_array), len(ID_array), xsize*ysize))
    #    print("  #Spectra,#Wavelength {0} ".format(np.shape(spec_y)))
        print("  Spec-y dimensions {0}".format(np.shape(spec_y)))

        # choose trial set or full data set
        #if (len(ID_array) != xsize*ysize):
         #     print(" *** using trial subset *** \n")   
        #else:
         #     print(" *** using full dataset ***")     
        #Spectra,#Wavelength (23, 2449) 
        #Spec-y dimensions (2449, 23)
        #print(len(rdf[:,0]))
        #print("ID length: ",len(ID_array), " Reduce Data Frame length: ",len(rdf)," image size: ", xsize*ysize)
        #print("(#Spectra,#Wavelength) : ", np.shape(rdf))
      
    length = xsize*ysize
    now = np.shape(spec_y)[0]  # number of wavelengths
    nos = np.shape(spec_y)[1]  # number of samples
    #num = np.shape(spec_y)[1] 
    
    ###################################
    #  create maps for:
    #  - Melt (water)
    #  - Host (Olivine (multiple ranges)
    #    range 1650 - 1675
    #    range 1765 - 1785
    #    range 2000 - 2020
    #  - Traces (CO2) (multiple ranges)
    #  - Peak2peak distance
    #  - Boolean Maps
    #    - suitable sample (boolean)
    #    - water (boolean)
    #    - olivine (boolean)
    #    - fringes (boolean)
    ############################################
    # store all information in MyClass-Object
    #################################################
    saveClass = MyClass(xsize,ysize,now,nos)
    saveClass.CO2 = chemistry['CO2']
    saveClass.FO = chemistry['FO']
    saveClass.chemistry = chemistry 
    saveClass.Refract_Host = chemistry['Refract_Host']
    saveClass.Refract_Melt = chemistry['Refract_Melt']
  
    variables1 = list(saveClass.__dict__.keys()) 
    #variables2 = list(chemistry.__dict__.keys()) 
    # print(variables1)
    
    # trace specify: Host, Melt or Trace-chemical 
    # trave specify: 
    # traces for each elementL - contains information on chemical element  
    # traces                   - contains wavenumber interval for peak    
    # traces                   - contains lower bounder 
    # traces                   - straight line or bend baseline fit
    # traces                   - lower interval for baseline 
    
    ##############################################################################
    # A vectorized version of the profilesthat can be used for other chemicals 
    ##############################################################################
    traces = vectorized_features.set_profiles() 
  
    # create an image folder    
    image_folder = imfo + '/images'
    isExist = os.path.exists(image_folder)
    if not isExist:
            os.makedirs(image_folder)
            
    ################################################################################
    #  Go over all different components /intervals 
    #  0. Fringes -Waves: single interval 
    #  1. Melt -Water: single interval
    #  2. Host -Olivine: multiple intervals: 3
    #  3. Traces -CO2: multiple intervals: 2

    # prepare to add error to data according to error distribution
    #NOTE: adding error to data in the fringe detect process might results in 0 fringes detected. (improvement needed)   
    i = np.arange(np.shape(spec_y)[1])
    ss = np.asarray(list(map(lambda xx: np.random.normal(loc=0, scale=error_distribution[xx], size=np.shape(spec_y)[0])     ,i ))).T
    #print("Shape ss", np.shape(ss))    
    #print("Shape y", np.shape(spec_y))    
    saveClass.Rscores, saveClass.booleans['Fringes'],saveClass.peak2peak_map, curve_corrected_section,fit_section  = vectorized_features.detect_fringes(spec_x, spec_y,folder= image_folder)
    print("Percentage ofgood fits: ", 100*np.sum(saveClass.booleans['Fringes'])/len(saveClass.booleans['Fringes']))  # 85 or 0.37 percent  without added error
    
    #Add error to data 
    spec_y = spec_y+ss

    # Always visualize for single spectra
    if nos ==1: 
        generate_graphs.fringes_graph(spec_x, spec_y,saveClass.Rscores,saveClass.booleans['Fringes'],saveClass.peak2peak_map, curve_corrected_section,fit_section, folder=image_folder )
        
    print("So far, so good, moving on to detecting Water ")
    saveClass.peak_boolean, y_fit, saveClass.booleans, interval, mid_interval,saveClass.water_map = vectorized_features.detect_trace(spec_x, spec_y,traces, imfo,"Melt",saveClass.peak_boolean, booleans=saveClass.booleans,debug_mode=True,xsize=xsize,ysize=ysize, error=error_distribution)
 
    if nos ==1: 
        generate_graphs.melt_graph(spec_x, spec_y,saveClass.peak_boolean, saveClass.booleans, y_fit,interval,folder= image_folder)
        saveClass.printing()

    print("So far, so good, moving on to detecting Olivine ")
    saveClass.peak_boolean, new_y_fit, saveClass.booleans, new_interval, mid_interval,saveClass.olivine_map = vectorized_features.detect_trace(spec_x, spec_y,traces, imfo,"Host",saveClass.peak_boolean, booleans=saveClass.booleans,debug_mode=True,xsize=xsize,ysize=ysize, error=error_distribution)
    if nos ==1: 
       generate_graphs.host_graph(spec_x, spec_y,saveClass.peak_boolean,saveClass.booleans, new_y_fit,new_interval,mid_interval,folder= image_folder)
    
    print("So far, so good, moving on to detecting CO2 in Melt ")
    saveClass.peak_boolean, new_y_fit2, saveClass.booleans, new_interval2,mid_interval2,saveClass.CO2_map = vectorized_features.detect_trace(spec_x, spec_y,traces, imfo,"Trace",saveClass.peak_boolean, booleans=saveClass.booleans,debug_mode=True,xsize=xsize,ysize=ysize, error=error_distribution)
    if nos ==1: 
        generate_graphs.trace_graph(spec_x, spec_y,saveClass.peak_boolean,saveClass.booleans, new_y_fit2,new_interval2,mid_interval2,folder= image_folder)

    #variables2 = list(saveClass.__dict__.keys()) 
    #print(variables2)
    #print("full", len(variables2))
    #print(set(variables1) ^ set(variables2))
    #print(variables2)
    print("   ----- FINISHED one loop of detect features ")
    return saveClass 
      
