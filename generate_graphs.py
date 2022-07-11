# Import the required Libraries
#import tkinter as tk 
#import easygui
#from tkinter import *
#from tkinter import ttk, filedialog, font
#from tkinter import Scrollbar
#from tkinter.filedialog import askopenfile
#from tkinter.filedialog import asksaveasfilename
#from tkinter.messagebox import showerror
#from tkinter import messagebox
#from tkinter.messagebox import showinfo

import matplotlib.pyplot as plt
#import matplotlib.colors as colors
#from matplotlib import cm
import matplotlib.colorbar
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#from matplotlib.figure import Figure 
matplotlib.use('Agg')

import numpy as np

def preliminary_errors_maps(img3,error,x_size,y_size):
        class MyNormalize(colors.Normalize):
            def __call__(self, value, clip=None):
                # function to normalize any input between vmin and vmax linearly to [0,1]
                n = lambda x: (x-self.vmin)/(self.vmax-self.vmin)
                # nonlinear function between [0,1] and [0,1]
                f = lambda x,a: (2*x)**a*(2*x<1)/2. +(2-(2*(1-1*x))**a)*(2*x>=1)/2.
                return np.ma.masked_array(f(n(value),0.5))
  
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        img3 = np.reshape(avg_,(y_size,x_size))   
        ax1.imshow(img3, cmap=plt.cm.copper, aspect='auto')
        ax1.invert_yaxis()
   
   
        img3 = np.reshape(max_,(y_size,x_size))
        ax2.imshow(img3, cmap=plt.cm.copper, aspect='auto')
        ax2.invert_yaxis()
   
        vmin_,vmax_ = np.min(error),np.max(error)
        #order between min/max  (error_std=23, error_beq=)
        print(vmin_, vmax_)
        
        
        norm=  MyNormalize(vmin=vmin_, vmax=vmax_)
        img3 = np.reshape(error,(y_size,x_size))
        im3 =  ax3.imshow(img3,norm=norm, aspect='auto')
        ax3.invert_yaxis()
    
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')

        
        #im = ax2.imshow(X[::-1,np.newaxis], norm=norm, cmap="coolwarm", aspect="auto")
        #fig.colorbar(im)
        
        

#        if max(error) < 0.1:
 #                 print("new log-colorbar")
      
#                  pcm = ax[2].pcolormesh(img3, norm=colors.PowerNorm(gamma=0.5), shading='auto')
 #                 fig.colorbar(pcm, ax=ax, shrink=1.0)
  #      else:
   #               print("regular color bar")
        #          im3 =  ax[2].imshow(img3,vmin=0, vmax=1, aspect='auto')
  #  #              ax[2].invert_yaxis()
     #             fig.colorbar(im3, ax=ax, shrink=1.0)
   
   
        plt.close(1)
        plt.close(2)
        fig.savefig('Map_error.png', bbox_inches='tight')
    #   plt.show()

def get_text_string(*args, **kwargs):
      boolean = args[0]
      switch = args[1]
      if (boolean['Suitable']):
                    suit_str =  'Sample: Suitable'
      else: 
                    suit_str =   'Sample: Unsuitable'
      
      if (boolean['Melt']):
                    melt_str =  'Sample: Basalt'
      else: 
                    melt_str =   'Sample: No-Basalt'
      
      if (boolean['Cont']):
                    cont_str =  'Sample: Contaminated'
      else: 
                    cont_str =   'Sample: Pure'
      
      if (boolean['Host']):
                    host_str =  'Sample: Host'
      else: 
                    host_str =   'Sample: No-Host'
      
      if (boolean['Trace']):
                    trace_str =  'Sample: Trace'
      else: 
                    trace_str =   'Sample: No-Trace'
      
      if switch == "Melt":
                #textstr = '\n'.join((r'$ %s$' % (water_str, ), r'$ %s$' % (suit_str, ),r'$ %s$' % (oli_str, ),r'$ %s$' % (fringe_str, ),r'$ %s$' % (cont_str, ),r'$ %s$' % (CO2_str, ) ))
                textstr = '\n'.join((r'$ %s$' % (suit_str, ),r'$ %s$' % (melt_str, ) ))
      if switch == "Host":
                #textstr = '\n'.join((r'$ %s$' % (water_str, ), r'$ %s$' % (suit_str, ),r'$ %s$' % (oli_str, ),r'$ %s$' % (fringe_str, ),r'$ %s$' % (cont_str, ),r'$ %s$' % (CO2_str, ) ))
                textstr = '\n'.join((r'$ %s$' % (suit_str, ),r'$ %s$' % (melt_str, ),r'$ %s$' % (host_str, ),r'$ %s$'  % (cont_str, )        ))
      if switch == "Trace":
                #textstr = '\n'.join((r'$ %s$' % (water_str, ), r'$ %s$' % (suit_str, ),r'$ %s$' % (oli_str, ),r'$ %s$' % (fringe_str, ),r'$ %s$' % (cont_str, ),r'$ %s$' % (CO2_str, ) ))
                textstr = '\n'.join((r'$ %s$' % (suit_str, ),r'$ %s$' % (melt_str, ),r'$ %s$' % (host_str, ),r'$ %s$'  % (cont_str, ),r'$ %s$' % (trace_str, )        ))
      
      
      
      if not boolean['Suitable']:
                nothing_str = ' '
  
                textstr = '\n'.join((r'$ %s$' % (suit_str, ), ' '))
         
      return textstr          

def fringes_graph(*args,**kwargs):
    spec_x = args[0]
    spec_y = args[1]
    R2_values =  args[2]
    fringes_bool  = args[3]
    peak2peak  = args[4]
    curve_corrected_section  = args[5]
    fit_section  = args[6]
    folder = kwargs.get("folder")
    interval = np.where((spec_x < 2250) & (spec_x > 2050))[0] 
    num = np.shape(spec_y)[1]            # number of spectra

    for j in range(0,num):
       fig, ax = plt.subplots() 
       
       ax.scatter(spec_x[interval],curve_corrected_section[:,j], linewidth=1,label='Baseline Corrected Data')
       ax.plot(spec_x[interval],fit_section[:,j],linestyle='dotted', linewidth=1,label='Fitted Sine')
       
       
       textstr2 = '\n'.join((r'$R-Score: %.3f$' % (R2_values[j], ) , r'$Accepted: %s$' % (fringes_bool[j], )))
       
       
       ax.text(0.05, 0.05, textstr2 , fontweight="bold", transform=ax.transAxes,bbox=dict(facecolor='white', alpha=0.5,boxstyle='round,pad=1'))   
       #ax.text(0.5, 0.5, textstr2, bbox=dict(facecolor='red', alpha=0.5))
      # ax.text(0.05, 0.05, textstr2 , fontweight="bold")   
      #print(R2, R2_values[j])
       
       
       #plt.plot(spec_x[new_index],spec_y[new_index,j])
    
       ax.legend(loc='upper right')
       ax.invert_xaxis()
       fig.savefig(''.join([folder,'/Graph_of_fringes_plus_fit',str(j) ,'.png']), bbox_inches='tight')
       plt.close(fig)
       
    return None     
    

def host_graph(*args,**kwargs):
    spec_x = args[0]
    spec_y = args[1]
    peak_boolean = args[2]
    booleans = args[3]
    y_fit = args[4]
    widest_interval = args[5]
    mid_interval = args[6]
    folder = kwargs.get("folder")
    
    min_relative_peak_value = 0.02
    relative_peak_val = spec_y[widest_interval,:] - y_fit
    
    
    mini = np.min(np.where(widest_interval))
    
    num = np.shape(spec_y)[1]            # number of spectra
    
    # exclude peaks outside interval
    mid_interval = np.logical_or.reduce(mid_interval, axis=0)
    temp = [mid_interval]*num
    temp2 = np.transpose(temp) & peak_boolean
    peak_boolean[~temp2]  = False
    
    # take the subset of the relative peaks that are within the peak intervals 
    relative_peak_val[~temp2[widest_interval,:]] = 0.000
    relative_peak_val[np.where(relative_peak_val <  min_relative_peak_value)] = 0.000
    
    #print(np.shape(relative_peak_val))
    
    peak_boolean[widest_interval,:][np.where(relative_peak_val <  min_relative_peak_value)] = False
    
    
    for j in range(0,num):
            #    print("\n ")
                fig, ax = plt.subplots()
                plt.plot(spec_x,spec_y[:,j])
              #    index = result['Peaks'].iloc[j]
                #print(peak_boolean[j])
                index = peak_boolean[:,j]
                
                
             #   plt.plot(spec_x[index],spec_y[index,j], "xk",color="black")
                
                
     #    
     #           index = result['Peaks'].iloc[j]
      #          if number_of_peaks[j] > 0:
       #             print(number_of_peaks[j])
        #            print("location: ",spec_x[widest_interval][result['Peaks'].iloc[j]])
         #           print("height: ",result['peak_heights'].iloc[j])
                 #   print("prominences: ",result['prominences'].iloc[j])
                #    print("widths: ",result['widths'].iloc[j])
                #    print("width_heights: ",result['width_heights'].iloc[j])

                #plt.vlines(spec_x[peak_boolean[:,j]],0, spec_y[peak_boolean[:,j],j])
                
          #      plt.vlines(spec_x[widest_interval,j],0, 1)
                # index for vertical lines 
                help_index1 = np.where(peak_boolean[:,j]) 
                # index for x-es on peaks 
                help_index2 = np.where(relative_peak_val[:,j] > 0.02)[0] 
                
                # plot vertical lines and x-es 
                plt.vlines(spec_x[help_index1],y_fit[help_index1-mini,j],y_fit[help_index1-mini,j]+ relative_peak_val[help_index1-mini,j],linestyles='dotted')
                plt.plot(spec_x[help_index2+mini],spec_y[help_index2+mini,j], "x",color="black")
 
                # plot host-bands 
                ax.axvspan(1650, 1675, facecolor='green', alpha=0.3)
                ax.axvspan(1765, 1785, facecolor='green', alpha=0.3)
                ax.axvspan(2000, 2020, facecolor='green', alpha=0.3)
                plt.gca().invert_xaxis()
                plt.xlabel("Wavenumber $cm^{-1}$")
                plt.ylabel("Absorbance")
 
                plt.plot(spec_x[index],spec_y[index,j], "x",color="black")
                plt.plot(spec_x[widest_interval],y_fit[:,j])
                
               
                textstr = get_text_string(booleans[j], "Host")
   
                # place a text box in upper left in axes coords
                props = dict(boxstyle='round', facecolor='white', alpha=0.5) 
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                fig.savefig(''.join([folder,'/Graph_of_host_plus_fit',str(j) ,'.png']), bbox_inches='tight') 
                plt.close()
    return None                     
#peak_boolean, y_fit, booleans, interval = vectorized_features.detect_trace(spec_x, spec_y,ID_array,traces, imfo,"Melt",peak_boolean)
# generate_graphs.melt_graph(spec_x, spec_y,peak_booleans, boolean, fit,interval)
    

def melt_graph(*args,**kwargs):
    spec_x = args[0]
    spec_y = args[1]
    peak_boolean = args[2]
    booleans = args[3]
    y_fit = args[4]
    widest_interval = args[5]
    folder = kwargs.get("folder")
    relative_peak_val = spec_y[widest_interval,:] - y_fit
    mini = np.min(np.where(widest_interval))
   
    num = np.shape(spec_y)[1]            # number of spectra
    print(num)
  #  if num == 1:
   #    booleans = np.transpose([booleans])
    print(booleans['Suitable'],booleans['Melt'])
    for j in range(0,num):
                #print("\n ")
                fig, ax = plt.subplots()
                ax.axvspan(3510, 3590, facecolor='green', alpha=0.3)
                plt.plot(spec_x,spec_y[:,j])
              #    index = result['Peaks'].iloc[j]
                #print(peak_boolean[j])
                index = peak_boolean[:,j] 
                help_index1 = np.where(peak_boolean[:,j]) 
                
                #plt.vlines(spec_x[help_index1],y_fit[help_index2,j],y_fit[help_index2,j]+ relative_peak_val[help_index2,j])
                #plt.vlines(spec_x[help_index1],y_fit[help_index1,j],y_fit[help_index1,j]+ relative_peak_val[help_index1-mini,j])
                
                plt.vlines(spec_x[help_index1],y_fit[help_index1-mini,j],y_fit[help_index1-mini,j]+ relative_peak_val[help_index1-mini,j], linestyles='dotted')
                
      #         
     #           index = result['Peaks'].iloc[j]
      #          if number_of_peaks[j] > 0:
       #             print(number_of_peaks[j])
        #            print("location: ",spec_x[widest_interval][result['Peaks'].iloc[j]])
         #           print("height: ",result['peak_heights'].iloc[j])
                 #   print("prominences: ",result['prominences'].iloc[j])
                #    print("widths: ",result['widths'].iloc[j])
                #    print("width_heights: ",result['width_heights'].iloc[j])

                #plt.vlines(spec_x[peak_boolean[:,j]],0, spec_y[peak_boolean[:,j],j])
                
          #      plt.vlines(spec_x[widest_interval,j],0, 1)
                #help_index1 = np.where(peak_boolean[widest_interval,j]) 
               # help_index2 = np.where(peak_boolean[widest_interval,j]) 
                
              #  plt.vlines(spec_x[help_index1],y_fit[help_index2,j],y_fit[help_index2,j]+ relative_peak_val[help_index2,j])
               # if len(help_index1) != 0:
                #   plt.plot(spec_x[help_index1],spec_y[help_index1,j], "xk",color="black")
 
                plt.plot(spec_x[index],spec_y[index,j], "x",color="black")
       
                plt.plot(spec_x[widest_interval],y_fit[:,j])
#     
                if (booleans[j]['Suitable']):
                    suit_str =  'Sample: Suitable'
                else: 
                    suit_str =   'Sample: Unsuitable'
      
                if (booleans[j]['Melt']):
                    melt_str =  'Sample: Basalt'
                else: 
                    melt_str =   'Sample: No Basalt'
      
                nothing_str = 'nothing'
      
                plt.gca().invert_xaxis()
                plt.xlabel("Wavenumber $cm^{-1}$")
                plt.ylabel("Absorbance")
   
                #textstr = '\n'.join((r'$ %s$' % (water_str, ), r'$ %s$' % (suit_str, ),r'$ %s$' % (oli_str, ),r'$ %s$' % (fringe_str, ),r'$ %s$' % (cont_str, ),r'$ %s$' % (CO2_str, ) ))
                textstr = '\n'.join((r'$ %s$' % (suit_str, ),r'$ %s$' % (melt_str, ) ))
                textstr = get_text_string(booleans[j], "Melt")
   
                # place a text box in upper left in axes coords
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
       
                #add_interval(fig, ax)
                fig.savefig(''.join([folder,'/Graph_of_melt_plus_fit',str(j) ,'.png']), bbox_inches='tight')
                plt.close()
    return None     

def trace_graph(*args,**kwargs):
    spec_x = args[0]
    spec_y = args[1]
    peak_boolean = args[2]
    booleans = args[3]
    y_fit = args[4]
    widest_interval = args[5]
    mid_interval = args[6]
    folder = kwargs.get("folder")
    
    relative_peak_val = spec_y[widest_interval,:] - y_fit
    mini = np.min(np.where(widest_interval))

    num = np.shape(spec_y)[1]            # number of spectra

    # exclude peaks outside interval
    mid_interval = np.logical_or.reduce(mid_interval, axis=0)
    temp = [mid_interval]*num
    temp2 = np.transpose(temp) & peak_boolean
    peak_boolean[~temp2]  = 0

    for j in range(0,num):
            #    print("\n ")
                fig, ax = plt.subplots()
                plt.plot(spec_x,spec_y[:,j])
              #    index = result['Peaks'].iloc[j]
                #print(peak_boolean[j])
                index = peak_boolean[:,j] 
                plt.plot(spec_x[index],spec_y[index,j], "x",color="black")
     #           index = result['Peaks'].iloc[j]
      #          if number_of_peaks[j] > 0:
       #             print(number_of_peaks[j])
        #            print("location: ",spec_x[widest_interval][result['Peaks'].iloc[j]])
         #           print("height: ",result['peak_heights'].iloc[j])
                 #   print("prominences: ",result['prominences'].iloc[j])
                #    print("widths: ",result['widths'].iloc[j])
                #    print("width_heights: ",result['width_heights'].iloc[j])

                #plt.vlines(spec_x[peak_boolean[:,j]],0, spec_y[peak_boolean[:,j],j])
                
          #      plt.vlines(spec_x[widest_interval,j],0, 1)
                help_index1 = np.where(peak_boolean[:,j]) 
            #    help_index2 = np.where(peak_boolean[widest_interval,j]) 
                
                #plt.vlines(spec_x[help_index1],y_fit[help_index2,j],1)
               #
                plt.vlines(spec_x[help_index1],y_fit[help_index1-mini,j],y_fit[help_index1-mini,j]+ relative_peak_val[help_index1-mini,j], linestyles='dotted')
               # if len(help_index1) != 0:
                #   plt.plot(spec_x[help_index1],spec_y[help_index1,j], "xk",color="black")
 
                ax.axvspan(1495, 1535, facecolor='green', alpha=0.3)
                ax.axvspan(1410, 1450, facecolor='green', alpha=0.3)
                   
                plt.gca().invert_xaxis()
                plt.xlabel("Wavenumber $cm^{-1}$")
                plt.ylabel("Absorbance")
 
                plt.plot(spec_x[widest_interval],y_fit[:,j])
               
                textstr = get_text_string(booleans[j], "Trace")
   
                # place a text box in upper left in axes coords
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                fig.savefig(''.join([folder,'/Graph_of_trace_plus_fit',str(j) ,'.png']), bbox_inches='tight')
                plt.close()
    return None                    
  
##############################################
# Add colorbands
################################################
def add_interval(fig, ax, anno="ON"):
   fig = args[0]
   ax = args[1]
     # wavelength larger than 3800
  # ax.axvspan(4000, 3800, facecolor='red', alpha=0.3, linewidth=1)
   #if indi = "ON":
   #ax.annotate('Noise', xy=(3900, 0.75),  xycoords='data',
    #        bbox=dict(boxstyle="round", fc="none", ec="black"),
     #       xytext=(75, 10), textcoords='offset points', ha='center',
      #      arrowprops=dict(arrowstyle="->"))
             
   # wavelength  3550 +/- 25   
   
   ax.axvspan(3575, 3525, facecolor='green', alpha=0.3)
   
   
   if anno == "ON":
      ax.annotate(r'$H_2O$ peak',(0.1,0.5),  xycoords='axes fraction', bbox=dict(boxstyle="round", fc="none", ec="black"),
          xytext=(50, 10), textcoords='offset points', ha='center',arrowprops=dict(arrowstyle="->"))
   
   # wavelength  2250 - 2400
   ax.axvspan(2250, 2400, facecolor='red', alpha=0.3)
   if anno == "ON":
      ax.annotate("Atmospheric noise",(0.6,0.60), xycoords='axes fraction',bbox=dict(boxstyle="round", fc="none", ec="black"),
          xytext=(-75, 10), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="->"))

   # wavelength 2000- 1600
   ax.axvspan(1600, 2050, facecolor='red', alpha=0.3)
   if anno == "ON":
        ax.annotate("Olivine",(0.8,0.75),xycoords='axes fraction', bbox=dict(boxstyle="round", fc="none", ec="black"),
           xytext=(0, 20), textcoords='offset points', ha='center',          arrowprops=dict(arrowstyle="->"))
             
   # wavelength 1515 +/ 20 
   # wavelength 1430 +/ 20 
   ax.axvspan(1495, 1535, facecolor='green', alpha=0.5)
   ax.axvspan(1410, 1450, facecolor='green', alpha=0.5)
   if anno == "ON":
      ax.annotate(r'$CO_2$ peak',(0.93,0.85),xycoords='axes fraction', bbox=dict(boxstyle="round", fc="none", ec="black"),
             xytext=(0, 20), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="->"))
      ax.annotate("", (0.97, 0.845),xycoords='axes fraction', xytext=(0, 12), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="->"))

   # wavelength smaller than 1400
   ax.axvspan(1400, 750, facecolor='red', alpha=0.3)
   #ax.annotate("Noise",(1000,1.2),bbox=dict(boxstyle="round", fc="none", ec="black"),
    #   xytext=(0, 20), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="->"))
   
   # add intervals    
   if anno == "ON":
     ax2 = ax.twiny()
     #ax2.set_xlim([3800, 1350])
     ax2.set_xlim(ax.get_xlim())
    # ticks = [3015, 3000,2500, 2250]
     ticks = [3575, 3525,2400, 2250, 2050,1600, 1535,1495,1450,1410]
     ax2.set_xticks(ticks)
     ax2.set_xticklabels(ax2.get_xticks(), rotation = 50)  
     ax2.tick_params(axis='x', labelsize=6, pad=0.2)
   
   return fig, ax

