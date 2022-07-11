# Import the required Libraries
import tkinter as tk 
import post_process
#import easygui
#from tkinter import *
from tkinter import ttk, filedialog, font
from tkinter import Scrollbar
from tkinter.filedialog import askopenfile
from tkinter.filedialog import asksaveasfilename
from tkinter.messagebox import showerror
from tkinter import messagebox
from tkinter.messagebox import showinfo
import math_support              #my own module
import vectorized_features  
import pandas as pd
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.colorbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure 
matplotlib.use('Agg')
#matplotlib.use('TkAgg')

# now called a function that returns a percentage
# Progress bar widget
#import pandas as pd
#import csv
#import copy
#import os    # for path control
#import multiprocessing
from pandas import read_csv as pd_read_csv
from os.path import basename as os_pathbasename
import numpy as np
#import time
import utility16
import density
import threading
import cProfile
import pstats
#import sys
import re

#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
#from matplotlib.backend_bases import key_press_handler

# define basic gray colors
# set colors 
gray1 = '#ececec'   
gray2 = '#e8ecf0' 
#gray3 = '#e3e3e3'
#gray4 = 'gainsboro'
# bg="gray"
# bg="grey"
debug_mode = True
#inter_color = "pink"

inter_color = gray1
root_background_color = gray1
button_relief = None
image_background = "white"

#inter_color = "pink"
#root_background_color = "yellow"
#button_relief = tk.RIDGE
#image_background = "green"

class VerticalScrolledFrame(tk.Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling

    """
    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)            
        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=tk.VERTICAL)
        vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
        canvas = tk.Canvas(self, bd=0, highlightthickness=0, yscrollcommand=vscrollbar.set, bg=gray1)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.FALSE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = tk.Frame(canvas, bg=inter_color)
        interior_id = canvas.create_window(0, 0, window=interior,anchor=tk.NW)
                                           
        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=parent.winfo_width())
                canvas.config(height=parent.winfo_height())
                
                #canvas.config(width=300)
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=parent.winfo_width())

        canvas.bind('<Configure>', _configure_canvas)


def profiler(method):
        profile = cProfile.Profile()
        profile.runcall(method)
        ps = pstats.Stats(profile)
        ps.sort_stats('calls', 'cumtime') 
        #ps.print_stats()
        ps.print_stats(10)
        #ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        

# --- classes ---
class MyWindow:
    #############################################
    # The correct way to deal with button pressed and loading 
    ############################################
    def correct(self,button,method):
              button.config(state=tk.DISABLED, text='Calculating')
              submit_thread1 = threading.Thread(name='t1',target=method, args=(button,))
              submit_thread1.start()
              return None
              
    def __init__(self, parent):
        # defining the callback function (observer) for tracing the global variable app 
        def my_callback(var, index, mode):
                #print ("Traced variable {}".format(self.Validate_bool.get()))
                if self.Validate_bool.get():
                      self.calc_button.config(state='normal')
        
        # Active the Extrapolation Tickbox ONlY ifthe thickness is estimated. 
        def activateCheck():
          if (self.estimate_thickness_bool.get()):
              self.e8b.config(state=tk.NORMAL)
          else:     
              self.e8b.config(state=tk.DISABLED)
              self.extrapolate.set(0)
              
        # inherit parent   
        self.parent = parent
        self.frame = VerticalScrolledFrame(self.parent, bg="green")
        self.frame.pack(expand=True,padx = 6,pady = 6)
        self.testframe1 = self.frame.interior
        
        # Use relief for testing design
        relief = button_relief

        # Creating a Font object of "TkDefaultFont"
        self.defaultFont = font.nametofont("TkDefaultFont")
        #self.defaultFont.configure(family="Segoe Script",size=15, weight=font.BOLD)
        #self.defaultFont.configure(size=15, weight=font.BOLD)
        
        # declaring variables storing name variables and entries
        self.scans_var=tk.IntVar()        #[0]
        self.height_var=tk.IntVar()        #[1]
        self.width_var=tk.IntVar()             #[2]
        self.chem_filename    =tk.StringVar()         #[3]
        self.spectra_filename =tk.StringVar()           #[4]
        self.thickness_var=tk.IntVar()             #[5]
        self.thickness_var.set(1)                  #[5]
        self.thickness_var_error=tk.IntVar()             #[6]
        self.thickness_var_error.set(1)             #[6]
        self.estimate_thickness_bool =tk.BooleanVar() #[7]
        self.extrapolate =tk.BooleanVar()    #[8]
        self.number_of_iterations = tk.IntVar()            #[9]
        self.number_of_iterations.set(300)                  #[9]
        self.lightsource = tk.BooleanVar()   #[10] 0 = synchotron, 1 = Globar
        self.lightsource.set(False)                    # [10]
        
        ##### OUTPUT VARIABLES
        self.H2O_map_bool    =tk.BooleanVar()       #[1]
        self.CO2_map_bool    =tk.BooleanVar()       #[2]
        self.Graph_bool    =tk.BooleanVar(False)       #[3]
        self.Validate_bool    =tk.BooleanVar(False)       #[64
        self.debug_bool =tk.BooleanVar() 
        self.debug_bool.set(True) 
        
        ##### CONSISTENCY CHECKER VARIABLES
        self.check_chem   =tk.BooleanVar() 
        self.check_chem.set(False) 
        self.check_data   =tk.BooleanVar() 
        self.check_data.set(False)
        self.loaded_scans = tk.BooleanVar(False)
        self.loaded_height = tk.BooleanVar(False)
        self.loaded_width = tk.BooleanVar(False)
        self.loaded_spectra = tk.BooleanVar(False)
        self.loaded_chem = tk.BooleanVar(False)

        self.data= None
        self.comp= None
        self.df = None
        self.spec_x = None
        self.spec_y = None
        self.data_type = tk.IntVar()

        # registering the observer
        self.Validate_bool.trace('w', my_callback)

        # create a folder/directory for images and reports 
        self.report_folder = utility16.get_report_folder()
  
        ##########################################
        # SETUP 4 FRAMES / LABELS / CANVAS       #
        ##########################################
      #  label0 = tk.LabelFrame(self.testframe1,font=('Helvetica', 18, 'bold'), bg=gray1)
       # label0.grid(row=1, column=1,padx=5,pady=5, sticky="nsew")
        #label0.pack(fill=tk.BOTH)
        label1 = tk.LabelFrame(self.testframe1, text=" INPUT / OUTPUT ",font=('Helvetica', 18, 'bold'), bg=gray1)
        label1.grid(row=1, column=1,padx=5,pady=5, columnspan=20, sticky="nw")
        label1.grid_columnconfigure(6,minsize=200)  # Here
        #label2 = tk.LabelFrame(self.testframe1, text=" OUTPUT ",font=('Helvetica', 18, 'bold'), bg=gray1)
        #label2.grid(row=1, column=12,padx=5,pady=5, sticky="ne")
        label3 = tk.LabelFrame(self.testframe1, text=" DIALOG ",font=('Helvetica', 18, 'bold'), bg=gray1)
        label3.grid(row=10, column=1,padx=5,pady=5, sticky="nwe", columnspan=12)
        label3.grid_columnconfigure(6,minsize=500)  # Here
         # Create the canvas, size in pixels.
        self.canvas = tk.Canvas(self.testframe1, width=600, height=600, bg=image_background)
        self.canvas.grid(row = 16, column =1,padx=5,pady=5)
        
        ###########################################
        # create labels for components 
        ###############################################
        self.l1 = tk.Label(label1, text = "Number of Scans",anchor="w",bg=gray1, relief=relief).grid(row = 0, column = 1,columnspan=2, pady = 2,sticky = 'w')
        self.l1 = tk.Label(label1, text = "Height", anchor="w",bg=gray1, relief=relief).grid(row = 1, column = 1,columnspan=2, pady = 2,sticky = 'w')
        self.l1 = tk.Label(label1, text = "Width", anchor="w", bg=gray1, relief=relief).grid(row = 2, column = 1,columnspan=2, pady = 2,sticky = 'w')
        self.l1 = tk.Label(label1, text = "Spectral (data)",anchor="w",bg=gray1, relief=relief).grid(row = 3, column = 1,columnspan=2, pady = 2,sticky = 'w')
        self.l1 = tk.Label(label1, text = "Compositions (data)",bg=gray1, relief=relief).grid(row = 4, column = 1, pady=2,columnspan=2,sticky = 'w')
        self.l1 = tk.Label(label1, text = "Thickness (\u03bcm) ",bg=gray1, relief=relief).grid(row = 5, column = 1, pady = 2,columnspan=2,sticky = 'w')
        self.l1 = tk.Label(label1, text = "Thickness error (\u03bcm) ",bg=gray1, relief=relief).grid(row = 6, column = 1, pady = 2,columnspan=2,sticky = 'w')
        self.l1 = tk.Label(label1, text = "Use fringes:",bg=gray1, relief=relief).grid(row = 7, column = 1, pady=2,columnspan=2,sticky = 'w')
        self.l1 = tk.Label(label1, text = "Extrapolate:",bg=gray1, relief=relief).grid(row = 8, column = 1, pady=2,columnspan=2,sticky = 'w')
        self.l1 = tk.Label(label1, text = "Number of iterations ",bg=gray1, relief=relief).grid(row = 9, column = 1,columnspan=2, pady = 2,sticky = 'w')
        self.l1 = tk.Label(label1, text = "Light source",bg=gray1, relief=relief).grid(row = 10, column = 1,columnspan=2, pady = 2,sticky = 'w')
      
        self.e1 = tk.Entry(label1,validate='key',textvariable = self.scans_var, relief=relief)
        self.e1['validatecommand'] = (self.e1.register(self.testVal),'%P','%d')
        self.e1.grid(row = 0, column = 5, pady = 2,sticky = 'W')
        
        self.e2 = tk.Entry(label1,validate="key",textvariable = self.height_var, relief=relief)
        self.e2.grid(row = 1, column = 5, pady = 2,sticky = 'W')
        self.e2['validatecommand'] = (self.e2.register(self.testVal),'%P','%d')

        self.e3 = tk.Entry(label1,validate="key",textvariable = self.width_var, relief=relief)
        self.e3.grid(row = 2, column = 5, pady = 2,sticky ='W')
        self.e3['validatecommand'] = (self.e3.register(self.testVal),'%P','%d')
        
        # load data buttons
        self.load_button4 = tk.Button(label1, text='LOAD SPECTRAL DATA',anchor="w", command=lambda: self.press_loadfile(1), relief=relief)
        self.load_button4.grid(row =3, column = 5, pady = 2, sticky='W')
        self.load_button5 = tk.Button(label1, text='LOAD CHEMICAL DATA',anchor="w", command=lambda: self.press_loadfile(2), relief=relief)
        self.load_button5.grid(row =4, column = 5, pady = 2, sticky='W')

        self.e6 = tk.Entry(label1,validate='key',textvariable = self.thickness_var, relief=relief)
        self.e6.grid(row = 5, column = 5, pady = 2,sticky = 'W')
        self.e6['validatecommand'] = (self.e6.register(self.testVal),'%P','%d')
        
        self.e7 = tk.Entry(label1,validate='key',textvariable = self.thickness_var_error, relief=relief)
        self.e7.grid(row = 6, column = 5, pady = 2,sticky = 'W')
        self.e7['validatecommand'] = (self.e7.register(self.testVal),'%P','%d')
        
        self.e8 = tk.Checkbutton(label1, text="", variable=self.estimate_thickness_bool,bg=gray1, command=activateCheck,relief=relief)
        self.e8.grid(row=7, column=5,  sticky='W')
        
        self.e8b = tk.Checkbutton(label1, text="", variable=self.extrapolate,bg=gray1, relief=relief,state=tk.DISABLED)
        self.e8b.grid(row=8, column=5,  sticky='W')
      
        # active submite button if validation is completed. 
        self.e9 =  tk.Entry(label1,validate='key',textvariable = self.number_of_iterations, relief=relief)
        self.e9.grid(row=9, column=5,  sticky='W')
  
        self.e10a = tk.Radiobutton(label1, text="Synchotron", var=self.lightsource, value=0,bg=gray1).grid(row=10, column=5,  sticky='W')
        self.e10b = tk.Radiobutton(label1, text="Globar", var=self.lightsource, value=1,bg=gray1).grid(row=11, column=5,  sticky='W')
  
  
        ################################
        # OUTPUT  
        #################################
        # ld = label dialog
        # ed = entry dialg
        self.ld1 = tk.Label(label1, text = "H20 Maps",anchor="w",bg=gray1, relief=relief).grid(row = 0, column = 12, pady = 5,sticky = 'W')
        self.eo1 = tk.Checkbutton(label1, text="", variable=self.H2O_map_bool,bg="seashell2", relief=relief).grid(row=0, column=15,pady=2,  sticky='W')
        
        self.out1 = tk.Label(label1, text = "CO2 Maps",anchor="w",bg=gray1, relief=relief).grid(row = 1, column = 12, pady = 5,sticky = 'W')
        var3 = tk.IntVar()
        tk.Checkbutton(label1, text="", variable=self.CO2_map_bool,bg=gray1, relief=relief).grid(row=1, column=15,pady=2,  sticky='W')
        
        self.out1 = tk.Label(label1, text = "Graphs",anchor="w",bg=gray1, relief=relief).grid(row = 2, column = 12, pady = 5,sticky = 'W')
        var4 = tk.IntVar()
        tk.Checkbutton(label1, text="", variable=self.Graph_bool,bg=gray1, relief=relief).grid(row=2, column=15,pady=2,  sticky='W')
      
        self.out1 = tk.Label(label1, text = "Debug",anchor="w",bg=gray1, relief=relief).grid(row = 3, column = 12, pady = 5,sticky = 'W')
        var5 = tk.IntVar()
        tk.Checkbutton(label1, text="", variable=self.debug_bool,bg=gray1, relief=relief).grid(row=3, column=15,pady=2,  sticky='W')
      
                
        ################################
        # DIALOG FRAME 
        #################################
        self.d1 = tk.Label(label3, text = " SPECTRAL DATA NOT LOADED "  ,anchor="w",bg=gray1, fg="red", relief=relief)
        self.d1.grid(row = 12, column = 1, pady = 2,sticky = 'w', columnspan=12)
        
        self.d2 = tk.Label(label3, text = " COMPOSITION DATA NOT LOADED ",anchor="w",bg=gray1, fg="red", relief=relief)
        self.d2.grid(row = 13, column = 1, pady = 2,sticky = 'w',columnspan=12)
      
        self.default_button = tk.Button(label3, text='DEFAULT',  command  =lambda: self.correct(self.default_button, self.set_default), relief=relief)
        self.default_button.grid(row = 14, column = 1,  columnspan=1,pady = 5, padx=5,sticky = 'w')
        
        self.default_button2 = tk.Button(label3, text='DEFAULT SINGLE',  command  =lambda: self.correct(self.default_button2, self.set_default2), relief=relief)
        self.default_button2.grid(row = 15, column = 1,  columnspan=1,pady = 5, padx=5,sticky = 'w')
        
        self.validate_button = tk.Button(label3, text='VALIDATE', command  =lambda: self.correct(self.validate_button,self.validate_all), relief=relief)
        self.validate_button.grid(row = 14, column = 2, columnspan=1,pady = 5, padx=5,sticky = 'w')
   
         # Calculate button which is activated once the validation is complete        
        self.calc_button = tk.Button(label3,text="CALCULATE", command=lambda: self.correct(self.calc_button, self.script_python), relief=relief)
        self.calc_button.grid(row = 14, column = 4, pady = 2,sticky = 'w')
        
        # active submite button if validation is completed. 
        if (self.Validate_bool.get()):
             print("All good to go")
             self.calc_button.config(state=tk.NORMAL)
        else:     
             self.calc_button.config(state=tk.DISABLED)
        
        #########################################
        # profiling lines     
        ##########################################
       #profiler(self.script_python)   #(run here, is a bit slow)   
        #profiler(self.validate_all)   #(runs at the end of set default is ok)   
        #root.update()
        #root.updat()
        ### END INIT ###

  #  def start_submit_thread(event):
   #   global submit_thread
    #  submit_thread = threading.Thread(target=submit)
     # submit_thread.daemon = True
    #  progressbar.start()
    #  submit_thread.start()
    #  root.after(20, check_submit_thread)

  #  def check_submit_thread():
   #   if submit_thread.is_alive():
    #    root.after(20, check_submit_thread)
     # else:
      # d    progressbar.stop()

    # validate if entry is a digit        
    def testVal(self,inStr,acttyp):
        if acttyp == '1': #insert  
            if not inStr.isdigit():
                return False
            return True

    def set_default(self, button):
           self.scans_var.set(32)
           self.height_var.set(57)  
           self.width_var.set(27)  
           
           folder = "/Users/davidhuijser/Documents/Freelance Work/Elaine - March 2021/Data/"
       #    os.chdir("/Users/davidhuijser/Documents/Freelance Work/Elaine - March 2021")

           filename2 = "BIROC-H2O_Input_Example.csv"
           #filename ="66189ME1-1a_BG256_15.66H_32scan_0.15PH_27x57_4umstep_CORR_DATAPTTABLE.0.dpt"
           filename ="TEST_FILE_15x15.dpt"
           self.height_var.set(15)  
           self.width_var.set(15)  

           self.estimate_thickness_bool.set(True)
           self.thickness_var.set(50)  
           self.thickness_var_error.set(50)  
           self.number_of_iterations.set(50)
           self.extrapolate.set(True)
           self.spectra_filename.set("".join([folder,filename]))
           #self.df = pd_read_csv(self.spectra_filename.get() , sep='\t')
           self.df = pd_read_csv(self.spectra_filename.get() , sep='\t+|,',engine="python",header=None)
           lst = np.asarray(self.df.values[:,0])
             # check if ascendent sorted:
           if not (all(lst[:-1] <= lst[1:]) or all(lst[:-1] >= lst[1:])):
                      self.df = pd_read_csv(self.spectra_filename.get() , sep='\t+|,',engine="python")
           self.check_data, self.spec_x,self.spec_y,self.nos,self.now = utility16.load_data(self.df, x_size = self.width_var.get(),y_size= self.height_var.get(), folder=self.report_folder, debug_mode=True)

          
          
           #print(np.shape(self.data),np.shape(self.df))   
           self.chem_filename.set("".join([folder,filename2]))
           self.df = pd_read_csv(self.chem_filename.get() )
           self.check_chem, self.comp,self.comp_error  = utility16.load_chemical_composition(self.df, debug_mode=True)

           print_data_name = "".join([ "Succesfully loaded file: " , os_pathbasename(self.spectra_filename.get())])
           self.d1.config(text=print_data_name, fg='black')
        
           print_chem_name = "".join(["Succesfully loaded file: ", os_pathbasename(self.chem_filename.get())])
           self.d2.config(text=print_chem_name, fg='black')
           self.default_button.config(text="Done")
           print("------------------------------finished Default------------------------------")
           return None 
         
    
    def set_default2(self, button):
           self.scans_var.set(32)
           self.height_var.set(1)  
           self.width_var.set(1)  
           
           folder = "/Users/davidhuijser/Documents/Freelance Work/Elaine - March 2021/Data/"
       #    os.chdir("/Users/davidhuijser/Documents/Freelance Work/Elaine - March 2021")

           filename2 = "BIROC-H2O_Input_Example.csv"
           #filename ="66189ME1-1a_BG256_15.66H_32scan_0.15PH_27x57_4umstep_CORR_DATAPTTABLE.0.dpt"
           filename ="1R MI 1.csv"
           self.height_var.set(1)  
           self.width_var.set(1)  

           self.estimate_thickness_bool.set(False)
           self.extrapolate.set(False)
           
           self.thickness_var.set(50)  
           self.thickness_var_error.set(50)  
           self.number_of_iterations.set(5)
           
           self.lightsource.set(1)
          
           self.spectra_filename.set("".join([folder,filename]))
           #self.df = pd_read_csv(self.spectra_filename.get() , sep='\t')
           #self.df = pd_read_csv(self.spectra_filename.get() , sep='\t+|,',engine="python",header=None)
           self.df = pd_read_csv(self.spectra_filename.get() , sep='\t+|,',engine="python",header=None)
           lst = np.asarray(self.df.values[:,0])
             # check if ascendent sorted:
           if not (all(lst[:-1] <= lst[1:]) or all(lst[:-1] >= lst[1:])):
                      self.df = pd_read_csv(self.spectra_filename.get() , sep='\t+|,',engine="python")
           
           self.check_data, self.spec_x,self.spec_y,self.nos,self.now = utility16.load_data(self.df, x_size = self.width_var.get(),y_size= self.height_var.get(), folder=self.report_folder, debug_mode=True)
  
          
          
           #print(np.shape(self.data),np.shape(self.df))   
           self.chem_filename.set("".join([folder,filename2]))
           self.df = pd_read_csv(self.chem_filename.get() )
           self.check_chem, self.comp,self.comp_error  = utility16.load_chemical_composition(self.df, debug_mode=True)

           print_data_name = "".join([ "Succesfully loaded file: " , os_pathbasename(self.spectra_filename.get())])
           self.d1.config(text=print_data_name, fg='black')
        
           print_chem_name = "".join(["Succesfully loaded file: ", os_pathbasename(self.chem_filename.get())])
           self.d2.config(text=print_chem_name, fg='black')
           self.default_button.config(text="Done")
           print("------------------------------finished Default------------------------------")
           return None 
         
    
      

    ################################
    # METHOD TO LOAD FILES (either Chem-file or Spectral file )  
    #################################
    def press_loadfile(*args, **kwargs):
        # load file name, and indicate if file ind = spectra(1) or chem(2)  
        self = args[0]
        ind_file = args[1]
        # Load SPECTRA 
        if ind_file == 1:
            name =  tk.filedialog.askopenfilename(filetypes=[('CSV',('*.csv','*.dpt')), ('Excel', ('*.xls', '*.xslm', '*.xlsx'))])
            #name =  tk.filedialog.askopenfilename(filetypes=[('DPT','*.dpt','*.csv','*.CSV'), ('Excel', ('*.xls', '*.xslm', '*.xlsx'))])    
          #  try:  
            if name:
                if name.endswith(('.dpt','.csv','.CSV')):
                   #self.df = pd_read_csv(name , sep='\t+|,',engine="python",header=None)
                   self.df = pd_read_csv(name , sep='\t+|,',engine="python",header=None)
                   lst = np.asarray(self.df.values[:,0])
                   # check if ascendent sorted:
                   if not (all(lst[:-1] <= lst[1:]) or all(lst[:-1] >= lst[1:])):
                         self.df = pd_read_csv(name , sep='\t+|,',engine="python")
                   #self.df = pd.read_csv(name,sep=',')
                else:
                   self.df = pd.read_excel(name)
                self.spectra_filename.set(name)
                self.loaded_spectra.set(True)
                
                # Load_data will check format
                self.check_data, self.spec_x, self.spec_y,self.nos, self.now = utility16.load_data(self.df, x_size = self.width_var.get(),y_size= self.height_var.get(), debug_mode=True,folder=self.report_folder)
  
                print_data_name = "Succesfully loaded file: " + os.path.basename(self.spectra_filename.get())
                self.d1.config(text=print_data_name, fg='black')
         #   except:
            else:
                print("Error loading Data file in [GUI.py]")
                messagebox.showerror("Error 1", "There seems to something wrong with the spectral data file.")
                self.loaded_spectra.set(False)
       
        #load chem
        if ind_file == 2:
          name =  tk.filedialog.askopenfilename(filetypes=[('CSV', '*.csv',), ('Excel', ('*.xls', '*.xslm', '*.xlsx'))])
          try: 
            if name:
              if name.endswith('.csv'):
                  self.df = pd.read_csv(name)
              else:
                  self.df = pd.read_excel(name)
              self.chem_filename.set(name)
              self.loaded_chem.set(True)
              # now we need to check the information entered
              self.check_chem, self.comp, self.comp_error = utility16.load_chemical_composition(self.df, debug_mode=True)
              print_chem_name = "Succesfully loaded file: " + os.path.basename(self.chem_filename.get())
              if self.check_chem  == False:
                   raise  
              self.d2.config(text=print_chem_name, fg='black')
          except:
                 self.loaded_chem.set(False)
                 messagebox.showerror("Error 2", "There seems to something wrong with the composition data file.")
        return None        



    def validate_all(self,button):
        validated = utility16.validate(self)
        if (validated):
           button.config(text='Done')        
        print("------ FINSISHED -------")
        return None


        #print("CHEM", self.check_chem.get())
        #if self.check_chem.get() == False:
        #    messagebox.showerror("Error 0", "There seem to something wrong with the chemistry composition data file.")
        #if self.check_data.get() == False:
        #    messagebox.showerror("Error 1", "There seem to something wrong with the spectral data file.")
        #if (height == 0) | (width==0)  :
        #    messagebox.showerror("Error 2", "Incorrect dimension entered.")
        
             #MyWindow.label3.submit_button.config(state=tk.NORMAL)
        #else:     
         #    self.Validate_bool.set(False)
        #print("Loaded " + filename)
       # if self.Validate_bool.get():
        #      self.submit_button.config(state=tk.NORMAL)
      #  else:
       #       self.submit_button.config(state=tk.DISABLED)
        
        
        
        #print(filename)
        
        #print("Loaded " , loaded_boolean)
        #extrapolate_bool = self.extrapolate_bool.get()
        

        
        # ask for file if not loaded yet
        #if self.df is None:
         #   self.load_file()
        
        
        #print(self.df)
        
    



 

    #def submit():
        #time.sleep(100) # put your stuff here    
     #   self.script_python
    
    def script_python(self,button):

        #####################################################################################
        # First do a sanity check and see if all numbers got transfered correctly. 
        #####################################################################################
        # import entered values 
        scans = self.scans_var.get()
        height = self.height_var.get()
        width =  self.width_var.get()
        thickness =  self.thickness_var.get()
      
        name_spectra = os_pathbasename(self.spectra_filename.get())
        name_chem = os_pathbasename(self.chem_filename.get())
        
        loaded_chem  =  self.loaded_chem.get()
        loaded_spectra =  self.loaded_spectra.get()

        H2O_map_bool = self.H2O_map_bool.get()
        CO2_map_bool=  self.CO2_map_bool.get()
        Graph_bool=  self.Graph_bool.get()
        
        print("-----------------SCRIPT STARTED------------------------------")
        print("Height ", height)
        print("Scans ", scans)
        print("Width ", width)
        print("Thicknes ", thickness)
        
        print("Loaded Data: " , loaded_spectra)
        print("Loaded Chem: " , loaded_chem)
        
        print("Flags:")
        print("H2O Map ", H2O_map_bool)
        print("CO2 Map ",CO2_map_bool)
        print("Graphs ",Graph_bool)
        
        #self.lightsource = tk.BooleanVar()   #[10] 0 = synchotron, 1 = Globar
        if self.lightsource.get():
          print("Light Source Type: Globar")
        else:
          print("Light Source Type: Synchrotron")
          
        test_mode = True
        print("Number of iteratios ", self.number_of_iterations.get())
        print("Extrapolate ", self.extrapolate.get())
        
        print("final")
        ###################################
        # obtain error distributions  
        ##################################
        error_ = utility16.calc_spectrum_error(self.spec_y,self.spec_x, int(width), int(height),self.lightsource.get(),folder=self.report_folder,scans=self.scans_var.get() )
        
        #for thread in threading.enumerate(): 
         #     print(thread.name)
        
        ###################################
        # Create Progress bar plus Popup-window
        ##################################
        N_itt = self.number_of_iterations.get()
        
        #start progress bar
        style = ttk.Style(root)
        style.layout('text.Horizontal.TProgressbar',
             [('Horizontal.Progressbar.trough',
               {'children': [('Horizontal.Progressbar.pbar',
                              {'side': 'left', 'sticky': 'ns'})],
                'sticky': 'nswe'}),
              ('Horizontal.Progressbar.label', {'sticky': ''})])
        style.configure('text.Horizontal.TProgressbar', text='0 %')
         
        popup = tk.Toplevel()
        
        tk.Label(popup, text="Processing. Please don't close this winodw. ").grid(row=0,column=0)

        progress = 0
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(popup, variable=progress_var, maximum=100,style='text.Horizontal.TProgressbar')
        progress_bar.grid(row=1, column=0)#.pack(fill=tk.X, expand=1, side=tk.BOTTOM)
        popup.pack_slaves()

        progress_step = float(100.0/N_itt)
          
        # my list where all the data is store as a list of class objectes   
        mylist = [] 
        thicknesses = []
        value = 1 
        def button_command(popup,progress_var, progress):
            for i in range(N_itt):
                popup.update()
                print("###############################################################")
                
                ##################################################################
                # draw refraction index from distribution 
                ####################################################################
                chemistry,Succes = utility16.calc_refraction_index(self.comp, self.comp_error,debug_mode=debug_mode)       
                ##################################################################
                # Detect features  
                ####################################################################
                ClassObject = utility16.detect_features_vectorized(self.spec_x, self.spec_y,chemistry, self.report_folder, error_ ,xsize=int(width), ysize= int(height), debug_mode=debug_mode, test=test_mode, GenImg=True,UseFringes = self.estimate_thickness_bool.get(),Extrapolate = self.extrapolate.get())
                ClassObject.printing()
                
                ##################################################################
                # Calculate Thickness 
                ##################################################################
                print("Calculate Thickness")
                print(self.estimate_thickness_bool.get())
                if self.estimate_thickness_bool.get():
                     thickness, Succes_thickness = vectorized_features.calculate_thickness_vectorized(ClassObject,chemistry,self.report_folder,int(width), int(height), output="ON",Extrapolate = self.extrapolate.get())
                if (self.estimate_thickness_bool.get() == False) or (Succes_thickness ==False):
                     print("Thickness can not be determine by fringes and/or extrapolation")
                     thickness = self.thickness_var.get()*np.ones(self.nos).reshape(self.nos)
                ClassObject.thickness_map= thickness
                ClassObject.H2O_concentration,ClassObject.CO2_concentration,ClassObject.Density = density.calculate_density(ClassObject) 

                ##################################################################
                # Append Result to list 
                ##################################################################
                mylist.append(ClassObject)
                
                #thicknesses.append(thickness)
                #thickness = vectorized_features.calculate_thickness_vectorized(ClassObject.peak2peak_map, ClassObject.water_map, ClassObject.olivine_map, ClassObject.booleans,Oli_refrac, Melt_refrac,self.report_folder,int(width), int(height), output="ON")
                #print(np.shape(ClassObject.thickness_map))
                #print(np.shape(thickness))
                pindex = ~np.isnan(thickness)
 #               print("pindex", np.where(pindex))


                progress += progress_step
                progress_var.set(progress)
                style.configure('text.Horizontal.TProgressbar', text='{:g} %'.format(progress))
                
                print("total water", np.sum(ClassObject.water_map))
                print("total Olivine", np.sum(ClassObject.olivine_map))
                print("total CO2", np.sum(ClassObject.CO2_map))



            return 0, mylist

        value, mylist = button_command(popup,progress_var,progress)
        print(np.shape(mylist))
        print("XXXXXXXXXXXXXXXXX FINISHED LOOP XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        
        #obtain all distributions 
        print("store datacube to file ")
        utility16.save_data(mylist,self.report_folder)
        # post-process and store to file 
        print("Post-process data and save to CSV ")
        dataframe = post_process.postprocess(mylist,self.report_folder) 
        
        #l = np.shape(mylist)[0]
        #new_peak_array = list(map(lambda x: mylist[x].peak2peak_map,range(l)))
        #print(len(new_peak_array)) 
        #print(len(np.unique(new_peak_array))) 
        #print(max(new_peak_array)-min(new_peak_array)) 
        
        if value ==0:
            messagebox.showinfo('', 'Completed. You can close this window to finish.')
          
            progress_bar.destroy() 
            popup.destroy()
            
            print("Finished this part")
              #start progress bar
            
            #succes_popup = tk.Toplevel()
            #tk.Label(succes_popup, text="Calculations succesfull.").grid(row=0,column=0)
            #succes_popup.update() 
            
        #tk.Button(root, text="Launch", command=button_command).pack()

      #  Button(self.testframe1, text='Start', command=step).pack()
       # root.update()
        #print("current thread",threading.get_ident() )
        print("Executed succesfully")    
        button.config(text='Done')  
        return None







if __name__ == '__main__':
    
    root = tk.Tk()
    root.title('BIROC-H2O')
    root.geometry("1000x800")
   
    root.configure(bg=root_background_color)
    root.update() 
    
    top = MyWindow(root)
    root.mainloop()
    
  
