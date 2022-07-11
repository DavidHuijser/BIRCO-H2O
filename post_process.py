import numpy as np
import pandas as pd


def postprocess(data,folder):
  if ((data[0].xsize == 1) & (data[0].ysize == 1)):
      df = single_postprocess(data)
      print("Single process")
  else: 
      print("Multi process")
      df =multi_postprocess(data)
  #save df 
  name = "".join([folder , "/out.csv"])  
  df.to_csv(name)  
  return df 


def single_postprocess(data):
    array_names = ['water_map', 'CO2_map', 'thickness_map','H2O_concentration', 'CO2_concentration']
    l = len(data)
    i = np.arange(l)
    # generate index for the 75% and 100% percent intervals            
    Index_100 = np.arange(l)         
    Index_80 = np.arange(int(round(0.7*l)))         
    # get booleans 
    bools = np.asarray(list(map(lambda x: getattr(data[x], 'booleans'),i)))
    Melt = np.average(bools['Melt'],axis=0)
    Host = np.average(bools['Host'],axis=0)
    Cont = np.average(bools['Cont'],axis=0)
    Fringes = np.average(bools['Fringes'],axis=0)
    x_size = data[0].xsize
    y_size = data[0].ysize
    for j in range(0,len(array_names)):
            print(array_names[j])
            # identify good samples  
            temp2 = np.asarray(list(map(lambda x: getattr(data[x], array_names[j]),i)))
            index_good = bools['Melt'] & bools['Suitable'] & ~bools['Cont']
            index_good_80 = bools['Melt'][Index_80] & bools['Suitable'][Index_80] & ~bools['Cont'][Index_80]
            # If the difference between 0 and 1, the sample is good AND considered 'converged'   
            unit = np.abs(np.std(temp2[Index_100][index_good],axis=0) - np.std(temp2[Index_80][index_good_80],axis=0)/np.std(temp2[Index_100][index_good],axis=0)) 
            CONVERGED =  ((unit < 1) & (unit >= 0))
            # Calculated the error of the 
            mean_val = np.mean(temp2[index_good],axis=0)
            # Calculated the error of the 
            error_val = np.std(temp2[index_good],axis=0)
            # Calculates for each pixel how often the 'water' is detected
            probability = np.mean(index_good, axis=0)[0]
            # add to dataframe 
            if (array_names[j] == 'water_map'):
                  data_line = {'WP':[mean_val],'WP Err':[error_val],'WP Prob':[probability]}
                  df = pd.DataFrame(data_line)
            if (array_names[j] == 'CO2_map'):
                  df['CP'] = mean_val
                  df['CP Err'] = error_val
                  df['CP Prob'] = probability
            if (array_names[j] == 'thickness_map'):
                  df['T'] = mean_val
                  df['T Err'] = error_val
                  df['T Prob'] = probability
            if (array_names[j] == 'H2O_concentration'):
                  df['H2O Con'] = mean_val
                  df['H2O Con Err'] = error_val
                  df['H2O Con Prob'] = probability
            if (array_names[j] ==  'CO2_concentration'):      
                  df['CO2 Con'] = mean_val
                  df['CO2 Con Err'] = error_val
                  df['CO2 Con Prob'] = probability
                  #empty.append(mean_val,error_val, probability)
                  #column_names.append()  
    return df

def multi_postprocess(data):
    array_names = ['water_map', 'CO2_map','H2O_concentration', 'CO2_concentration']
    l = len(data) 
    i = np.arange(l)
    # generate index for the 75% and 100% percent intervals            
    Index_100 = np.arange(l)         
    Index_80 = np.arange(int(round(0.7*l)))         
    # get booleans 
    bools = np.asarray(list(map(lambda x: getattr(data[x], 'booleans'),i)))
    Melt = np.average(bools['Melt'],axis=0)
    Host = np.average(bools['Host'],axis=0)
    Cont = np.average(bools['Cont'],axis=0)
    Fringes = np.average(bools['Fringes'],axis=0)
    x_size = data[0].xsize
    y_size = data[0].ysize
    for j in range(0,len(array_names)):
            print(array_names[j])
            # identify good samples  
            temp2 = np.asarray(list(map(lambda x: getattr(data[x], array_names[j]),i)))   
            index_good = bools['Melt'] & bools['Suitable'] & ~bools['Cont']
            index_good_80 = bools['Melt'][Index_80] & bools['Suitable'][Index_80] & ~bools['Cont'][Index_80]
            # set fault indices to NaN 
            temp2[~index_good] = np.nan
            # If the difference between 0 and 1, the sample is good AND considered 'converged'   
            unit = np.abs(np.nanstd(temp2[Index_100,:],axis=0) - np.nanstd(temp2[Index_80,:],axis=0)/np.nanstd(temp2[Index_100,:],axis=0)) 
            CONVERGED =  ((unit < 1) & (unit >= 0))
            # Calculated the error of the 
            mean_val = np.mean(temp2,axis=0)
            # Calculated the error of the 
            error_val = np.std(temp2,axis=0)
            # Calculates for each pixel how often the 'water' is detected
            probability = np.mean(index_good, axis=0)
            # add to dataframe 
            if (array_names[j] == 'water_map'):
                  data_line = {'WP':mean_val,'WP Err':error_val,'WP Prob':probability}
                  df = pd.DataFrame(data_line)
            if (array_names[j] == 'CO2_map'):
                  df['CP'] = mean_val
                  df['CP Err'] = error_val
                  df['CP Prob'] = probability
            if (array_names[j] == 'thickness_map'):
                  df['T'] = mean_val
                  df['T Err'] = error_val
                  df['T Prob'] = probability
            if (array_names[j] == 'H2O_concentration'):
                  df['H2O Con'] = mean_val
                  df['H2O Con Err'] = error_val
                  df['H2O Con Prob'] = probability
            if (array_names[j] ==  'CO2_concentration'):      
                  df['CO2 Con'] = mean_val
                  df['CO2 Con Err'] = error_val
                  df['CO2 Con Prob'] = probability
                  #empty.append(mean_val,error_val, probability)
                  #column_names.append()  
    return df
