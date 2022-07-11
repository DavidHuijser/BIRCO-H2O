import numpy as np
###################################
# Calculate the densities 
###################################
# Read in 10 rows: SiO2, TiO2,Al2O3,Fe2O3,FeO,MnO,MgO,CaO,Na2O,K2O,H2O
# The first iteration start withH20-density of 0 
def calculate_density(data):
      dixo = ['SiO2_melt', 'TiO2_melt', 'Al2O3_melt', 'Fe2O3_melt', 'FeO_melt', 'MnO_melt', 'MgO_melt', 'CaO_melt', 'Na2O_melt', 'K2O_melt','H2O'] 
      dix_weight = [60.0843, 79.8788,101.9602,159.6922,71.8464,70.9375,40.3044,56.0774,61.9774,94.1960, 18.01528]
    #  volume = [26.86,23.16,37.42,42.13,13.65, 0.0, 11.69,16.53,28.88,45.07,26.27,18.01528,26.27]
      volume = [26.86,23.16,37.42,42.13,13.65,np.nan,11.69,16.53,28.88,45.07,26.27]
      alpha = np.double([0.0000,0.00724,0.00262,0.00909,0.00292,np.nan,0.00327,0.00374,0.00768,0.01191,0.00946])
      beta = np.double([-0.000189,-0.000231,-0.000226,-0.000253,-0.000045,np.nan,0.000027,0.000034,-0.00024,-0.000675,-0.000315])
      Temp_Kelvin = 293
      Pressure = 1
      chemistry = data.chemistry
      #find index all elements presents in  chemistry.keys()
      index = np.where(np.intersect1d(list(chemistry.keys()) , dixo))[0]
      # sum of all relative weights 
      weigth_percentages = np.array(list(chemistry.values()))[index]
      weigth_percentages = np.append(weigth_percentages,[0.0])
      #weigth_percentages = np.array([45.13,3.07,15.39,2.03,10.36,0.16,5.24,12.54,4.45,1.50,0.00])
      peaks = data.water_map 
      # Anhydrous
      molecular_proportions  = (weigth_percentages/dix_weight)*(100-weigth_percentages[10])/np.nansum(weigth_percentages)
      mole_fraction = molecular_proportions/np.nansum(molecular_proportions)
      #
      VT =mole_fraction*(volume+alpha*(Temp_Kelvin-1673.0)+beta*(Pressure-1.0))
      XiMi = mole_fraction*dix_weight
      # exclude MnO from summation
      XiMi[5] = np.nan
      #
      rho = np.nansum(XiMi)/np.nansum(VT)
      Density = 1000*rho
      #densities
      H20_peak = data.water_map.ravel()
      #Thickness = np.array([0.005,0.005]
      Thickness = data.thickness_map.ravel()
      print(Thickness)
      if (np.max(Thickness) < 30e-6):
             Thickness = np.array(len(Thickness)*0.005)
      Molar_absorbtion_H20 = 63
      Molar_weight_H20 = 18.01528
      #
      H2O_mass_fraction  = H20_peak*Molar_weight_H20 / (Density*Thickness*Molar_absorbtion_H20)
      densities = H2O_mass_fraction*100
      #
      l = len(alpha)        
      for i in range(10):
              # insert weights 
              weigth_percentages = np.array(len(H20_peak)*[np.array(list(chemistry.values()))[index]])
              weigth_percentages = np.append(weigth_percentages,[densities]).reshape(len(H20_peak),11)
              #weigth_percentages = np.array(len(H20_peak)*[np.array([45.13,3.07,15.39,2.03,10.36,0.16,5.24,12.54,4.45,1.50,0.00])])
              #weigth_percentages[:,10] = densities
              # hydrous
              tot = np.transpose(l*[(100-weigth_percentages[:,10])/np.nansum(weigth_percentages[:,:-1], axis=1)])
              molecular_proportions  = tot*(weigth_percentages/dix_weight)
              mole_fraction = molecular_proportions/np.transpose(l*[np.nansum(molecular_proportions,axis=1)])
              #       
              VT =mole_fraction*(volume+alpha*(Temp_Kelvin-1673.0)+beta*(Pressure-1.0))
              XiMi = mole_fraction*dix_weight
              XiMi[:,5] =np.nan
              rho = np.nansum(XiMi)/np.nansum(VT)
              Density = 1000*rho
              # Anhydrous    
              H2O_mass_fraction  = H20_peak*Molar_weight_H20 / (Density*Thickness*Molar_absorbtion_H20)
              densities = H2O_mass_fraction*100
              #H2O_mass_fraction  = H20_peak*Molar / (Sample_density*Sample_thickness*Molar_absorbtion)
              #densities = H2O_mass_fraction*100
     # calculate concentrations CO2
      Molar_absorbtion_CO2 = chemistry['CO2'] 
      Molar_weight_CO2 = 44.01
      #CO2_peak = np.array([0.33546464,0.33546464])
      CO2_peak = data.CO2_map.ravel()
      CO2_mass_fraction = CO2_peak*Molar_weight_CO2 / (Density*Thickness*Molar_absorbtion_CO2)
      CO2_ppm = CO2_mass_fraction*10000
      return  densities, CO2_ppm,Density 
    
