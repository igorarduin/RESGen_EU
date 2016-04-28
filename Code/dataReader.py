import os
import pandas as pd
import numpy as np    
from datetime import datetime, time
import gc
import math
from scipy.interpolate import interp1d
import random

class expando:
    pass 


#Function needed to define distances between nodes from longitudes and latitudes
def distance_from_long_lat(lat1, long1, lat2, long2):
    # Convert latitude and longitude to spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0  
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    # Compute spherical distance from spherical coordinates.
    # For two locations in spherical coordinates (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) = sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + 
           math.cos(phi1)*math.cos(phi2))
    if cos>1:#numerical approximations can bring to a number slightly >1
        cos=1
    arc = math.acos( cos )
    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    R_earth = 6371 #km
    arc = arc * R_earth
    return arc


#climatology cdf/inv_cdf function 
def clim_cdf(input_data_t, input_data_hd, cdf_keyword):
    if len(input_data_t)>0:
        quantiles = {}
        probabilities = np.arange(1,len(input_data_t)+1)/(float(len(input_data_t))+1)     
        quantiles = sorted(input_data_t)
        quantiles_extended = \
        np.concatenate([[-1], quantiles, [1]])
#        np.concatenate([[min(quantiles)*0 - 1], quantiles, [max(quantiles)*2 + 1000]])
        quantiles_extended[quantiles_extended < 0] = 0.
#        quantiles_extended = np.concatenate([[-1],quantiles_extended,[10e8]])
        quantiles_extended = np.concatenate([[-0.001],quantiles_extended,[1]])
        probabilities_extended = np.concatenate([[-0.0000001,0.],probabilities,[1,1.000001]])
    else:
        quantiles_extended = np.array([0.0,max(input_data_hd)])
        probabilities_extended = np.zeros(len(quantiles_extended))
    if cdf_keyword == 'cdf':
        interpolation = interp1d(quantiles_extended, probabilities_extended)
    elif cdf_keyword == 'inv_cdf':
        interpolation = interp1d(probabilities_extended, quantiles_extended)
    return interpolation


class dataReader:
    
     def __init__(self, countries,max_number_loc,renewable_type,data_type,start_time,
                  end_time,nbr_leadTimes,folder_location):         
        self._set_attributes(countries,max_number_loc,renewable_type,data_type,start_time,
                             end_time,nbr_leadTimes,folder_location)
        self._check_countries()  
        self._load_observations()
        self._tod_observations()
        self._load_forecasts()
        self._get_distances()
        self._set_climatology()
        print('Data has been imported!')  
        
        pass
        

     def _set_attributes(self, countries,max_number_loc,renewable_type,data_type,start_time,
                         end_time,nbr_leadTimes,folder_location):
                            
        self.attributes = expando()
        self.attributes.renew_type = renewable_type
        self.attributes.data_type = data_type
        self.attributes.folder_loc = folder_location
        self.attributes.start_time = start_time
        self.attributes.end_time = end_time
        self.attributes.nbr_leadT = nbr_leadTimes
        self.attributes.countries = countries
        self.attributes.max_number_loc = max_number_loc
        
        self.metadata = expando()
        

        pass
    
    
     #Function that check the input countries and display and error message if they
     #don't correspond. Returns the available countries and indices from the nodes
     def _check_countries(self):

         self.metadata.network_nodes = pd.read_csv(self.attributes.folder_loc+'Metadata/network_nodes.csv', 
                                     sep=',')
         available_countries = set(self.metadata.network_nodes.country)
         countries = self.attributes.countries
         if bool(countries-available_countries.intersection(countries)): 
             print ', '.join(countries-available_countries.intersection(countries)) + \
             ' are not in the country list. ' + 'See in:' + ', '.join(available_countries)
         self.attributes.countries = list(available_countries.intersection(countries))
         
         ix_net_nodes_bool = np.in1d(self.metadata.network_nodes.country, self.attributes.countries)
         self.metadata.ix_nodes = np.where(ix_net_nodes_bool)[0]+1
         if len(self.metadata.ix_nodes)>self.attributes.max_number_loc:
             self.metadata.ix_nodes = np.sort(random.sample(self.metadata.ix_nodes, self.attributes.max_number_loc))
             print 'The number of nodes selected was higher than the maximum number of locations (' +\
             str(self.attributes.max_number_loc) + ') and therefore reduced.'
         
         pass
    
    
     def _load_observations(self):
        
        #loading of observations data under data_observations
        filename = self.attributes.folder_loc + 'Nodal_TS/' + self.attributes.renew_type + \
        '_signal_' + self.attributes.data_type + '.csv'
        data_observations_aux = pd.read_csv(filename, sep=',')
        ix_time_bool = np.in1d(data_observations_aux.Time, 
                               [self.attributes.start_time,self.attributes.end_time])
        ix_time = np.where(ix_time_bool)[0]
        ix_net_nodes = np.append(0, self.metadata.ix_nodes)
        data_observations = data_observations_aux.ix[ix_time[0]:ix_time[len(ix_time)-1], 
                                                     ix_net_nodes]
        del data_observations_aux, filename
        data_observations.Time = pd.to_datetime(data_observations.Time)  
        new_col_names = [None] * len(data_observations.columns)
        new_col_names[0] = 'Time'
        for icol, col_name in enumerate(data_observations.columns[1:], start=1):
            new_col_names[icol] = 'id_' + col_name
        data_observations.columns = new_col_names    
        
        self.metadata.id_nodes = new_col_names[1:]
        self.obs = data_observations
        
        pass
    
    
     def _tod_observations(self):
        
        #Assumption of an hourly day discretisation, to be adapted better if 
        #intraday market or other kinds are to be considered
        time_of_day = [time(ih,0,0,0) for ih in range(24)]
        tod_name = [None] * len(time_of_day)
        #defining the repartition in day for later climatology application
        for index,itime in enumerate(time_of_day):
            if itime.hour<10: nb_name= '0' + str(itime.hour)
            else: nb_name = str(itime.hour)
            tod_name[index] = 'h_'+ nb_name
        
        self.metadata.tod = time_of_day
        self.metadata.tod_label = tod_name

        pass
 
    
     def _load_forecasts(self):
        
        forecast_ahead = [None] * self.attributes.nbr_leadT
        for leadT in range(1,self.attributes.nbr_leadT+1):
           if leadT<10: nb_name= '0' + str(leadT)
           else: nb_name = str(leadT)
           forecast_ahead[leadT-1] = 'leadT_' + nb_name
        self.metadata.fore_leadT = forecast_ahead
        
        #loading of forecasts data under data_forecasts
        data_forecasts = expando()
        for iforecast in os.listdir(self.attributes.folder_loc + 'Nodal_FC/'):
            iforecast_asDate = datetime(int(iforecast[:4]), int(iforecast[4:6]), int(iforecast[6:8]),
                                        int(iforecast[8:]),0,0)
            iforecast_asDate = iforecast_asDate.strftime("%Y-%m-%d %H:%M:%S")
            if iforecast_asDate>=self.attributes.start_time and iforecast_asDate<=self.attributes.end_time:
                filename = self.attributes.folder_loc + 'Nodal_FC/' + iforecast + \
                '/' + self.attributes.renew_type + '_forecast.csv'
                data_forecasts_aux = pd.read_csv(filename, sep=',')
                for leadT, leadT_name in  enumerate(self.metadata.fore_leadT, start=1):
                    data_forecasts_temp = data_forecasts_aux.ix[leadT-1,self.metadata.ix_nodes]
                    data_forecasts_temp = pd.DataFrame({'_'+iforecast: data_forecasts_temp})        
                    try:
                        exec('data_forecasts.' + leadT_name + ' = pd.concat([data_forecasts.' + leadT_name + ',data_forecasts_temp], axis=1)')
                    except:
                        exec('data_forecasts.' + leadT_name + ' = data_forecasts_temp')
                    del data_forecasts_temp
                del data_forecasts_aux, filename
                
        gc.collect
        
        self.fore = data_forecasts
        
        pass
    
    
    
     def _get_distances(self):
        
        dist_df = pd.DataFrame(index=self.metadata.id_nodes, columns=self.metadata.id_nodes)
        for loc_ref in self.metadata.id_nodes:
            id_ref = int(loc_ref[3:])
            ix_ref = np.where(self.metadata.network_nodes.ID == id_ref)[0]
            for loc_comp in self.metadata.id_nodes:
                id_comp = int(loc_comp[3:])
                ix_comp = np.where(self.metadata.network_nodes.ID == id_comp)[0]
                dist_df.loc[(loc_ref, loc_comp)] = \
                distance_from_long_lat(self.metadata.network_nodes.latitude.values[ix_ref],
                                       self.metadata.network_nodes.longitude.values[ix_ref],
                                       self.metadata.network_nodes.latitude.values[ix_comp],
                                       self.metadata.network_nodes.longitude.values[ix_comp])
        self.metadata.distances = dist_df
        del dist_df
        
        pass
    
    
     def _set_climatology(self):         
    
        tod = self.metadata.tod #time of day
        half_day_ix = tod[len(tod)/2].hour #index of half day in order to get approx max value later
        
        #Making Climatology time-of-day transformation
        climatology = expando()
        for location in self.metadata.id_nodes:
            setattr(climatology, location, expando())
            for index,itime in enumerate(tod):
                setattr(getattr(climatology, location), self.metadata.tod_label[index], 
                        getattr(self.obs,location)[self.obs.Time.dt.time == itime])
                
        #Making the cdf and the inv_cdf for the climatology to use as a transformation
        self.clim = expando()
        self.clim.cdf = expando()
        self.clim.inv_cdf = expando()
        for location in self.metadata.id_nodes:
            setattr(self.clim.cdf, location, expando())
            setattr(self.clim.inv_cdf, location, expando())
            clim_loc_hd = getattr(getattr(climatology,location), 
                                  self.metadata.tod_label[half_day_ix])    
            for itime in self.metadata.tod_label:  
                clim_loc_t = getattr(getattr(climatology,location),itime)
                setattr(getattr(self.clim.cdf, location), itime, 
                        clim_cdf(clim_loc_t, clim_loc_hd, 'cdf'))
                setattr(getattr(self.clim.inv_cdf, location), itime, 
                        clim_cdf(clim_loc_t, clim_loc_hd, 'inv_cdf'))
    
        pass
    
     
    
    
    

    
    
    
