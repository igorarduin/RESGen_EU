# License: BSD_3_clause
#
# Copyright (c) 2015, Jan Emil Banning Iversen, Pierre Pinson, Igor Arduin
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the
# distribution.
#
# Neither the name of the Technical University of Denmark (DTU)
# nor the names of its contributors may be used to endorse or
# promote products derived from this software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
import sys
import pandas as pd
import numpy as np    
from datetime import datetime, time
import gc
import math
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



class dataReader:
    
     def __init__(self, countries,max_number_loc,renewable_type,data_type,start_time,
                  end_time,fore_start_time,fore_end_time,nbr_leadTimes,folder_location):         
        self._set_attributes(countries,max_number_loc,renewable_type,data_type,start_time,
                             end_time,fore_start_time,fore_end_time,nbr_leadTimes,folder_location)
        self._check_countries()  
        self._load_observations()
        self._tod_observations()
        self._load_forecasts()
        self._get_distances()
        print('Data has been imported!')  
        
        pass
        
     #Function that stores all inputs as attributes of the output
     def _set_attributes(self, countries,max_number_loc,renewable_type,data_type,start_time,
                         end_time,fore_start_time,fore_end_time,nbr_leadTimes,folder_location):
                            
        self.attributes = expando()
        self.attributes.renew_type = renewable_type
        self.attributes.data_type = data_type
        self.attributes.folder_loc = folder_location
        self.attributes.start_time = start_time
        self.attributes.end_time = end_time
        self.attributes.fore_start_time = fore_start_time
        self.attributes.fore_end_time = fore_end_time
        self.attributes.nbr_leadT = nbr_leadTimes
        self.attributes.countries = countries
        self.attributes.max_number_loc = max_number_loc
        
        self.metadata = expando()
        
        pass
    
    
     #Function that check input countries and display an error message if they
     #don't correspond. Returns the available countries and indices from nodes
     def _check_countries(self):

         self.metadata.network_nodes = pd.read_csv(self.attributes.folder_loc+'/Metadata/network_nodes.csv', 
                                                   sep=',')
         available_countries = set(self.metadata.network_nodes.country)
         countries = self.attributes.countries
         if bool(countries-available_countries.intersection(countries)): 
             print(', '.join(countries-available_countries.intersection(countries)) + \
             ' are not in the country list. ' + 'See in:' + ', '.join(available_countries))
         self.attributes.countries = list(available_countries.intersection(countries))
         
         ix_net_nodes_bool = np.in1d(self.metadata.network_nodes.country, self.attributes.countries)
         self.metadata.ix_nodes = np.where(ix_net_nodes_bool)[0]+1
         if self.attributes.max_number_loc != None and len(self.metadata.ix_nodes)>self.attributes.max_number_loc: 
             self.metadata.ix_nodes = np.sort(random.sample(list(self.metadata.ix_nodes), 
                                                            self.attributes.max_number_loc))
             print('The number of nodes selected was higher than the maximum number of locations (' +\
             str(self.attributes.max_number_loc) + ') and therefore reduced.')
         
         pass
    
     #Function that loads observations and stores them in the 'obs' attribute of output
     def _load_observations(self):
        
        filename = self.attributes.folder_loc + '/Nodal_TS/' + self.attributes.renew_type + \
        '_signal_' + self.attributes.data_type + '.csv'
        data_observations_aux = pd.read_csv(filename, sep=',')
        
        #Getting observations of training period
        ix_time_bool = np.in1d(data_observations_aux.Time, 
                               [self.attributes.start_time,self.attributes.end_time])
        ix_time = np.where(ix_time_bool)[0]
        if len(ix_time) == 1:
            sys.exit('Training period contains only one element.'+ \
            'There must be an error in the definition of starting/ending dates.'+\
            'Check day, month and year selected. Remember that data are available hourly only.')
        ix_net_nodes = np.append(0, self.metadata.ix_nodes)
        data_observations = data_observations_aux.ix[ix_time[0]:ix_time[len(ix_time)-1], 
                                                     ix_net_nodes]
        data_observations.Time = pd.to_datetime(data_observations.Time)  
        del ix_time_bool, ix_time   
        
        #Getting observations of testing period
        ix_time_bool = np.in1d(data_observations_aux.Time, 
                               [self.attributes.fore_start_time,self.attributes.fore_end_time])
        ix_time = np.where(ix_time_bool)[0]
        data_observations_cf = data_observations_aux.ix[ix_time[0]:ix_time[len(ix_time)-1],
                                                        ix_net_nodes]
        data_observations_cf.Time = pd.to_datetime(data_observations_cf.Time)  
        
        #Define colnames with locations
        new_col_names = [None] * len(data_observations.columns)
        new_col_names[0] = 'Time'
        for icol, col_name in enumerate(data_observations.columns[1:], start=1):
            new_col_names[icol] = 'id_' + col_name
        self.metadata.id_nodes = new_col_names[1:]
        
        data_observations.columns = new_col_names  
        data_observations_cf.columns = new_col_names
        
        data_observations.reset_index(drop=True, inplace=True)
        data_observations_cf.reset_index(drop=True, inplace=True)
        
        del data_observations_aux, filename
        
        self.obs = data_observations
        self.current_fore = expando()
        self.current_fore.obs = data_observations_cf
        
        pass
    
     #Function that defines the time of day horizon of predictions/observations
     #Dataset contains only hourly information but it can be adapted for other
     #markets
     def _tod_observations(self):
        
        #Assumption of an hourly day discretisation, to be adapted better if 
        #intraday market or other kinds are to be considered
        time_of_day = [time(ih,0,0,0) for ih in range(24)]
        tod_name = [None] * len(time_of_day)
        #defining the repartition in day for later climatology application
        for index,itime in enumerate(time_of_day):
            if itime.hour<10: h_name= '0' + str(itime.hour)
            else: h_name = str(itime.hour)
            if itime.minute<10: min_name= '0' + str(itime.minute)
            else: min_name = str(itime.minute)
            tod_name[index] = 'h_'+ h_name + '_' + min_name
            
        self.metadata.tod = time_of_day
        self.metadata.tod_label = tod_name

        pass
 
     #Function that loads predictions and stores them in the 'fore' attribute of output
     def _load_forecasts(self):
        
        #Define lead times labels
        forecast_ahead = [None] * self.attributes.nbr_leadT
        for leadT in range(1,self.attributes.nbr_leadT+1):
           if leadT<10: nb_name= '0' + str(leadT)
           else: nb_name = str(leadT)
           forecast_ahead[leadT-1] = 'leadT_' + nb_name
        self.metadata.fore_leadT = forecast_ahead
        
        #loading of forecasts data under data_forecasts
        data_forecasts = expando()
        data_forecasts_cf = expando()
        empty_df = pd.DataFrame(columns=self.obs.columns)
        for leadT_name in  self.metadata.fore_leadT:
            setattr(data_forecasts, leadT_name, empty_df)
            setattr(data_forecasts_cf, leadT_name, empty_df)
             
        for iforecast in os.listdir(self.attributes.folder_loc + '/Nodal_FC/'):
            iforecast_asDate = datetime(int(iforecast[:4]), int(iforecast[4:6]), int(iforecast[6:8]),
                                        int(iforecast[8:]),0,0)
            iforecast_asDate = iforecast_asDate.strftime("%Y-%m-%d %H:%M:%S")
            if iforecast_asDate>=self.attributes.start_time and iforecast_asDate<=self.attributes.end_time:
                filename = self.attributes.folder_loc + '/Nodal_FC/' + iforecast + \
                '/' + self.attributes.renew_type + '_forecast.csv'
                data_forecasts_aux = pd.read_csv(filename, sep=',')
                for leadT, leadT_name in  enumerate(self.metadata.fore_leadT, start = 1):
                    temp_df = pd.DataFrame(np.nan, index=[0],columns=self.obs.columns)
                    temp_df.loc[0,('Time')] = iforecast_asDate
                    for iloc, location in enumerate(self.metadata.id_nodes):
                        temp_df.loc[0,(location)] = data_forecasts_aux.ix[leadT,self.metadata.ix_nodes[iloc]]
                    setattr(data_forecasts, leadT_name, 
                            getattr(data_forecasts, leadT_name).append(temp_df, ignore_index=True))
                    del temp_df
                del data_forecasts_aux, filename
                
            if iforecast_asDate>=self.attributes.fore_start_time and iforecast_asDate<=self.attributes.fore_end_time:
                filename = self.attributes.folder_loc + '/Nodal_FC/' + iforecast + \
                '/' + self.attributes.renew_type + '_forecast.csv'
                data_forecasts_aux = pd.read_csv(filename, sep=',')
                for leadT, leadT_name in  enumerate(self.metadata.fore_leadT, start = 1):
                    temp_df = pd.DataFrame(np.nan, index=[0],columns=self.obs.columns)
                    temp_df.loc[0,('Time')] = iforecast_asDate
                    for iloc, location in enumerate(self.metadata.id_nodes):
                        temp_df.loc[0,(location)] = data_forecasts_aux.ix[leadT,self.metadata.ix_nodes[iloc]]
                    setattr(data_forecasts_cf, leadT_name, 
                            getattr(data_forecasts_cf, leadT_name).append(temp_df, ignore_index=True))
                    del temp_df
                del data_forecasts_aux, filename
                
        gc.collect
        
        for leadT_name in  self.metadata.fore_leadT:
            getattr(data_forecasts, leadT_name).Time = \
            pd.to_datetime(getattr(data_forecasts, leadT_name).Time)  
            getattr(data_forecasts_cf, leadT_name).Time = \
            pd.to_datetime(getattr(data_forecasts_cf, leadT_name).Time)  
            
        self.fore = data_forecasts
        self.current_fore.fore = data_forecasts_cf
        
        pass
    
    
     #Function that calculates and stores distances between nodes 
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
    
    
    
     
    
    
    

    
    
    
