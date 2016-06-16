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
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
from modelEstimation import get_label_time

#Combined exp and Cauchy for correlation matrice fitting
def mix_func(X, beta, tau, a, b):
    x,y = X
    return np.exp(-(tau*x)/(1+a*y**(2*b))**beta)/(1+a*y**(2*b))


#Function to define the forecast name from time
def get_time_label(time):
    if time.month<10: name_month= '0' + str(time.month)
    else: name_month = str(time.month)
    if time.day<10: name_day= '0' + str(time.day)
    else: name_day = str(time.day)
    if time.hour<10: name_hour= '0' + str(time.hour)
    else: name_hour = str(time.hour)
    if time.minute<10: name_minute= '0' + str(time.minute)
    else: name_minute = str(time.minute)
    time_name = '_'+str(time.year)+name_month+name_day+name_hour+name_minute
    return time_name
    

#Function to save scenarios in csv files
def save_scenarios(scenarios,folder_output):
    for idate in scenarios.simulation.__dict__:
        if not os.path.isdir(folder_output+'/'+idate):
            os.makedirs(folder_output+'/'+idate)
        for iloc in getattr(scenarios.simulation, idate).__dict__:
            if iloc != 't_actual':
                getattr(getattr(scenarios.simulation, idate), iloc).to_csv(folder_output+'/'+idate+'/'+iloc+idate+'.csv')
        
    
    

class expando:
    pass

class scenarioGeneration:
    
    def __init__(self, model, data, improv_forecast, nb_scenarios):         
        self._set_attributes(improv_forecast, nb_scenarios)  
        self._get_covariance(model, data)
        self._get_scenarios(model, data)  
        print('Scenarios computed!')       
        pass
    
    
    def _set_attributes(self, improv_forecast, nb_scenarios):
        self.attributes = expando()
        self.attributes.improv_forecast = improv_forecast
        self.attributes.nb_scenarios = nb_scenarios
        pass


    def _get_covariance(self, model, data):
        
        self.correlation_matrix = pd.DataFrame(columns = model.corr.pivot_columns, 
                                               index = model.corr.pivot_columns)
        for id_ref in data.metadata.id_nodes:
            for id_loc in data.metadata.id_nodes: 
                for leadT_ref in data.metadata.fore_leadT:
                    for leadT_loc in data.metadata.fore_leadT:
                        dist_loc = getattr(getattr(data.metadata.distances, id_ref), id_loc)
                        dleadT = abs(int(leadT_loc[6:]) - int(leadT_ref[6:]))
                        self.correlation_matrix.loc[(id_ref,leadT_ref),(id_loc,leadT_loc)] = \
                        model.corr.fit.combined.func((dist_loc, dleadT))

        
        self.correlation_matrix = self.correlation_matrix.astype(float)
        pass


    

    def _get_scenarios(self, model, data):     
        
        self.simulation = expando()      
        dates_of_issue = getattr(data.current_fore.fore, data.metadata.fore_leadT[0]).Time

        for i_date_issue, date_issue in enumerate(dates_of_issue):
            
            date_issue_name = get_time_label(date_issue)
            print(date_issue_name)
            setattr(self.simulation, date_issue_name, expando())
                                           
            t_actual = expando()    
            for ileadT, leadT in enumerate(data.metadata.fore_leadT, start=1):
                t_actual_temp = date_issue + data.current_fore.obs.Time.dt.freq * ileadT
                setattr(t_actual, leadT, t_actual_temp)
            setattr(getattr(self.simulation, date_issue_name), 't_actual', t_actual)  
            mean = np.zeros(model.corr.correlation_matrix.shape[1])     
            #First we simulate uniform variables with the appropriate interdependence structure. 
            #This is easily done by first simulating Gaussian varialbes with the same interdependence
            #sturcture and the transforming them to the uniform domain by their marginals.
            
#            rv_mvnorm = multivariate_normal(mean, self.correlation_matrix)
#            simulation_mvnorm = rv_mvnorm.rvs(self.attributes.nb_scenarios)
            rv_mvnorm = multivariate_normal(mean, model.corr.correlation_matrix)
            simulation_mvnorm = rv_mvnorm.rvs(self.attributes.nb_scenarios)
            simulation_uniform = pd.DataFrame(data = norm.cdf(simulation_mvnorm), 
                                              columns = model.corr.pivot_columns) 


            #Having obtained the simulated variables in the uniform domain, we need to get them into the transformed 
            #domain. This is done by using the inverse cummulative density function (inv_cdf) for each region and 
            #lead time. As the marginals depend on the predicted values, the predictions are required. 
            #Here the predictions that came with the data are used.
        
            #first we put the transformed predictions on the appropriate form. To do this we need a set of 
            #multi horizon point predictions spanning the locations considered and the prediction horizons.
            #Futher we need a starting time. In this implementation we simply choose a starting time from
            #the forecast data and choose the associated forecasts.
            scen_label = [None] * self.attributes.nb_scenarios
            for iscen in range(1, self.attributes.nb_scenarios+1):
               scen_label[iscen-1] = 'scen_' + str(iscen)
            scen_label.insert(0,'forecasts')
            scen_label.insert(0,'init_forecasts')
            self.attributes.scen_label = scen_label

            for id_loc in data.metadata.id_nodes:
                simulation_loc = pd.DataFrame(0, columns = data.metadata.fore_leadT, 
                                              index = scen_label)
                
                for leadT in data.metadata.fore_leadT:
                    
                    predict_simulation = getattr(data.current_fore.fore, leadT).loc[i_date_issue,(id_loc)]
                    simulation_loc.loc[(scen_label[0],leadT)] = predict_simulation
                    
                    label_time = get_label_time(getattr(t_actual, leadT).time())

                    #Get the prediction out of seasonality effects
                    clim_cdf = \
                    getattr(getattr(model.clim.cdf, id_loc), label_time)
                    predict_transf_simulation = clim_cdf(predict_simulation)
                    #Improve the forecast from the weighted persistence-climatology model
                    if self.attributes.improv_forecast == 1: 
                        #Makes sense only for wind predictions so far
                        #Get the observation at time of issue out of seasonality effects, 
                        #which will represent persistence value
                        label_time_issue = get_label_time(date_issue.time())
                        clim_cdf = \
                        getattr(getattr(model.clim.cdf, id_loc), label_time_issue)
                        fore_obs_loc = getattr(data.current_fore.obs, id_loc)
                        pers_obs_loc_trans = \
                        clim_cdf(fore_obs_loc[np.where(data.current_fore.obs.Time == date_issue)[0][0]])
                        #
                        coeff_imp = \
                        getattr(getattr(model.imp_fore, id_loc), leadT)
                        
                        predict_transf_simulation = coeff_imp.intercept + \
                        coeff_imp.beta_pers * pers_obs_loc_trans+ \
                        coeff_imp.beta_fore* predict_transf_simulation
                        predict_transf_simulation = predict_transf_simulation
                        #When the transformation is made, the prediction might exceed 1 which is unrealistic
                        if predict_transf_simulation > 1:
                            predict_transf_simulation = 1
                            
                    #If the prediction is 0, we attributes scenarios to be zeros (for solar night)
                    if predict_transf_simulation> 0:
                        #Having obtained the predictions on the transformed domain, we can simulate on the 
                        #transformed domain. This is done by converting the simulated uniforms to the appropriate
                        #transformed values through the inv_cdf transformation.
                        inv_cdf = getattr(getattr(model.inv_cdf, id_loc), leadT)  
                        conditional_inv_cdf = inv_cdf(predict_transf_simulation)
                        clim_inv_cdf = getattr(getattr(model.clim.inv_cdf,id_loc),label_time)
                        for iscen in range(self.attributes.nb_scenarios):
                            simulation_transformed_temp = \
                            float(conditional_inv_cdf(getattr(getattr(simulation_uniform, id_loc), leadT)[iscen]))
                            #Transformes the simulations on the transformed domain back to the original domain. Here we
                            #need to define an initial time to transform the simulation to the original power domain:
                            simulation_loc.loc[(scen_label[iscen+1],leadT)] = clim_inv_cdf(simulation_transformed_temp)  
                        #Save the modified input forecast
                        simulation_loc.loc[(scen_label[1],leadT)] = clim_inv_cdf(predict_transf_simulation)
                        
                setattr(getattr(self.simulation, date_issue_name), id_loc, simulation_loc)   
                
        pass







