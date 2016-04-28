import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
from datetime import datetime, timedelta

class expando:
    pass

class scenarioGeneration:
    
    def __init__(self, model, data, fore, nb_scenarios):         
        self._set_attributes(data, nb_scenarios)  
        self._get_covariance(model, data)
        self._get_scenarios(model, data, fore)  
        print('Scenarios computed!')
        
        pass
    
    
    def _set_attributes(self, data, nb_scenarios):
        self.attributes = expando()
        self.attributes.nb_scenarios = nb_scenarios

        pass


    def _get_covariance(self, model, data):
        
        columns_name = []
        for location in data.metadata.id_nodes:
            for leadT in data.metadata.fore_leadT:
                columns_name.append((location, leadT))
        self.pivot_columns = pd.MultiIndex.from_tuples(columns_name, names=['location', 'ltname'])
        self.correlation_matrix = pd.DataFrame(columns = columns_name, index = columns_name)

        for id_ref in data.metadata.id_nodes:
            for id_loc in data.metadata.id_nodes: 
                for leadT_ref in data.metadata.fore_leadT:
                    for leadT_loc in data.metadata.fore_leadT:
                        dist_loc = getattr(getattr(data.metadata.distances, id_ref), id_loc)
                        dist_corr = model.corr.fit.dist.func(dist_loc, 
                                                               model.corr.fit.dist.coeff[0])
                        dleadT = abs(int(leadT_loc[6:]) - int(leadT_ref[6:]))
                        dt_corr = model.corr.fit.dt.func(dleadT, 
                                                             model.corr.fit.dt.coeff[0])
                        self.correlation_matrix.loc[(id_ref,leadT_ref),(id_loc,leadT_loc)] =  dist_corr*dt_corr
        
        pass


    

    def _get_scenarios(self, model, data, fore):     
        
        self.simulation = expando()
        
        dates_of_issue = getattr(fore.fore, fore.metadata.fore_leadT[0]).keys()
            
        for date_issue in dates_of_issue:
            print(date_issue)
            setattr(self.simulation, date_issue, expando())
            
            date_id = date_issue[1:]
            t_issue = datetime(int(date_id[:4]), int(date_id[4:6]), int(date_id[6:8]), 
                               int(date_id[8:]), 0, 0)
                               
            t_actual = expando()    
            for leadT in fore.metadata.fore_leadT:
                leadT_num = int(leadT[6:])
                t_actual_temp = t_issue + timedelta(hours=leadT_num)
                setattr(t_actual, leadT, t_actual_temp)
            setattr(getattr(self.simulation, date_issue), 't_actual', t_actual)  
            mean = np.zeros(model.corr.correlation_matrix.shape[1])     
            #First we simulate uniform variables with the appropriate interdependence structure. 
            #This is easily done by first simulating Gaussian varialbes with the same interdependence
            #sturcture and the transforming them to the uniform domain by their marginals.
            
            rv_mvnorm = multivariate_normal(mean, model.corr.correlation_matrix)
            simulation_mvnorm = rv_mvnorm.rvs(self.attributes.nb_scenarios)
            simulation_uniform = pd.DataFrame(data = norm.cdf(simulation_mvnorm), 
                                              columns = model.corr.pivot_columns) 
#            rv_mvnorm = multivariate_normal(mean, self.correlation_matrix)
#            simulation_mvnorm = rv_mvnorm.rvs(self.attributes.nb_scenarios)
#            simulation_uniform = pd.DataFrame(data = norm.cdf(simulation_mvnorm), 
#                                              columns = self.pivot_columns)
                                              
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

            for id_loc in fore.metadata.id_nodes:
                id_num = id_loc[3:]
                
                simulation_loc = pd.DataFrame(0, columns = fore.metadata.fore_leadT, 
                                              index = scen_label)
                
                for leadT in fore.metadata.fore_leadT:
                    
                    simulation_loc.loc[(scen_label[0],leadT)] = getattr(getattr(fore.fore, leadT),date_issue)[id_num]
                
                    bool_hour = [getattr(t_actual, leadT).time() == itime for itime in fore.metadata.tod]
                    ix_hour = [ix for ix, ibool in enumerate(bool_hour) if ibool][0]
                    
                    predict_simulation = getattr(getattr(fore.fore, leadT),date_issue)[id_num]

                    #Get the prediction out of seasonality effects
                    clim_cdf = \
                    getattr(getattr(data.clim.cdf, id_loc), fore.metadata.tod_label[ix_hour])
                    predict_transf_simulation = clim_cdf(predict_simulation)
                    #Improve the forecast from the weighted persistence-climatology model
                    coeff_imp = \
                    getattr(getattr(model.imp_fore, id_loc), leadT)
                    fore_obs_loc = getattr(fore.obs, id_loc)
                    predict_transf_simulation = coeff_imp.intercept + \
                    coeff_imp.beta_pers * fore_obs_loc[fore_obs_loc.index[np.where(fore.obs.Time == t_issue)[0]]] + \
                    coeff_imp.beta_fore* predict_transf_simulation
                    #If the prediction is 0, we attributes scenarios to be zeros (for solar night)
                    if predict_simulation> 0:
                        #Having obtained the predictions on the transformed domain, we can simulate on the 
                        #transformed domain. This is done by converting the simulated uniforms to the appropriate
                        #transformed values through the inv_cdf transformation.
                        inv_cdf = getattr(getattr(model.inv_cdf, id_loc), leadT)  
                        conditional_inv_cdf = inv_cdf(predict_transf_simulation)
                        clim_inv_cdf = getattr(getattr(data.clim.inv_cdf,id_loc),fore.metadata.tod_label[ix_hour])
                        for iscen in range(self.attributes.nb_scenarios):
                            simulation_transformed_temp = \
                            float(conditional_inv_cdf(getattr(getattr(simulation_uniform, id_loc), leadT)[iscen]))
                            #Transformes the simulations on the transformed domain back to the original domain. Here we
                            #need to define an initial time to transform the simulation to the original power domain:
                            simulation_loc.loc[(scen_label[iscen+1],leadT)] = clim_inv_cdf(simulation_transformed_temp)  
                        #Save the modified input forecast
                        simulation_loc.loc[(scen_label[1],leadT)] = clim_inv_cdf(predict_transf_simulation)

                setattr(getattr(self.simulation, date_issue), id_loc, simulation_loc)   
                
        pass







