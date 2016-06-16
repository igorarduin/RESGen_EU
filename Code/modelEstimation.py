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


import numpy as np                 
import pandas as pd  
import gc
import warnings

from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit        
from sklearn import linear_model  
import statsmodels.formula.api as smf
          
class expando:
    pass 
    
#Combined exp and Cauchy for correlation matrice fitting
def mix_func(X, beta, tau, a, b):
    x,y = X
    return np.exp(-(tau*x)/(1+a*y**(2*b))**beta)/(1+a*y**(2*b))

#climatology cdf/inv_cdf function 
def clim_cdf(input_data_t, loc_NC, max_factor, cdf_keyword):
    probabilities = np.arange(1,len(input_data_t)+1)/(float(len(input_data_t))+1)     
    quantiles = np.array(sorted(input_data_t))
    quantiles[quantiles < 0] = 0.
    if (quantiles == np.zeros(len(quantiles))).all():
        #night of solar cases
        quantiles_extended = np.array([0.0, loc_NC])
        probabilities_extended = np.zeros(len(quantiles_extended))
    else:
        #Extension of quantiles to reach nominal capacity. The climatology 
        #functions are built with observations. This extension is to prevent the cases
        #when forecast are higher than observations and therefore out of range.
        #The value 1.2 is the lowest fit found so far. Could be generalize using directly 
        #the real nominal capacity
        quantiles_extended = \
        np.concatenate([[-1e-5,0], quantiles, [quantiles.max()*max_factor, loc_NC*max_factor]])
        probabilities_extended = np.concatenate([[-1e-5,0.],probabilities,[1,1+1e-5]])
    
    if cdf_keyword == 'cdf':
        interpolation = interp1d(quantiles_extended, probabilities_extended)
    elif cdf_keyword == 'inv_cdf':
        interpolation = interp1d(probabilities_extended, quantiles_extended)
    
    return interpolation

#Function to labelize the hours when climatology will be applied. 
#Format 'h_HH_MM'. Ex: in intraday 'h_18_45'
def get_label_time(time):
    if time.hour<10: name_hour= '0' + str(time.hour)
    else: name_hour = str(time.hour)
    if time.minute<10: name_minute= '0' + str(time.minute)
    else: name_minute = str(time.minute)
    label_time = 'h_'+name_hour+'_'+name_minute
    return label_time


#cdf from conditional quantile regression
def cqr_cdf(prediction, betas, cdf_keyword): 
    prob = betas.loc[:,('probabilities')].values
    quantiles = np.zeros(len(prob))
    for i in range(len(prob)):
        quantiles[i] = float(betas.loc[i,('intercept')] + betas.loc[i,('coefficient')]*prediction)
    quantiles[quantiles < 0] = 0
    quantiles[quantiles > 1] = 1
    quantiles_extended = np.concatenate([[0], sorted(quantiles), [1]])
    probabilities_extended = np.concatenate([[0],prob,[1]])
    if cdf_keyword == 'cdf':
        interpolation = interp1d(quantiles_extended, probabilities_extended)
    elif cdf_keyword == 'inv_cdf':
        interpolation = interp1d(probabilities_extended, quantiles_extended)
    return interpolation
 


class modelEstimation:

    def __init__(self, data): 
        print('Climatology transformation')
        self._set_climatology(data)
        self._apply_climatology(data)
        self._get_concurrent_clim_set(data)
        print('Getting the estimators to improve forecast')
        self._improvement_forecast(data)   
        print('Quantiles calculation')
        self._set_quantiles(data)
        self._set_cdf(data)
        print('CDF transformation')
        self._apply_cdf(data)
        print('Generalization of correlation matrice')
        self._get_corr(data)
        self._generalize_corr(data)
        print('Estimation finished!')
        pass
    
    #Function that defines climatology cdf for every corresponding time horizon,
    #using training data.
    def _set_climatology(self, data):       
        #time of day
        tod = data.metadata.tod 
        #Making Climatology time-of-day transformation
        climatology = expando()
        for location in data.metadata.id_nodes:
            setattr(climatology, location, expando())
            for index,itime in enumerate(tod):
                setattr(getattr(climatology, location), data.metadata.tod_label[index], 
                        getattr(data.obs,location)[data.obs.Time.dt.time == itime])
                
        #Making the cdf and the inv_cdf for the climatology to use as a transformation
        self.clim = expando()
        self.clim.cdf = expando()
        self.clim.inv_cdf = expando()
        #As observations might not reach nominal capacity of the farms while forecasts 
        #might predict it, necessity to define a factor to multiply to the maximum 
        #of observations in the definition of climatology cdf/inv_cdf
        #This factor will not be needed if NC of farms are included/used
        if data.attributes.renew_type == 'wind':
            max_factor = 1.05
        elif data.attributes.renew_type == 'solar':
            max_factor = 1.2
        #For each location and time of day, creation of climatology cdf/inv_cdf   
        for location in data.metadata.id_nodes:
            setattr(self.clim.cdf, location, expando())
            setattr(self.clim.inv_cdf, location, expando())
            loc_NC = max(getattr(data.obs,location)) #close to nominal capacity of farm
            for itime in data.metadata.tod_label:  
                clim_loc_t = getattr(getattr(climatology,location),itime)
                setattr(getattr(self.clim.cdf, location), itime, 
                        clim_cdf(clim_loc_t, loc_NC, max_factor, 'cdf'))
                setattr(getattr(self.clim.inv_cdf, location), itime, 
                        clim_cdf(clim_loc_t, loc_NC, max_factor, 'inv_cdf'))
        pass    
    
    
    #Function that applies climatology transformations to all observations and predictions.
    def _apply_climatology(self, data):    
        
        #Transforming observations and predictions (by climatology) to get rid of seasonality
        #First transforming the observed power
        self.clim.obs = pd.DataFrame(np.nan, index=range(len(data.obs)),
                                     columns=['t_actual']+data.metadata.id_nodes)                            
        self.clim.obs.loc[:,('t_actual')] = data.obs.Time      
        
        for location in data.metadata.id_nodes:
            for itime, time in data.obs.Time.dt.time.iteritems():
                label_time = get_label_time(time)
                clim_cdf_loc_t = getattr(getattr(self.clim.cdf,location), label_time)  
                self.clim.obs.loc[itime,(location)] = float(clim_cdf_loc_t(data.obs.loc[itime,(location)]))                       
                                              
    
        #Second, the predicted power is transformed
        self.clim.fore = expando()
        for ileadT, leadT in enumerate(data.metadata.fore_leadT, start = 1):
            data_fore_leadT = getattr(data.fore, leadT)
            
            init_df = pd.DataFrame(np.nan, index=range(len(data_fore_leadT.Time)),
                                   columns=['t_issue','t_actual']+data.metadata.id_nodes)
            init_df.loc[:,('t_issue')] = data.fore.leadT_01.Time
            init_df.loc[:,('t_actual')] = init_df.loc[:,('t_issue')] + data.obs.Time.dt.freq * ileadT

            for location in data.metadata.id_nodes: 
                for idate in range(len(data_fore_leadT.Time)):
                    power_to_be_transformed = getattr(data_fore_leadT,location)[idate]
                    label_time = get_label_time(init_df.loc[idate,('t_actual')])
                    clim_cdf_loc_t = getattr(getattr(self.clim.cdf, location),label_time)
                    init_df.loc[idate,(location)] = float(clim_cdf_loc_t(power_to_be_transformed))         
                    
            setattr(self.clim.fore, leadT, init_df)
            del init_df   

        pass
    
    #Function that organizes data per lead time in order to have an easy access to 
    #predictions and corresponding observations. Persistence values are also stored
    #for the improvement forecast phase.
    def _get_concurrent_clim_set(self, data):   
        
        self.clim.concurr = expando()
        for location in data.metadata.id_nodes:   
            setattr(self.clim.concurr, location, expando())
            clim_concurr_loc = getattr(self.clim.concurr, location)
            for leadT in data.metadata.fore_leadT:
                clim_fore_leadT = getattr(self.clim.fore, leadT)
                
                #Getting observations corresponding to this leadT predictions
                ix_match_obs_bool = np.in1d(self.clim.obs.t_actual, clim_fore_leadT.t_actual)   
                ix_match_obs = np.where(ix_match_obs_bool)[0]
                observations = self.clim.obs.loc[ix_match_obs,(location)]
                observations.reset_index(drop=True, inplace=True)
                
                t_actual = self.clim.obs.t_actual[ix_match_obs]
                t_actual.reset_index(drop=True, inplace=True)
                
                #Getting prediction of this leadT, in accordance with previous observations
                ix_match_pred_bool = np.in1d(clim_fore_leadT.t_actual, self.clim.obs.t_actual)   
                ix_match_pred = np.where(ix_match_pred_bool)[0]
                predictions = clim_fore_leadT.loc[ix_match_pred, (location)]
                predictions.reset_index(drop=True, inplace=True)
                
                #Getting the observation at time of issue, corresponding to the persistence value
                ix_match_per_bool = np.in1d(self.clim.obs.t_actual, clim_fore_leadT.loc[ix_match_pred,('t_issue')])   
                ix_match_per = np.where(ix_match_per_bool)[0]
                persistences = self.clim.obs.loc[ix_match_per,(location)]
                persistences.reset_index(drop=True, inplace=True)
                
                concurr_temp = pd.DataFrame({'observations':observations, \
                'predictions':predictions, 'persistences':persistences, 't_actual':t_actual})
                setattr(clim_concurr_loc, leadT, concurr_temp)                

        pass

    #Function that fits coefficients for a weighted combination of persistence 
    #and actual forecasts in order to generate better point forecasts. Calculations
    #are made on the training sample
    def _improvement_forecast(self, data):
        self.imp_fore = expando()
        
        for location in data.metadata.id_nodes:  
            betas = pd.DataFrame(columns = data.metadata.fore_leadT, 
                                 index = ['intercept','beta_pers','beta_fore'])
                              
            for leadT in data.metadata.fore_leadT:
                clim_concurr_loc_leadT = getattr(getattr(self.clim.concurr, location), leadT)  
                #Fitting observations with persistence and predicted values
                for_fitting = np.zeros((len(clim_concurr_loc_leadT.observations),2))
                for_fitting[:,0] = clim_concurr_loc_leadT.persistences
                for_fitting[:,1] = clim_concurr_loc_leadT.predictions
                regr_leadT = linear_model.LinearRegression()
                regr_leadT.fit(for_fitting.reshape((len(for_fitting),2)), 
                               clim_concurr_loc_leadT.observations)
                betas.loc[('intercept', leadT)] = regr_leadT.intercept_
                betas.loc[('beta_pers', leadT)] = regr_leadT.coef_[0]
                betas.loc[('beta_fore', leadT)] = regr_leadT.coef_[1]
                del for_fitting, regr_leadT
            setattr(self.imp_fore, location, betas)
        pass


    def _set_quantiles(self, data): 
        
        #Compute quantiles for the transformed power conditional on the transformed power prediction
        #for a specific location and a specific lead time.
        
        #smf.quantreg generates warning - see documentation for more details
        #warning off just for this section
        warnings.filterwarnings("ignore")
        #Performs the actual quantile regression and stores the variables of 
        prob = np.concatenate([[0.001],np.arange(0.05,0.951,0.05),[0.999]])
        self.betas = expando()
        for location in data.metadata.id_nodes: 
            print(location)
            setattr(self.betas, location, expando())
            for ileadT, leadT in enumerate(data.metadata.fore_leadT, start = 1):
                
                clim_concurr_loc_leadT = getattr(getattr(self.clim.concurr, location), leadT)                
                
                betas_aux = pd.DataFrame(0, columns = ['probabilities','intercept', 'coefficient'], 
                                         index = range(len(prob)))  
                betas_aux.loc[:,('probabilities')] = prob                            
                #For solar cases, all quantiles are kepts to zeros
                if not np.all(clim_concurr_loc_leadT.observations == 0.): 
                    mod = smf.quantreg('observations ~ predictions', clim_concurr_loc_leadT)
                    for iq,q in enumerate(prob):
                        res = mod.fit(q=q)
                        betas_aux.loc[iq,('intercept')] =  res.params['Intercept']
                        betas_aux.loc[iq,('coefficient')] = res.params['predictions']
                        del res
                    del mod

                setattr(getattr(self.betas,location), leadT, betas_aux)
                del betas_aux
                gc.collect()
        #warning on
        warnings.filterwarnings("always")
        pass
    
    def _set_cdf(self, data):    
        
        #In order to use the copula approach we need to transform to uniform marginal distributions.
        #This is achieved by using the predictive marginal densities on the transformed domain to 
        #do a second transformation to get uniformly distributed marginals. For a complete and 
        #acessable introduction see the wikipedia page on Copulas.
    
        #First define the marginal cummulative density functions. They are stored as the cummulative 
        #density function (cdf) and for easy use we also define the inverse cumulative density 
        #function inv_cdf. Each is defined for every location and every lead time.

        self.cdf = expando()
        self.inv_cdf = expando()
        
        for location in data.metadata.id_nodes:
            setattr(self.cdf, location, expando())
            setattr(self.inv_cdf, location, expando())
    
            for leadT in data.metadata.fore_leadT:
                betas_loc_leadT = getattr(getattr(self.betas,location),leadT)

                cdf_loc_leadT = \
                lambda prediction, betas=betas_loc_leadT, cdf_keyword='cdf': \
                cqr_cdf(prediction, betas, cdf_keyword)
                setattr(getattr(self.cdf, location), leadT, cdf_loc_leadT)
                   
                inv_cdf_loc_leadT = \
                lambda prediction, betas=betas_loc_leadT, cdf_keyword='inv_cdf': \
                cqr_cdf(prediction, betas, cdf_keyword)
                setattr(getattr(self.inv_cdf, location), leadT, inv_cdf_loc_leadT)
            
        pass


    def _apply_cdf(self, data):    
        
        #Using the defined cummulative density function (cdf) we can now convert every observation 
        #into the uniform domain. This is done for every location and every lead time.
        
        self.uniform = expando()
        for location in data.metadata.id_nodes:
            print(location)
            setattr(self.uniform, location, expando())

            for ileadT, leadT in enumerate(data.metadata.fore_leadT, start = 1):
                cdf_loc_leadT = getattr(getattr(self.cdf, location), leadT)
                
                observations = getattr(getattr(self.clim.concurr, location), leadT).observations
                predictions = getattr(getattr(self.clim.concurr, location), leadT).predictions
                t_actual = getattr(getattr(self.clim.concurr, location), leadT).t_actual
    
                unif_aux = {}    
                unif_aux['value'] = {}
                unif_aux['time'] = {}
                unif_aux['date'] = {}
                
                unif_aux['t'] = t_actual
                unif_aux['t'].index = range(len(observations))
                
                for index in unif_aux['t'].keys():  
                    conditional_cdf_loc_leadT = cdf_loc_leadT(predictions[index])
                    unif_aux['value'][index] = float(conditional_cdf_loc_leadT(observations[index]))
                    unif_aux['time'][index] = unif_aux['t'][index].time()
                    unif_aux['date'][index] = unif_aux['t'][index].date()
                    del conditional_cdf_loc_leadT
                unif_aux = pd.DataFrame(unif_aux,columns=['t','value','time','date'])
                
                setattr(getattr(self.uniform, location), leadT, unif_aux)
                
            del unif_aux, observations, predictions
            gc.collect()
            
        pass


    def _get_corr(self, data):    
        
        #Next we estimate the correlation matrix for the uniform variables. To facilitate this the
        #uniform variables are put on an appropriate form for computing a correlation matrix. This
        #is done through using a pivot table      
        uniform_df = pd.DataFrame({'location': [], 't': [], 'value': [], 'ltname': [],\
        'date': [], 'time': []})
        for location in data.metadata.id_nodes:
            for leadT in data.metadata.fore_leadT:
                uniform_loc_leadT = getattr(getattr(self.uniform, location), leadT)
                
                df_loc_leadT_temp = pd.DataFrame({'location': location, 't': uniform_loc_leadT.t, \
                'value': uniform_loc_leadT.value, 'ltname': leadT, 'date': uniform_loc_leadT.date, \
                'time': uniform_loc_leadT.time})      
                
                uniform_df = pd.concat([uniform_df, df_loc_leadT_temp])
                del df_loc_leadT_temp
    
    
        uniform_df['value']=uniform_df['value'].astype(float)
        uniform_pivot = uniform_df.pivot_table(index='date',columns=('location','ltname'),values='value')
        
        norm_df =  uniform_df
        norm_df['value'] = norm.ppf(uniform_df['value'])
        norm_pivot = norm_df.pivot_table(index='date',columns=('location','ltname'),values='value')
               
        #From the observations in the uniform domain we can now compute the correlation matrix. 
        #The correlation matrix specifies the Gaussian copula used for combining the different models. 
        #Where the computed correlation is NaN we set it to zero.
        correlation_matrix_na = norm_pivot.corr()
        where_are_NaNs = np.isnan(correlation_matrix_na)
        correlation_matrix = correlation_matrix_na
        correlation_matrix[where_are_NaNs] = 0.
        if not np.all(np.diag(correlation_matrix) == 1.):
            print('All diagonal values of correlation matrix are not 1!')
            np.fill_diagonal(correlation_matrix.values, 1.)
        
        self.corr = expando()
        self.corr.correlation_matrix = correlation_matrix
        self.corr.pivot_columns = uniform_pivot.columns
            
        pass



    def _generalize_corr(self, data):    
        #The purpose of this function is to extrapolate the correlation values to
        #different distances and delta lead times. In order to do so, 
        #an exponential function is used for fitting.
        self.corr.fit = expando()
        self.corr.fit.combined = expando()
        corr_original = []
        loc_to_compare_with = data.metadata.id_nodes[:]
        for id_ref in data.metadata.id_nodes:
            for id_loc in loc_to_compare_with: 
                dist_locs = data.metadata.distances.loc[(id_ref,id_loc)]
                leadT_to_compare_with = data.metadata.fore_leadT[:]
                for leadT_ref in data.metadata.fore_leadT:
                    for leadT in leadT_to_compare_with:
                        dleadT = abs(int(leadT[6:]) - int(leadT_ref[6:]))
                        new_corr = self.corr.correlation_matrix[(id_ref, leadT_ref)][(id_loc, leadT)]
                        new_value = np.matrix([[dist_locs, dleadT, new_corr]])                
                        temp_values = corr_original
                        try:
                            corr_original = np.concatenate((temp_values,new_value))
                        except:
                            corr_original = new_value
                    if id_ref == id_loc:
                        leadT_to_compare_with.remove(leadT_ref) 
            loc_to_compare_with.remove(id_ref)
    
        self.corr.fit.original = pd.DataFrame(corr_original, columns = ['distances','dt','correlation'])
        
        #Fitting part, using curve_fit 
        x = np.squeeze(np.asarray(corr_original[:,0])) 
        y = np.squeeze(np.asarray(corr_original[:,1])) 
        z = np.squeeze(np.asarray(corr_original[:,2]))   
        coeff, pcov = curve_fit(mix_func, (x, y), z)
        self.corr.fit.combined.coeff = coeff
        self.corr.fit.combined.func = lambda X, beta=coeff[0], tau=coeff[1], a=coeff[2], b=coeff[3]: \
        mix_func(X, coeff[0], coeff[1], coeff[2], coeff[3])
          
        pass


    
    
    
        

