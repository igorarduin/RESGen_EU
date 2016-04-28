import numpy as np                 
import pandas as pd  
import gc

from scipy.stats import norm
from scipy.optimize import fmin
from scipy.interpolate import interp1d
from datetime import datetime, timedelta 
from scipy.optimize import curve_fit        
from sklearn import linear_model  
          
class expando:
    pass 

#Small functions for the definition of quantiles/cdf
def model(beta,x):
     return np.polyval(beta,x)
def rho(y,tau):
    taum=tau-1
    return np.where(y<0,y*taum,y*tau)    
#loss function needs to be minimized (cf wikipedia)
def loss_function(b,observation,prediction,tau): 
    return np.sum(rho(observation - model(b,prediction), tau))
    
#Powered exponential function for correlation matrice fitting
def pow_exp_func(x, a, b):
    return np.exp(-(a*x)**b)
#Exponential function for correlation matrice fitting
def exp_func(x, a):
    return np.exp(-(a*x))
#Cauchy function for correlation matrice fitting
def cauchy_func(x, a, b):
    return 1./(1+a*x**(2*b))
#Combined exp and Cauchy for correlation matrice fitting
def mix_func(x, beta, tau, a, b):
    return np.exp(-(tau*x[:,0])/(1+a*x[:,1]**(2*b))**beta)/(1+a*x[:,1]**(2*b))


    
#cdf from conditional quantile regression
def cqr_cdf(prediction, prob, betas, cdf_keyword): 
    quantiles = {}
    for i in range(0,len(prob)):
        quantiles[i] = float(model(betas[i],prediction))
    quantiles_extended = np.concatenate([[0], sorted(quantiles.values()), [1]])
    quantiles_extended[quantiles_extended < 0] = 0
    quantiles_extended[quantiles_extended > 1] = 1
    probabilities_extended = np.concatenate([[0],prob,[1]])
    if cdf_keyword == 'cdf':
        interpolation = interp1d(quantiles_extended, probabilities_extended)
    elif cdf_keyword == 'inv_cdf':
        interpolation = interp1d(probabilities_extended, quantiles_extended)
    return interpolation



class modelEstimation:
    
    
    def __init__(self, data): 
        print('Climatology transformation')
        self._apply_climatology(data)
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
    
    def _apply_climatology(self, data):    
        
        #Transforming observations and predictions (by climatology) to get rid of seasonality
        #First transforming the observed power and storing it in data_transformed
        self.clim_transfo = expando()
        self.clim_transfo.obs = expando()
        for location in data.metadata.id_nodes: 
            
            data_loc = getattr(data.obs,location)       
            data_transformed_aux = {}
            data_transformed_aux['t_actual'] = data.obs.Time
            data_transformed_aux['value'] = {}
            
            for index, time in data.obs.Time.dt.time.iteritems():
                clim_cdf_loc_t = getattr(getattr(data.clim.cdf,location), 
                                         data.metadata.tod_label[time.hour])
                data_transformed_aux['value'][index] = float(clim_cdf_loc_t(data_loc[index]))  
            
            data_transformed_aux = pd.DataFrame(data_transformed_aux,columns=['t_actual','value'])
            data_transformed_aux.t_actual = pd.to_datetime(data_transformed_aux.t_actual) 
            setattr(self.clim_transfo.obs, location, data_transformed_aux)
            del index, time, data_transformed_aux
    
    
        #Second, the predicted power is transformed and stored in data_forecast_transformed
        t_fore = pd.DataFrame(np.nan, index=range(len(data.fore.leadT_01.keys())), 
                              columns=['t_issue','t_actual'])
        for idate, corr_date in enumerate(data.fore.leadT_01.keys()):
            date_id = corr_date[1:]
            date_issue = datetime(int(date_id[:4]), int(date_id[4:6]), int(date_id[6:8]), 
                                  int(date_id[8:]),0,0)
            t_fore.loc[idate,('t_issue')] = date_issue         
                
        self.clim_transfo.fore = expando()
        for ileadT, leadT in enumerate(data.metadata.fore_leadT, start = 1):
            #print ileadT, leadT
            data_fore_leadT =getattr(data.fore, leadT)
            t_fore.t_actual = t_fore.t_issue + timedelta(hours=ileadT)
            data_forecast_transformed_aux = expando() 
            for location in data.metadata.id_nodes: 
                id_loc = location[3:]
                data_fore_transf_aux_location = {}
                data_fore_transf_aux_location['t_issue'] = t_fore.t_issue
                data_fore_transf_aux_location['t_actual'] = t_fore.t_actual
                data_fore_transf_aux_location['value'] = {}
                for idate, date_issue in enumerate(data_fore_leadT.keys()):
                    #print idate, date_issue
                    power_to_be_transformed = getattr(data_fore_leadT,date_issue)[id_loc]
                    clim_cdf_loc_t = getattr(getattr(data.clim.cdf, location),
                                             data.metadata.tod_label[t_fore.t_issue[idate].hour])
                    data_fore_transf_aux_location['value'][idate] = \
                    float(clim_cdf_loc_t(power_to_be_transformed))
                
                data_fore_transf_aux_location = \
                pd.DataFrame(data_fore_transf_aux_location, columns=['t_issue','t_actual','value'])
                data_fore_transf_aux_location.t_issue = pd.to_datetime(data_fore_transf_aux_location.t_issue) 
                data_fore_transf_aux_location.t_actual = pd.to_datetime(data_fore_transf_aux_location.t_actual) 
                setattr(data_forecast_transformed_aux, location, data_fore_transf_aux_location)
            setattr(self.clim_transfo.fore, leadT, data_forecast_transformed_aux)

        pass


    def _improvement_forecast(self, data):
        self.imp_fore = expando()
        
        for location in data.metadata.id_nodes:  
            data_tran_loc = getattr(self.clim_transfo.obs, location)
            betas = pd.DataFrame(columns = data.metadata.fore_leadT, 
                                          index = ['intercept','beta_pers','beta_fore'])
                                          
            for leadT in data.metadata.fore_leadT:
                data_tran_fore_leadT = getattr(getattr(self.clim_transfo.fore, leadT), location)                
                
                ix_match_obs_bool = np.in1d(data_tran_loc.t_actual, data_tran_fore_leadT.t_actual)   
                ix_match_obs = np.where(ix_match_obs_bool)[0]
                observations = data_tran_loc.value[ix_match_obs]
                
                ix_match_pred_bool = np.in1d(data_tran_fore_leadT.t_actual, data_tran_loc.t_actual)   
                ix_match_pred = np.where(ix_match_pred_bool)[0]
                data_tran_fore_leadT_sub = data_tran_fore_leadT.loc[ix_match_pred]
                predictions = data_tran_fore_leadT_sub.value
                
                ix_match_per_bool = np.in1d(data_tran_loc.t_actual, data_tran_fore_leadT_sub.t_issue)   
                ix_match_per = np.where(ix_match_per_bool)[0]
                persistences = data_tran_loc.value[ix_match_per]

                for_fitting = np.zeros((len(observations),2))
                for_fitting[:,0] = persistences
                for_fitting[:,1] = predictions
                
                regr_leadT = linear_model.LinearRegression()
                regr_leadT.fit(for_fitting.reshape((len(for_fitting),2)),observations)
                betas.loc[('intercept', leadT)] = regr_leadT.intercept_
                betas.loc[('beta_pers', leadT)] = regr_leadT.coef_[0]
                betas.loc[('beta_fore', leadT)] = regr_leadT.coef_[1]

            setattr(self.imp_fore, location, betas)
        pass


    def _set_quantiles(self, data):    
        #Compute quantiles for the transformed power conditional on the transformed power prediction
        #for a specific location and a specific lead time.
        
        #In order to use conditional quantile regression we first need to define the type of model 
        #that we are looking at. The model takes as input some parameters and an input value and 
        #returns the appropriate quantile. If regression type is defined as "poly", the quantiles will
        #be polynomial interpolations. If it is defined as "spline", the calculations will use spline 
        #in R. It is therefore needed to download package rpy2.
        
        
        #Performs the actual quantile regression and stores the variables of 
        self.prob = np.concatenate([[0.001],np.arange(0.05,0.951,0.05),[0.999]])
        self.betas = expando()
        beta0 = np.zeros(2)
        for location in data.metadata.id_nodes: 
            print(location)
            setattr(self.betas, location, expando())
            data_tran_loc = getattr(self.clim_transfo.obs,location)
        
            for ileadT, leadT in enumerate(data.metadata.fore_leadT, start = 1):
                data_fore_fore_leadT = getattr(getattr(self.clim_transfo.fore, leadT),location)
    
                ix_match_obs_bool = np.in1d(data_tran_loc.t_actual, data_fore_fore_leadT.t_actual)   
                ix_match_obs = np.where(ix_match_obs_bool)[0]
                observation = data_tran_loc.value[ix_match_obs]
                
                ix_match_fore_bool = np.in1d(data_fore_fore_leadT.t_actual, data_tran_loc.t_actual)   
                ix_match_fore = np.where(ix_match_fore_bool)[0]
                prediction = data_fore_fore_leadT.value[ix_match_fore]
         
                betas_aux = {}
                for i,tau in enumerate(self.prob):
                    beta = \
                    fmin(loss_function, args=(observation,prediction,tau), x0=beta0, 
                         xtol=1e-8, disp=False, maxiter=1e20)
                    betas_aux[i] = beta
                
                setattr(getattr(self.betas,location),leadT,betas_aux.values())

                del observation, prediction
                gc.collect()

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
                lambda prediction, prob=self.prob, betas=betas_loc_leadT, cdf_keyword='cdf': \
                cqr_cdf(prediction, prob, betas, cdf_keyword)
                setattr(getattr(self.cdf, location), leadT, cdf_loc_leadT)
                   
                inv_cdf_loc_leadT = \
                lambda prediction, prob=self.prob, betas=betas_loc_leadT, cdf_keyword='inv_cdf': \
                cqr_cdf(prediction, prob, betas, cdf_keyword)
                setattr(getattr(self.inv_cdf, location), leadT, inv_cdf_loc_leadT)
            
        pass


    def _apply_cdf(self, data):    
        
        #Using the defined cummulative density function (cdf) we can now convert every observation 
        #into the uniform domain. This is done for every location and every lead time.
        
        self.uniform = expando()
        for location in data.metadata.id_nodes:
            print(location)
            setattr(self.uniform, location, expando())
            data_tran_loc = getattr(self.clim_transfo.obs, location)
            for ileadT, leadT in enumerate(data.metadata.fore_leadT, start = 1):
                cdf_loc_leadT = getattr(getattr(self.cdf, location), leadT)
                data_tran_fore_leadT = getattr(getattr(self.clim_transfo.fore, leadT), location)
    
                ix_match_obs_bool = np.in1d(data_tran_loc.t_actual, data_tran_fore_leadT.t_actual)   
                ix_match_obs = np.where(ix_match_obs_bool)[0]
                observation = data_tran_loc.value[ix_match_obs]
                observation.index = range(len(observation))
                
                ix_match_fore_bool = np.in1d(data_tran_fore_leadT.t_actual, data_tran_loc.t_actual)   
                ix_match_fore = np.where(ix_match_fore_bool)[0]
                prediction = data_tran_fore_leadT.value[ix_match_fore]
                prediction.index = observation.index
    
                unif_aux = {}    
                unif_aux['value'] = {}
                unif_aux['time'] = {}
                unif_aux['date'] = {}
                
                unif_aux['t'] = data_tran_loc.t_actual[ix_match_obs]
                unif_aux['t'].index = range(len(observation))
                
                for index in unif_aux['t'].keys():  
                    conditional_cdf_loc_leadT = cdf_loc_leadT(prediction[index])
                    unif_aux['value'][index] = float(conditional_cdf_loc_leadT(observation[index]))
                    unif_aux['time'][index] = unif_aux['t'][index].time()
                    unif_aux['date'][index] = unif_aux['t'][index].date()
                    del conditional_cdf_loc_leadT
                unif_aux = pd.DataFrame(unif_aux,columns=['t','value','time','date'])
                
                setattr(getattr(self.uniform, location), leadT, unif_aux)
                
            del unif_aux, observation, prediction
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
        correlation_matrix[where_are_NaNs] = 0.000
        
        self.corr = expando()
        self.corr.correlation_matrix = correlation_matrix
        self.corr.pivot_columns = uniform_pivot.columns
            
        pass



    def _generalize_corr(self, data):    
        #The purpose of this function is to extrapolate the correlation values to
        #different distances and delta lead times. In order to do so, 
        #an exponential function is used for fitting.
        self.corr.fit = expando()
        self.corr.fit.dist = expando()
        self.corr.fit.dt = expando()
        self.corr.fit.combined = expando()
        self.corr.fit.dist.func = exp_func
        self.corr.fit.dt.func = cauchy_func
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
        y = np.squeeze(np.asarray(corr_original[:,2]))   
        self.corr.fit.dist.coeff, pcov = curve_fit(self.corr.fit.dist.func, x, y, p0 = np.array([0.01]))        
        del x
        x = np.squeeze(np.asarray(corr_original[:,1])) 
        self.corr.fit.dt.coeff, pcov = curve_fit(self.corr.fit.dt.func, x, y, p0 = np.array([1,1]))
        
#        tau = self.corr.fit.dist.coeff
#        a = self.corr.fit.dt.coeff[0]
#        b = self.corr.fit.dt.coeff[1]
#        self.corr.fit.combined.func = lambda x_i, beta : mix_func(x_i, beta, tau, a, b)
        self.corr.fit.combined.func = mix_func
        
        del x    
        x = np.squeeze(np.asarray(corr_original[:,0:2]))   
        self.corr.fit.combined.coeff, pcov = curve_fit(self.corr.fit.combined.func, x, y)
          
        pass


    
    
    
        

