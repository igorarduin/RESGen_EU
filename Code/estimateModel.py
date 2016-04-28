def estimateModel(ipts, data):
                      
    import numpy as np                 
    import pandas as pd  
    import gc
    
    from scipy.stats import norm
    from scipy.optimize import fmin
    from scipy.interpolate import interp1d
    from datetime import datetime, timedelta           
              
    class expando:
        pass 

    
    time_of_day = data.obs.Time.dt.time[range(24)]
    half_day_ix = time_of_day[len(time_of_day)/2].hour
    tod_name = [None] * len(time_of_day)
    #Making Climatology time-of-day transformation
    climatology = expando()
    for location in data.obs.keys()[1:]:
        exec("climatology." + location + " = expando()") 
        for index,itime in enumerate(time_of_day):
            if itime.hour<10: nb_name= '0' + str(itime.hour)
            else: nb_name = str(itime.hour)
            tod_name[index] = 'h_'+ nb_name
            exec("climatology."+location+"."+tod_name[index]+" = data.obs." + location + "[data.obs.Time.dt.time == itime]")
            
    #Making the cdf and the inv_cdf for the climatology to use as a transformation
    climatology_cdf = expando()
    climatology_inv_cdf = expando()
    for location in data.obs.keys()[1:]:
        exec("climatology_cdf." + location + " = expando()")
        exec("climatology_inv_cdf." + location + " = expando()")
        exec("clim_loc = climatology." + location)
        
        exec("clim_loc_hd = climatology." + location + "." + tod_name[half_day_ix])         
        
        for time in tod_name:        
            exec("clim_loc_t = climatology." + location + "." + time)
            len_clim = len(clim_loc_t)
            
            def cdf_function(time=time,location=location):
                if (len_clim > 0):
                    quantiles = {}
                    probabilities = np.arange(1,len_clim+1)/(float(len_clim)+1)     
                    quantiles = sorted(clim_loc_t)
                    quantiles_extended = \
                    np.concatenate([[min(quantiles)*0 - 1], quantiles, [max(quantiles)*2 + 1000]])
                    quantiles_extended[quantiles_extended < 0] = float(0)
                    quantiles_extended = np.concatenate([[-1],quantiles_extended,[10e8]])
                    probabilities_extended = np.concatenate([[0.0,0.0001],probabilities,[1,1]])
                else:
                    quantiles_extended = np.array([0.0,max(clim_loc_hd)])
                    probabilities_extended = np.zeros(len(quantiles_extended))
                interpolation = interp1d(quantiles_extended, probabilities_extended)
                return interpolation
            exec("climatology_cdf." + location + "." + time + " = cdf_function()")
               
            def inv_cdf_function(time=time,location=location):
                if (len_clim > 0):
                    quantiles = {}
                    probabilities = np.arange(1,len_clim+1)/(float(len_clim)+1)     
                    quantiles = sorted(clim_loc_t)
                    quantiles_extended = \
                    np.concatenate([[min(quantiles)*0 - 1], quantiles, [max(quantiles)*2 + 1000]])
                    quantiles_extended[quantiles_extended < 0] = 0
                    quantiles_extended = np.concatenate([[-1],quantiles_extended,[10e4]])
                    probabilities_extended = np.concatenate([[-0.0000001,0],probabilities,[1,1.01]])
                else:
                    quantiles_extended = np.array([0.0,max(clim_loc_hd)])
                    probabilities_extended = np.zeros(len(quantiles_extended))
                interpolation = interp1d(probabilities_extended,quantiles_extended)
                return interpolation
            exec("climatology_inv_cdf." + location + "." + time + " = inv_cdf_function()")


    #Transforming observations and predictions (by climatology) to get rid of seasonality
    #First transforming the observed power and storing it in data_transformed
    print('Climatology transformation')
    data_transformed = expando()
    for location in data.obs.keys()[1:]: 
        exec("data_transformed." + location + " = expando()")
        exec("data_loc = data.obs." + location)  
        
        data_transformed_aux = {}
        data_transformed_aux['t_actual'] = data.obs.Time
        data_transformed_aux['value'] = {}
        
        for index, time in data.obs.Time.dt.time.iteritems():
            exec("clim_cdf_loc_t = climatology_cdf." + location + "." + tod_name[time.hour])
            data_transformed_aux['value'][index] = float(clim_cdf_loc_t(data_loc[index]))  
        
        data_transformed_aux = pd.DataFrame(data_transformed_aux,columns=['t_actual','value'])
        exec("data_transformed." + location + " = data_transformed_aux")
        del index, time, data_transformed_aux

    
    #Second, the predicted power is transformed and stored in data_forecast_transformed
    t_fore = pd.DataFrame(np.nan, index=range(len(data.fore.leadT_01.keys())), columns=['t_issue','t_actual'])
    for idate, corr_date in enumerate(data.fore.leadT_01.keys()):
        date_id = corr_date[1:]
        date_issue = datetime(int(date_id[:4]), int(date_id[4:6]), int(date_id[6:8]), int(date_id[8:]),0,0)
        t_fore.t_issue[idate] = date_issue
            
    data_fore_transformed = expando()
    for ileadT, leadT in enumerate(data.fore_leadT, start = 1):
        exec("data_fore_transformed." + leadT + " = expando()")
        exec("data_fore_leadT = data.fore." + leadT)  
        t_fore.t_actual = t_fore.t_issue + timedelta(hours=ileadT)
        data_forecast_transformed_aux = expando() 
        for location in data.obs.keys()[1:]: 
            id_loc = location[3:]
            data_fore_transf_aux_location = {}
            data_fore_transf_aux_location['t_issue'] = t_fore.t_issue
            data_fore_transf_aux_location['t_actual'] = t_fore.t_actual
            data_fore_transf_aux_location['value'] = {}
            for idate, date_issue in enumerate(data_fore_leadT.keys()):
                exec("power_to_be_transformed= data_fore_leadT."+date_issue+"[id_loc]")                  
                exec("clim_cdf_loc_t = climatology_cdf." + location + "." + tod_name[t_fore.t_issue[idate].hour])
                data_fore_transf_aux_location['value'][idate] = float(clim_cdf_loc_t(power_to_be_transformed))
            
            data_fore_transf_aux_location = pd.DataFrame(data_fore_transf_aux_location, columns=['t_issue','t_actual','value'])
            exec("data_forecast_transformed_aux." + location + "= data_fore_transf_aux_location")
        exec("data_fore_transformed." + leadT + "= data_forecast_transformed_aux")   
            



#    Compute quantiles for the transformed power conditional on the transformed power prediction
#    for a specific location and a specific lead time.
#
#    In order to use conditional quantile regression we first need to define the type of model 
#    that we are looking at. The model takes as input some parameters and an input value and 
#    returns the appropriate quantile. If regression type is defined as "poly", the quantiles will
#    be polynomial interpolations. If it is defined as "spline", the calculations will use spline in R. 
#    It is therefore needed to download package rpy2.


    betas = {}
    beta0 = np.zeros(2)
    def model(beta,x):
         return np.polyval(beta,x)
    def rho(y,tau):
        taum=tau-1
        return np.where(y<0,y*taum,y*tau)    
    def loss_function(b,observation,prediction,tau): #loss function needs to be minimized (cf wikipedia)
        return np.sum(rho(observation - model(b,prediction), tau))

#    Performs the actual quantile regression and stores the variables of 
    print('Quantiles calculation')
    probabilities = np.concatenate([[0.001],np.arange(0.05,0.951,0.05),[0.999]])
    betas = expando()
    for location in data.obs.keys()[1:]: 
        print(location)
        id_loc = location[3:]
        exec("betas."+location+" = expando()")
        exec("data_tran_loc = data_transformed." + location) 
    
        for ileadT, leadT in enumerate(data.fore_leadT, start = 1):
            print(leadT) 
            exec("data_fore_fore_leadT = data_fore_transformed." + leadT + "." + location)

            ix_match_obs_bool = np.in1d(data_tran_loc.t_actual, data_fore_fore_leadT.t_actual)   
            ix_match_obs = np.where(ix_match_obs_bool)[0]
            observation = data_tran_loc.value[ix_match_obs]
            
            ix_match_fore_bool = np.in1d(data_fore_fore_leadT.t_actual, data_tran_loc.t_actual)   
            ix_match_fore = np.where(ix_match_fore_bool)[0]
            prediction = data_fore_fore_leadT.value[ix_match_fore]
     
            betas_aux = {}
            for i,tau in enumerate(probabilities):
                beta = \
                fmin(loss_function,args=(observation,prediction,tau),x0=beta0, xtol=1e-8, disp=False, maxiter=1e20)
                betas_aux[i] = beta
                
            exec("betas."+location+"."+ leadT +" = betas_aux.values()")
            
            del observation, prediction
            gc.collect()
                      
         

#    In order to use the copula approach we need to transform to uniform marginal distributions.
#    This is achieved by using the predictive marginal densities on the transformed domain to 
#    do a second transformation to get uniformly distributed marginals. For a complete and 
#    acessable introduction see the wikipedia page on Copulas.

#    First define the marginal cummulative density functions. They are stored as the cummulative 
#    density function (cdf) and for easy use we also define the inverse cumulative density 
#    function inv_cdf. Each is defined for every location and every lead time.

    
    cdf = expando()
    inv_cdf = expando()
    
    for location in data.obs.keys()[1:]:
        exec("cdf." + location + " = expando()")
        exec("inv_cdf." + location + " = expando()")

        for leadT in data.fore_leadT:
            exec("betas_loc_leadT = betas."+location+"."+ leadT)
            
            def cdf_function(x, prediction,leadT=leadT,location=location): 
                quantiles = {}
                for i in range(0,len(probabilities)):
                    quantiles[i] = float(model(betas_loc_leadT[i],prediction))
                quantiles_extended = np.concatenate([[0], sorted(quantiles.values()), [1]])
                quantiles_extended[quantiles_extended < 0] = 0
                quantiles_extended[quantiles_extended > 1] = 1
                probabilities_extended = np.concatenate([[0],probabilities,[1]])
                interpolation = interp1d(quantiles_extended, probabilities_extended)
                return interpolation(x)
                
            exec("cdf." + location + "." + leadT + " = cdf_function")
               
            def inv_cdf_function(x, prediction,leadT=leadT,location=location):  
                quantiles = {}
                for i in range(0,len(probabilities)):
                    quantiles[i] = float(model(betas_loc_leadT[i],prediction))
                quantiles_extended = np.concatenate([[0], sorted(quantiles.values()), [1]])  
                quantiles_extended[quantiles_extended < 0] = 0
                quantiles_extended[quantiles_extended > 1] = 1
                probabilities_extended = np.concatenate([[0],probabilities,[1]])
                interpolation = interp1d(probabilities_extended, quantiles_extended)
                return interpolation(x)
                
            exec("inv_cdf." + location + "." + leadT + " = inv_cdf_function")


#    Using the defined cummulative density function (cdf) we can now convert every observation 
#    into the uniform domain. This is done for every location and every lead time.

    
    print('CDF transformation')
    uniform = expando()
    for location in data.obs.keys()[1:]:
        print(location)
        exec("uniform." + location + " = expando()")
         
        exec("data_tran_loc = data_transformed." + location) 
        for ileadT, leadT in enumerate(data.fore_leadT, start = 1):
            print(leadT)          
            exec("cdf_loc_leadT = cdf." + location + "." + leadT)
            exec("data_fore_fore_leadT = data_fore_transformed." + leadT + "." + location)

            ix_match_obs_bool = np.in1d(data_tran_loc.t_actual, data_fore_fore_leadT.t_actual)   
            ix_match_obs = np.where(ix_match_obs_bool)[0]
            observation = data_tran_loc.value[ix_match_obs]
            observation.index = range(len(observation))
            
            ix_match_fore_bool = np.in1d(data_fore_fore_leadT.t_actual, data_tran_loc.t_actual)   
            ix_match_fore = np.where(ix_match_fore_bool)[0]
            prediction = data_fore_fore_leadT.value[ix_match_fore]
            prediction.index = observation.index

            unif_aux = {}    
            unif_aux['value'] = {}
            unif_aux['time'] = {}
            unif_aux['date'] = {}
            
            unif_aux['t'] = data_tran_loc.t_actual[ix_match_obs]
            unif_aux['t'].index = range(len(observation))
            
            for index in unif_aux['t'].keys():  
                unif_aux['value'][index] = float(cdf_loc_leadT(observation[index],prediction[index]))
                unif_aux['time'][index] = unif_aux['t'][index].time()
                unif_aux['date'][index] = unif_aux['t'][index].date()
            unif_aux = pd.DataFrame(unif_aux,columns=['t','value','time','date'])
            exec("uniform." + location + "." + leadT +" = unif_aux")
        del unif_aux, observation, prediction
        gc.collect()
    

#    Next we estimate the correlation matrix for the uniform variables. To facilitate this the
#    uniform variables are put on an appropriate form for computing a correlation matrix. This
#    is done through using a pivot table 

    
    location_vect = [None] * len(data.obs.keys()[1:])
    leadT_vect = [None] * len(data.obs.keys()[1:])
    for iloc, location in enumerate(data.obs.keys()[1:]):
        location_vect[iloc] = location
    for ileadT, leadT in enumerate(data.fore_leadT):
        leadT_vect[ileadT] = leadT
        
    for location in data.obs.keys()[1:]:
        for leadT in data.fore_leadT:
            exec("uniform_loc_leadT = uniform."+location+"."+leadT) 
            df_loc_leadT_temp = pd.DataFrame({'location': location, 't': uniform_loc_leadT.t, 'value': uniform_loc_leadT.value, 'ltname': leadT, 'date': uniform_loc_leadT.date, 'time': uniform_loc_leadT.time})
            try:             
                uniform_df = pd.concat([uniform_df, df_loc_leadT_temp])
            except:
                uniform_df = df_loc_leadT_temp
            del df_loc_leadT_temp


    uniform_df['value']=uniform_df['value'].astype(float)
    uniform_pivot = uniform_df.pivot_table(index='date',columns=('location','ltname'),values='value')
    
    norm_df =  uniform_df
    norm_df['value'] = norm.ppf(uniform_df['value'].astype(float))
    norm_pivot = norm_df.pivot_table(index='date',columns=('location','ltname'),values='value')
           
           
    
#    From the observations in the uniform domain we can now compute the correlation matrix. 
#    The correlation matrix specifies the Gaussian copula used for combining the different models. 
#    Where the computed correlation is NaN we set it to zero.
    

    correlation_matrix_na = norm_pivot.corr()
    where_are_NaNs = np.isnan(correlation_matrix_na)
    correlation_matrix = correlation_matrix_na
    correlation_matrix[where_are_NaNs] = 0.000
    
    class expando:
        pass        
    
    output = expando()

    output.climatology_cdf = climatology_cdf
    output.climatology_inv_cdf = climatology_inv_cdf
    output.cdf = cdf
    output.inv_cdf = inv_cdf
    output.correlation_matrix = correlation_matrix
    output.pivot_columns = uniform_pivot.columns
    output.data = data
    output.betas = betas
    
    print('Estimation finished!')
        

    return output
