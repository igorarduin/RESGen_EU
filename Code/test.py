import sys
sys.path.insert(0, 'C:/Users/Igor/Documents/DTU/Thesis/Code/IgorV1')

from dataReader import dataReader
from modelEstimation import modelEstimation
from scenarioGeneration import scenarioGeneration

#Overall variables
folder_location = 'C:/Users/Igor/Documents/DTU/Thesis/Code/IgorV1/RE-Europe_dataset_package/'
renewable_type = 'wind' #wind or solar
data_type = 'COSMO' #COSMO or ECMWF

#Model construction variables
countries = {'DNK','FR','PR'}
max_number_loc = 20
start_time = '2012-01-01 00:00:00' #YYYY-MM-DD HH:MM:SS
end_time = '2012-09-01 00:00:00' #YYYY-MM-DD HH:MM:SS
nbr_leadTimes = 20 #number of lead times studied (up to 91)

data = dataReader(countries,max_number_loc,renewable_type,data_type,start_time,
                  end_time,nbr_leadTimes,folder_location)

model = modelEstimation(data)

#current Forecast
data_fore = dataReader(countries,max_number_loc,renewable_type,data_type,start_time,
                       end_time,nbr_leadTimes,folder_location)
fore_start_time = '2012-09-02 00:00:00' #YYYY-MM-DD HH:MM:SS
fore_end_time = '2012-09-03 00:00:00' #YYYY-MM-DD HH:MM:SS
current_fore = dataReader(countries,max_number_loc,renewable_type,data_type,start_time,
                          end_time,nbr_leadTimes,folder_location)


scenarios = scenarioGeneration(model, data_fore, current_fore, 10)



network_nodes = pd.read_csv(folder_location+'Metadata/network_nodes.csv', sep=',')
available_countries = set(network_nodes.country)

