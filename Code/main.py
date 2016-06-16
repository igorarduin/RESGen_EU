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



##INPUTS TO BE MODIFIED
#Path where all .py files are stored
#Path to data
#Output path to store scenarios in csv
#Renewable type to be studied: 'wind' or 'solar'
renewable_type = 'wind' #
#Data type: 'COSMO' or 'ECMWF' (COSMO recommended)
data_type = 'COSMO'
#Countries to be studied - see in documentation for list of countries keywords
countries = {'FRA'}
#Mawimum number of locations (random selection among all selected countries)
max_number_loc = 10
#Number of lead times to be studied (up to 91)
nbr_leadTimes = 10
#Starting and ending time of training period ('YYYY-MM-DD HH:MM:SS')
start_time = '2012-01-02 00:00:00'
end_time = '2012-12-31 00:00:00' 
#Starting and ending time of testing period - when scenarios will be generated ('YYYY-MM-DD HH:MM:SS')
fore_start_time = '2014-09-01 00:00:00'
fore_end_time = '2014-09-10 00:00:00'
#Use of the improved forecast model (0:no - 1:yes) - only relevant for wind case
improv_forecast = 1
#Number of scenarios to be computed
nb_scenarios = 50 


##CODE STRUCTURE - DON'T MODIFY IF ONLY USE
import sys
sys.path.insert(0, folder_code)
from dataReader import dataReader
from modelEstimation import modelEstimation
from scenarioGeneration import scenarioGeneration, save_scenarios

data = dataReader(countries,max_number_loc,renewable_type,data_type,start_time,
                  end_time,fore_start_time,fore_end_time,nbr_leadTimes,folder_data)
model = modelEstimation(data)
scenarios = scenarioGeneration(model, data, improv_forecast, nb_scenarios)
save_scenarios(scenarios, folder_output)
    
