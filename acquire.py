### IMPORTS
import pandas as pd
import os
import env
directory='/Users/chellyannmoreno/codeup-data-science/Methodologies I/'

### FUNCTIONS
def get_db_url(database):
  return f'mysql+pymysql://{env.username}:{env.password}@{env.host}/{database}'



### GET TELCO 
def new_telco_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a connection_url to mySQL
    - return a df of the given query from the telco_db
    """
    url = get_db_url('telco_churn')
    
    return pd.read_sql(SQL_query, url)



def get_telco_data():
   " this function will help pull the data from SQL and freate a new file if necessary or open the existing file with the name ~telco.csv~"
   SQL_query="""SELECT * FROM telco_churn.customers
            left join telco_churn.payment_types using (payment_type_id)
            left join telco_churn.internet_service_types using (internet_service_type_id)
            left join telco_churn.contract_types using (contract_type_id);"""
   url = get_db_url('telco_churn')
   filename='telco.csv'
   if os.path.exists(directory+filename): 
        df = pd.read_csv(filename)
        return df
   else:
        df = new_telco_data(SQL_query)

        df.to_csv(filename)
        return df