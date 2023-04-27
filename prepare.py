#Imports'
import pandas as pd
import numpy as np
import env
import acquire as acq
from sklearn.model_selection import train_test_split


# FUNCTIONS


def prep_telco(df):
    """ This function helps clean and prepare the telco data.
    Generating encoded columns, renaming columns, and dropping that would not be of used"
    """
    df=df.drop(columns=['internet_service_type_id', 'contract_type_id', 'payment_type_id'])
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
    dummy_df = pd.get_dummies(df[['multiple_lines', 'online_security', 'online_backup','device_protection',  'tech_support', 'streaming_tv', 'streaming_movies', 'contract_type',  'internet_service_type','payment_type']],
                                  )
    
    df = pd.concat( [df,dummy_df], axis=1 )
    
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
    df=df.rename(columns={"contract_type_Month-to-month":"no_contract"})
    return df





def split_data(df,variable):
    """This function helps divide the data into train, validate, and testing"
    """
    train, test = train_test_split(df,
                                   random_state=123, test_size=.20, stratify= df[variable])
    train, validate = train_test_split(train, random_state=123, test_size=.25, stratify= train[variable])
    return train, validate, test




    