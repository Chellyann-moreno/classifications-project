# Project description:
   We would be looking at customer data from a the Telcom, both the telecom company data are not real. 
  This data would not be used on future customers or for real life prediction.
# Project goals:
  In this project we would analyze the different factors as to why the customers are leaving the company (churning).
  We would try to predict what factors may cause clients to churn.
    Some of these factors are: monthly charges, contract type, age, tech support access, etc.
    Focusing on these factors would help us come up with ideas on how to keep our current clients from churning,
    and how the company could improve and reduce the rate of churning customers. 
# Initial hypotheses and questions about the data:
  1. Does clients with a contract churn less than clients that pay month to month?
  2. Clients with dependent tend to churn less than clients with no dependents?
  3. Does senior clients churn less than client who are not seniors?
  4. Clients with tech support tend to churn less? Does the type of internet service affects the churning rate?
  5. Clients with lower monthly charges tend to churn less than clients with a higher monhtly charge?
  6. Is there a difference between gender churning rate? Does having streaming services affects the churning rate?
# Data dictionary:
![Alt text](https://github.com/Chellyann-moreno/classifications-project/blob/98ee7df94144ca83ee074de4b9c4c9e386f63132/Data%20Dictionary.png
)
- For information please visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# Project planning, layout of science pipeline:
  1. Project Planning- During this process we asked ourselves important questions about the project. Data planning will be shown in this readme.
  2. Data acquisition- We would be acquiring the data from Codeup Data Server in MYSQL. Raw data would be downloaded and "telco.csv" has been created which would be use to pull the data during this project.
  3. Data preparation- The telco data would be clean and prepared for exploration. Columns were listed along with data types. columns with incorrect data types were transformed, columns were created to encode a int/float instead of a string, and columns that are not to be used were dropped.
  4. Data exploration- During the data exploration we would visualize and answer our questions and hypotheses. We would be using statistical tests and plots, to help proving our hypotheses and understand the reasoning for clients churning.
  5. Data modeling- We would be using the most important features(6 to 7) to check the accuracy and recall scores. By looking on how to prevent customers from churning we would be using our true positive(TP) conditions, focusing on our recall rate.
  6. Data Delivery- We would be using a jupyter notebook, were we would be showing our visualizations, questions and findings. We would also show the  score moetrics of our best model. This project would be a written and a verbal presentation. 
 # Instructions on how to reproduce this project:
  For an user to  succesfully reproduce this project, they must have a connection to the CodeUp Server in MySQL. User must have a "env.py" with the username, password, and database name,  to establish a connection.
  Also, the acquire.py, explore.py and prepare.py must be download in the same repository/folder as the final_report to run it successfully.
  Once all files are download, user may run the final_report notebook.
 # Key findings, recommendations and takeaways:
   - About 26.5% of the customers have churned the company.
   - Customers with access to tech support tend to stay at the company.
   - Customers with dependents and extra add-ons, have a lower chance terminating the services.
   - Senior citizens tend to churn more. However, the senior citizen population is low.
   - Customers with a month to month contract tend to churn at a higher rate than those with a one or two year contract.
   - Based on our observations, we recomment using the machine model to identify customers with a high probablity of churning. Offer them tech support services, discounts and free add-ons to keep them from churning.
 
    
   

