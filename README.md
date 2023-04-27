# Project description:
   we would be looking at customer data from a the Telcom, both the telecom company data are not real. 
  This data would not be used on future customers or for real life prediction.
# Project goals:
  In this project we would analyze the different factors as to why the customers are leaving the company (churning).
  We would try to predict what factors may cause clients to churn.
    Some of these factors are: monthly charges, contract type, age, the access to tech support, etc.
    Focusing on these factors would help us come up with ideas on how to keep our current clients from churning the company,
    and how the company could improve in order to reduce the rate of churning customers. 
# Inital hypotheses and questions about thee data:
  1. Does clients with a contract churn less than clients that pay month to month?
  2. Clients with dependent tend to churn less than clients with no dependents?
  3. Does senior clients churn less than client who are not seniors?
  4. Clients with tech support tend to churn less? Does the type of internet service affects the churning rate?
  5. Clients with lower monthly charges tend to churn less than clients with a higher monhtly charge?
  6. Is there a difference between gender churning rate? Does having streaming services affects the churning rate?
# Data dictionary:
# Project planning, layout of science pipeline:
  1. Project Planning- During this process we asked ourselves important questions about the project. Data planning will be shown in this readme.
  2. Data acquisition- We would be acquiring the data from Codeup Data Server in MYSQL. Raw data would be downloaded and "telco.csv" has been created which would be use to pull the data during this project.
  3. Data preparation- The telco data would be clean and prepared for exploration. Columns were listed along with data types. columns with incorrect data types were transformed, columns were created to encode a int/float instead of a string, and columns that are not to be used were dropped.
  4. Data exploration- During data exploration we would be visualizing and answering our questions and hypotheses. We would be using statistical tests and plots, to help proving our hypotheses and understand the reasoning for clients churning.
  5. Data modeling- We would be using the most important features(6 to 7) to check the accuracy and recall scores. By looking on how to prevent customers from churning we would be using our true positive(TP) conditions, focusing on our recall rate.
  6. Data Delivery- We would be using a jupyter notebook were we would be showing our visualizations, questions and findings. We would also show the  score moetrics of our best model. This project would be a written and a verbal presentation. 
 # Instructions on how to reproduce this project:
  For a user to be able to reproduce this project they must have a connection to the CodeUP Server in MySQL. The user must have a "env.py" with the username, password, and database name, to establish the connection.
  Also, the acquire.py, explore.py and prepare.py must be downloaded in the folder as the final_report, to be able to run it successfully.
  Once all files are downloaded, user may run the final report notebook.
 # Key findings, recommendations and takeaways:
   - About 26.5% of the customers have churned the company.
   - 
   - 
 
    
   
