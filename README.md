# Data_Science_Capstone

# Health Condition Predictor
<br>

## Selected Topic
### Analyzing heart disease dataset from CDC website.
- TIn this project we are trying to analyze different factors that may contribute to heart disease using Machine learning algorithms and various tools.   

## Reason for Selecting Topic
- We want to build a model that will predict the heart condition good or bad depending on different factors. 
- This information can be used by physicians or health insurance companies and by any individual user who are trying to find out their heart health condition.

## Description of the Source Data
CDC website - https://www.cdc.gov/brfss/annual_data/annual_2020.html
- This dataset has 401,958 records for 2020 
- This file for the combined landline and cell phone data set was exported from SAS V9.3 in the XPT transport format. This file contains 279 variables.
- This link https://www.cdc.gov/brfss/annual_data/2020/summary_matrix_20.html has the Summary Matrix of Calculated Variables (CV) in the 2020 Data File
- This dataset has multitude of features such as but not limited to the below ones 
BMI, Age, Sex, Physical Activity, Cancer, Ashthma, Race, Drinking, Smoking, Mental Health, Physical Health, General Health etc.
- The definition of each of the columns are also mentioned in https://www.cdc.gov/brfss/annual_data/2020/pdf/2020-calculated-variables-version4-508.pdf
- For the sake of our analysis, we reduced the 279 columns to 22 columns

## Questions we hope to answer with the data
- How do different cfeatures in the given dataset impact Heart health condition?
- Does any Feature have more impact over the other?
 

## Communication Protocols
- In order to keep updated on the status of each of our parts of the project, we message each other regularly through Slack and organized regular zoom meetings.

## Tools
- Creating Database
    - Jupyter Notebook
    - PostgreSQL
- Connecting to Database
    - 
    - Psycopg2
    - SQLAlchemy
- Analyzing Data
    - Pandas
- Machine Learning
    - Scikit-Learn
    - Logistic Regession
- Dashboard
    - Tableau
    - Javascript
    - Flask
    - HTML
    - CSS


## Machine Learning Model
- The dataset includes a columns with data as Calculated Variables. 
- We followed the Summary Matrix presented by CDC to go through each of the 279 columns to retrive the 22 ones which we intend to use in our program.
- We also used the Calculated Variables document to decode the SAS codes to get to know our data and what are the likely values for each of the columns. 
- In the Python code, we used pyreadstat module to read the ".XPT" file 
- Next we renamed the columns for easier readability
- After connecting to the Postgres database, we created the dataFrame to a table(Heart_Disease_22_Full) using df.to_sql().  This copies the python dataframe to a SQL table.
- As all the values in the dataset were encoded with SAS codes, we utilized some of the SQL update statements('Case When' statements) to bring back the actual values of the data to make sense when doing visualizations.
- This massaged data is exported to a CSV which is then used by Tableau for visualizations and also by the Machine learning program.
- Our dataset is labeled, so we can consider Supervised Machine Learning.
- In the Machine learning python code, target Feature was identified as the HeartDiease column.
- The rest of teh Features were used to train/test the model.
- This data is read and splitted based on categorical and numerical features. 
- The categorical features were encoded using LabelEncoder().
- The data was split into training and test data using the train_test_split function. We used the default 75% to 25% split.
- The data was then scaled using StandardScaler()
- Logistic regression was used to train the model as we are trying to arrive at a HeartDisease Yes/No answer. So its a classification technique hence Logistic Regression.
 

## Presentation

- Our project presentation is drafted in here [Google SlideShow][https://docs.google.com/presentation/d/1YGsOvHFZg9-QRkvNQFwV-EOQtRx9CBKobDLYu7l_36k/edit?usp=sharing]
## Dashboard
- We used Tableau as a part of our dashboard. Our Tableau analysis can be found here [Tableau Dashboard][https://public.tableau.com/app/profile/mark.vogel/viz/Heart_Disease_Visualizations_CDC_2020/FinalProjectTableau?publish=yes] 
- The other part of our dashboard is an interactive webpage using machine learning to calculate a heart disease yes/no. It includes an interactive element, users are able to select data that pertains to them (age, sex, race, smoking, etc.) and click a button that will give the a prediction for heart disease yes/no.
- The link to the  dashboard is here (TBD)
 
