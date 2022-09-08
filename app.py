from flask import Flask, render_template, jsonify, json, request, redirect
# from joblib import dump, load
from pickle import dump as dump_p, load as load_p
import numpy as np
import pandas as pd
import joblib
from sklearn. preprocessing import OneHotEncoder

# load the saved model
with open('saved_model.pkl','rb') as file:
    model = load_p(file)
encoder = joblib.load('encoder.joblib')
# Load the original dataset
df_orig = pd.read_pickle("encode.pkl")


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = 0
    user_data = {}

    if request.method == "POST":
        print(request.form)
        # read form data inputed by user
        user_Sex = int(request.form["personGender"])
        user_AgeCategory = int(request.form["Age"])
        user_Race = int(request.form["Race"])
        user_health = int(request.form["Health"])        
        user_GenHealth = int(request.form["healthnotgood"])
        user_bmi = int(request.form["BMI"])
        user_Diabetic = int(request.form["diabetes"])
        user_MentalHealth = int(request.form["mhealthnotgood"])        
        user_AlchoholDrinking = int(request.form["drink"])
        user_Smoking = int(request.form["smoke"])
        user_Asthma = int(request.form["asthma"])
        user_KidneyDisease = int(request.form["kidney"])
        user_Stroke = int(request.form["stroke"])
        user_colonoscopy = int(request.form["cancer"])
        user_PhysicalActivity = int(request.form["physact"])       
        user_sleephours = int(request.form["sleephours"])

        
        print(user_AgeCategory)
        
         # Place user inputs into a datFrame
        input_df = pd.DataFrame({
            "Birth_Sex": [user_Sex],   
            "Age": [user_AgeCategory],
            "Race": [user_Race], 
            "Overall_Health": [user_health],
            "Physical_Health": [user_GenHealth], 
            "BMI_CDC_Categories": [user_bmi],
            "Diabetes": [user_Diabetic], 
            "Mental_Health": [user_MentalHealth],    
            "Alcohol_Usage": [user_AlchoholDrinking],  
            "Tobacco_Usage": [user_Smoking], 
            "Asthma_History": [user_Asthma],  
            "Kidney_Disease": [user_KidneyDisease],
            "Stroke": [user_Stroke], 
            "Colonoscopy": [user_colonoscopy],                                                                              
            "Physical_Activity": [user_PhysicalActivity],  
            "Avg_Hours_of_Sleep": [user_sleephours],  
        })
       
        print("User entered:")
        print(input_df)
        print("------------------------------------------------------------------------")
        encode_col = ['Age', 'Race', 'Physical_Health', 'BMI_CDC_Categories', 'Diabetes', 'Mental_Health', 'Colonoscopy']
        
        # Drop the target Feature 
        df_orig_m = df_orig.drop(columns="Heart_Disease")

        # Concat the original datFrame loaded from pickle file and the user entered data in another dataFrame
        input_to_encode_df = pd.concat([input_df, df_orig_m], axis=0)

        # Fit and transform the OneHotEncoder using the categorical variable list
        encode_df = pd.DataFrame(encoder.fit_transform(input_to_encode_df[encode_col]))
        encode_df.columns = encoder.get_feature_names(encode_col)

        # Merge one-hot encoded features and drop the originals
        input_to_encode_df = input_to_encode_df.merge(encode_df,left_index=True, right_index=True)
        input_to_encode_df = input_to_encode_df.drop(encode_col,1)

        print('after input_df')
      
        input_to_encode_df = input_to_encode_df[:1]

        print(input_to_encode_df)
        print("------------------------------------------------------------------------")
        
        # Extract the prediction to get to know the health condition

        prediction_heart = model.predict(input_to_encode_df)
        
        # Extract the probability to get to know the health condition

        prediction_proba = model.predict_proba(input_to_encode_df)
        # Extract the probability to get 1
        print(prediction_proba)
        print("------------------------------------------------------------------------")

        prediction = prediction_proba[0][1]
        print(prediction)
        print("------------------------------------------------------------------------")


        print(f"Possibility of having a bad heart condition : {prediction_heart[0]}")
        print(f"Probablity of having a bad heart condition is: {round(prediction*100, 2)}")
        
        
        # Dict of user inputs to reload

        user_data = {
            "Birth_Sex": user_Sex,   
            "Age": user_AgeCategory,
            "Race": user_Race, 
            "Overall_Health": user_health,
            "Physical_Health": user_GenHealth, 
            "BMI_CDC_Categories": user_bmi,
            "Diabetes": user_Diabetic, 
            "Mental_Health": user_MentalHealth,    
            "Alcohol_Usage": user_AlchoholDrinking,  
            "Tobacco_Usage": user_Smoking, 
            "Asthma_History": user_Asthma,  
            "Kidney_Disease": user_KidneyDisease,
            "Stroke": user_Stroke, 
            "Colonoscopy": user_colonoscopy,                                                                              
            "Physical_Activity": user_PhysicalActivity,  
            "Avg_Hours_of_Sleep": user_sleephours 
        }

    print(user_data)
    # return render_template("index.html")
    return render_template("index.html", predict=round(prediction*100, 2), form_reuse=user_data)
    
if __name__ == "__main__":
    app.run(debug=True)