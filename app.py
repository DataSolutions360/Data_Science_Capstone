from flask import Flask, render_template, jsonify, json, request, redirect
# from joblib import dump, load
from pickle import dump as dump_p, load as load_p
import numpy as np
import pandas as pd
from sklearn. preprocessing import LabelEncoder


with open('saved_model.pkl','rb') as file:
    model = load_p(file)
print('i am here 0')


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = 0
    user_data = {}

    if request.method == "POST":
        print(request.form)
        # read form data inputed by user
        user_Sex = request.form["personGender"]
        user_AgeCategory = request.form["Age"]
        user_Race = request.form["Race"]
        user_health = request.form["Health"]        
        user_GenHealth = request.form["healthnotgood"]
        user_bmi = request.form["BMI"]
        user_Diabetic = request.form["diabetes"]
        user_MentalHealth = request.form["mhealthnotgood"]        
        user_AlchoholDrinking = request.form["drink"]
        user_Smoking = request.form["smoke"]
        user_Asthma = request.form["asthma"]
        user_KidneyDisease = request.form["kidney"]
        user_Stroke = request.form["stroke"]
        user_colonoscopy = request.form["cancer"]
        user_PhysicalActivity = request.form["physact"]       
        user_sleephours = request.form["sleephours"]

        
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

        print(input_df)
        print('after input_df')

        # Run the pipeline (Scaler and model) on user inputs
        
        # Extract the prediction to get to know the health condition

        prediction_heart = model.predict(input_df)
        
        # Extract the probability to get to know the health condition

        prediction_proba = model.predict_proba(input_df)
        # Extract the probability to get 1

        prediction = prediction_proba[0][1]
        print(prediction)

        print(f"Possibility of having a bad heart condition : {prediction_heart[0]}")
        print(f"Probablity of having a bad heart condition is: {prediction_proba[0][1]}")
        # if prediction_heart == 0:
        #     prediction_text = "NO" 
        # else:
        #     prediction_text = "YES"

        
        
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
            "Avg_Hours_of_Sleep": user_sleephours,  
        }

    print(user_data)
    # return render_template("index.html")
    return render_template("index.html", predict=prediction, form_reuse=user_data)
    
if __name__ == "__main__":
    app.run(debug=True)