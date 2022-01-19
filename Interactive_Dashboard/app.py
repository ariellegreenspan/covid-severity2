# Code resource: https://www.analyticsvidhya.com/blog/2021/08/quick-hacks-to-save-machine-learning-model-using-pickle-and-joblib/
from flask import Flask, render_template, jsonify, json, request, redirect
from joblib import dump, load
from pickle import dump as dump_p, load as load_p
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import os
import psycopg2


# Load pipeline
pipeline = load("Interactive_Dashboard/ml/pipeline.joblib")
# Load the label encoders
le_sex = load_p(open('Interactive_Dashboard/ml/le_sex.pkl', 'rb'))
le_state = load_p(open('Interactive_Dashboard/ml/le_state.pkl', 'rb'))
le_race = load_p(open('Interactive_Dashboard/ml/le_race.pkl', 'rb'))
le_ethnicity = load_p(open('Interactive_Dashboard/ml/le_ethnicity.pkl', 'rb'))
le_age = load_p(open('Interactive_Dashboard/ml/le_age.pkl', 'rb'))

# Run app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('FINAL_PROJECT_DATA', '')

# Remove tracking modifications
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
class Booster(db.Model):
    __tablename__ = 'booster_table'

    county_name = db.Column(db.String, primary_key=True)
    state = db.Column(db.String)
    cases = db.Column(db.Integer)
    deaths = db.Column(db.Integer)
    completely_vaccinated_population = db.Column(db.Numeric)
    dose_1_population = db.Column(db.Numeric)
    booster_dose_population = db.Column(db.Numeric)
    
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = 0
    a = {}

    if request.method == "POST":
        print(request.form)
        # read form data inputed by user
        user_sex = request.form["sex"]
        user_state = request.form["state"]
        user_race = request.form["race"]
        user_ethnicity = request.form["ethnicity"]
        user_age = request.form["age_range"]
        

        # Place user inputs into a list and create df for label encoding
        inputs = [user_sex, user_state, user_race, user_ethnicity, user_age]
        inputs_pd = pd.DataFrame([inputs, inputs])
        # Encode user inputs
        #inputs_pd[1] = le_sex.transform(inputs_pd[1])
        #inputs_pd[2] = le_state.transform(inputs_pd[2])
        #inputs_pd[3] = le_race.transform(inputs_pd[3])
        #inputs_pd[4] = le_ethnicity.transform(inputs_pd[4])
        #inputs_pd[5] = le_age.transform(inputs_pd[5])

        # Run the pipeline (Scaler and rf_model) on user inputs
        prediction_vector = pipeline.predict_proba(inputs_pd)
        # Extract the probability to get 1(Serious or Fatal crash)
        prediction = prediction_vector[0][1]
        print(prediction)
        
        # Dict of user inputs to reload
        a = {
        "sex": user_sex,
        "state": user_state,
        "race": user_race,
        "ethnicity": user_ethnicity,
        "age_range": user_age,
        }
        print(a)

    return render_template("index.html", predict=5 * prediction, form_reuse=a)

@app.route("/summary")
def summary():
    return render_template("home.html")

@app.route("/booster_table")
def booster_table():
    mycursor.execute("SELECT * FROM booster_table")
    data = mycursor.fetchall()
    print(Booster.query.all()[0:10])
    #return render_template('booster_table.html', data=data)

if __name__ == "__main__":
    app.run(debug=True)
