from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)


def init():
    # load the saved model.
    global  predictionmodal
    predictionmodal = joblib.load("WQP_RFC_ML_Model.pkl")


@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    total_score = 0
    try: 
        if request.method == 'POST':
            print(22)
            fixed_acidity = float(request.form['Fixed_Acidity'])
            volatile_acidity = float(request.form['Volatile_Acidity'])
            citric_acid = float(request.form['Citric_Acid'])
            residual_sugar = float(request.form['Residual_Sugar'])
            chlorides = float(request.form['Chlorides'])
            total_sulfur_dioxide = float(request.form['Total_Sulfur_Dioxide'])
            free_sulfur_dioxide	= float(request.form['Free_Sulfur_Dioxide'])
            density = float(request.form['Density'])
            pH = float(request.form['PH'])
            sulphates = float(request.form['Sulphates'])
            alcohol = float(request.form['Alcohol'])
            #print(sulphates)
            print(21)

            # Predict Apparent temperature
            # Same order as the x_train dataframe
            features = [np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,total_sulfur_dioxide, density, pH, sulphates, alcohol])]
            prediction = predictionmodal.predict(features)
            finalresult = 'Quality : ' + str(prediction[0])
        return render_template('index.html', result = finalresult)

    except Exception as e:
        print(e)
        return 'Calculation Error' + str(e), 500

if __name__ == '__main__':
    init()
    app.run(debug=True)
