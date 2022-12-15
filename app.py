from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# App Name
app = Flask(__name__)

# Create Route 
@app.route("/")
def home():
    # print("TEST", svmPredict)
    return render_template("index.html")
  

@app.route("/svm", methods=['GET', 'POST'])
def rain_detector_svm():
    if request.method == 'GET':
        return render_template("svm.html")    
    elif request.method == 'POST':
        rain_features = dict(request.form).values()
        rain_features = np.array([float(x) for x in rain_features])
        module_dir = os.path.dirname(__file__)
        f = os.path.join(module_dir, 'train_model/predict_rain_model_svm_fix.pkl')
        model = joblib.load(f)
        robust_scaler = model[0]
        model_svm = model[2]
        rain_features = robust_scaler.transform([rain_features])
        result = model_svm.predict(rain_features)
        rain = {
            '0': 'Not Rain',
            '1': 'Rain'
        }
        result = rain[str(result[0])]
        return render_template('svm.html', result=result)
    else:
        return "Invalid Input"  

@app.route("/logreg", methods=['GET', 'POST'])
def rain_detector_logreg():
    if request.method == 'GET':
        return render_template("logreg.html")    
    elif request.method == 'POST':
        rain_features = dict(request.form).values()
        rain_features = np.array([float(x) for x in rain_features])
        module_dir = os.path.dirname(__file__)
        f = os.path.join(module_dir, 'train_model/predict_rain_model_logreg_fix.pkl')
        model = joblib.load(f)
        robust_scaler = model[0]
        model_logreg = model[2]
        rain_features = robust_scaler.transform([rain_features])
        result = model_logreg.predict(rain_features)
        rain = {
            '0': 'Not Rain',
            '1': 'Rain'
        }
        result = rain[str(result[0])]
        return render_template('logreg.html', result=result)
    else:
        return "Invalid Input"  

# Main Function
if __name__ == "__main__":
    app.run(debug=True)