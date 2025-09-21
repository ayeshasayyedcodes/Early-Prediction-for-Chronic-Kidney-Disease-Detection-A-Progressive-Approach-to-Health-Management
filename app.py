from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

# --- Setup paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # project root
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
MODEL_DIR = os.path.join(BASE_DIR, "Models")

# --- Initialize Flask with correct template folder ---
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Debug prints
print(">>> Current Working Directory:", os.getcwd())
print(">>> Flask template folder:", app.template_folder)
print(">>> Absolute path:", os.path.abspath(app.template_folder))
print(">>> Files in template folder:", os.listdir(app.template_folder))

# --- Load ML model, scaler, and features ---
model = joblib.load(os.path.join(MODEL_DIR, 'kidney_disease_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))


# --- Routes ---
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1) Get form data
        form_data = {col: request.form[col] for col in request.form}
        input_data = pd.DataFrame([form_data])

        # ✅ Drop "id" if it exists (not needed by model)
        if "id" in input_data.columns:
            input_data = input_data.drop(columns=["id"])

        # Ensure input_data columns match model's feature columns
        input_data = input_data.reindex(columns=feature_columns, fill_value=0) 
        
 

        # 2) Clean categorical values
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm',
                            'cad', 'appet', 'pe', 'ane']
        for c in categorical_cols:
            if c in input_data.columns:
                input_data[c] = input_data[c].astype(str).str.strip().str.lower()
            else:
                input_data[c] = np.nan

        # 3) Convert numeric features
        numeric_features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu',
                            'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        for f in numeric_features:
            if f in input_data.columns:
                input_data[f] = pd.to_numeric(input_data[f], errors='coerce')
            else:
                input_data[f] = np.nan

        # 4) One-hot encode categorical features
        input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

        # 5)  Drop unwanted columns (like 'id' if it sneaks in)
        if 'id' in input_data.columns:
            input_data = input_data.drop(columns=['id'])

        # 6) Reorder columns
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)

        print("Input Data:\n", input_data.head())
        print("Expected Features:\n", feature_columns)

        # 7) Scale data
        scaled_data = scaler.transform(input_data)

        # 8) Predict
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)


        # 9) Interpret result
        if int(prediction[0]) == 1:
            result = "Oops, you have chronic disease 😟"
            confidence = float(probability[0][1]) * 100
            image="danger.jpg" 
        else:
            result = "Great, you don't have the disease 😃"
            confidence = float(probability[0][0]) * 100
            image = "happy.jpg"

        
       

        return render_template('result.html',
                               prediction=result,
                               confidence=round(confidence, 2))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('result.html',
                               prediction=f"Error: {str(e)}",
                               confidence=0)


if __name__ == '__main__':
    # Run Flask
    app.run(debug=True, host='0.0.0.0', port=5000)
