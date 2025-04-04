import joblib
import numpy as np
from flask import Flask, jsonify

app = Flask(__name__)

#POSTMAN para verificar
@app.route('/predict', methods=['GET'])
def predict():
    X_text = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    prediction = model.predict(X_text.reshape(1, -1))
    return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
    # Load the model
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080)