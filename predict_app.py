from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Carregar models
models = {}
for name in ["logistic_regression", "svm", "decision_tree", "knn"]:
    with open(f'models/{name}_model.pkl', 'rb') as f:
        dv, scaler, model = pickle.load(f)
        models[name] = (dv, scaler, model)

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if model_name not in models:
        return jsonify({"error": "Model not found"}), 404
    
    data = request.json
    dv, scaler, model = models[model_name]
    X = dv.transform([data])
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled).tolist()
    
    return jsonify({"prediction": y_pred[0], "probability": y_prob})

if __name__ == '__main__':
    app.run(debug=True)
