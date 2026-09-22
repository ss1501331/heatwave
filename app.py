from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model/heatwave_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [[
        data["max_temp"],
        data["min_temp"],
        data["humidity"],
        data["wind_speed"],
        data["previous_temp"]
    ]]
    prediction = model.predict(features)[0]
    return jsonify({"prediction": "Heatwave" if prediction == 1 else "No Heatwave"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
