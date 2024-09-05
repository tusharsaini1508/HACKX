from flask import Flask, jsonify, request
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and label encoder from files
def load_model_and_encoder():
    with open('model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return best_model, label_encoder

# Define the Weather class
class Weather:
    def __init__(self, temp, feelsLike, pressure, humidity, clouds, visibility, wind, rain, snow, conditionId, main, description, icon):
        self.temp = temp
        self.feelsLike = feelsLike
        self.pressure = pressure
        self.humidity = humidity
        self.clouds = clouds
        self.visibility = visibility
        self.wind = wind
        self.rain = rain
        self.snow = snow
        self.conditionId = conditionId
        self.main = main
        self.description = description
        self.icon = icon

    def to_dict(self):
        return {
            'temp': self.temp,
            'feelsLike': self.feelsLike,
            'pressure': self.pressure,
            'humidity': self.humidity,
            'clouds': self.clouds,
            'visibility': self.visibility,
            'wind_deg': self.wind['deg'],
            'wind_gust': self.wind['gust'],
            'wind_speed': self.wind['speed'],
            'rain': self.rain,
            'snow': self.snow,
            'conditionId': self.conditionId,
            'main': self.main,
            'description': self.description
        }

# Create a Weather object from a dictionary
def create_weather_from_dict(data):
    return Weather(
        temp=data['temp']['cur'],
        feelsLike=data['feelsLike']['cur'],
        pressure=data['pressure'],
        humidity=data['humidity'],
        clouds=data['clouds'],
        visibility=data['visibility'],
        wind=data['wind'],
        rain=data['rain'],
        snow=data['snow'],
        conditionId=data['conditionId'],
        main=data['main'],
        description=data['description'],
        icon=data['icon']
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    current_weather = create_weather_from_dict(data)
    current_weather_df = pd.DataFrame([current_weather.to_dict()])

    # Load trained model and encoder
    best_model, label_encoder = load_model_and_encoder()

    # Predict disaster type for current weather
    current_weather_encoded = best_model.predict(current_weather_df)
    predicted_disaster_type = label_encoder.inverse_transform(current_weather_encoded)[0]

    # Predict probabilities for current weather
    probabilities = best_model.predict_proba(current_weather_df)[0]
    disaster_probabilities = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}

    filtered_probabilities = {disaster: prob for disaster, prob in disaster_probabilities.items() if prob >= 0.12}

    result = {
        'predicted_disaster_type': predicted_disaster_type,
        'disaster_probabilities': filtered_probabilities
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
