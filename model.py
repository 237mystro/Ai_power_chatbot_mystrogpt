import requests
from datetime import datetime

# Crop database
SEASONAL_CROP_DATA = {
    'rainy': ['maize', 'groundnuts', 'cassava'],
    'dry': ['millet', 'sorghum', 'beans'],
    'harmattan': ['onions', 'carrots', 'spinach']
}

CROP_INFO = {
    'maize': {'duration_days': 90, 'price_per_kg': 250},
    'groundnuts': {'duration_days': 100, 'price_per_kg': 300},
    'cassava': {'duration_days': 270, 'price_per_kg': 150},
    'millet': {'duration_days': 100, 'price_per_kg': 220},
    'sorghum': {'duration_days': 110, 'price_per_kg': 200},
    'beans': {'duration_days': 90, 'price_per_kg': 350},
    'onions': {'duration_days': 120, 'price_per_kg': 500},
    'carrots': {'duration_days': 80, 'price_per_kg': 450},
    'spinach': {'duration_days': 60, 'price_per_kg': 300}
}

def get_weather_data(city_name, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            print("Error from weather API:", data.get("message", "Unknown error"))
            return None, None

        temperature = data['main']['temp']
        weather = data['weather'][0]['description']
        return round(temperature), weather.lower()
    except Exception as e:
        print("Exception during weather API call:", e)
        return None, None

def get_season_by_month():
    month = datetime.now().month
    if 5 <= month <= 9:
        return 'rainy'
    elif 1 <= month <= 4:
        return 'dry'
    else:
        return 'harmattan'

def recommend_crops(city):
    API_KEY = "6853988b7365e74c17453cc3b877a850"
    temperature, weather_description = get_weather_data(city, API_KEY)
    season = get_season_by_month()

    if temperature is None:
        return {"error": "Could not retrieve weather data."}

    crops = SEASONAL_CROP_DATA.get(season, [])
    crop_recommendations = []

    for crop in crops:
        info = CROP_INFO[crop]
        crop_recommendations.append({
            "crop": crop,
            "grow_duration_days": info['duration_days'],
            "price_per_kg": info['price_per_kg']
        })

    return {
        "location": city,
        "current_temperature_celsius": temperature,
        "weather": weather_description,
        "season": season,
        "recommended_crops": crop_recommendations
    }

# Run test
if __name__ == "__main__":
    city = input("Enter your city or village name: ")
    recommendations = recommend_crops(city)
    print(recommendations)
