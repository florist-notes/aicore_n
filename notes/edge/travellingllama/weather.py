import requests
import json

# Your API key from OpenWeatherMap
api_key = "your_api_key"

# Location for which you want to fetch the weather data (e.g., Golaghat, India)
location = "Golaghat,IN"

# URL for OpenWeatherMap API (Current Weather Data)
url = f"https://api.openweathermap.org/data/2.5/weather?q=Golaghat&units=metric&appid=99b9922597ba486ef77f40305e0beab6"  # Metric for Celsius

# Send a GET request to fetch data
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    weather_data = response.json()

    # Print weather details
    print("City:", weather_data['name'])
    print("Temperature (C):", weather_data['main']['temp'])
    print("Weather:", weather_data['weather'][0]['description'])
    print("Humidity:", weather_data['main']['humidity'], "%")
    print("Wind Speed:", weather_data['wind']['speed'], "m/s")

    # Save the data as a JSON file
    with open('realtime_weather.json', 'w') as json_file:
        json.dump(weather_data, json_file, indent=4)

    print("Weather data saved to 'realtime_weather.json'")
else:
    print("Error fetching data. Status Code:", response.status_code)
