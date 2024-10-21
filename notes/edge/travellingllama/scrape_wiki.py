import requests
import json

# Define the page title
page_title = "Golaghat"

# Wikipedia API URL
url = f"https://en.wikipedia.org/w/api.php"

# Set parameters to get the full content
params = {
    "action": "query",
    "format": "json",
    "prop": "extracts",
    "explaintext": True,  # Get plain text without HTML formatting
    "titles": page_title
}

# Send a GET request to the API
response = requests.get(url, params=params)
data = response.json()

# Save the full-length content to a JSON file
with open('golaghat_full_wikipedia_data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("Full data saved to golaghat_full_wikipedia_data.json")
