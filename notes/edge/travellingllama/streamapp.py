from langchain_groq import ChatGroq
import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import AIMessage, HumanMessage
import geopandas as gpd
import requests
import os
import folium 
import json
from northeastern_states import northeastern_states_and_districts
from district_coordinates import district_coordinates


st.set_page_config(page_title="Travel Llama.AI", page_icon="🌍")
st.title("Wander Llama 🦙")

template = """
You are a travel assistant chatbot named Travel Llama.AI, designed to help users plan their trips and provide travel-related information. Here are some scenarios you should be able to handle:
1. Booking Flights: Assist users with booking flights to their desired destinations.
2. Booking Hotels: Help users find and book accommodations.
3. Booking Rental Cars: Facilitate the booking of rental cars.
4. Destination Information: Provide information about popular travel destinations.
5. Travel Tips: Offer practical travel tips and advice.
6. Weather Updates: Give current weather updates for specific destinations.
7. Local Attractions: Suggest local attractions and points of interest.
8. Customer Service: Address customer service inquiries and provide assistance.

Chat history:
{chat_history}

User question:
{user_question}
"""
prompt = ChatPromptTemplate.from_template(template)

def get_response(user_query, chat_history):
    llm = ChatGroq(
        groq_api_key="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",
        model="llama-3.1-70b-versatile",
        temperature=0,
        streaming=True
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "chat_history": chat_history,
        "user_question": user_query,
    })

    return response

def download_map_data(lat, lon):
    # Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Overpass query to get all data around the specified latitude and longitude for a 10 km radius
    overpass_query = f"""
    [out:json];
    (
      node(around:10000,{lat},{lon});  // All nodes
      way(around:10000,{lat},{lon});   // All ways
      relation(around:10000,{lat},{lon}); // All relations
    );
    out body;
    """
    
    response = requests.get(overpass_url, params={'data': overpass_query})
    
    if response.status_code == 200:
        # Convert the response to GeoJSON format
        data = response.json()
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        point_names = []  # List to store names of all points

        # Convert Overpass API data to GeoJSON format
        for element in data['elements']:
            feature = {
                "type": "Feature",
                "properties": {
                    "tags": element.get('tags', {}),
                    "name": element.get('tags', {}).get('name', 'Unnamed')  # Get the name if available
                },
            }
            if element['type'] == 'node':
                # Ensure 'lat' and 'lon' exist for nodes
                if 'lat' in element and 'lon' in element:
                    feature["geometry"] = {
                        "type": "Point",
                        "coordinates": [element['lon'], element['lat']]
                    }
                    geojson['features'].append(feature)
                    point_names.append(feature["properties"]["name"])  # Save the name of the node
            elif element['type'] == 'way' and 'geometry' in element:
                # Ensure 'geometry' exists for ways
                coordinates = [(node['lon'], node['lat']) for node in element['geometry'] if 'lat' in node and 'lon' in node]
                if coordinates:
                    feature["geometry"] = {
                        "type": "LineString",
                        "coordinates": coordinates
                    }
                    geojson['features'].append(feature)
            elif element['type'] == 'relation' and 'members' in element:
                # For relations, handle members
                coordinates = []
                for member in element['members']:
                    if member['type'] == 'node' and 'lat' in member and 'lon' in member:
                        coordinates.append((member['lon'], member['lat']))
                if coordinates:
                    feature["geometry"] = {
                        "type": "Polygon",  # or "MultiPolygon" based on your needs
                        "coordinates": [coordinates]  # Wrap coordinates in another list
                    }
                    geojson['features'].append(feature)
        
        # Save the GeoJSON to a file
        with open("map_data.geojson", "w") as f:
            f.write(json.dumps(geojson, indent=4))

        # Save the names of the points to a text file
        with open("point_names.txt", "w") as f:
            for name in point_names:
                f.write(name + "\n")

        return "map_data2.geojson"
    else:
        return None


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Travel Llama.AI. How can I help you?"),
    ]

# Use the northeastern_states_and_districts dictionary
# Example usage in your Streamlit app
# Create two columns for the state and district selection
col1, col2 = st.columns(2)

# Use the northeastern_states_and_districts dictionary
with col1:
    selected_state = st.selectbox("Select State", list(northeastern_states_and_districts.keys()))

with col2:
    selected_district = st.selectbox("Select District", northeastern_states_and_districts[selected_state])

# Fetch latitude and longitude for the selected district
if selected_district in district_coordinates:
    lat, lon = district_coordinates[selected_district]

    # Update the user location with the fetched coordinates
    st.session_state.user_location = (lat, lon)

# Display the selected district's coordinates
st.write(f"Coordinates for {selected_district}: Latitude {lat}, Longitude {lon}")

# Initialize user location if not already set
if "user_location" not in st.session_state:
    st.session_state.user_location = (26.0, 93.0)  # Default location (latitude, longitude)

# Create a single row for latitude and longitude inputs
col_lat, col_lon = st.columns(2)

with col_lat:
    latitude = st.number_input("Latitude", value=st.session_state.user_location[0], format="%.6f", step=0.000001)

with col_lon:
    longitude = st.number_input("Longitude", value=st.session_state.user_location[1], format="%.6f", step=0.000001)

# Update user location when latitude or longitude changes
if st.button("Update Location"):
    st.session_state.user_location = (latitude, longitude)

# Display the updated map with the new user location and hide the Mapbox logo
st.map(data={"lat": [st.session_state.user_location[0]], "lon": [st.session_state.user_location[1]]}, zoom=6)

# Download map data for the selected state
if st.button("Download Map Data"):
    file_name = download_map_data(latitude, longitude)
    if file_name:
        st.success(f"Map data downloaded successfully as {file_name}")
        with open(file_name, "rb") as f:
            st.download_button("Download Map Data", f, file_name)
    else:
        st.error("Failed to download map data.")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input for chat
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    response = get_response(user_query, st.session_state.chat_history)

    response = response.replace("AI response:", "").replace("chat response:", "").replace("bot response:", "").strip()

    with st.chat_message("AI"):
        st.write(response)

    st.session_state.chat_history.append(AIMessage(content=response))
