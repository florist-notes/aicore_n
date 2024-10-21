import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run headlessly
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

# Specify the path to your ChromeDriver
service = Service(executable_path='/usr/local/bin/chromedriver')  # Update the path if necessary

# Initialize the Chrome WebDriver with the service
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL of the TripAdvisor page you want to scrape
url = "https://www.tripadvisor.in/Attractions-g12379827-Activities-oa0-Golaghat_District_Assam.html"

# Send a GET request to fetch the raw HTML content
driver.get(url)

# Wait for the attractions to load
try:
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'listing_title'))
    )
except Exception as e:
    print("Error while waiting for attractions to load:", e)

# Get page source after JavaScript has executed
html_content = driver.page_source

# Initialize lists to store the data
attraction_names = []
attraction_ratings = []
attraction_links = []

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find the section containing the attraction information
attractions = soup.find_all('div', class_='listing_title')

# Print the number of attractions found
print("Attractions Found:", len(attractions))

# Iterate over each attraction entry
for attraction in attractions:
    # Extract the name of the attraction
    name = attraction.get_text(strip=True)
    attraction_names.append(name)

    # Extract the URL for the attraction's page (add the base URL if necessary)
    link = "https://www.tripadvisor.in" + attraction.a['href']
    attraction_links.append(link)

    # Try to find the rating, if it exists (sometimes ratings might not be available)
    rating_tag = attraction.find_previous('span', class_='ui_bubble_rating')
    if rating_tag:
        # Ensure that the rating attribute exists
        if 'alt' in rating_tag.attrs:
            rating = rating_tag['alt'].split()[0]  # Extract the rating value (e.g., "4.5 of 5 bubbles")
            attraction_ratings.append(rating)
        else:
            attraction_ratings.append('No rating')
    else:
        attraction_ratings.append('No rating')

# Print the scraped data
for name, rating, link in zip(attraction_names, attraction_ratings, attraction_links):
    print(f"Attraction: {name}, Rating: {rating}, Link: {link}")

# Check if any data has been collected
if attraction_names:
    # Save the data into a Pandas DataFrame and export to CSV (optional)
    df = pd.DataFrame({
        'Attraction': attraction_names,
        'Rating': attraction_ratings,
        'Link': attraction_links
    })

    # Save the DataFrame to a CSV file
    df.to_csv('tripadvisor_golaghat_attractions.csv', index=False)
    print("\nData saved to 'tripadvisor_golaghat_attractions.csv'")
else:
    print("\nNo data found to save.")

# Quit the driver
driver.quit()
