import pandas as pd
import numpy as np
import folium
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from streamlit_folium import folium_static
import pickle
import base64

#from geopy.geocoders import Nominatim

# Load the restaurant data from the CSV file
data = pd.read_csv('KNNImputed_ne3_cleaned_fixed_distances.csv')

data = data.rename(columns={"name": "Restaurant Name", "Type of Restau": "Type of Restaurant", "city": "City"})

cities = ["Quezon City", "Makati", "Manila", "Taguig", "Mandaluyong", "Pasig", "Paranaque", "Muntinlupa", "San Juan", "Las Pinas", "Pasay", "Marikina", "Caloocan",
          "Malabon", "Valenzuela", "Navotas"]

data = data.rename(columns = {f"distance to {city}": f"Distance to {city}" for city in cities})
     
cuisines = ["Filipino", "American", "Chinese", "Japanese", "Fusion", "Fast Food", "Cafe", "Italian", "Desserts", "Mediterranean/Persian/Greek/Indian", "Korean",
            "European", "Beverage", "Mexican", "Bar", "Spanish", "Thai",  "Vietnamese", "Cuban"]

types_of_restau = ["Ethnic Cuisine", "Specialty", "Casual Dining", "Fast Food", "Beverage",  "Entertainment"]

data['City'] = pd.Categorical(data['City'], categories=cities, ordered=True)
data['Cuisine'] = pd.Categorical(data['Cuisine'], categories=cuisines, ordered=True)
data['Type of Restaurant'] = pd.Categorical(data['Type of Restaurant'], categories=types_of_restau, ordered=True)

data = data.sort_values(['Cuisine', 'Type of Restaurant', 'City']).reset_index()

with open('city_boundaries.pkl', 'rb') as handle:
    city_boundaries = pickle.load(handle)


# Create an instance of the OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the categorical columns
categorical_cols = ['Cuisine', 'Type of Restaurant', 'City']

encoded_cols = encoder.fit_transform(data[categorical_cols]).toarray()

# Normalize the feature values
scaler = MaxAbsScaler()

dist_cols =  []

for city in cities:
    dist_cols.append(f"Distance to {city}")

input_data = pd.concat([data[dist_cols], pd.DataFrame(encoded_cols)], axis=1)
input_data.columns = input_data.columns.astype(str)
normalized_cols = scaler.fit_transform(input_data)

features = normalized_cols

def get_recommendations(cuisine, restau_type, city, selected_price_buckets, num_recommendations):

    max_review_count = 5
    min_rating = 4
    price_bucket_mask = data['Price Bucket'].isin(selected_price_buckets)
    user_features = encoder.transform([[cuisine, restau_type, city]]).toarray() 
    user_features = np.insert(user_features, 0, 0, axis=1)

    # Compute the cosine similarity matrix
    
    cols_mask  = [True if (i == cities.index(city)) or (i > 15) else False for i in range((features.shape[1]))]

    similarity_matrix = cosine_similarity(features[:, cols_mask], user_features)
    data['Similarity Score']= similarity_matrix[:, 0]
    
    mask = (data['Cuisine'] == cuisine) & (data['Type of Restaurant'] == restau_type) & (data['City'] == city) & \
           (data['Price Bucket'].isin(selected_price_buckets))  & (data['Rating'] >= min_rating) & (data['Review Count'] <= max_review_count)
           

    exact_matches = data[mask]
    if len(exact_matches) > 0:
        # Calculate the average similarity scores for each similar restaurant
       
        exact_indices = exact_matches.index.tolist()

        # Sort exact matches by low Review Count but high Rating
        sorted_exact_indices = sorted(range(len(exact_indices)), key=lambda x: (exact_matches.iloc[x]['Review Count'], -exact_matches.iloc[x]['Rating']))

        # Get the top recommendations from exact matches
        top_exact_indices = [exact_indices[i] for i in sorted_exact_indices[:num_recommendations]]


        if len(exact_matches) >= num_recommendations:
            # Return the recommended restaurants from exact matches as a DataFrame
            return data.loc[top_exact_indices, ['Restaurant Name', 'Cuisine', 'Type of Restaurant', 'City', 'Review Count', 'Rating', 'Latitude', 'Longitude', 'Price Bucket', 'Reviews', 'Similarity Score', f"Distance to {city}"]]
        else:
            remaining_recommendations = num_recommendations - len(exact_matches)
 
            # Get all restaurants matching the price bucket options selected except exact matches

            remaining_indices =  data[price_bucket_mask].query("(Rating >= @min_rating) & (`Review Count` <= @max_review_count)").index.difference(exact_indices)
           
            partial_matches = data[data.index.isin(remaining_indices)]
        
            remaining_similarity_scores = similarity_matrix[remaining_indices, 0]
        
            # Sort remaining restaurants by Cuisine, Type of Restaurant, Review Count, Rating, and city
            sorted_remaining_indices = sorted(range(len(remaining_indices)), key=lambda x: (-remaining_similarity_scores[x], partial_matches.iloc[x]['Review Count'], -partial_matches.iloc[x]['Rating']))

            # Get the top recommendations from remaining restaurants
            top_remaining_indices = [remaining_indices[i] for i in sorted_remaining_indices[:remaining_recommendations]]

            # Combine recommendations from exact matches and remaining restaurants
            recommended_indices = top_exact_indices + top_remaining_indices

            # Return the recommended restaurants as a DataFrame
            return data.loc[recommended_indices, ['Restaurant Name', 'Cuisine', 'Type of Restaurant', 'City', 'Review Count', 'Rating', 'Latitude', 'Longitude', 'Price Bucket', 'Reviews', 'Similarity Score', f"Distance to {city}"]]
    else:
        # No exact matches, fetch all restaurants
        all_similarity_scores = similarity_matrix[data.index, 0]
        
        sorted_indices = sorted(range(len(data)), key=lambda x: (-all_similarity_scores[x], data.iloc[x]['Review Count'], -data.iloc[x]['Rating']))

        # Get the top N recommendations
        
        top_indices = [index for index in sorted_indices if (data.loc[index, 'Price Bucket'] in selected_price_buckets) \
                        and (data.loc[index, 'Rating'] >= min_rating) and (data.loc[index, 'Review Count'] <= max_review_count)][:num_recommendations]

        # Return the recommended restaurants as a DataFrame
        return data.loc[top_indices, ['Restaurant Name', 'Cuisine', 'Type of Restaurant', 'City', 'Review Count', 'Rating', 'Latitude', 'Longitude', 'Price Bucket', 'Reviews', 'Similarity Score', f"Distance to {city}"]]

def format_distance(distance):

    if (distance < 1000):
        return f"{distance:.2f} m"
    else:
        return f"{distance/1000:.2f} km"
    
def display_map(recommendations, city):
    #center of Metro Manila
    latitude = 14.6091
    longitude = 121.0223

    #create folium map centered on Metro Manila
    map = folium.Map(location=[latitude, longitude], zoom_start=12)

    # for address reverse lookup
    # geolocator = Nominatim(user_agent="MM_Restau_Recomm_System")  
    
    #add restaurant markers 
    for i, row in recommendations.iterrows():

        # reverse lookup restaurant address based on longitude and latitude
        
        #disabling for now, a bit slow.
        #restau_lat = row['Latitude']
        #restau_lon = row['Longitude']        
        #address = geolocator.reverse(f"{restau_lat}, {restau_lon}").address
    
        distance_to_city = recommendations[f"Distance to {city}"].loc[i]
        city_popup = f"<br>City: {recommendations['City'].loc[i]}" if (distance_to_city > 0) else ""
        distance_popup =  f"<br>Distance to {city}: {format_distance(distance_to_city)} (Straight-line distance to the closest border of the city)" if (distance_to_city > 0) else ""

        # create popup text

        price_bucket_descriptions = {'inexpensive': 'below ₱250', 'moderate': '₱250 - ₱450',
                                     'expensive': '₱450 - ₱650', 'very expensive': 'above ₱650' }
        popup = folium.Popup(f"Restaurant Name: {row['Restaurant Name']}<br>Cuisine: {row['Cuisine']}<br>Type: {row['Type of Restaurant']}<br>Rating: {row['Rating']}\
                             <br>Price Bucket: {row['Price Bucket']} (cost per person: {price_bucket_descriptions[row['Price Bucket']]})" + city_popup + distance_popup, max_width=500)
                             #<br>Address: {address}", max_width=500)
        
        # add restaurant name as label for markers
        icon = folium.DivIcon(html=f'<div><strong>{row["Restaurant Name"]}</strong></div>')
        # set marker based on lat/lon location and add popup
        label_marker = folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            icon=icon
        )

        
        marker = folium.Marker(location=[row['Latitude'], row['Longitude']],
                               popup=popup)
        
        label_marker.add_to(map)
        marker.add_to(map)

        

    # Convert the latitude and longitude columns to numeric values
    recommendations['Latitude_num'] = pd.to_numeric(recommendations['Latitude'])
    recommendations['Longitude_num'] = pd.to_numeric(recommendations['Longitude'])

    
    # Get the bounds of the markers
    bounds = [
        [recommendations['Latitude_num'].min(), recommendations['Longitude_num'].min()], # southwest corner
        [recommendations['Latitude_num'].max(), recommendations['Longitude_num'].max()]  # northeast corner
    ]


    # Add margin to the bounds
    lat_margin = 0.01 # degrees
    lon_margin = 0.005 # degrees
    bounds[0][0] -= lat_margin
    bounds[0][1] -= lon_margin
    bounds[1][0] += lat_margin
    bounds[1][1] += lon_margin

    # Fit the map to the bounds
    map.fit_bounds(bounds)

    boundaries = city_boundaries[city]['coords']
        
    for boundary in boundaries:
        folium.Polygon(locations=boundary, color='blue', opacity = 0.5,fill_color = 'blue', fill_opacity=0.01, weight = 5).add_to(map)

    #display map
    folium_static(map)

# Define a formatting function that rounds the value to 2 decimal places
def format_1_decimal_place(value):
    return f"{value:.1f}"

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        body {
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    add_bg_from_local('background.jpg')
    # Streamlit app
    st.title('MM Foodies')
    
    st.markdown('<style>#placeholder{display:block;}</style>', unsafe_allow_html=True)
    st.markdown('<div id="placeholder">Looking for the perfect restaurant? Our recommendation engine has got you covered! Simply input your <em>city</em>, preferred <em>cuisine</em>, <em/>type of restaurant</em> and <em>price range</em> and let us do the rest. Our engine will <strong>prioritize</strong> <em>less popular</em> but <em>highly rated</em> restaurants and even recommend nearby options if there are less exact matches than you wanted. So sit back, relax and let us help you find your next favorite dining spot!</div>', unsafe_allow_html=True)
    
    sort_options = ['Default', 'Cuisine', 'Type of Restaurant', 'City', 'Rating']

    with st.sidebar.form(key='my_form'):
        
        st.title('What are you in the mood for?')

        # Dropdown menus for cuisine, restaurant type, and city
        # Create two columns
        cuisine = st.selectbox('Cuisine', cuisines)
        col1a, col2a= st.columns(2)
        restau_type = col1a.selectbox('Type of Restaurant', types_of_restau)
        city = col2a.selectbox('City', cities)

        # Define the price bucket options
        st.write('Price Bucket Filter')
        options = ['inexpensive', 'moderate', 'expensive', 'very expensive']

        # Create a dictionary to store the checkbox values
        price_buckets = {}

        ## Create two columns
        col1b, col2b = st.columns(2)

        # Create a checkbox for each option
        for i, option in enumerate(options):
            # Display the first two options in the first column
            if i < 2:
                price_buckets[option] = col1b.checkbox(option, value = True)
            # Display the remaining options in the second column
            else:
                price_buckets[option] = col2b.checkbox(option, value = True)
            # Slider for the number of recommendations
            
        num_recommendations = st.slider('Number of Recommendations', min_value=1, max_value=20, value=5)

        # Button to trigger the recommendations
        submit_button = st.form_submit_button('Get Recommendations')

    # Display recommendations    when the button is clicked
    if submit_button:
        
        selected_price_buckets = [key for key,value in price_buckets.items() if value == True]

        st.markdown('<style>#placeholder{display:none;}</style>', unsafe_allow_html=True)
        
        if len(selected_price_buckets) == 0:
            st.error("Please check at least one price bucket.")
            
        else:
            recommendations = get_recommendations(cuisine, restau_type, city, selected_price_buckets, num_recommendations)
            recommendations_for_table = recommendations.drop(['Latitude', 'Longitude', 'Reviews', 'Review Count', 'Similarity Score', f'Distance to {city}'], axis = 1) # f"Distance to {city}" ], axis = 1)
            recommendations_for_map = recommendations.drop(['Reviews'], axis = 1)
            
            # CSS to inject contained in a string
            hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

            # Inject CSS with Markdown
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            
            #AgGrid(recommendations_for_table, fit_columns_on_grid_load=True)
            
            # Apply the formatting function Rating
            formatted_data = recommendations_for_table.style.format({'Rating': format_1_decimal_place})
            
            # Display the formatted table and map in Streamlit
    
            st.table(formatted_data)
            display_map(recommendations_for_map, city)

if __name__ == "__main__":
    main()
