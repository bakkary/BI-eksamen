import os
import pandas as pd
import streamlit as st
import pycountry_convert as pc

def DataPreparation():
    st.title('Data Preparation')
    st.write('We are working with two different datasets that show different data about pollution in different countries.')

    # Adjust these paths to match your actual data locations
    data_folder = 'DataSæt'  # Or the correct folder name
    dataset1_path = os.path.join('..', data_folder, 'global air pollution dataset.csv')
    dataset2_path = os.path.join('..', data_folder, '2017_-_Cities_Community_Wide_Emissions.csv')

    # Attempt to load the datasets
    try:
        df = pd.read_csv(dataset1_path)
        df2 = pd.read_csv(dataset2_path)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return

    with st.expander("Dataset 1 Preview"):
        st.dataframe(df.head())
    
    with st.expander("Dataset 2 Preview"):
        st.dataframe(df2.head())

    st.write('After that, we will merge the two datasets to get a better overview of the data, but before that, we will do some data cleaning and data wrangling to make the data more readable and understandable.')

    st.write('First, we will do some correction mapping for the names to make sure the name of the diffrent coutnries lines up with the other dataset.')

    correction_mapping = {
        "United States of America": "USA",
        "Viet Nam": "Vietnam",
        "Russian Federation": "Russia",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
        "Iran (Islamic Republic of)": "Iran",
        "Syrian Arab Republic": "Syria",
        "Republic of Korea": "South Korea",
        "Lao People's Democratic Republic": "Laos",
    }

    # Now, you can print this dictionary using st.write()
    st.write('We will do it like this:', correction_mapping)

    st.write('Next we correct the countries for df,df2 and merge them')


    # Apply the correction mapping
    df['Country'] = df['Country'].replace(correction_mapping).str.strip()
    df2['Country'] = df2['Country'].replace(correction_mapping).str.strip()

    # Merge the datasets
    df_merged = pd.merge(df, df2, on='Country', how='inner')

    # Rename 'City_x' to 'City'
    df_merged.rename(columns={'City_x': 'City'}, inplace=True)

    # Drop the extra 'City_y' column
    df_merged.drop(columns=['City_y'], inplace=True)
    
    # Rearrange the columns
    column_order = ['Country', 'City', 'AQI Value', 'AQI Category', 'CO AQI Value', 'CO AQI Category', 'Ozone AQI Value', 'Ozone AQI Category', 'NO2 AQI Value', 'NO2 AQI Category', 'PM2.5 AQI Value', 'PM2.5 AQI Category', 'Account number', 'Organization', 'Region', 'C40', 'Access', 'Reporting year', 'Accounting year', 'Boundary', 'Protocol', 'Protocol column', 'Gases included', 'Total emissions (metric tonnes CO2e)', 'Total Scope 1 Emissions (metric tonnes CO2e)', 'Total Scope 2 Emissions (metric tonnes CO2e)', 'Comment', 'Increase/Decrease from last year', 'Reason for increase/decrease in emissions', 'Population', 'Population year', 'GDP', 'GDP Currency', 'GDP Year', 'GDP Source', 'Average annual temperature (in Celsius)​', '​Average altitude (m)', '​Land area (in square km)', 'City Location', 'Country Location']
    # Reorder the DataFrame columns
    df_merged = df_merged[column_order]
   
    with st.expander("Merged Dataset Preview"):
        st.dataframe(df_merged.sample(10)) 


    st.write('Now we have the merged dataset, we can check the count')
    with st.expander("Merged Dataset Count"):
        df_merged_count = df_merged
        st.write(df_merged_count.count())    

     
    st.write('we can also check the null values')
    with st.expander("Merged Dataset Null Values"):
        df_merged_null = df_merged.copy()  # Create a copy of df_merged
        st.write(df_merged_null.isnull().sum())


    st.write('we got some columns that are not needed, so we will drop them, and we will drop all the null values')
    df_merged.drop(columns=['Gases included'], inplace=True)
    df_merged.drop(columns=['Protocol column'], inplace=True)
    df_merged.drop(columns=['Comment'], inplace=True)
    df_merged.drop(columns=['Total Scope 1 Emissions (metric tonnes CO2e)'], inplace=True)
    df_merged.drop(columns=['Total Scope 2 Emissions (metric tonnes CO2e)'], inplace=True)
    df_merged.drop(columns=['Account number'], inplace=True)
    df_merged.drop(columns=['Organization'], inplace=True)
    df_merged.drop(columns=['Accounting year'], inplace=True)
    df_merged.drop(columns=['Boundary'], inplace=True)
    df_merged.drop(columns=['Protocol'], inplace=True)
    df_merged.drop(columns=['Increase/Decrease from last year'], inplace=True)
    df_merged.drop(columns=['Reason for increase/decrease in emissions'], inplace=True)
    df_merged.drop(columns=['Population year'], inplace=True)
    df_merged.drop(columns=['GDP Currency'], inplace=True)
    df_merged.drop(columns=['GDP Source'], inplace=True)
    df_merged.drop(columns=['Access'], inplace=True)



    # Convert nulls/NaNs to 'False'
    df_merged['C40'] = df_merged['C40'].fillna('False')

    # Convert any cell that contains "C40" to 'True', assuming "C40" indicates a true condition
    # Adjust the condition as needed to match your data's specific representation of true
    df_merged['C40'] = df_merged['C40'].apply(lambda x: 'True' if 'C40' in str(x) else 'False')
 
    # Delete the null values from the data frame
    df_merged = df_merged.dropna()

    st.write('we can now check the null values again')
    with st.expander("Merged Dataset Null Values"):
        st.write(df_merged.isnull().sum())

    st.write('for working with maps we wanna split the location data into latitude and longitude')    
    # Extracting latitude and longitude from "City Location" and "Country Location" into new columns
    df_merged[['City Latitude', 'City Longitude']] = df_merged['City Location'].str.extract(r'\(([^,]+), ([^)]+)\)')
    df_merged[['Country Latitude', 'Country Longitude']] = df_merged['Country Location'].str.extract(r'\(([^,]+), ([^)]+)\)')

    # Convert the latitude and longitude columns from strings to floats
    df_merged['City Latitude'] = pd.to_numeric(df_merged['City Latitude'], errors='coerce')
    df_merged['City Longitude'] = pd.to_numeric(df_merged['City Longitude'], errors='coerce')
    df_merged['Country Latitude'] = pd.to_numeric(df_merged['Country Latitude'], errors='coerce')
    df_merged['Country Longitude'] = pd.to_numeric(df_merged['Country Longitude'], errors='coerce')
    
    df_merged.drop(columns=['City Location'], inplace=True)
    df_merged.drop(columns=['Country Location'], inplace=True)
    
    # Displaying the first few rows to ensure the transformation was successful
    with st.expander("Transformed Dataset Preview"):
        st.dataframe(df_merged.head())

    st.write('we also want a new column country for the merged dataset, this we can apply with the help of pycountry')
    # Applying continent mapping
    def country_to_continent(country_name):
        try:
            country_alpha2 = pc.country_name_to_country_alpha2(country_name)
            country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
            return pc.convert_continent_code_to_continent_name(country_continent_code)
        except:
            return None

    # Apply the conversion function to your DataFrame
    df_merged['Continent'] = df_merged['Country'].apply(country_to_continent)
    # Filter for other continents
    north_american_countries_df = df_merged[df_merged['Continent'] == 'North America']
    south_american_countries_df = df_merged[df_merged['Continent'] == 'South America']
    asian_countries_df = df_merged[df_merged['Continent'] == 'Asia']
    african_countries_df = df_merged[df_merged['Continent'] == 'Africa']
    oceania_countries_df = df_merged[df_merged['Continent'] == 'Oceania']
    Europe_df = df_merged[df_merged['Continent'] == 'Europe']

    st.write('we can now check the unique values of the continent')
    with st.expander("Unique Values of the Continent"):
        st.write(df_merged['Continent'].unique())


    st.write('we do have some duplicates in the dataset, so we will drop them')
    # Drop duplicates
    df_merged.drop_duplicates(inplace=True)
    st.expander('total count after duplicates removed')
    st.write(df_merged.count())    

    st.write('lastly we can do some 1 hot encoding for the C40 column')
    # Convert 'C40' from strings "True"/"False" to actual booleans
    df_merged['C40'] = df_merged['C40'].map({'True': True, 'False': False})

    # Create two new columns: 'C40_True' and 'C40_False'
    df_merged['C40_True'] = df_merged['C40'].astype(int)  # This will convert True to 1 and False to 0
    df_merged['C40_False'] = (~df_merged['C40']).astype(int)  # This inverts the boolean and then converts to 0/1
    df_merged.drop(columns=['C40'], inplace=True)
    
    with st.expander("Transformed Dataset Preview"):
        st.dataframe(df_merged.head())

    df = df_merged
    

    st.write('taking a look at the count for the data we can see that we have about 150k entires wich are alot of data')
    with st.expander("Transformed Dataset Count"):
        st.write(df.count())

    st.write('we are gonna check for copies')
    with st.expander("Transformed Dataset Duplicates"):
        st.write(df.duplicated().sum())

    st.write('we have a lot of copies so we will drop them')
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    st.write(df.duplicated().sum())
    
            
    st.write('lastly we can transform the data to a pickle file that we will be working way thoughout the set')