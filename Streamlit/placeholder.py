import pandas as pd
import numpy as np
from DataLoader import load_data




def show_placeholder():
 
 df = load_data()  # Load your DataFrame

 # Assuming df is your DataFrame containing city data

 # List of known C40 member cities
 c40_member_cities = [
    'Brazil', 'Italy', 'Poland', 'France', 'USA', 'Germany',
       'Netherlands', 'Colombia', 'Indonesia', 'Finland', 'South Africa',
       'United Kingdom', 'Philippines', 'New Zealand', 'Mexico', 'Japan',
       'Turkey', 'Canada', 'Switzerland', 'Denmark', 'Australia',
       'Portugal', 'Spain', 'Ecuador', 'Argentina', 'Chile', 'Greece',
       'Norway', 'Jordan'
]

 # Filter DataFrame to include only C40 member cities
 c40_df = df[df['City'].isin(c40_member_cities)]

 # Identify cities that don't exist in the dataset
 non_existing_cities = list(set(c40_member_cities) - set(c40_df['City']))

 # Sample 50 random cities that are not part of the C40 initiative
 np.random.seed(42)  # for reproducibility
 random_non_c40_cities = np.random.choice(non_existing_cities, size=50, replace=False)

 # Create DataFrame for random non-C40 cities
 random_non_c40_df = pd.DataFrame({'City': random_non_c40_cities, 'C40_member': False})

 # Combine C40 member cities DataFrame with random non-member cities DataFrame
 df_test_df = pd.concat([c40_df, random_non_c40_df])

 # Output results
 print("Cities that are members of the C40 initiative and exist in the dataset:")
 print(c40_df)

 print("\nCities that are members of the C40 initiative but don't exist in the dataset:")
 print(non_existing_cities)

 print("\n50 random cities that are not part of the C40 initiative:")
 print(random_non_c40_df)

 print("\nDataFrame for training/testing containing both C40 member cities and 50 random non-member cities:")
 print(df_test_df)