# Extended dataset: ev_market_data.csv
# This dataset contains a detailed EV market dataset with additional variables for analysis

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 5000  # Increased dataset size

# Define city categories
cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kolkata"]
vehicle_types = ["Two-Wheeler", "Three-Wheeler", "Four-Wheeler", "Fleet"]

# Create dataset
df = pd.DataFrame({
    'City': np.random.choice(cities, num_samples),  # Random city allocation
    'Vehicle_Type': np.random.choice(vehicle_types, num_samples),  # Random vehicle type
    'Income': np.random.randint(200000, 3000000, num_samples),  # Annual Income (INR)
    'Daily_Travel_km': np.random.randint(5, 200, num_samples),  # Distance traveled per day
    'Age': np.random.randint(18, 70, num_samples),  # Age of potential EV buyers
    'City_EV_Adoption_Rate': np.random.uniform(0.05, 0.95, num_samples),  # % of EVs in city
    'Charging_Accessibility': np.random.randint(1, 5, num_samples),  # Rating (1-5) for charging infra
    'Fuel_Cost_Savings': np.random.randint(5000, 50000, num_samples),  # Expected savings using EVs
    'Government_Subsidy': np.random.choice([50000, 100000, 150000], num_samples),  # Subsidy amount
    'EV_Purchase_Intention': np.random.choice(["High", "Medium", "Low"], num_samples, p=[0.4, 0.4, 0.2])
})

# Save dataset
df.to_csv("ev_market_data.csv", index=False)
print("Extended EV market dataset saved as ev_market_data.csv")
