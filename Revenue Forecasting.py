# revenue_forecasting.py
# This script estimates revenue for the EV and vehicle booking market using Fermi estimation with a dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("market_data.csv")

# Assumed parameters for EV market
ev_potential_customers = df[df['Market'] == 'EV']['Potential_Customers'].sum()
ev_conversion_rate = df[df['Market'] == 'EV']['Conversion_Rate'].mean()
ev_avg_revenue_per_unit = df[df['Market'] == 'EV']['Avg_Revenue_Per_Unit'].mean()
ev_market_growth_rate = df[df['Market'] == 'EV']['Growth_Rate'].mean()

ev_revenue = ev_potential_customers * ev_conversion_rate * ev_avg_revenue_per_unit
future_ev_revenue = ev_revenue * (1 + ev_market_growth_rate) ** 5  # 5-year projection

print(f"Estimated EV Market Revenue: ₹{ev_revenue/1e9:.2f} billion")
print(f"Projected EV Market Revenue (5 years): ₹{future_ev_revenue/1e9:.2f} billion")

# Assumed parameters for vehicle booking market
vb_potential_customers = df[df['Market'] == 'Vehicle_Booking']['Potential_Customers'].sum()
vb_conversion_rate = df[df['Market'] == 'Vehicle_Booking']['Conversion_Rate'].mean()
vb_avg_fare = df[df['Market'] == 'Vehicle_Booking']['Avg_Fare'].mean()
vb_avg_rides_per_month = df[df['Market'] == 'Vehicle_Booking']['Avg_Rides_Per_Month'].mean()
vb_market_growth_rate = df[df['Market'] == 'Vehicle_Booking']['Growth_Rate'].mean()

vb_monthly_revenue = vb_potential_customers * vb_conversion_rate * vb_avg_fare * vb_avg_rides_per_month
vb_annual_revenue = vb_monthly_revenue * 12
future_vb_annual_revenue = vb_annual_revenue * (1 + vb_market_growth_rate) ** 5  # 5-year projection

print(f"Estimated Vehicle Booking Market Monthly Revenue: ₹{vb_monthly_revenue/1e9:.2f} billion")
print(f"Estimated Vehicle Booking Market Annual Revenue: ₹{vb_annual_revenue/1e9:.2f} billion")
print(f"Projected Vehicle Booking Market Revenue (5 years): ₹{future_vb_annual_revenue/1e9:.2f} billion")

# Save revenue estimates to a CSV file
revenue_data = {
    "Market": ["EV", "Vehicle_Booking"],
    "Monthly_Revenue": [ev_revenue / 12, vb_monthly_revenue],
    "Annual_Revenue": [ev_revenue, vb_annual_revenue],
    "Projected_5_Year_Revenue": [future_ev_revenue, future_vb_annual_revenue]
}

revenue_df = pd.DataFrame(revenue_data)
revenue_df.to_csv("revenue_forecast.csv", index=False)

# Plot revenue projections
plt.figure(figsize=(10, 5))
markets = ["EV", "Vehicle Booking"]
current_revenue = [ev_revenue / 1e9, vb_annual_revenue / 1e9]
future_revenue = [future_ev_revenue / 1e9, future_vb_annual_revenue / 1e9]

x = np.arange(len(markets))
plt.bar(x - 0.2, current_revenue, width=0.4, label="Current Revenue")
plt.bar(x + 0.2, future_revenue, width=0.4, label="Projected 5-Year Revenue")

plt.xlabel("Market Segments")
plt.ylabel("Revenue (₹ Billion)")
plt.title("Current vs Projected Market Revenue")
plt.xticks(ticks=x, labels=markets)
plt.legend()
plt.savefig("revenue_projection.png")
plt.show()

print("Revenue forecasting complete. Results saved as revenue_forecast.csv")