import pandas as pd
import numpy as np

# File Path
input_csv = "nomadlist.csv"

# Read the CSV file
df = pd.read_csv(input_csv)

# Replace null values with 0
df.fillna(0, inplace=True)

# Normalize columns to a 0–1 scale
columns_to_normalize = [
    "Total score", "Cost", "Quality of life score", "Internet",
    "Safety", "Fun", "Walkability", "Nightlife", "Friendly to foreigners",
    "Freedom of speech", "English speaking", "Food safety", "Places to work from"
]

for column in columns_to_normalize:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

print(df.head(10))

# Ask user for input preferences
user_preferences = {}
attributes = [
    "Cost", "Safety", "Internet", "Fun", "Quality of life score", 
    "Nightlife", "English speaking", "Food safety", 
    "Freedom of speech", "Places to work from"
]

print("Enter your preferences on a scale of 1 to 10 for each attribute:")
for attribute in attributes:
    weight = float(input(f"Weight for {attribute}: "))
    user_preferences[attribute] = weight

# Calculate Prior (P(H))
df["Prior"] = df["Total score"] / df["Total score"].sum()

# Calculate Likelihood (P(E|H)) using distance-based scoring
def calculate_likelihood(row):
    total_distance = 0
    for feature, user_weight in user_preferences.items():
        city_value = row[feature]
        if feature == "Cost":  # Lower cost is better
            city_value = 1 - city_value  # Invert cost
        # Distance between user preference and city value
        distance = abs(user_weight / 10 - city_value)
        total_distance += distance ** 2  # Squared distance

    # Convert distance to likelihood (smaller distance = higher likelihood)
    return np.exp(-np.sqrt(total_distance))

df["Likelihood"] = df.apply(calculate_likelihood, axis=1)

# Calculate Posterior (P(H|E))
df["Posterior"] = df["Prior"] * df["Likelihood"]

# Normalize Posterior to sum to 1
df["Posterior"] /= df["Posterior"].sum()

# Sort cities by Posterior probability
df_sorted = df.sort_values(by="Posterior", ascending=False)

# Display the ranked cities
print("\nRanked Cities Based on Posterior Probability:")
print(df_sorted[["city", "Posterior"]])
