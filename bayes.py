import pandas as pd
import numpy as np

# File Path
input_csv = "nomadlist.csv"

# Read the CSV file
df = pd.read_csv(input_csv)

# Replace null values with 0
df.fillna(0, inplace=True)

# Normalize columns to a 1-10 scale
columns_to_normalize = [
    "Total score", "Cost", "Quality of life score", "Internet",
    "Safety", "Fun", "Walkability", "Nightlife", "Friendly to foreigners",
    "Freedom of speech", "English speaking", "Food safety", "Places to work from"
]

for column in columns_to_normalize:
    df[column] = 1 + 9 * (df[column] - df[column].min()) / (df[column].max() - df[column].min())

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

# Normalize user preferences so they sum to 1
total_weight = sum(user_preferences.values())
normalized_weights = {k: v / total_weight for k, v in user_preferences.items()}

# Calculate Prior (P(H))
df["Prior"] = df["Total score"] / df["Total score"].sum()

# Calculate Likelihood (P(E|H))
likelihoods = []
for _, row in df.iterrows():
    likelihood = 1  
    for feature, weight in normalized_weights.items():
        if feature == "Cost":
            normalized_value = 10 - row[feature]
        else:
            normalized_value = row[feature]
        
        likelihood *= np.exp(weight * normalized_value)
    likelihoods.append(likelihood)

df["Likelihood"] = likelihoods

# Calculate Posterior (P(H|E))
df["Posterior"] = df["Prior"] * df["Likelihood"]

# Normalize Posterior to sum to 1 (optional for better interpretation)
df["Posterior"] /= df["Posterior"].sum()

# Sort cities by Posterior probability
df_sorted = df.sort_values(by="Posterior", ascending=False)

# Display the ranked cities
print("\nRanked Cities Based on Posterior Probability:")
print(df_sorted[["city", "Posterior"]])
