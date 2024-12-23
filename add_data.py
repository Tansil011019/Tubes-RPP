import pandas as pd

# File Path
input_csv = "nomadlist.csv"
output_clp = "expert-system.clp"

# Read the CSV file
df = pd.read_csv(input_csv)

# Replace null values with 0
df.fillna(0, inplace=True)

# List of columns to normalize (numerical columns)
columns_to_normalize = [
    "Total score", "Cost", "Quality of life score", "Internet",
    "Safety", "Fun", "Walkability", "Nightlife", "Friendly to foreigners",
    "Freedom of speech", "English speaking", "Food safety", "Places to work from"
]

# Apply normalization to scale data from 1 to 10
for column in columns_to_normalize:
    df[column] = 1 + 9 * (df[column] - df[column].min()) / (df[column].max() - df[column].min())

# Open the CLIPS file and write the normalized data as facts
with open(output_clp, mode='a') as clpfile:
    clpfile.write("\n; Added normalized city data\n")
    clpfile.write("(deffacts city-data\n")
    for _, row in df.iterrows():
        clpfile.write(
            f'\t(city (name "{row["city"]}") '
            f'(total-score {row["Total score"]:.2f}) '
            f'(cost {row["Cost"]:.2f}) '
            f'(quality-of-life {row["Quality of life score"]:.2f}) '
            f'(internet {row["Internet"]:.2f}) '
            f'(safety {row["Safety"]:.2f}) '
            f'(fun {row["Fun"]:.2f}) '
            f'(walkability {row["Walkability"]:.2f}) '
            f'(nightlife {row["Nightlife"]:.2f}) '
            f'(friendly-to-foreigners {row["Friendly to foreigners"]:.2f}) '
            f'(freedom-of-speech {row["Freedom of speech"]:.2f}) '
            f'(english-speaking {row["English speaking"]:.2f}) '
            f'(food-safety {row["Food safety"]:.2f}) '
            f'(places-to-work-from {row["Places to work from"]:.2f}))\n'
        )
    clpfile.write(")\n")  # Close the deffacts
print("Normalized city data appended successfully.")