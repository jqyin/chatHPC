import pandas as pd
import sys

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <csv_file>")
    sys.exit(1)

# Get the CSV file path from command line arguments
csv_file = sys.argv[1]

# Read the CSV file into a DataFrame
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print("Error: CSV file not found.")
    sys.exit(1)

# Selecting only numerical columns and filtering out columns with name "Unnamed"
numerical_columns = df.select_dtypes(include=[float, int]).drop(columns=df.filter(like='Unnamed').columns)

# Dictionary to store column names and their mean values
mean_values = {}

# Calculating mean for each numerical column
for column in numerical_columns.columns:
    mean_values[column] = numerical_columns[column].mean()

# Creating a DataFrame from the dictionary
mean_df = pd.DataFrame.from_dict(mean_values, orient='index', columns=['Mean'])

print(mean_df)

values = mean_df.values.squeeze()

score = ( ((values[2]+values[5])/2.) * ((values[6]+ values[9])/2.) * (1/values[10]) )**(1.0/3.0)
print(score)
