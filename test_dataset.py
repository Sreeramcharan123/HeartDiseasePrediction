import pandas as pd
data = pd.read_csv("dataset/heart.csv")
print("First 5 rows:")
print(data.head())
print("\nShape:", data.shape)
print("\nColumns:")
print(data.columns)