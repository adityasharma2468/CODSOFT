import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Dataset Load

df = pd.read_csv(
    r"C:\Users\aditya\Desktop\Internship Project\MOVIE RATING PREDICTION WITH PYTHON\IMDb Movies India.csv",
    encoding="latin1"
)

print(df.head())
print(df.shape)

#Dataset Information

print(df.info())

# Data Cleaning

# Duplicate Rows Remove
df.drop_duplicates(inplace=True)

# Missing Values Check 
print(df.isnull().sum())

# Missing Rows Remove
df = df.dropna(subset=[
    "Rating",
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3",
    "Duration",
    "Votes",
    "Year"
])

# Index Reset 
df.reset_index(drop=True, inplace=True)

# Dataset Statistics

print(df.describe())

# Remove missing values.
df = df.dropna()

# Convert Votes
df["Votes"] = df["Votes"].str.replace(",", "")
df["Votes"] = df["Votes"].astype(int)

# Convert Duration
df["Duration"] = df["Duration"].str.replace(" min", "")
df["Duration"] = df["Duration"].astype(int)

# Convert Year
df["Year"] = df["Year"].str.extract("(\d{4})")
df["Year"] = df["Year"].astype(int)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Genre"] = le.fit_transform(df["Genre"])
df["Director"] = le.fit_transform(df["Director"])
df["Actor 1"] = le.fit_transform(df["Actor 1"])
df["Actor 2"] = le.fit_transform(df["Actor 2"])
df["Actor 3"] = le.fit_transform(df["Actor 3"])

# Features
X = df[["Genre","Director","Actor 1","Actor 2","Actor 3","Votes","Duration","Year"]]
y = df["Rating"]

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train,y_train)

prediction = model.predict(X_test)

# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
print("MAE :", mean_absolute_error(y_test,prediction))
print("RMSE :", np.sqrt(mean_squared_error(y_test,prediction)))
print("R2 Score :", r2_score(y_test,prediction))


# 1. Distribution Of Movie Rating (Visualization)

plt.figure(figsize=(8,5))
sns.histplot(df["Rating"], bins=20, color="skyblue")
plt.title("Distribution of Movie Ratings")
plt.show()

# 2. Top 10 Genres (Visualization)

plt.figure(figsize=(10,5))
df["Genre"].value_counts().head(10).plot(kind="bar", color="#98AF88")
plt.title("Top 10 Movie Genres")
plt.xticks(rotation=45)
plt.show()

# 3. Top Directors (Visualization)

plt.figure(figsize=(10,5))
df["Director"].value_counts().head(10).plot(kind="bar", color="#BC1880")
plt.title("Top 10 Directors")
plt.xticks(rotation=45)
plt.show()

# 4. Actual vs Predicted Graph (Visualization)

plt.figure(figsize=(8,6))
plt.scatter(y_test,prediction,color="#FF9C04")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Rating")
plt.show()

# 5. Rating vs Votes (Visualization)

plt.scatter(df["Votes"], df["Rating"])
plt.xlabel("Votes")
plt.ylabel("Rating")
plt.title("Votes vs Rating")
plt.show()

