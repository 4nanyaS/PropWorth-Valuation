import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("DubaiPropertyForSale.csv")

# Beds
df['Beds'] = (
    df['Beds']
    .replace('Studio', 0)
    .astype(str)
    .str.extract(r'(\d+)')
    .fillna(0)
    .astype(int)
)

# Baths
df['Baths'] = (
    df['Baths']
    .astype(str)
    .str.extract(r'(\d+)')
    .fillna(0)
    .astype(int)
)

# Area
df['Area_sqft'] = (
    df['Area_sqft']
    .astype(str)
    .str.extract(r'(\d+\.?\d*)')
    .fillna(0)
    .astype(float)
)

# Furnishing
df['Furnishing'] = df['Furnishing'].map({
    'Unfurnished': 0,
    'Furnished': 1,
}).fillna(0)

# Price
df['Price'] = (
    df['Price']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.extract(r'(\d+\.?\d*)')
    .astype(float)
)

# Model
X = df[['Beds', 'Baths', 'Area_sqft', 'Furnishing']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("price_model.pkl", "wb") as f:
    pickle.dump(model, f)