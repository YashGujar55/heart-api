import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("heart.csv")

# Features & target
X = df.drop("output", axis=1)
y = df["output"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained & saved!")


import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")