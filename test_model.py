import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# The model was trained on scaled data, so new data must be scaled the same way.
# Rebuild the scaler exactly like diabetes_rs.py does (same data, same split).
data = pd.read_csv("diabetes.csv")
x = data.drop("Outcome", axis=1)
y = data["Outcome"]
x_train, _, _, _ = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(x_train)

test_data = pd.DataFrame([[3, 75, 76, 29, 3, 26.6, 0.351, 10]], columns=x.columns)
test_data = scaler.transform(test_data)

prediction = model.predict(test_data)

if prediction[0] == 1:
    print("This patient was potentially diagnosed with diabetes!")
else:
    print("This patient was not diagnosed with diabetes!")
