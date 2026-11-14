import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

test_data = [[3, 75, 76, 29, 3, 26.6, 0.351, 10]]

prediction = model.predict(test_data)

if prediction == 1:
    print(f"This patient was potentially diagnosed with cancer!")
else:
    print(f"This patient was not diagnosed with cancer!")