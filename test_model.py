import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

test_data = [[3, 75, 76, 29, 3, 26.6, 0.351, 31]]

prediction = model.predict(test_data)

print(prediction[0])