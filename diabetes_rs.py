import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report


# import library to summarize statistic
# from ydata_profiling import ProfileReport

# read data from comma-separated values files
data = pd.read_csv("diabetes.csv")

# profile = ProfileReport(data, title = "Diabetes Report", explorative= True)
# profile.to_file("diabetes.html")
# # Print first 10 rows
# print(data.head(10))
#
# # Print out the basic stats and information of feature and outcome
# print(data.info())
# print(data.describe())
#
# # Print out the correlation
# print(data.corr())

target = "Outcome"
# Drop the Outcome column
x = data.drop(target, axis = 1)

# Obtain Outcome column
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Print out number of rows
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Preprocessing data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Fit support vector machine model
model = SVC()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
# for i, j in zip(y_predict, y_test):
#     print(f"Prediction: {i}. Actual values: {j}")


# print(f"Acc: {accuracy_score(y_test, y_predict)}")
# print(f"Precision: {precision_score(y_test, y_predict)}")
# print(f"Recall: {recall_score(y_test, y_predict)}")
# print(f"F1 score: {f1_score(y_test, y_predict)}")

print(f"Acc: {classification_report(y_test, y_predict)}")




