import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from lazypredict.Supervised import LazyClassifier

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
# Drop the Outcome column, axis = 1 is column, axis = 0 is row
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
# model = SVC()
# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)

# Fit logistic regression model
# model = LogisticRegression()
# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)

# Fit random forest model
# model = RandomForestClassifier()
# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)

# params = {
#     "n_estimators" : [50, 100, 200],
#     "criterion" : ["gini", "entropy", "log_loss"],
#     "max_depth": [None, 2, 5]
# }
#
# # Find the most optimal hyper parament for a model
# GridSCVModel = GridSearchCV(
#     # What model you use
#     estimator= RandomForestClassifier(random_state=42),
#
#     # What hyperparameter you use
#     param_grid= params,
#
#     # What metric you evaluate
#     scoring= "precision",
#
#     # How many times of k fold cross validation
#     cv=6,
#     verbose= 2,
#
#     # How many threads (processors) that you want to use
#     #n_jobs= 8
# )
# GridSCVModel.fit(x_train, y_train)
# y_predict =GridSCVModel.predict(x_test)
# print(GridSCVModel.best_score_, GridSCVModel.best_params_)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)

print(x_test)
# for i, j in zip(y_predict, y_test):
#     print(f"Prediction: {i}. Actual values: {j}")

# threshold = 0.3
#
# for score in y_predict:
#     if score[1] > threshold:
#         print("Class 1")
#     else:
#         print("Class 0")

# print(f"Acc: {accuracy_score(y_test, y_predict)}")
# print(f"Precision: {precision_score(y_test, y_predict)}")
# print(f"Recall: {recall_score(y_test, y_predict)}")
# print(f"F1 score: {f1_score(y_test, y_predict)}")

# Use the classification report
# We want to focus in recall for class 1 because we want to get as many patient with potential of cancer as we can

#print(f"{classification_report(y_test, y_predict)}")




