# Diabetes Prediction

This project uses machine learning to guess if a patient has diabetes, based on simple health measurements like blood sugar, BMI and age.

I tried several different models on the same data and compared their results. The results are below.

## The data

The file `diabetes.csv` is the **Pima Indians Diabetes** dataset.

- 768 patients (rows)
- 8 health measurements (input columns)
- 1 answer column called `Outcome`: `1` = has diabetes, `0` = no diabetes
- 500 patients have no diabetes and 268 have diabetes, so the data is not balanced

| Column | What it means |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Blood sugar level |
| BloodPressure | Blood pressure (mm Hg) |
| SkinThickness | Skin fold thickness (mm) |
| Insulin | Insulin level |
| BMI | Body mass index |
| DiabetesPedigreeFunction | A score for diabetes in the family |
| Age | Age in years |
| Outcome | 1 = diabetes, 0 = no diabetes |

`diabetes.html` is a data report made with `ydata_profiling`. Open it in a web browser to see charts and statistics for each column.

## Files in this project

| File | What it does |
|---|---|
| `diabetes_rs.py` | Main script. Loads the data, trains a model, prints the scores and saves the model to `model.pkl` |
| `test_model.py` | Loads `model.pkl` and makes a prediction for one example patient |
| `model.pkl` | The saved, trained model (Random Forest) |
| `diabetes.csv` | The dataset |
| `diabetes.html` | Data profile report |
| `Dockerfile` | Runs `diabetes_rs.py` inside a Docker container |

## How the testing works

Every model is tested the same way:

1. Split the data: **80% for training** (614 patients) and **20% for testing** (154 patients). `random_state=42` is used so the split is the same every time.
2. Scale the numbers with `StandardScaler` so all columns are on a similar scale.
3. Train the model on the training part.
4. Ask the model to predict the test part, then compare its guesses with the real answers.

### What the scores mean

- **Accuracy**: how many guesses were right overall.
- **Precision**: when the model says "diabetes", how often it is right.
- **Recall**: out of all the real diabetes patients, how many the model found.
- **F1 score**: one number that balances precision and recall.

For a health check, **recall is very important**: we don't want to miss sick patients.

## Results

### Test 1: Three main models

These are the models in `diabetes_rs.py` with default settings. Precision, recall and F1 are for the "has diabetes" group.

| Model | Accuracy | Precision | Recall | F1 score |
|---|---|---|---|---|
| Support Vector Machine (SVC) | 0.734 | 0.646 | 0.564 | 0.602 |
| **Logistic Regression** | **0.753** | 0.649 | **0.673** | **0.661** |
| Random Forest | 0.721 | 0.607 | 0.618 | 0.613 |

**Result:** Logistic Regression did best here. It had the highest accuracy and found the most diabetes patients (highest recall).

Note: the Random Forest in `diabetes_rs.py` does not set a `random_state`, so its score changes a little each run. Over 10 runs, its accuracy was between **0.71 and 0.77**. The row above uses `random_state=42`.

### Test 2: Random Forest with tuned settings (GridSearchCV)

`GridSearchCV` tries many settings and keeps the best one. It tested 27 combinations with 6-fold cross validation and picked the settings with the best **precision**.

- Settings tested:
  - `n_estimators`: 50, 100, 200
  - `criterion`: gini, entropy, log_loss
  - `max_depth`: None, 2, 5
- Best settings found: `criterion="entropy"`, `max_depth=2`, `n_estimators=100`

| Model | Accuracy | Precision | Recall | F1 score |
|---|---|---|---|---|
| Random Forest (tuned) | **0.779** | **0.800** | 0.509 | 0.622 |

**Result:** Tuning gave the best accuracy and precision so far. But recall dropped to about 0.51, so it **misses about half** of the diabetes patients. That happened because the search was told to optimize precision. For a medical test, it would be better to set `scoring="recall"` or `scoring="f1"`.

### Test 3: Try many models at once (LazyClassifier)

`LazyClassifier` from `lazypredict` trains 27 different models automatically and ranks them. These are the top 10:

| Rank | Model | Accuracy | Balanced Accuracy | ROC AUC | F1 score* |
|---|---|---|---|---|---|
| 1 | AdaBoost | 0.78 | 0.77 | 0.77 | 0.78 |
| 2 | Quadratic Discriminant Analysis | 0.78 | 0.76 | 0.76 | 0.78 |
| 3 | Gaussian Naive Bayes | 0.77 | 0.75 | 0.75 | 0.77 |
| 4 | Ridge Classifier CV | 0.77 | 0.74 | 0.74 | 0.77 |
| 5 | Decision Tree | 0.75 | 0.74 | 0.74 | 0.75 |
| 6 | Ridge Classifier | 0.76 | 0.74 | 0.74 | 0.76 |
| 7 | Linear Discriminant Analysis | 0.76 | 0.74 | 0.74 | 0.76 |
| 8 | Logistic Regression | 0.75 | 0.73 | 0.73 | 0.75 |
| 9 | Linear SVC | 0.75 | 0.73 | 0.73 | 0.75 |
| 10 | Calibrated Classifier CV | 0.75 | 0.73 | 0.73 | 0.75 |

\* LazyClassifier's F1 is a weighted average over both groups, so it is not the same as the F1 in Tests 1 and 2.

**Result:** AdaBoost was at the top. Most models scored between 0.73 and 0.78 accuracy, which is close. This tells us the dataset is probably the limit, not the choice of model.

### Summary

| Goal | Best model |
|---|---|
| Highest accuracy | Tuned Random Forest (0.779), AdaBoost (0.78) |
| Find the most diabetes patients (recall) | Logistic Regression (0.673) |
| Best balance (F1) | Logistic Regression (0.661) |

All models land at about **73–78% accuracy**. Simple models like Logistic Regression do just as well as more complex ones on this small dataset.

## How to run

### On your computer

```bash
pip install pandas scikit-learn
python diabetes_rs.py     # train the model, print scores, save model.pkl
python test_model.py      # load model.pkl and predict one patient
```

To try a different model, open `diabetes_rs.py` and uncomment the model you want (SVC, Logistic Regression, GridSearchCV or LazyClassifier). LazyClassifier also needs `pip install lazypredict`.

### With Docker

```bash
docker build -t diabetes-app .
docker run diabetes-app
```

## Known issues

- `test_model.py` gives the model **raw numbers**, but the model was trained on **scaled numbers** (from `StandardScaler`). This can give wrong answers. For example, the sample patient in the file (glucose 75, BMI 26.6) is predicted as "diabetes" with raw numbers, but "no diabetes" when the numbers are scaled correctly. To fix it, save the scaler together with the model and use it in `test_model.py`.
- The messages in `test_model.py` say "cancer", but this project is about **diabetes**.
- Some columns (like `Glucose`, `BloodPressure`, `BMI` and `Insulin`) have `0` values, which are not possible in real life. They are probably missing values and could be cleaned up to improve results.
