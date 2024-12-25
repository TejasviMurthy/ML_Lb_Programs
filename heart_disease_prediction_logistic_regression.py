

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart (3).csv')



print("Number of records in each label are")
print(df['target'].value_counts())


print("\nPercentage of records in each label are")
print(df['target'].value_counts() * 100 / df.shape[0], "\n")


df.head()



def sigmoid(x):
    return pd.Series(1 / ( 1 + np.exp(-x)))

df['chol'].describe()

plt.figure(figsize = (12,4), dpi = 96)
plt.title("Histogram for cholesterol values")
plt.hist(df['chol'], bins = 'sturges', edgecolor = 'black')
plt. show()

def standard_scalar(series):
    new_series = (series - series.mean()) / series.std()
    return new_series
scaled_chol = standard_scalar(df['chol'])

plt.figure(figsize = (12,4))
plt.title("Histogram for cholesterol values")
plt.hist(scaled_chol, bins = 'sturges', edgecolor = 'black')
plt.show()

chol_sig_output = sigmoid(df['chol'])
chol_sig_output.describe()

scaled_chol_sig_output = sigmoid(scaled_chol)
scaled_chol_sig_output.describe()

def predict(sig_output, threshold):
    y_pred = [ 1 if output >= threshold else 0 for output in sig_output]
    return pd.Series(y_pred)

threshold = 0.5
heart_disease_pred = predict(scaled_chol_sig_output, threshold)

plt.figure(figsize=(13,3), dpi = 96)
plt.scatter(scaled_chol, heart_disease_pred)
plt.axhline(y = threshold, label = f'y = { threshold }', color = 'r')
plt. legend()
plt.show()

print(f"Threshold value: {threshold}")
print(f"\nPredicted value counts:\n{heart_disease_pred.value_counts()}")
print(f"\nActual value counts:\n{df['target']. value_counts()}")

from sklearn.metrics import confusion_matrix

print(confusion_matrix(df['target'], heart_disease_pred))

from sklearn.metrics import classification_report

print(classification_report(df['target'], heart_disease_pred))


from sklearn.model_selection import train_test_split

X = df.drop(columns = 'target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


from sklearn.linear_model import LogisticRegression

log_clf_1 = LogisticRegression()
log_clf_1.fit(X_train, y_train)
print(log_clf_1.score(X_train, y_train))


y_train_pred = log_clf_1.predict(X_train)

print("\n Confusion Matrix \n")
print(confusion_matrix(y_train, y_train_pred))

print("\n Classification Report\n")
print(classification_report(y_train, y_train_pred))



y_test_pred = log_clf_1.predict(X_test)

print(f"{'Test Set'.upper()}\n{'-' * 75}\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report")
print(classification_report(y_test, y_test_pred))


def standard_scaler(series):
  new_series = (series - series.mean()) / series.std()
  return new_series

norm_X_train = X_train.apply(standard_scaler, axis = 0)
norm_X_test = X_test.apply(standard_scaler, axis = 0)

norm_X_train.describe()

norm_X_test.describe()


from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


dict_rfe = {}


for i in range(1, len(X_train.columns) + 1):
  lg_clf_2 = LogisticRegression()
  rfe = RFE(lg_clf_2,n_features_to_select=i)
  rfe.fit(norm_X_train, y_train)

  rfe_features = list(norm_X_train.columns[rfe.support_])
  rfe_X_train = norm_X_train[rfe_features]


  lg_clf_3 = LogisticRegression()
  lg_clf_3.fit(rfe_X_train, y_train)


  y_test_pred = lg_clf_3.predict(norm_X_test[rfe_features])

  f1_scores_array = f1_score(y_test, y_test_pred, average = None)
  dict_rfe[i] = {"features": list(rfe_features), "f1_score": f1_scores_array}

dict_rfe


pd.options.display.max_colwidth = 100
f1_df = pd.DataFrame.from_dict(dict_rfe, orient = 'index')
f1_df


lg_clf_4 = LogisticRegression()
rfe = RFE(lg_clf_4, n_features_to_select = 3)

rfe.fit(norm_X_train, y_train)

rfe_features = norm_X_train.columns[rfe.support_]
print(rfe_features)
final_X_train = norm_X_train[rfe_features]

lg_clf_4 = LogisticRegression()
lg_clf_4.fit(final_X_train, y_train)

y_test_predict = lg_clf_4.predict(norm_X_test[rfe_features])
final_f1_scores_array = f1_score(y_test, y_test_predict, average = None)
print(final_f1_scores_array)