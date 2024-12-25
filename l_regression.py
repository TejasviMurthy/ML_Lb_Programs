import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('heart (3).csv')


# Print the number of records with and without heart disease.
print("Number of records in each label are")
print(df['target'].value_counts())

# Print the percentage of each label
print("\nPercentage of records in each label are")
print(df['target'].value_counts() * 100 / df.shape[0], "\n")

# Print the first five rows of Dataframe.
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


#Split the training and testing data
from sklearn.model_selection import train_test_split

X = df.drop(columns = 'target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


#Create a multivariate logistic regression model. Also, predict the target values for the train set.
from sklearn.linear_model import LogisticRegression

log_clf_1 = LogisticRegression()
log_clf_1.fit(X_train, y_train)
print(log_clf_1.score(X_train, y_train))

#Predict the target values for the train set.
y_train_pred = log_clf_1.predict(X_train)

print("\n Confusion Matrix \n")
print(confusion_matrix(y_train, y_train_pred))

print("\n Classification Report\n")
print(classification_report(y_train, y_train_pred))



# Predict the target values for the test set.

y_test_pred = log_clf_1.predict(X_test)

print(f"{'Test Set'.upper()}\n{'-' * 75}\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report")
print(classification_report(y_test, y_test_pred))



#Normalise the train and test data-frames using the standard normalisation method.
def standard_scaler(series):
  new_series = (series - series.mean()) / series.std()
  return new_series

norm_X_train = X_train.apply(standard_scaler, axis = 0)
norm_X_test = X_test.apply(standard_scaler, axis = 0)

norm_X_train.describe()



#Display descriptive statistics for the normalised values of the features for the test data-frames.
norm_X_test.describe()


#Create a dictionary containing the different combination of features selected by RFE and their corresponding f1-scores.
# Import the libraries
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

# Create the empty dictionary.
dict_rfe = {}

# Create a loop
for i in range(1, len(X_train.columns) + 1):
  lg_clf_2 = LogisticRegression()
  rfe = RFE(lg_clf_2,n_features_to_select=i) # 'i' is the number of features to be selected by RFE to fit a logistic regression model on norm_X_train and y_train.
  rfe.fit(norm_X_train, y_train)

  rfe_features = list(norm_X_train.columns[rfe.support_]) # A list of important features chosen by RFE.
  rfe_X_train = norm_X_train[rfe_features]

  # Build a logistic regression model using the features selected by RFE.
  lg_clf_3 = LogisticRegression()
  lg_clf_3.fit(rfe_X_train, y_train)

  # Predicting 'y' values only for the test set as generally, they are predicted quite accurately for the train set.
  y_test_pred = lg_clf_3.predict(norm_X_test[rfe_features])

  f1_scores_array = f1_score(y_test, y_test_pred, average = None)
  dict_rfe[i] = {"features": list(rfe_features), "f1_score": f1_scores_array} # 'i' is the number of features to be selected by RFE.

  # Print the dictionary created
  dict_rfe

  # Convert the dictionary to the dataframe
  pd.options.display.max_colwidth = 100
  f1_df = pd.DataFrame.from_dict(dict_rfe, orient='index')
  f1_df

  # Logistic Regression with the ideal number of features.
  lg_clf_4 = LogisticRegression()
  rfe = RFE(lg_clf_4, n_features_to_select=3)

  rfe.fit(norm_X_train, y_train)

  rfe_features = norm_X_train.columns[rfe.support_]
  print(rfe_features)
  final_X_train = norm_X_train[rfe_features]

  lg_clf_4 = LogisticRegression()
  lg_clf_4.fit(final_X_train, y_train)

  y_test_predict = lg_clf_4.predict(norm_X_test[rfe_features])
  final_f1_scores_array = f1_score(y_test, y_test_predict, average=None)
  print(final_f1_scores_array)