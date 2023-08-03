# import EDA modules
import pandas as pd
import numpy as np


# read csv file to create pd DataFrame.
closeobj = pd.read_csv('NASA Close Object Project\EarthCloseObjects.csv')
# Exploratory Data Analysis.  I begin with .head() to learn about the column names
# print(closeobj.head())      # uncomment this to see the first few lines

# # .describe() to look at descriptive statistics for each col, .info() to learn about NaN values and dtypes, isna().any() to see which cols have null values
# print(closeobj.describe())      # uncomment this line for descriptive statistics
# print(closeobj.info())          # uncomment this line for count of non-null values and data types
# print(closeobj.isna().any())    # uncomment this line to check which cols have null values

# check for skewness in data for the Y variables
print(closeobj.groupby('hazardous')['hazardous'].count())

# drop cols that do not have any variance/do not appear relevant to regression
closeobj.drop(columns=['id','name','orbiting_body','sentry_object'], axis=1, inplace=True)

# change non-float columns to float vals
closeobj['hazardous'] = closeobj['hazardous'].astype(int)


# split closeobj DF into X and Y.
# X: explanatory vars, almost every column except name, id, sentry_object, orbiting_body.  Latter 2 were deemed irrelevant due to lack of variance
# Y: Hazardous: Must turn into 0 and 1.  False = 0 , True = 1
y = np.array(closeobj['hazardous'].values)
closeobj.drop(columns='hazardous', inplace=True)
X = np.array(closeobj.values)

# import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# split data into training and testing sets, set random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)
logreg = LogisticRegression(max_iter=150, penalty='l2')

history = logreg.fit(X_train, y_train)

# have model predict X_test values
yhat = logreg.predict(X_test)

# import relevant metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# calculate and print confusion matrix
result_cm = confusion_matrix(y_test, yhat)
result_accuracy = accuracy_score(y_test, yhat)
print(result_cm)
print("Prediction Accuracy: " + "{:.2%}".format(result_accuracy))

# extract and print coefficients for explanatory variables
coefs = logreg.coef_[0]
for i in range(len(closeobj.columns)):
    print(closeobj.columns[i] + " coefficient: " + str("{:.5}".format(coefs[i])))
