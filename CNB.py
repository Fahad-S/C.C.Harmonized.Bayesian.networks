# Import required modules
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris # load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

# Loading the dataset
dataset = load_iris()# load_diabetes()#load_wine()
X = dataset.data
y = dataset.target

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)

# Creating and training the Complement Naive Bayes Classifier
classifier = ComplementNB() #MultinomialNB()#
classifier.fit(X_train, y_train)

# Evaluating the classifier
prediction = classifier.predict(X_test)
prediction_train = classifier.predict(X_train)

print(f"Training Set Accuracy : {accuracy_score(y_train, prediction_train) * 100} %\n")
print(f"Test Set Accuracy : {accuracy_score(y_test, prediction) * 100} % \n\n")
print(f"Classifier Report : \n\n {classification_report(y_test, prediction)}")

