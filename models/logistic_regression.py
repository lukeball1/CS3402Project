import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve

#Input your own path for the file!!!
data = pd.read_csv("C:\\Users\\Zahren\\Downloads\\CS3402Project\\datasets\\ai_vs_human_dataset.csv")

#Text Data
X = data["Text"]
#Label: 0 for AI, 1 for Human
Y = data["Author"]

#Sets test data to use 20% of the dataset
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = None)

percentages = ["10", "30", "50", "100"]

for percentage in percentages:
    #Sets training data to inputted percentage of the rest of the dataset
    if percentage == "100":
        X_train, Y_train = X_train_full, Y_train_full
    else:
        ratio = int(percentage) / 100
        X_train, _, Y_train, _ = train_test_split(X_train_full, Y_train_full, train_size = ratio, random_state = None)

    #Creates Pipeline Shortcut
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    #Train Data
    pipeline.fit(X_train, Y_train)

    #Test Model and print Accuracy Results
    predictions = pipeline.predict(X_test)
    print(classification_report(Y_test, predictions))

    #Print Test and Training Errors
    train_pred = pipeline.predict(X_train)
    train_acc = accuracy_score(Y_train, train_pred)
    test_acc = accuracy_score(Y_test, predictions)

    print("Training accuracy:", train_acc)
    print("Testing accuracy:", test_acc)
    print("Training error:", 1 - train_acc)
    print("Testing error:", 1 - test_acc)

    #Print Confusion Matrix
    confusion = confusion_matrix(Y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix = confusion)
    disp.plot()
    plt.show()

#Create Learning_curve
training_sizes = [0.1, 0.3, 0.5, 1.0]

train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, Y, train_sizes = training_sizes, cv = 5, scoring = "accuracy", n_jobs = -1
)

train_mean = train_scores.mean(axis = 1)
test_mean = test_scores.mean(axis = 1)

plt.plot(train_sizes, train_mean, label = "Training Accuracy")
plt.plot(train_sizes, test_mean, label = "Testing Accuracy")

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.show()