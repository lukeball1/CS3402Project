import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve


def run_logistic_regression(data, label_data, text_data):
    X = data.drop(columns = [label_data])
    Y = data[label_data]

    #Splits data into numeric and categorical categories
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    #Removes text data from categorical columns
    if text_data and text_data in categorical_cols:
        categorical_cols.remove(text_data)
    
    transformers = []

     # Text pipeline
    if text_data:
        transformers.append(("text", TfidfVectorizer(max_features=5000, ngram_range=(1,2)), text_data))

    # Numeric pipeline
    if numeric_cols:
        num_pipeline = Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())])
        transformers.append(("num", num_pipeline, numeric_cols))

    # Categorical pipeline
    if categorical_cols:
        cat_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers)

    #Creates Pipeline Shortcut
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    #Sets test data to use 20% of the dataset
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 97)

    percentages = ["10", "30", "50", "100"]

    metric_results = {"10": {}, "30": {}, "50": {}, "100": {}}

    for percentage in percentages:
        #Sets training data to inputted percentage of the rest of the dataset
        if percentage == "100":
            X_train, Y_train = X_train_full, Y_train_full
        else:
            ratio = int(percentage) / 100
            X_train, _, Y_train, _ = train_test_split(X_train_full, Y_train_full, train_size = ratio, random_state = 97)

        #Train Data
        pipeline.fit(X_train, Y_train)

        #Test Model and print Accuracy Results
        predictions = pipeline.predict(X_test)
        print(classification_report(Y_test, predictions))

        #Print Test and Training Errors
        train_pred = pipeline.predict(X_train)
        train_acc = accuracy_score(Y_train, train_pred)
        test_acc = accuracy_score(Y_test, predictions)

        print("Training error:", 1 - train_acc)
        print("Testing error:", 1 - test_acc)

        #Calculates MSE, RMSE, and R2 score for each model
        le = LabelEncoder()
        Y_test_encoded = le.fit_transform(Y_test)
        predictions_encoded = le.transform(predictions)
        
        mse = mean_squared_error(Y_test_encoded, predictions_encoded)
        rmse = mse ** 0.5
        r2 = r2_score(Y_test_encoded, predictions_encoded)

        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R2:", r2)
        
        metric_results[percentage] = {"MSE": mse, "RMSE": rmse, "R2": r2}

    #Organize metrics for graphing
    x = np.arange(len(percentages))
    width = 0.25

    mse_vals  = [metric_results[p]["MSE"]  for p in percentages]
    rmse_vals = [metric_results[p]["RMSE"] for p in percentages]
    r2_vals   = [metric_results[p]["R2"]   for p in percentages]

    #Graph of MSE
    plt.figure()
    plt.bar([f"{p}%" for p in percentages], mse_vals, color="steelblue")
    plt.title("MSE Across Training Sizes")
    plt.xlabel("Training Data %")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.show()

    #Graph of RMSE
    plt.figure()
    plt.bar([f"{p}%" for p in percentages], rmse_vals, color="orange")
    plt.title("RMSE Across Training Sizes")
    plt.xlabel("Training Data %")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.show()

    #Graph of R2
    plt.figure()
    plt.bar([f"{p}%" for p in percentages], r2_vals, color="green")
    plt.title("R² Across Training Sizes")
    plt.xlabel("Training Data %")
    plt.ylabel("R²")
    plt.tight_layout()
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







#ai_vs_human_dataset
data1 = pd.read_csv("../datasets/ai_vs_human_dataset.csv")
run_logistic_regression(data1, label_data = "Author", text_data = "Text")

#spam_email_dataset
data2 = pd.read_csv("../datasets/spam_email_dataset.csv")
data2 = data2.drop(columns=[
    "email_id",
    "sender_email",
    "num_characters",
    "sender_domain",
    "sender_reputation_score",
    "has_attachment",
    "subject"
]) 
run_logistic_regression(data2, label_data = "label", text_data = "email_text")

#Ecommerce_customer_churn_dataset
data3 = pd.read_csv("../datasets/ecommerce_customer_churn_dataset.csv")
data3 = data3.drop(columns=[
    "City",
    "Signup_Quarter",
    "Lifetime_Value",
    "Credit_Balance",
]) 
run_logistic_regression(data3, label_data = "Churned", text_data = None)