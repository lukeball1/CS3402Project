import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.layers import Input

def get_representative_subset(X, y, percentage, random_state=42):

    if percentage >= 1.0:
        return X, y

    X_subset, _, y_subset, _ = train_test_split(
        
        X, y, 
        train_size=percentage, 
        stratify=y, 
        random_state=random_state
    )
    return X_subset, y_subset

def prepare_data(X, y):
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'string']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    X_reshaped = X_processed.reshape((X_processed.shape[0], X_processed.shape[1], 1))
    
    return X_reshaped, X_processed.shape[1]

def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim, 1)), # Explicit Input layer
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    file_name = '../datasets/ecommerce_customer_churn_dataset.csv'
    df = pd.read_csv(file_name)
    
    X = df.drop(columns=['Churned'])
    y = df['Churned']
    
    sizes = [0.10, 0.20, 0.50, 1.0]
    
    for size in sizes:
        print(f"\n--- Testing with {size*100}% of data ---")
        
        X_sub, y_sub = get_representative_subset(X, y, percentage=size)
        X_ready, num_features = prepare_data(X_sub, y_sub)
        
        X_train, X_test, y_train, y_test = train_test_split(X_ready, y_sub, test_size=0.2, random_state=42)
        
        model = build_model(num_features)
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Accuracy at {size*100}%: {accuracy:.4f}")