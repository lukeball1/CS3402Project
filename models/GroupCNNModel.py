import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input
import string

DATASET_CONFIG = {
    '../datasets/ai_vs_human_dataset.csv': {
        'target_col': 'Author',
        'categorical_target': True,
        'cols_to_drop': ['Author', 'Label'],
        'text_col': 'Text'
    },
    '../datasets/spam_email_dataset.csv': {
        'target_col': 'label',
        'categorical_target': False,
        'cols_to_drop': ['label', 'email_id', 'subject', 'email_text', 'sender_email', 'sender_domain'],
        'text_col': None
    },
    '../datasets/ecommerce_customer_churn_dataset.csv': {
        'target_col': 'Churned',
        'categorical_target': False,
        'cols_to_drop': ['Churned'],
        'text_col': None
    }
}

def extract_text_features(df, text_col):
    text = df[text_col].fillna('')
    features = pd.DataFrame()
    features['num_words'] = text.str.split().str.len()
    features['num_characters'] = text.str.len()
    features['avg_word_length'] = text.apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
    features['num_punctuation'] = text.apply(lambda x: sum(1 for c in x if c in string.punctuation))
    features['num_uppercase'] = text.apply(lambda x: sum(1 for c in x if c.isupper()))
    features['uppercase_ratio'] = features['num_uppercase'] / features['num_characters'].replace(0, 1)
    features['num_unique_words'] = text.apply(lambda x: len(set(x.lower().split())))
    features['unique_word_ratio'] = features['num_unique_words'] / features['num_words'].replace(0, 1)
    return features

def get_representative_subset(X, y, percentage, random_state=42):
    if percentage >= 1.0:
        return X, y
    X_subset, _, y_subset, _ = train_test_split(
        X, y, train_size=percentage, stratify=y, random_state=random_state
    )
    return X_subset, y_subset

def prepare_data(X):
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'string']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
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
        Input(shape=(input_dim, 1)),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    plt.figure(figsize=(10, 6))
    
    for file_name, config in DATASET_CONFIG.items():
        print(f"\nProcessing {file_name}...")
        df = pd.read_csv(file_name)
        X = df.drop(columns=[col for col in config['cols_to_drop'] if col in df.columns])

        if config['text_col'] and config['text_col'] in df.columns:
            text_features = extract_text_features(df, config['text_col'])
            X = pd.concat([X.reset_index(drop=True), text_features.reset_index(drop=True)], axis=1)
            X = X.drop(columns=[config['text_col']], errors='ignore')
        
        y = df[config['target_col']]
        if config['categorical_target']:
            y = LabelEncoder().fit_transform(y)

        sizes = [0.10, 0.20, 0.50, 1.0]
        train_accs = []
        test_accs = []

        for size in sizes:
            X_sub, y_sub = get_representative_subset(X, y, percentage=size)
            X_ready, num_features = prepare_data(X_sub)
            X_train, X_test, y_train, y_test = train_test_split(X_ready, y_sub, test_size=0.2, random_state=42)
            model = build_model(num_features)
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
            
            _, train_acc = model.evaluate(X_train, y_train, verbose=0)
            _, test_acc = model.evaluate(X_test, y_test, verbose=0)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)

        label_name = file_name.split('/')[-1]
        plt.plot([s*100 for s in sizes], train_accs, marker='o', linestyle='--', label=f'{label_name} (Train)')
        plt.plot([s*100 for s in sizes], test_accs, marker='o', linestyle='-', label=f'{label_name} (Test)')

    plt.title('Training vs. Testing Accuracy Comparison')
    plt.xlabel('Training Data Size (%)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_vs_test_accuracy.png')
    print("\nPlot saved as 'train_vs_test_accuracy.png'")
