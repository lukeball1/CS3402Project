# =============================================================================
# mlp_experiment.py
# CS 3402 - Intro to Data Science | Project 3
# MLP Implementation: Impact of Training Data Size on Model Performance
#
# HOW TO RUN:
#   1. Install dependencies:
#        pip install torch scikit-learn pandas numpy matplotlib seaborn
#
#   2. Place your dataset CSV files in the same folder as this script.
#
#   3. Update the three dataset blocks in the "MAIN" section at the bottom:
#        - Set the correct filename for each dataset
#        - Set the correct label column name for each dataset
#        - The input_dim is computed automatically from your data
#
#   4. Run the script:
#        python mlp_experiment.py
#
#   OUTPUT:
#        - Learning curve plots saved as PNG files in the same folder
#        - A summary table printed to the console with mean +/- std accuracy
# =============================================================================


# ── IMPORTS ──────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")

# Set a global random seed for reproducibility across numpy and torch
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── MLP MODEL ─────────────────────────────────────────────────────────────────
#
# FlexibleMLP accepts any input dimension, so it works across all 3 datasets
# without any changes to the class itself. Just pass the correct input_dim
# when creating the model, which is handled automatically in run_experiment().
#
# Architecture:
#   Input → Linear → BatchNorm → ReLU → Dropout
#         → Linear → BatchNorm → ReLU → Dropout
#         → Output (num_classes)
#
# BatchNorm stabilizes training. Dropout helps prevent overfitting, which is
# especially relevant for this experiment since we want to observe how
# overfitting changes as training data size increases.

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=2, dropout=0.3):
        """
        Args:
            input_dim   : number of input features (set automatically per dataset)
            hidden_dims : list of hidden layer sizes (default: two layers, 128 and 64)
            num_classes : number of output classes (default: 2 for binary classification)
            dropout     : dropout rate for regularization (default: 0.3)
        """
        super(FlexibleMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers dynamically based on hidden_dims list
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Output layer — no activation here, CrossEntropyLoss handles that
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ── TRAINING FUNCTION ─────────────────────────────────────────────────────────
#
# Trains the MLP for a fixed number of epochs and returns accuracy curves
# for both training and test sets across epochs. These curves are used later
# to plot how performance evolves during training.

def train_model(X_train, y_train, X_test, y_test, input_dim,
                hidden_dims=[128, 64], epochs=50, batch_size=64, 
                lr=0.001, use_class_weights=False):
    """
    Trains a FlexibleMLP and returns per-epoch train/test accuracy.

    Args:
        X_train   : training features (numpy array)
        y_train   : training labels (numpy array)
        X_test    : test features (numpy array, fixed across all experiments)
        y_test    : test labels (numpy array, fixed across all experiments)
        input_dim : number of input features
        epochs    : number of training epochs (default: 50)
        batch_size: mini-batch size (default: 64)
        lr        : learning rate for Adam optimizer (default: 0.001)

    Returns:
        train_accs : list of training accuracy per epoch
        test_accs  : list of test accuracy per epoch
    """

    # Convert numpy arrays to PyTorch tensors
    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_te = torch.FloatTensor(X_test)
    y_te = torch.LongTensor(y_test)

    # Wrap training data in a DataLoader for mini-batch iteration
    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer fresh for each run
    model     = FlexibleMLP(input_dim=input_dim, hidden_dims=hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if use_class_weights:
        class_counts  = np.bincount(y_train)
        class_weights = torch.FloatTensor(1.0 / class_counts)
        class_weights = class_weights / class_weights.sum()
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    train_accs = []
    test_accs  = []

    for epoch in range(epochs):

        # --- Training phase ---
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        # --- Evaluation phase (no gradient needed) ---
        model.eval()
        with torch.no_grad():
            train_preds = model(X_tr).argmax(dim=1).numpy()
            test_preds  = model(X_te).argmax(dim=1).numpy()

            train_acc = accuracy_score(y_train, train_preds)
            test_acc  = accuracy_score(y_test,  test_preds)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

    # ── OPTIONAL: Save model weights ─────────────────────────────────────────
    # Not needed for this experiment since results are captured statistically.
    # Uncomment the line below if you want to save the weights for later use.
    #
    # torch.save(model.state_dict(), f"mlp_weights_dim{input_dim}.pth")
    # ─────────────────────────────────────────────────────────────────────────

    return train_accs, test_accs


# ── EXPERIMENT FUNCTION ───────────────────────────────────────────────────────
#
# Core of Project 3: trains the MLP on subsets of increasing size and records
# performance. Each fraction is repeated n_repeats times with different random
# subsamples to get a stable mean +/- std estimate, as required by the project.

def run_experiment(X, y, input_dim, dataset_name,
                    fractions=[0.1, 0.3, 0.5, 1.0],
                    n_repeats=3, epochs=50, hidden_dims=[128, 64],
                    use_class_weights=False):
    """
    Runs the full training-size experiment for one dataset.

    Args:
        X            : full feature matrix (numpy array)
        y            : full label array (numpy array)
        input_dim    : number of input features
        dataset_name : name used for printing and saving plots
        fractions    : list of training data fractions to test
        n_repeats    : number of times to repeat each fraction (for mean +/- std)
        epochs       : epochs per training run

    Returns:
        results : dict mapping each fraction to mean/std train and test accuracy
    """

    print(f"\n{'='*60}")
    print(f"  Running experiment: {dataset_name}")
    print(f"  Input features : {input_dim}")
    print(f"  Total samples  : {len(X)}")
    print(f"  Fractions      : {fractions}")
    print(f"  Repeats each   : {n_repeats}")
    print(f"{'='*60}")

    # Scale features — important for MLP performance
    scaler = StandardScaler()

    # Fixed 80/20 train/test split — test set never changes across experiments
    # random_state=42 ensures the same split every time you run the script
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # Fit scaler on training data only, then apply to both sets
    X_train_full = scaler.fit_transform(X_train_full)
    X_test       = scaler.transform(X_test)

    results = {}

    for frac in fractions:
        repeat_train_accs = []
        repeat_test_accs  = []

        for repeat in range(n_repeats):

            # Randomly subsample the training set at this fraction
            n_samples = max(1, int(len(X_train_full) * frac))
            idx       = np.random.choice(len(X_train_full), n_samples, replace=False)
            X_sub     = X_train_full[idx]
            y_sub     = y_train_full[idx]

            # Train and collect final-epoch accuracy for this repeat
            tr_curve, te_curve = train_model(
                X_sub, y_sub, X_test, y_test,
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                epochs=epochs,
                use_class_weights=use_class_weights
            )

            repeat_train_accs.append(tr_curve[-1])
            repeat_test_accs.append(te_curve[-1])

        # Summarize across repeats
        results[frac] = {
            'train_mean' : np.mean(repeat_train_accs),
            'train_std'  : np.std(repeat_train_accs),
            'test_mean'  : np.mean(repeat_test_accs),
            'test_std'   : np.std(repeat_test_accs),
            'n_samples'  : int(len(X_train_full) * frac)
        }

        print(f"  Fraction {frac*100:5.0f}% | "
                f"n={results[frac]['n_samples']:>6} | "
                f"Train: {results[frac]['train_mean']:.3f} +/- {results[frac]['train_std']:.3f} | "
                f"Test:  {results[frac]['test_mean']:.3f} +/- {results[frac]['test_std']:.3f}")

    return results


# ── PLOTTING FUNCTION ─────────────────────────────────────────────────────────
#
# Generates a learning curve showing train and test accuracy vs. data fraction.
# The shaded band around the test curve represents +/- 1 standard deviation
# across the repeated runs, visualizing result stability.

def plot_results(results, dataset_name):
    """
    Plots and saves the learning curve for one dataset's experiment results.

    Args:
        results      : dict returned by run_experiment()
        dataset_name : used for the plot title and saved filename
    """

    fractions   = list(results.keys())
    train_means = [results[f]['train_mean'] for f in fractions]
    train_stds  = [results[f]['train_std']  for f in fractions]
    test_means  = [results[f]['test_mean']  for f in fractions]
    test_stds   = [results[f]['test_std']   for f in fractions]
    n_samples   = [results[f]['n_samples']  for f in fractions]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot train accuracy line
    ax.plot(fractions, train_means, 'o-', color='steelblue',
            linewidth=2, markersize=7, label='Train Accuracy')
    ax.fill_between(fractions,
                    [m - s for m, s in zip(train_means, train_stds)],
                    [m + s for m, s in zip(train_means, train_stds)],
                    alpha=0.15, color='steelblue')

    # Plot test accuracy line with std shading
    ax.plot(fractions, test_means, 's-', color='darkorange',
            linewidth=2, markersize=7, label='Test Accuracy')
    ax.fill_between(fractions,
                    [m - s for m, s in zip(test_means, test_stds)],
                    [m + s for m, s in zip(test_means, test_stds)],
                    alpha=0.15, color='darkorange')

    # Add sample count annotations along the x-axis
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{int(f*100)}%\n(n={n})" for f, n in zip(fractions, n_samples)])

    ax.set_xlabel('Training Data Fraction', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'MLP Learning Curve — {dataset_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    filename = f"{dataset_name.replace(' ', '_')}_learning_curve.png"
    plt.savefig(filename, dpi=150)
    print(f"\n  Plot saved: {filename}")
    plt.show()


# ── CONFUSION MATRIX ─────────────────────────────────────────────────────────
#
# Generates and saves a confusion matrix for the final trained model on each
# dataset. Uses the 100% training fraction results since that represents the
# model's best performance with all available data.

def plot_confusion_matrix(X, y, input_dim, dataset_name, 
                            hidden_dims=[128, 64], epochs=50,
                            use_class_weights=False):
    """
    Trains a final model on 80% of the data and plots the confusion matrix
    evaluated on the held-out 20% test set.

    Args:
        X            : full feature matrix (numpy array)
        y            : full label array (numpy array)
        input_dim    : number of input features
        dataset_name : used for plot title and saved filename
        hidden_dims  : hidden layer sizes matching the experiment config
        epochs       : epochs matching the experiment config
        use_class_weights : whether to use class weights in loss function
    """

    # Use the same fixed split as the experiment so results are comparable
    scaler = StandardScaler()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    X_train_full = scaler.fit_transform(X_train_full)
    X_test       = scaler.transform(X_test)

    # Train one final model on 100% of the training data
    print(f"\n  Generating confusion matrix for {dataset_name}...")
    tr_curve, te_curve = train_model(
        X_train_full, y_train_full, X_test, y_test,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        epochs=epochs,
        use_class_weights=use_class_weights
    )

    # Generate predictions on the test set for the confusion matrix
    X_te     = torch.FloatTensor(X_test)
    model    = FlexibleMLP(input_dim=input_dim, hidden_dims=hidden_dims)

    # Re-train to get the final model state
    # (train_model returns curves but not the model itself)
    X_tr_tensor = torch.FloatTensor(X_train_full)
    y_tr_tensor = torch.LongTensor(y_train_full)
    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)

    if use_class_weights:
        class_counts  = np.bincount(y_train_full)
        class_weights = torch.FloatTensor(1.0 / class_counts)
        class_weights = class_weights / class_weights.sum()
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Get final predictions
    model.eval()
    with torch.no_grad():
        test_preds = model(X_te).argmax(dim=1).numpy()

    # Plot confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {dataset_name}', 
                    fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f"{dataset_name.replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(filename, dpi=150)
    print(f"  Confusion matrix saved: {filename}")
    plt.show()


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
#
# Prints a formatted comparison table across all datasets and fractions.
# Useful for copying into your project report.

def print_summary_table(all_results):
    """
    Prints a summary table of test accuracy (mean +/- std) for all datasets.

    Args:
        all_results : dict mapping dataset_name -> results dict from run_experiment()
    """

    print(f"\n{'='*70}")
    print("  SUMMARY TABLE — Test Accuracy (mean +/- std)")
    print(f"{'='*70}")

    # Collect all fractions across datasets
    all_fractions = sorted({f for res in all_results.values() for f in res.keys()})

    # Header row
    header = f"{'Fraction':<12}" + "".join([f"{name:<25}" for name in all_results.keys()])
    print(header)
    print("-" * len(header))

    # One row per fraction
    for frac in all_fractions:
        row = f"{frac*100:>6.0f}%     "
        for res in all_results.values():
            if frac in res:
                m = res[frac]['test_mean']
                s = res[frac]['test_std']
                row += f"{m:.3f} +/- {s:.3f}       "
            else:
                row += f"{'N/A':<25}"
        print(row)

    print(f"{'='*70}\n")
    

# ── DATA PREPROCESSING FOR SPAM EMAILS ─────────────────────────────────────
def preprocess_spam_email(df):
    """
    Cleans and encodes the spam email dataset for use with the MLP.
    
    Drops non-numeric columns that can't be meaningfully encoded,
    encodes sender_domain as a numeric label, and returns a clean
    feature matrix X and label array y.
    """

    # Drop columns that are identifiers or raw text
    # email_id is just a row number, subject and email_text would need
    # full NLP pipelines (tokenization, embeddings) to be useful here
    df = df.drop(columns=["email_id", "subject", "email_text", "sender_email"])

    # Encode sender_domain — converts each unique domain string to an integer
    # e.g. "outlook.com" -> 0, "gmail.com" -> 1, etc.
    le = LabelEncoder()
    df["sender_domain"] = le.fit_transform(df["sender_domain"].astype(str))

    # Separate features from label
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    return X, y

# ── DATA PREPROCESSING FOR CUSTOMER CHURN ─────────────────────────────────
def preprocess_churn(df):
    """
    Cleans and encodes the customer churn dataset for use with the MLP.

    Handles missing values, encodes categorical columns, and returns
    a clean feature matrix X and label array y.
    """

    # ── Handle missing values ─────────────────────────────────────────────
    # Fill numeric columns with the median of that column rather than 0
    # Median is preferred over mean here because it's more robust to outliers
    # which are common in behavioral data like session duration and credit balance
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != "Churned":
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Fill categorical columns with the most frequent value
    for col in ["Gender", "Country", "City", "Signup_Quarter"]:
        most_frequent = df[col].mode()[0]
        df[col] = df[col].fillna(most_frequent)
    # ─────────────────────────────────────────────────────────────────────

    # Encode categorical string columns
    le = LabelEncoder()
    for col in ["Gender", "Country", "City", "Signup_Quarter"]:
        df[col] = le.fit_transform(df[col].astype(str))

    # Separate features from label
    X = df.drop(columns=["Churned"]).values.astype(np.float32)
    y = df["Churned"].values.astype(np.int64)

    return X, y

def preprocess_ai_text(df):
    """
    Cleans and encodes the AI vs Human text dataset for use with the MLP.

    Label in this dataset is actually a row index, not a class label.
    Author contains the real class information: "AI" or "Human".
    Text is converted to TF-IDF features since raw strings can't be
    fed into the MLP directly.
    """

    # Drop the Label column — it's just a row index in this dataset
    df = df.drop(columns=["Label"])

    # Encode Author: "AI" -> 0, "Human" -> 1
    le = LabelEncoder()
    y = le.fit_transform(df["Author"].astype(str)).astype(np.int64)

    # Convert Text to TF-IDF feature matrix
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df["Text"].astype(str)).toarray().astype(np.float32)

    return X, y


# ── MAIN ──────────────────────────────────────────────────────────────────────
#
# HOW TO CONFIGURE EACH DATASET:
#
#   For each dataset block below, update two things:
#     1. The filename in pd.read_csv(...)
#     2. The label_column — the name of the column that holds the class label
#
#   Everything else (input_dim, running the experiment, plotting) is automatic.
#
#   DATASET 1 — spam Email Classification
#     - 10,000 entries, 19 features, 1 label column
#     - Example: df = pd.read_csv("spam_emails.csv")
#                label_column = "label"
#
#   DATASET 2 — Your second dataset
#     - 50,000 entries, features determined automatically
#     - Example: df = pd.read_csv("dataset2.csv")
#                label_column = "class"   ← update to your actual column name
#
#   DATASET 3 — Your third dataset
#     - 100,000 entries, features determined automatically
#     - Example: df = pd.read_csv("dataset3.csv")
#                label_column = "target"  ← update to your actual column name
#
#   If your label column contains string values like "spam"/"ham", the script
#   will encode them automatically using pandas category codes.

if __name__ == "__main__":

    # Fractions of training data to test — matches Project 3 requirements
    # (project says 10/30/50/100 but we added 80% as your group specified)
    FRACTIONS = [0.1, 0.3, 0.5, 1.0]

    # Number of repeated runs per fraction (project requires at least 3)
    N_REPEATS = 3

    # Number of training epochs per run
    EPOCHS = 50

    # Stores results from all datasets for the summary table
    all_results = {}

    # ── DATASET 1: spam Email Classification ─────────────────────────────────
    # 10,000 entries | 19 features | binary classification
    #
    # TO RUN: Replace "spam_emails.csv" with your actual filename.
    #         Replace "label" with your actual label column name.
    # ─────────────────────────────────────────────────────────────────────────

    print("\nLoading Dataset 1: Spam Email Classification...")
    df1 = pd.read_csv("../datasets/spam_email_dataset.csv")

    # Use the dedicated preprocessor instead of the generic loading approach
    X1, y1 = preprocess_spam_email(df1)

    input_dim_1 = X1.shape[1]  # will now reflect the actual usable feature count
    dataset_name_1 = "Spam Email"

    results1 = run_experiment(X1, y1, input_dim_1, dataset_name_1,
                            fractions=FRACTIONS, n_repeats=N_REPEATS, epochs=EPOCHS)
# No changes needed for Dataset 1 — defaults are fine
    plot_results(results1, dataset_name_1)
    plot_confusion_matrix(X1, y1, input_dim_1, dataset_name_1)
    all_results[dataset_name_1] = results1


    # ── DATASET 2 ─────────────────────────────────────────────────────────────
    # 50,000 entries | features determined automatically
    #
    # TO RUN: Replace "dataset2.csv" with your actual filename.
    #         Replace "label" with your actual label column name.
    # ─────────────────────────────────────────────────────────────────────────

    print("\nLoading Dataset 2...")
    df2 = pd.read_csv("../datasets/ecommerce_customer_churn_dataset.csv")
    
    X2, y2 = preprocess_churn(df2)

    input_dim_2 = X2.shape[1]
    dataset_name_2 = "Customer Churn"

    df2_check = pd.read_csv("../datasets/ecommerce_customer_churn_dataset.csv")

# ─────────────────────────────────────────────────────────────────────────

    results2 = run_experiment(X2, y2, input_dim_2, dataset_name_2,
                            fractions=FRACTIONS, n_repeats=N_REPEATS, epochs=100,
                            hidden_dims=[256, 128, 64],
                            use_class_weights=True)
# Deeper layers, more epochs, and class weights enabled to handle imbalance
    plot_results(results2, dataset_name_2)
    plot_confusion_matrix(X2, y2, input_dim_2, dataset_name_2,
                        hidden_dims=[256, 128, 64],
                        epochs=100,
                        use_class_weights=True)
    all_results[dataset_name_2] = results2


    # ── DATASET 3 ─────────────────────────────────────────────────────────────
    # 100,000 entries | features determined automatically
    #
    # TO RUN: Replace "dataset3.csv" with your actual filename.
    #         Replace "label" with your actual label column name.
    # ─────────────────────────────────────────────────────────────────────────

    print("\nLoading Dataset 3...")
    df3 = pd.read_csv("../datasets/ai_vs_human_dataset.csv")  # <-- update filename

    X3, y3 = preprocess_ai_text(df3)

    input_dim_3 = X3.shape[1]   # will be 1000 from TF-IDF max_features
    dataset_name_3 = "AI vs Human Text"

    print(df3["Author"].value_counts(normalize=True))

    results3 = run_experiment(X3, y3, input_dim_3, dataset_name_3,
                            fractions=FRACTIONS, n_repeats=N_REPEATS, epochs=EPOCHS)
    plot_results(results3, dataset_name_3)
    plot_confusion_matrix(X3, y3, input_dim_3, dataset_name_3)
    all_results[dataset_name_3] = results3


    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    # Prints a table comparing all datasets side by side.
    # Copy this output directly into your project report.

    print_summary_table(all_results)
    print("All experiments complete.")