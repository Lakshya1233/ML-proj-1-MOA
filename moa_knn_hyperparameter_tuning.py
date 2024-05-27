from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# Load and preprocess data
train_features = pd.read_csv('train_features.csv')
train_targets = pd.read_csv('train_targets_scored.csv')
train_features = train_features.drop('sig_id', axis=1)
train_targets = train_targets.drop('sig_id', axis=1)

# Split data into train and test sets
train_features, test_features, train_scored, test_scored = train_test_split(
    train_features, train_targets, test_size=0.5, random_state=42)

# Limit data for faster iteration
train_features = train_features[:5000]
test_features = test_features[:5000]
train_scored = train_scored[:5000]
test_scored = test_scored[:5000]

# Define one-hot encoding function
def encode_categorical(data):
    discrete_features = ['cp_dose', 'cp_time', 'cp_type']
    return pd.get_dummies(data, columns=discrete_features, dtype=int)

# Define cross-entropy loss function
def cross_entropy(predicted_probabilities, true_labels):
    epsilon = 1e-15
    predicted_probabilities = np.clip(
        predicted_probabilities, epsilon, 1 - epsilon)
    cross_entropy_loss = -np.mean(true_labels * np.log(predicted_probabilities) + (
        1 - true_labels) * np.log(1 - predicted_probabilities))
    return cross_entropy_loss

# Grid search parameters
param_grid = {'n_neighbors': [100, 200, 300, 400, 500]}

best_loss = float('inf')
best_params = None

neighbor_counts = []
testing_losses = []

# Grid search over the number of neighbors
for n_neighbors in param_grid['n_neighbors']:
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    X_train = train_features
    y_train = train_scored

    X_test = test_features
    y_test = test_scored

    # One-hot encoding
    X_train_encoded = encode_categorical(X_train)
    X_test_encoded = encode_categorical(X_test)

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # PCA transformation
    pca = PCA(n_components=30)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    loss = 0
    loss_train = 0

    # Training and evaluation
    for col in tqdm(y_train.columns, desc=f'Processing columns with {n_neighbors} neighbors'):
        if y_train[col].nunique() == 1:
            y_pred = [0.001] * len(X_test)
            y_train_pred = [0.001] * len(X_train)
        else:
            model.fit(X_train_pca, y_train[col])
            y_train_pred = model.predict_proba(X_train_pca)[:, 1]
            y_pred = model.predict_proba(X_test_pca)[:, 1]

        loss += log_loss(y_test[col].values, y_pred, labels=[0, 1])
        loss_train += log_loss(y_train[col].values,
                               y_train_pred, labels=[0, 1])

    average_loss_train = loss_train / len(y_train.columns)
    average_loss = loss / len(y_train.columns)

    neighbor_counts.append(n_neighbors)
    testing_losses.append(average_loss)

    print(f"Training loss with {n_neighbors} neighbors: {average_loss_train}")
    print(f"Testing loss with {n_neighbors} neighbors: {average_loss}")

    # Update best hyperparameters
    if average_loss < best_loss:
        best_loss = average_loss
        best_params = {'n_neighbors': n_neighbors}

print(f"Best hyperparameters: {best_params}")
print(f"Best testing loss: {best_loss}")

# Plotting neighbor count vs loss
plt.plot(neighbor_counts, testing_losses, marker='o')
plt.title('Neighbor Count vs Testing Loss')
plt.xlabel('Number of Neighbors')
plt.ylabel('Testing Loss')
plt.show()
