from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import skew, boxcox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

train_features = pd.read_csv('lish-moa/train_features.csv')
train_targets = pd.read_csv('lish-moa/train_targets_scored.csv')
train_features = train_features.drop('sig_id', axis=1)
train_targets = train_targets.drop('sig_id', axis=1)

train_features, test_features, train_scored, test_scored = train_test_split(
    train_features, train_targets, test_size=0.5, random_state=42)

train_features = train_features.reset_index(drop=True)
train_targets = train_targets.reset_index(drop=True)
train_scored = train_scored.reset_index(drop=True)
test_scored = test_scored.reset_index(drop=True)

train_features = train_features[:5000]
test_features = test_features[:5000]
train_scored = train_scored[:5000]
test_scored = test_scored[:5000]

test_features.shape


def oneHotEncoding(data):
    # performing one hot encoding on discrete features
    discrete_features = ['cp_dose', 'cp_time', 'cp_type']
    data = pd.get_dummies(data, columns=discrete_features, dtype=int)
    return data


def cross_entropy(predicted_probabilities, true_labels):
    # Compute cross-entropy loss
    # A small constant to prevent numerical instability (avoid log(0))
    epsilon = 1e-15
    predicted_probabilities = np.clip(
        predicted_probabilities, epsilon, 1 - epsilon)  # Clip probabilities
    cross_entropy_loss = -np.mean(true_labels * np.log(predicted_probabilities) + (
        1 - true_labels) * np.log(1 - predicted_probabilities))
    return cross_entropy_loss


model = MLPClassifier(hidden_layer_sizes=(
    3, 3, 3, 3), activation='relu', solver='adam', max_iter=1000, random_state=42)

X_train = train_features
y_train = train_scored

X_test = test_features
y_test = test_scored

X_train = oneHotEncoding(X_train)
X_test = oneHotEncoding(X_test)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=10)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# from sklearn.preprocessing import StandardScaler

# Assuming you have a function oneHotEncoding that performs one-hot encoding

# Define your hyperparameter grids
solvers = ['adam']
activations = ['identity', 'logistic', 'tanh', 'relu']

best_loss = float('inf')
best_solver = None
best_activation = None

training_losses = []
testing_losses = []

for solver in solvers:
    for activation in activations:
        print(f"Testing solver: {solver}, activation: {activation}")

        model = MLPClassifier(hidden_layer_sizes=(
            3, 3, 3, 3), activation=activation, solver=solver, max_iter=10000, random_state=42)

        loss = 0
        loss_train = 0

        for col in tqdm(y_train.columns, desc='Processing columns'):
            if y_train[col].unique().shape[0] == 1:
                y_pred = [0.001] * len(X_test)
                y_train_pred = [0.001] * len(X_train)
            else:
                model.fit(X_train, y_train[col])
                y_train_pred = model.predict_proba(X_train)[:, 1]
                y_pred = model.predict_proba(X_test)[:, 1]

            loss += log_loss(y_test[col].values, y_pred, labels=[0, 1])
            loss_train += log_loss(y_train[col].values,
                                   y_train_pred, labels=[0, 1])

        average_training_loss = loss_train / len(y_train.columns)
        average_testing_loss = loss / len(y_train.columns)

        training_losses.append(average_training_loss)
        testing_losses.append(average_testing_loss)

        print(f"Training loss : {average_training_loss}")
        print(f"Testing loss : {average_testing_loss}")

        # Update the best hyperparameters if needed
        if average_testing_loss < best_loss:
            best_loss = average_testing_loss
            best_solver = solver
            best_activation = activation

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(len(solvers) * len(activations)),
         training_losses, label='Training Loss', marker='o')
plt.plot(range(len(solvers) * len(activations)),
         testing_losses, label='Testing Loss', marker='o')
plt.xlabel('Configuration Index')
plt.ylabel('Log Loss')
plt.title('Solver and Activation vs Loss')
plt.xticks(range(len(solvers) * len(activations)),
           [f'{solver}-{activation}' for solver in solvers for activation in activations], rotation=45)
plt.legend()
plt.show()

print(f"\nBest solver: {best_solver}")
print(f"Best activation: {best_activation}")
print(f"Corresponding testing loss: {best_loss}")
