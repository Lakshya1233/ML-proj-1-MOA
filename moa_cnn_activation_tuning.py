from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# Load and preprocess data
features_df = pd.read_csv('lish-moa/train_features.csv')
targets_df = pd.read_csv('lish-moa/train_targets_scored.csv')
features_df.drop('sig_id', axis=1, inplace=True)
targets_df.drop('sig_id', axis=1, inplace=True)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features_df, targets_df, test_size=0.2, random_state=42)

# One-hot encode categorical features
def encode_categorical(data):
    categorical_cols = ['cp_dose', 'cp_time', 'cp_type']
    return pd.get_dummies(data, columns=categorical_cols, dtype=int)

X_train_encoded = encode_categorical(X_train)
X_test_encoded = encode_categorical(X_test)

# Standardize the feature sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Reshape data for 1D CNN input
X_train_reshaped = X_train_pca.reshape(-1, 50, 1)
X_test_reshaped = X_test_pca.reshape(-1, 50, 1)

# Define activation functions to test
activation_functions = ['relu', 'sigmoid', 'tanh']

best_loss = float('inf')
best_activation = None

train_losses = []
test_losses = []
activation_test_losses = {}

# Loop through activation functions and train models
for activation in activation_functions:
    print(f"Testing activation function: {activation}")
    
    # Build the model
    model = Sequential([
        Conv1D(64, kernel_size=3, activation=activation, input_shape=(50, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(rate=0.2),
        Flatten(),
        Dense(128, activation=activation),
        Dropout(rate=0.2),
        Dense(y_train.shape[1], activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_reshaped, y_train.values, epochs=50, batch_size=32, verbose=0)
    
    # Evaluate the model on training and test sets
    train_loss = model.evaluate(X_train_reshaped, y_train.values, verbose=0)[0]
    test_loss = model.evaluate(X_test_reshaped, y_test.values, verbose=0)[0]
    
    print(f"Training loss: {train_loss}")
    print(f"Testing loss: {test_loss}")
    
    activation_test_losses[activation] = test_loss
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    if test_loss < best_loss:
        best_loss = test_loss
        best_activation = activation

# Plot the test losses for different activation functions
plt.figure(figsize=(10, 6))
plt.bar(activation_test_losses.keys(), activation_test_losses.values())
plt.xlabel('Activation Function')
plt.ylabel('Loss')
plt.title('Activation Function vs Loss')
plt.show()

print(f"Best activation function: {best_activation}")
print(f"Best test loss: {best_loss}")
