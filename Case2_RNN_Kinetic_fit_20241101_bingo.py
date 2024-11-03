import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Generate synthetic data for the reaction A -> B -> C -> A
def generate_data(num_samples=550, timesteps=100):
    k1_true = np.random.uniform(0.5, 1.0, num_samples)
    # k2_true = np.random.uniform(0.5, 1.0, num_samples)
    k2_true = k1_true*0.8
    # k3_true = np.random.uniform(0.5, 1.0, num_samples)
    k3_true = k1_true*0.3
    
    concentration_data = []
    rate_constants = []

    for i in range(num_samples):
        A = [1.0]
        B = [0.0]
        C = [0.0]
        dt = 0.1
        
        for t in range(1, timesteps):
            dA = (-k1_true[i] * A[t-1] + k3_true[i] * C[t-1]) * dt
            dB = (k1_true[i] * A[t-1] - k2_true[i] * B[t-1]) * dt
            dC = (k2_true[i] * B[t-1] - k3_true[i] * C[t-1]) * dt
            
            A.append(A[t-1] + dA)
            B.append(B[t-1] + dB)
            C.append(C[t-1] + dC)
        
        concentration_data.append(np.column_stack((A, B, C)))
        rate_constants.append([k1_true[i], k2_true[i], k3_true[i]])
    
    return np.array(concentration_data), np.array(rate_constants)

# Generate data
X, y = generate_data()

# Split into training and test sets
X_train, X_test = X[:500], X[500:]
y_train, y_test = y[:500], y[500:]

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(15, activation='selu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')
# Train the model
history = model.fit(X_train, y_train, epochs=200, validation_split=0.1, verbose=1)
# early_stop = EarlyStopping(monitor='val_loss', patience=50)
# history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stop])
# history = model.fit(X_train, y_train, epochs=200, validation_split=0.1, callbacks=[early_stop])


"""
# Build the improved RNN model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    LSTM(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(3)
])

# Compile the model with a custom learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Train the model with increased epochs and patience
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)
"""

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
y_pred = model.predict(X_test)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(y_test[:, 0], y_test[:, 0], 'r', label='True k1')
plt.scatter(y_test[:, 0], y_pred[:, 0], label='Predicted k1')
plt.plot(y_test[:, 1], y_test[:, 1], 'g', label='True k2')
plt.scatter(y_test[:, 1], y_pred[:, 1], label='Predicted k2')
plt.plot(y_test[:, 2], y_test[:, 2], 'b', label='True k3')
plt.scatter(y_test[:, 2], y_pred[:, 2], label='Predicted k3')
plt.legend()
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Rate Constants')
plt.show()

