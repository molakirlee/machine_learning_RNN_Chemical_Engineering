import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf




def generate_data(timesteps, k1, k2):
    A = np.zeros(timesteps)
    B = np.zeros(timesteps)
    C = np.zeros(timesteps)

    A[0] = 1  # Initial concentration
    dt = 0.1  # Time interval
    time = np.linspace(0, timesteps*dt, timesteps)  # Time array

    for t in range(1, timesteps):
        dA = -k1 * A[t-1] * dt
        dB = (k1 * A[t-1] - k2 * B[t-1]) * dt
        dC = k2 * B[t-1] * dt
               
        A[t] = A[t-1] + dA
        B[t] = B[t-1] + dB
        C[t] = C[t-1] + dC

    return time, np.stack((A, B, C), axis=1)



# Generate training data
timesteps = 200
samples = 1000
data = []
targets = []

k1 = 0.5
k2 = 0.4 
for _ in range(samples):
    time, concentrations = generate_data(timesteps, k1, k2)
    # data.append(time[:-1].reshape(-1, 1))
    data.append(np.column_stack((time[:-1], np.full(timesteps-1, k1), np.full(timesteps-1, k2))))
    targets.append(concentrations[1:])

data = np.array(data)
targets = np.array(targets)


####################
# Build the model
####################
model = Sequential()
# model.add(SimpleRNN(50, activation='relu', input_shape=(timesteps-1, 1), return_sequences=True))
model.add(SimpleRNN(50, activation='relu', input_shape=(timesteps-1, 3), return_sequences=True))
"""
50:This is the number of units (or neurons) in the SimpleRNN layer. Increasing this number can allow the model to capture more complex patterns, but also increases computational cost and the risk of overfitting.
activation='relu':This specifies the activation function used by the layer.'relu' stands for Rectified Linear Unit, which returns max(x, 0) for any input x.ReLU is often preferred as it helps mitigate the vanishing gradient problem and allows for faster training.
input_shape=(timesteps-1, 3):This defines the shape of the input data.timesteps-1 is the number of time steps in each input sequence.3 represents the number of features at each time step (in your case, concentrations of A, B, and C).
return_sequences=True:When set to True, the layer returns the full sequence of outputs for each time step.This is useful when you're stacking multiple RNN layers or when you need output for each time step.If False, only the last output in the sequence would be returned.

stateful:If set to True, the states for each batch element are reused as initial states for the batch element in the next batch.Default is False.
dropout and recurrent_dropout:These can be used to apply dropout to the inputs and recurrent state, respectively.Helpful for preventing overfitting.
kernel_initializer and recurrent_initializer:These specify the initializer for the input and recurrent weights matrices.Default is often 'glorot_uniform'.
bias_initializer:Specifies the initializer for the bias vector.
kernel_regularizer, recurrent_regularizer, bias_regularizer:These can be used to apply regularization to the weights and biases.
"""
model.add(Dense(3))
"""
Dimensionality Adjustment.The SimpleRNN layer outputs a sequence with shape (timesteps-1, 50), where 50 is the number of RNN units. However, we need to predict 3 values (concentrations of A, B, and C) for each time step. The Dense layer with 3 units helps transform the RNN output to the required dimensionality.
"""

# model.compile(optimizer='adam', loss='mse')

# Define custom RMSE loss function
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# Use the custom loss function in model compilation
model.compile(optimizer='adam', loss=rmse)

"""
optimizer='adam': This specifies the optimization algorithm to use. Adam (Adaptive Moment Estimation) is a popular choice because it adapts the learning rate for each parameter, often leading to faster convergence.
loss='mse': This defines the loss function. 'mse' stands for Mean Squared Error, which is appropriate for regression tasks like predicting continuous values (in this case, chemical concentrations).
metrics: List of metrics to be evaluated by the model during training and testing.
learning_rate: You can set a custom learning rate if not using the default.
"""

model.summary()

# Train the model
# model.fit(data, targets, epochs=200, batch_size=32,validation_split=0.2)
early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(data, targets, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stop])


# Test the model
test_time, test_concentrations = generate_data(timesteps, k1, k2)
# test_input = test_time[:-1].reshape(1, timesteps-1, 1)
test_input = np.column_stack((test_time[:-1], np.full(timesteps-1, k1), np.full(timesteps-1, k2)))
test_input = test_input.reshape(1, timesteps-1, 3)
predicted = model.predict(test_input).reshape(timesteps-1, 3)


# Calculate RMSE for each species
rmse_A = np.sqrt(np.mean((predicted[:, 0] - test_concentrations[1:, 0])**2))
rmse_B = np.sqrt(np.mean((predicted[:, 1] - test_concentrations[1:, 1])**2))
rmse_C = np.sqrt(np.mean((predicted[:, 2] - test_concentrations[1:, 2])**2))

# Visualize the results
plt.figure('f1',figsize=(6, 6))
# plt.plot(test_time[:-1], predicted[:, 0], 'o', label='Predicted A')   
# plt.plot(test_time[:-1], predicted[:, 1], 's', label='Predicted B')
# plt.plot(test_time[:-1], predicted[:, 2], '^', label='Predicted C')
plt.plot(test_time[:-1], predicted[:, 0], 'o', label=f'Predicted A (RMSE: {rmse_A:.4f})', markersize=4)   
plt.plot(test_time[:-1], predicted[:, 1], 's', label=f'Predicted B (RMSE: {rmse_B:.4f})', markersize=4)
plt.plot(test_time[:-1], predicted[:, 2], '^', label=f'Predicted C (RMSE: {rmse_C:.4f})', markersize=4)
plt.plot(test_time[:-1], test_concentrations[:-1, 0], label='True A', linestyle='dashed')
plt.plot(test_time[:-1], test_concentrations[:-1, 1], label='True B', linestyle='dashed')
plt.plot(test_time[:-1], test_concentrations[:-1, 2], label='True C', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.show()

"""
# Generate a new time array with more timesteps
new_timesteps = timesteps * 2  # Double the original timesteps
new_time = np.linspace(0, new_timesteps*0.1, new_timesteps)

# Reshape the new time array to match the model's input shape
new_input = new_time[:-1].reshape(1, new_timesteps-1, 1)

# Use the model to predict concentrations
new_predicted = model.predict(new_input).reshape(new_timesteps-1, 3)

# Ensure all predicted concentrations are non-negative
new_predicted = np.maximum(new_predicted, 0)

# Visualize the results
plt.figure('f2',figsize=(6, 6))
plt.plot(new_time[:-1], new_predicted[:, 0], 'o', label='Predicted A', markersize=2)
plt.plot(new_time[:-1], new_predicted[:, 1], 's', label='Predicted B', markersize=2)
plt.plot(new_time[:-1], new_predicted[:, 2], '^', label='Predicted C', markersize=2)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Predicted Concentrations for Extended Timeframe (Non-negative)')
plt.legend()
plt.ylim(bottom=0)  # Set the lower y-axis limit to 0
plt.show()
"""