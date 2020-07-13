import numpy as np
# np.random.seed(0)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Input datasets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[1], [0], [0], [1]])

epochs = 10000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# Random weights and bias initialization
hidden_weights = np.array([[0.25879808,  0.64155191],
                           [0.51179979,  0.33730641]])
hidden_bias = np.array([[0.6756413,  0.16551511]])
output_weights = np.array([[0.18202619],
                           [0.68994281]])
output_bias = np.array([[0.69256705]])

print("Initial hidden weights: ", end='')
print(*hidden_weights)
print("Initial hidden biases: ", end='')
print(*hidden_bias)
print("Initial output weights: ", end='')
print(*output_weights)
print("Initial output biases: ", end='')
print(*output_bias)


# Training algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * \
        sigmoid_derivative(predicted_output)  # (1000*1) .* (1000 * 1)

    error_hidden_layer = d_predicted_output.dot(
        output_weights.T)  # (1000 * 1) * (2 * 1)' = 1000 * 2

    d_hidden_layer = error_hidden_layer * \
        sigmoid_derivative(hidden_layer_output)  # (1000 * 2) .* (1000 * 2)

    # Updating Weights and Biases

    # (2 * 1000) * (1000 * 1) = (2 * 1)
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr

    # add a single number to each wight
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr

    # (2 * 1000) * (1000 *2) = (2 * 2)
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    # hidden_bias + [[-0.00124674 -0.00107953]]
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

print("Final hidden weights: ", end='')
print(*hidden_weights)
print("Final hidden bias: ", end='')
print(*hidden_bias)
print("Final output weights: ", end='')
print(*output_weights)
print("Final output bias: ", end='')
print(*output_bias)

print("\nOutput from neural network after 10,000 epochs: ", end='')
print(*predicted_output)
