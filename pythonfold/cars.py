# import math
import math
inputs = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
    (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
    (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
    (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] # 0 = keep, 1 = sell

weights = [0.1, 0.2]

learning_rate = 0.5
bias = 0.3
epochs = 400




def predict(inputs):
    return sum([inputs[i] * weights[i] for i in range(len(inputs))]) + bias

# sigmoid activation function
def activate(x):
    return 1 / (1 + math.exp(-x))

# cost function for binary classification
def log_loss(act, target):
    return -(target * math.log(act) + (1 - target) * math.log(1 - act))

# Training the network
for epoch in range(epochs):
    pred = [predict(inp) for inp in inputs]
    act = [activate(p) for p in pred]
    
    errors = [(targets[i] - pred[i]) ** 2 for i in range(len(targets))] # calculate the error for each input and store in errors array
    
    cost = sum(log_loss(a, t) for a, t in zip(act, targets)) / len(targets) # calculate the cost of the network (indicates how well the network is performing)
    
    print(f"ep: {epoch}, c: {cost:.2f}")

    # Back propagation
    errors_d = [((act[i] - targets[i])) for i in range(len(targets))] # calculate the derivative of the error for each input and store in errors_d array
    weights_d = [[e * i for i in inp] for e, inp in zip(errors_d, inputs)] # calculate the derivative of the weight for each input and store in weight_d array
    weights_d_T = list(zip(*weights_d)) # transpose the weights_d array
    for i in range(len(weights)):
        weights[i] -= learning_rate * sum(weights_d_T[i]) / len(targets) # update the weight of the network

    bias_d = [(errors_d[i] * 1) for i in range(len(targets))]

    

    bias -= learning_rate*sum(bias_d) / len(bias_d)
    

# test the network with new input (test data)
test_inputs = [(0.1600, 0.1391), (0.5600, 0.3046),
    (0.7600, 0.8013), (0.9600, 0.3046), (0.1600, 0.7185)]
test_targets = [0, 0, 1, 0, 0]

pred = [predict(inp) for inp in test_inputs]
act = [activate(p) for p in pred]
for a, t in zip(act, test_targets):
    # print the predicted and actual values in 0 decimal places
    print("Target: {1:.0f}, Predicted: {0:.0f}".format(a, t))




