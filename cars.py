inputs = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
    (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
    (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
    (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290, 870, 1545, 1480, 1750, 1845, 1790, 1955]

weight1 = 0.1
weight2 = 0.2

learning_rate = 0.4
bias = 0.3
epochs = 400




def predict(input1, input2):
    return (weight1 * input1 + weight2 * input2 + bias)

# Training the network
for epoch in range(epochs):
    pred = [predict(i1, i2) for i1, i2 in inputs]
    errors = [(targets[i] - pred[i]) ** 2 for i in range(len(targets))] # calculate the error for each input and store in errors array
    cost = sum(errors)/len(targets)

    # Back propagation
    errors_d = [(2 * (pred[i] - targets[i])) for i in range(len(targets))] # calculate the derivative of the error for each input and store in errors_d array
    weights1_d = [(errors_d[i] * inputs[i][0]) for i in range(len(inputs))] # w*i 
    weights2_d = [(errors_d[i] * inputs[i][1]) for i in range(len(inputs))] # w*i 

    bias_d = [(errors_d[i] * 1) for i in range(len(targets))]

    # Update network
    weight1 -= learning_rate*sum(weights1_d) / len(weights1_d)
    weight2 -= learning_rate*sum(weights2_d) / len(weights2_d)

    bias -= learning_rate*sum(bias_d) / len(bias_d)
    



    # print the weight and cost of the network with 2 decimal places
    print(f"Epoch: {epoch} Cost: {cost:.2f}")
print(f"Weight1: {weight1:.2f}, Weight2: {weight2:.2f}, Bias: {bias:.2f}")
#  test the network
print(f"Prediction: {predict(1.0, 20000):.2f}")

