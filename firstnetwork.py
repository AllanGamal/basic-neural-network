# neural network with one neuran, one input and one output

weight = 0.1 # arbitrary value for the slope
learning_rate = 0.1
bias = 0.1 # arbitrary value for the bias for testing. Use random values for real

inputs = [1, 2, 3, 4,]
targets = [12, 14, 16, 18,]


# function that takes input and returns slope*input
def predict(input):
    return weight*input+bias


# train the network
weights = []
costs = []
epochs = 30 # number of times to train the network (iterations)

for _ in range(epochs):
    pred = [predict(i) for i in inputs] # predict the output for each input and store in pred array
    errors = [(targets[i] - pred[i]) ** 2 for i in range(len(targets))] # calculate the error for each input and store in errors array
    cost = sum(errors) / len(targets) # calculate the cost of the network (indicates how well the network is performing)

    # print the weight and cost of the network with 2 decimal places
    print(f"Weight: {weight:.2f}, Bias: {bias:.2f}, Cost: {cost:.2f}")
    
    errors_d = [(2 * (pred[i] - targets[i])) for i in range(len(targets))] # calculate the derivative of the error for each input and store in errors_d array
    weight_d = [(errors_d[i] * inputs[i]) for i in range(len(targets))] # calculate the derivative of the weight for each input and store in weight_d array
    bias_d = [(errors_d[i] * 1) for i in range(len(targets))] # calculate the derivative of the bias for each input and store in bias_d array

    weight_d_avg = sum(weight_d) / len(targets) # calculate the average derivative of the weight for each input and store in weight_d_avg array
    weight -= weight_d_avg * learning_rate # update the weight of the network
    bias -= (sum(bias_d)/len(bias_d))*learning_rate # update the bias of the network

    weights.append(weight) # store the weight in the weights array
    costs.append(cost) # store the cost in the costs array
    



# show a graph of the costs and iteration as points with green and red lines
# cost function on graph
import matplotlib.pyplot as plt
plt.plot(costs)
plt.ylabel('Cost')
plt.xlabel('Iteration')
# show a grid in the plot
plt.grid(True)
# shot every data point in the plot
plt.scatter(range(epochs), costs, color='green')
# expand x axis and y axis to show all data points so the graph is not squashed
plt.axis([min(costs)-0.5, epochs+1, min(costs)-0.5, max(costs)+1])
# show clear line at x-axis and y-axis
plt.axhline(y=0, color='red')
plt.axvline(x=0, color='red')
plt.show()

# test the network with new input (test data)
test_input = [5, 6]
test_targets = [20, 22]
pred = [predict(i) for i in test_input]
# print the predicted output and the actual output
for i in range(len(test_targets)):
    print(f"Input: {test_input[i]}, Target: {test_targets[i]}, Prediction: {pred[i]}")
    