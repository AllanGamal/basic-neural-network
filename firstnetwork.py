# neural network with one neuran, one input and one output

weight = 0.9 # arbitrary value for the slope
learning_rate = 0.1

inputs = [1, 2, 3, 4,]
targets = [2, 4, 6, 8,]


# function that takes input and returns slope*input
def predict(input):
    return weight*input


# train the network
weights = []
costs = []
iteration = 25
for i in range(iteration):
    pred = [predict(i) for i in inputs] # predict the output for each input and store in pred array
    errors = [targets[i] - pred[i] for i in range(len(targets))] # calculate the error for each input and store in errors array
    cost = sum(errors) / len(targets) # calculate the cost of the network (indicates how well the network is performing)

    # print the weight and cost of the network with 2 decimal places
    print(f"Weight: {weight:.2f}, Cost: {cost:.2f}")
    
    
    weight += cost * learning_rate# use the cost value to adjuste the weight/slope value
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
plt.scatter(range(iteration), costs, color='green')
# expand x axis and y axis to show all data points so the graph is not squashed
plt.axis([min(costs)-0.5, iteration+1, min(costs)-0.5, max(costs)+1])
# show clear line at x-axis and y-axis
plt.axhline(y=0, color='red')
plt.axvline(x=0, color='red')
plt.show()

# test the network with new input (test data)
test_input = [5, 6]
test_targets = [10, 12]
pred = [predict(i) for i in test_input]
# print the predicted output and the actual output
for i in range(len(test_targets)):
    print(f"Input: {test_input[i]}, Target: {test_targets[i]}, Prediction: {pred[i]}")
    