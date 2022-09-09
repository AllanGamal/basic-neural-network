from audioop import bias
import flowersdata as data
import math

weights = [[0.1, 0.2], [0.15, 0.25], [0.18, 0.1]]
biases = [0.3, 0.4, 0.35] 
epochs = 1
learning_rate = 0.1

# softmax function
def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p - m) for p in predictions]
    total = sum(temp)
    return [t / total for t in temp]

# train the network
for epoch in range(epochs): 
    pred = [[sum([w * i for w, i in zip(we, inp)]) + bi for we, bi in zip(weights, biases)] for inp in data.inputs]
    act = [softmax(p) for p in pred]
    print(act)