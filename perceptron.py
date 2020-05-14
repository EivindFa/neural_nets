import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def setup():
    training_set = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_outputs = np.array([[0,1,1,0]]).T
    np.random.seed(1)
    r = np.random.random((3,1))
    synaptic_weights = 2 * r - 1
    return training_set, training_outputs, synaptic_weights

#print(synaptic_weights)
def training(training_set, weights, facts):
    for iteration in range(2000):
        input_layer = training_set
        training_outputs = facts
        outputs = sigmoid(np.dot(input_layer, weights)) # sig( sum(xi * wi) )

        # Calculate the error, which is the difference between the 
        # output we got, and the actual output
        error = training_outputs - outputs

        adjustments = error *sigmoid_derivative(outputs)

        # Update weights
        weights += np.dot(input_layer.T, adjustments)
    return outputs

def main():
    training_set, training_outputs, weights = setup()

    outputs = training(training_set, weights, training_outputs)
    
    print("Training done, should be close to [0, 1, 1, 0]: ")
    print(outputs)

if __name__=="__main__":
    main()