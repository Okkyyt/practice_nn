import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

train_data = pd.read_csv("../mnist_dataset/book_1/mnist_train.csv", header=None, sep=",")
test_data = pd.read_csv("../mnist_dataset/book_1/mnist_test.csv", header=None, sep=",")

# 画像の表示
all_values = train_data.iloc[0].values
image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap="Greys", interpolation="None")
plt.show()

# ニューラルネットワークのクラス
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))

        self.activation_function = lambda x: sc.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        inputs_hidden = np.dot(self.wih, inputs)
        outputs_hidden = self.activation_function(inputs_hidden)
        final_inputs = np.dot(self.who, outputs_hidden)
        final_outputs = self.activation_function(final_inputs)  
        outputs_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, outputs_errors)
        self.who += self.lr * np.dot((outputs_errors * final_outputs * (1.0 - final_outputs)), np.transpose(outputs_hidden))
        self.wih += self.lr * np.dot((hidden_errors * outputs_hidden * (1.0 - outputs_hidden)), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        inputs_hidden = np.dot(self.wih, inputs)
        outputs_hidden = self.activation_function(inputs_hidden)
        final_inputs = np.dot(self.who, outputs_hidden)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

for record in train_data.values:
    inputs = (np.asfarray(record[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(record[0])] = 0.99
    n.train(inputs, targets)
    pass

scorecard = []
for record in test_data.values:
    correct_label = int(record[0])
    inputs = (np.asfarray(record[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)

# エポック:学習数を増やす
epochs = 5
for e in range(epochs):
    for record in train_data.values:
        inputs = (np.asfarray(record[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(record[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

scorecard = []
for record in test_data.values:
    correct_label = int(record[0])
    inputs = (np.asfarray(record[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)