import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

train_data_100 = pd.read_csv("../mnist_dataset/book_1/mnist_train_100.csv", header=None, sep=",")
test_data_10 = pd.read_csv("../mnist_dataset/book_1/mnist_test_10.csv", header=None, sep=",")

all_values = train_data_100.iloc[0].values
# print(all_values)
image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap="Greys", interpolation="None")
plt.show()

# すべての値を0.01から1.0に変換
scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)

# 出力ノード数
onodes = 10
targets = np.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
print(targets)

# ニューラルネットワーククラス
class neuralNetwork:
    #ニューラルネットワークの初期化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))

        self.lr = learningrate
        self.activation_function = lambda x: sc.special.expit(x)

        pass

    #ニューラルネットワークの学習
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),np.transpose(inputs))
        pass

    #ニューラルネットワークへの照会
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    pass

# ニューラルネットワークのインスタンスを生成
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# ニューラルネットワークの学習
for record in train_data_100.values:
    inputs = (np.asfarray(record[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(record[0])] = 0.99
    n.train(inputs, targets)
    pass

# ニューラルネットワークの照会
scorecard = []

for record in test_data_10.values:
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
    print(f'pred:{label}, act:{correct_label}')

print(scorecard)