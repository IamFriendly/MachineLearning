import numpy as np
import copy
from sklearn import datasets

iris = datasets.load_iris()

#Input array
inputData=iris.data

#Output
Z=iris.target

targets= []
for i in range(len(Z)):
	if Z[i] == 0:
		targets.append([1,0,0])
	if Z[i] == 1:
		targets.append([0,1,0])
	if Z[i] == 2:
		targets.append([0,0,1])


#Sigmoid Function
def sigmoid (x):
	return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
	return x * (1 - x)

#Variable initialization
epoch=1500 #Setting training iterations
lr=.1 #Setting learning rate
inputlayer_neurons = 4 #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 3 #number of neurons at output layer
errorPerEpoch =[]

class neuron:
	def __init__(self, inputNum):
		self.inputNum = inputNum
		self.weights = np.random.uniform(low = -1.0, high= 1.0, size=inputNum)
	def getOutput(self, inputs):
		return sigmoid(np.sum(self.weights*inputs))
	def adjustWeights(self, Err, a):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] - lr*Err*a

hNeurons = []
for i in range(hiddenlayer_neurons):
	hNeurons.append(neuron(inputlayer_neurons))


oNeurons = []
for i in range(output_neurons):
	oNeurons.append(neuron(hiddenlayer_neurons))

outPutErr = copy.deepcopy(targets)
outputs = copy.deepcopy(targets)
for i in range(epoch):


	for j in range(len(inputData)):
	#Forward Propogation
		hLayer_Outputs = []
		for k in range(hiddenlayer_neurons):
			hLayer_Outputs.append(1)
		for k in range(len(hNeurons)):
			hLayer_Outputs[k] = hNeurons[k].getOutput(inputData[j])
		for k in range(len(oNeurons)):
			outputs[j][k] = oNeurons[k].getOutput(hLayer_Outputs)

	#BackPropagation

		outputDelta = []
		for k in range(output_neurons):
			outputDelta.append(1)
		for k in range(len(oNeurons)):
			outPutErr[j][k] = outputs[j][k] - targets[j][k]
			outputDelta[k] = derivatives_sigmoid(outputs[j][k])*(outputs[j][k]-targets[j][k])
			oNeurons[k].adjustWeights(outputDelta[k],outputs[j][k])
		for k in range(len(hNeurons)):
			x = 0
			for l in range(len(outputDelta)):
				x += outputDelta[l]*oNeurons[l].weights[k]
			hNeurons[k].adjustWeights(derivatives_sigmoid(hLayer_Outputs[k])*x,hLayer_Outputs[k])
	errorPerEpoch.append(np.sum(outPutErr)/150)
x=0
for i in range(len(targets)):
	if abs(targets[i][0] - outputs[i][0]) < .5 and abs(targets[i][1] - outputs[i][1]) < .5 and abs(targets[i][2] - outputs[i][2]) <.5:
		x += 1
print(x/150)
print(outputs)
