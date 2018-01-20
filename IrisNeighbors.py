import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

iris_data_train, iris_data_test, iris_target_train, iris_target_test = train_test_split(
	iris.data, iris.target, test_size = 0.33, random_state = 42)

classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(iris_data_train, iris_target_train)

targets_predicted = model.predict(iris_data_test)

i = 0
accuracy = 0
while i < len(targets_predicted) :
	if(targets_predicted[i] == iris_target_test[i]) :
		accuracy += 1
	i += 1
print("sklearn algorithm: ",accuracy,"/50")
class NeighborsClassifier:
	def __init__(self,n_neighbors) :
		self.n_neighbors = n_neighbors
	def mpredict(self, dTrain, tTrain, dTest) : 
		predictions = []
		distances=[]
		i = 0
		while i < len(dTest) : 
			j=0
			s = 0
			ve = 0
			vi = 0
			while j < len(dTrain) :
				distance = abs((sum(dTrain[j]))-(sum(dTest[i])))
				distances.append([distance,tTrain[j]])
				j += 1
			distances.sort()
			k=0
			while k < self.n_neighbors :
				if(distances[k][1] == 0) :
					s += 1
				if(distances[k][1] == 1) :
					ve += 1
				if(distances[k][1] == 2) :
					vi += 1
				k += 1
			distances.clear()
			if(s >= ve and s >= vi) :
				predictions.append(0)
			elif(ve > s and ve >= vi) :
				predictions.append(1)
			elif(vi > s and vi > ve) :
				predictions.append(2)
			i += 1
		return predictions

	def epredict(self, dTrain, tTrain, dTest) : 
		predictions = []
		distances=[]
		i = 0
		while i < len(dTest) : 
			j=0
			s = 0
			ve = 0
			vi = 0
			while j < len(dTrain) :
				distance = np.linalg.norm(dTrain[j] - dTest[i])
				distances.append([distance,tTrain[j]])
				j += 1
			distances.sort()
			k=0
			while k < self.n_neighbors :
				if(distances[k][1] == 0) :
					s += 1
				if(distances[k][1] == 1) :
					ve += 1
				if(distances[k][1] == 2) :
					vi += 1
				k += 1
			distances.clear()
			if(s >= ve and s >= vi) :
				predictions.append(0)
			elif(ve > s and ve >= vi) :
				predictions.append(1)
			elif(vi > s and vi > ve) :
				predictions.append(2)
			i += 1
		return predictions

classifier = NeighborsClassifier(n_neighbors=3)
mPredictions = classifier.mpredict(iris_data_train, iris_target_train, iris_data_test)
i = 0
accuracy = 0
while i < len(mPredictions) :
	if(mPredictions[i] == iris_target_test[i]) :
		accuracy += 1
	i += 1
print("mlearn algorithm: ",accuracy,"/50")

ePredictions = classifier.epredict(iris_data_train, iris_target_train, iris_data_test)
i = 0
accuracy = 0
while i < len(ePredictions) :
	if(ePredictions[i] == iris_target_test[i]) :
		accuracy += 1
	i += 1
print("elearn algorithm: ",accuracy,"/50")

