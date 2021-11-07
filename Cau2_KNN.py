import numpy as np
import math
from sklearn import datasets
import operator

#dataset to test
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

randIndex = np.arange(iris_X.shape[0])
np.random.shuffle(randIndex)

iris_X = iris_X[randIndex]
iris_y = iris_y[randIndex]

X_train = iris_X[:100,:]
X_test = iris_X[100:,:]
y_train = iris_y[:100]
y_test = iris_y[100:]
#

#model
class KNN:
    def __init__(self):
        return

    def calculate_distance(self,p1, p2):
        dimension = len(p1)
        distance = 0
        for i in range(dimension):
            distance += (p1[i] - p2[i])*(p1[i] - p2[i])
        return math.sqrt(distance)

    def get_k_neighbors(self, X_train, Y_train, point, knn):
        distances = []
        neighbors = []
        for i in range(len(X_train)):
            distance = self.calculate_distance(X_train[i], point)
            distances.append((distance,Y_train[i]))
        distances.sort(key=operator.itemgetter(0)) #sort by distances
        for i in range(knn):
            neighbors.append(distances[i][1])
        return neighbors

    def highest_votes(self, labels):
        labels = np.array(labels)
        values, counts = np.unique(labels, return_counts=True)
        labels_count = list(map(lambda x,y: [x,y], values, counts))
        max_ids, max_value = max(labels_count, key=lambda item: item[1])
        return max_ids

    def predict(self, X_train, Y_train, X_test, knn):
        res = []
        for point in X_test:
            neighbors_labels = self.get_k_neighbors(X_train, Y_train, point, knn)
            tmp = self.highest_votes(neighbors_labels)
            res.append(tmp)
        return res

    def accuracy_score(self, predicts, labels):
        total = len(predicts)
        correct_count = 0
        for i in range(total):
            if predicts[i] == labels[i]:
                correct_count +=1
        accuracy = correct_count/ total
        return str(accuracy*100) + "%"
#

model = KNN()
res  = model.predict(X_train, y_train, X_test, 4) 
print(res)
print(y_test)
print(model.accuracy_score(res, y_test))

