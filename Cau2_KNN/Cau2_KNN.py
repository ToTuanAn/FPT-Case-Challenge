import numpy as np
import pandas as pd
import math
import operator
from sklearn.metrics import confusion_matrix

# Chi tiết về dữ liệu mẫu https://archive.ics.uci.edu/ml/datasets/iris
# Sửa lại tên file và tham số sep dựa vào ký tự ngăn cách các từ, nếu dữ liệu không có tên cột thì thêm tham số header = None
train = pd.read_csv("input.txt" , sep = ",") 
test = pd.read_csv("output.txt" , sep = ",") 
# Mặc định cột cuối là label
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values  
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values # nếu không có y test thì xóa dòng này và sửa dòng trên thành lấy hết


# Class của model
class KNeighborsClassifier:
    def __init__(self , knn = 5):
        self.knn = knn # số điểm gần nhất
        self.X_train = [] # tập X được fit vào train
        self.y_train = [] # tập y được fit vào train
        return 
    
    # Trả về khoảng cách L2 giữa p1 và p2
    def calculate_distance(self,p1, p2):
        dimension = len(p1)
        distance = 0
        for i in range(dimension):
            distance += (p1[i] - p2[i])*(p1[i] - p2[i])
        return math.sqrt(distance)

    # Trả về nhãn của k điểm gần nhất
    def get_k_neighbors(self, point):
        distances = []
        neighbors = []
        for i in range(len(self.X_train)):
            distance = self.calculate_distance(self.X_train[i], point)
            distances.append((distance, self.y_train[i]))
        distances.sort(key=operator.itemgetter(0)) #sort by distances
        for i in range(self.knn):
            neighbors.append(distances[i][1])
        return neighbors

    #  Trả về nhãn chiếm đa số trong k điểm gần nhất
    def highest_votes(self, labels):
        labels = np.array(labels)
        values, counts = np.unique(labels, return_counts=True)
        labels_count = list(map(lambda x,y: [x,y], values, counts))
        max_ids, max_value = max(labels_count, key=lambda item: item[1])
        return max_ids
    
    # Truyền vào X_train và y_train để fit cho class
    def fit(self , X , y):
        self.X_train = X
        self.y_train = y
    
    # Trả về độ chính xác của mô hình trên tập test
    def accuracy_score(self, predicts, labels):
        total = len(predicts)
        correct_count = 0
        for i in range(total):
            if predicts[i] == labels[i]:
                correct_count +=1
        accuracy = correct_count/ total
        return str(accuracy*100) + "%"
            
        
    # Trả về mảng kiểu numpy.array gồm m phần tử chứa các nhãn phân loại của dữ liệu test
    def predict(self, X_test):
        res = []
        for point in X_test:
            neighbors_labels = self.get_k_neighbors(point)
            tmp = self.highest_votes(neighbors_labels)
            res.append(tmp)
        return res


# Chọn số điểm gần nhất trong trường hợp này là 4 
model = KNeighborsClassifier(knn = 4) 
res = model.fit(X_train , Y_train)

# Chạy model trên tập test và kiểm tra độ chính xác
y_predict  = model.predict(X_test) 
print("Mảng kiểu numpy.array gồm m phần tử chứa các nhãn phân loại của dữ liệu test:")
print(y_predict)
print("\nCác thông số khác")
print("Độ chính xác trên tập test: ", model.accuracy_score(y_test , y_predict))
print("Confusion Matrix của tập test")
cm = confusion_matrix(y_test, y_predict)
print(cm)

