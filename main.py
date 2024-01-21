import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = np.genfromtxt('wine.csv', delimiter=',')
# Tách features (X) và labels (y)
X = data[:, 1:]  
y = data[:, 0]   

# chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#xây dựng mô hình
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# Dự đoán nhãn cho tập kiểm tra
y_pred = knn.predict(X_test)

print("Các nhãn dự đoán: ", y_pred)
print("Các nhãn thực tế: ", y_test)
#so sánh nhãn dự đoán với nhãn thực tế 
for i in range(len(y_pred)):
      if y_pred[i] == y_test[i]:
            print("Mẫu ", i, "được phân loại chính xác")
      else:
            print("Mẫu ", i, "được phân loại sai, nhãn dự đoán là", y_pred[i], ", nhãn thực tế là", y_test[i])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
