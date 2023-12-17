import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = {
    "Age": [25, 35, 45, 20, 35, 52, 23, 40, 60, 48, 33, 29],
    "Loan": [40000, 60000, 80000, 20000, 120000, 180000, 95000, 62000, 100000, 220000, 150000, 300000],
    "Default": ['N', 'N', 'N', 'N', 'N', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N']
}

# Tạo DataFrame 
df = pd.DataFrame(data)

# Xác định features và target
X = df[['Age', 'Loan']]
y = df['Default']

# Tạo giá trị KNN với k=3 
knn = KNeighborsClassifier(n_neighbors=5)

# Huấn luyện mô hình
knn.fit(X, y)
train_predictions = knn.predict(X)
# Nhập thông tin của một người mới từ bàn phím
new_age = int(input("Nhập tuổi của người mới: "))
new_loan = int(input("Nhập số tiền vay của người mới: "))

# Dự đoán khả năng vỡ nợ
new_data = {
    "Age": [new_age],
    "Loan": [new_loan]
}
new_df = pd.DataFrame(new_data)
prediction = knn.predict(new_df)
print(f"Predicted Default: {prediction[0]}")
if prediction[0]  == 'Y':
    print("Predicted Default (Y)")
else:
    print("Predicted Default (N)")
    
# Tính toán độ chính xác của mô hình
train_accuracy = accuracy_score(y, train_predictions)
print(f"Training Accuracy: {train_accuracy}") 