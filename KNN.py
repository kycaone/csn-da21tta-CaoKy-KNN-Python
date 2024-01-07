from math import sqrt
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd

file_path = 'khanang_vono.xlsx'  
data = pd.read_excel(file_path )

# Tính khoảng cách
def euclidean_distance(x1, x2):
    distance = sqrt((x1['Age'] - x2['Age']) ** 2 + (x1['Loan'] - x2['Loan']) ** 2)
    return distance

K = 5
# Chia dữ liệu, huấn luyện và dự đoán
train_data, test_data = train_test_split(data.to_dict('records'), test_size=0.2, random_state=42)
predicted_labels = []
actual_labels = [item['Default'] for item in test_data]

for new_person in test_data:
    closest_points = sorted(train_data, key=lambda x: euclidean_distance(x, new_person))[:K]
    nearest_labels = [point['Default'] for point in closest_points]
    predicted_label = Counter(nearest_labels).most_common(1)[0][0]
    predicted_labels.append(predicted_label)
                   
new_age = int(input("Nhập tuổi của người mới: "))
new_loan = int(input("Nhập số tiền vay của người mới: "))
new_person = {'Age': new_age, 'Loan': new_loan, 'Default': ''}
closest_points = sorted(data.to_dict('records'), key=lambda x: euclidean_distance(x, new_person))[:K]
distances = [(idx, euclidean_distance(row, new_person)) for idx, row in enumerate(data.to_dict('records'))]
sorted_distances = sorted(distances, key=lambda x: x[1])[:K]
nearest_labels = [data.to_dict('records')[idx]['Default'] for idx, _ in sorted_distances]
predicted_label = Counter(nearest_labels).most_common(1)[0][0]

new_person = {'Age': new_age, 'Loan': new_loan, 'Default': ''}
data = pd.concat([data, pd.DataFrame([new_person])], ignore_index=True)
print(f"Mục tiêu dự đoán cho tuổi {new_age}, số tiền vay {new_loan} là: {predicted_label}")
for i, point in enumerate(closest_points, 1):
    print(f"Điểm {i}: Age={point['Age']}, Loan={point['Loan']}, Default={point['Default']}")
                         
                         