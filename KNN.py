from math import sqrt
from collections import Counter
from sklearn.model_selection import train_test_split
data = [
    {'Age': 25, 'Loan': 40000, 'Default': 'N'},
    {'Age': 35, 'Loan': 60000, 'Default': 'N'},
    {'Age': 45, 'Loan': 80000, 'Default': 'N'},
    {'Age': 20, 'Loan': 20000, 'Default': 'N'},
    {'Age': 35, 'Loan': 120000, 'Default': 'N'},
    {'Age': 52, 'Loan': 18000, 'Default': 'N'},
    {'Age': 23, 'Loan': 95000, 'Default': 'Y'},
    {'Age': 40, 'Loan': 62000, 'Default': 'Y'},
    {'Age': 60, 'Loan': 100000, 'Default': 'Y'},
    {'Age': 48, 'Loan': 220000, 'Default': 'Y'},
    {'Age': 33, 'Loan': 150000, 'Default': 'Y'},
    {'Age': 48, 'Loan': 142000, 'Default': 'Y'}
]
# Tính khoảng cách
def euclidean_distance(row1, row2):
    distance = sqrt((row1['Age'] - row2['Age']) ** 2 + (row1['Loan'] - row2['Loan']) ** 2)
    return distance
K = 3

#chia dữ liệu , huấn luyện và dự đoán

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
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
closest_points = sorted(data, key=lambda x: euclidean_distance(x, new_person))[:K]
distances = [(idx, euclidean_distance(row, new_person)) for idx, row in enumerate(data)]
sorted_distances = sorted(distances, key=lambda x: x[1])[:K]
nearest_labels = [data[idx]['Default'] for idx, _ in sorted_distances]
predicted_label = Counter(nearest_labels).most_common(1)[0][0]

data.append(new_person)
print(f"Nhãn dự đoán cho age {new_age}, loan {new_loan} là : {predicted_label}")
for i, point in enumerate(closest_points, 1):
    print(f"Điểm {i}: Age={point['Age']}, Loan={point['Loan']}, Default={point['Default']}")

# Tính độ chính xác
accuracy = sum(1 for p, a in zip(predicted_labels, actual_labels) if p == a) / len(actual_labels)
print(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")