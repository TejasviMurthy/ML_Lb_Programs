import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

file_path = 'knn_data.xlsx'
data = pd.read_excel(file_path)

X = data[['cgpa', 'assessment', 'project']].values
y = data['result'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify = y_encoded
)
k = 3
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

for i, test_point in enumerate(X_test):
    distances = np.sqrt(((X_train - test_point) ** 2).sum(axis=1))
    nearest_neighbors_indices = np.argsort(distances)[:k]
    print(f"\nTest Point {i+1}: {test_point}")
    for j, idx in enumerate(nearest_neighbors_indices):
        print(f"Neighbor {j+1}: Distance = {distances[idx]:.2f}, Label = {label_encoder.inverse_transform([y_train[idx]])[0]}")