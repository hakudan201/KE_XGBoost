from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from XGBoostClassifier import XGBoostClassifier
from sklearn.metrics import confusion_matrix
import seaborn
from matplotlib import pyplot

# generate data for classification problem
base_data_origin, predicted_data_origin = make_classification(
    n_samples=1000, n_features=10, n_classes=2, random_state=42)
base_data_origin_train, base_data_origin_test, predicted_data_origin_train, predicted_data_origin_test = train_test_split(
    base_data_origin, predicted_data_origin, test_size=0.2, random_state=42)

# build xgboost for classification
xgb_classification = XGBoostClassifier()
xgb_classification.fit(base_data_origin_train, predicted_data_origin_train)

# make predictions
predictions = xgb_classification.predict(base_data_origin_test)

# demonstrate on confusion matrix
matrix = confusion_matrix(predicted_data_origin_test, predictions)
# print(matrix)

# Plot confusion matrix using seaborn
pyplot.figure(figsize=(6, 4))
seaborn.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
pyplot.xlabel('Predicted labels')
pyplot.ylabel('True labels')
pyplot.title('Confusion Matrix')
pyplot.show()
