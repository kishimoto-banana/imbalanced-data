from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

X, y = make_classification(n_samples = 1000000, n_features = 10, weights = [0.99, 0.01], n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

num_pos = y_train.sum()
sampler = RandomUnderSampler(ratio={0:num_pos, 1:num_pos}, random_state=42)
X_train_sampled, y_train_sampled = sampler.fit_sample(X_train, y_train)

model = LogisticRegression(solver='lbfgs', random_state=42)
model.fit(X_train_sampled, y_train_sampled)

y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))
print(f'precision = {precision}')
print(f'recall = {recall}')
print(f'f1 = {f1}')