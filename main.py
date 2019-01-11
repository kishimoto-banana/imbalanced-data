from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np


for r in reversed(range(1, 11)):
    positive_rate = r / 1000

    X, y = make_classification(n_samples = 1000000, n_features = 10, weights = [1-positive_rate, positive_rate], n_classes=2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    num_pos = y_train.sum()
    sampler = RandomUnderSampler(ratio={0:num_pos, 1:num_pos}, random_state=42)
    X_train_sampled, y_train_sampled = sampler.fit_sample(X_train, y_train)

    model = LogisticRegression(solver='lbfgs', random_state=42)
    model.fit(X_train_sampled, y_train_sampled)

    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(confusion_matrix(y_test, y_pred))
    print(f'precision = {precision}')
    print(f'recall = {recall}')
    print(f'f1 = {f1}')

    # PR曲線
    precisions, recalls, th = precision_recall_curve(y_test, y_proba[:,1])
    ap = average_precision_score(y_test, y_proba[:,1])

    # PR曲線をプロット
    plt.plot(precisions, recalls, label=f'positive rate = {positive_rate} (auc = %.2f)'%ap)
    plt.legend()
    plt.title('PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    # plt.show()
plt.savefig('fig/prcurve_per_rate.png')