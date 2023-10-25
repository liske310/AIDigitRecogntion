# Example from https://www.toptal.com/data-science/machine-learning-number-recognition
# Example from https://www.youtube.com/watch?v=bte8Er0QhDg
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Load digits from MNIST Database
digits = load_digits()

(X_train, X_test, y_train, y_test) = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=42
)

ks = np.arange(2, 10)
scores = []
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(model, X_train, y_train, cv=5)
    score.mean()
    scores.append(score.mean())

plt.plot(scores, ks)
plt.xlabel('accuracy')
plt.ylabel('k')
plt.show()
