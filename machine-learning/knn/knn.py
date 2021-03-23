import numpy as np
from sklearn import datasets
import tensorflow as tf

# load data: total 150 samples
iris = datasets.load_iris()
x = np.array([i for i in iris.data])
y = np.array(iris.target)
print(x.shape)
print(y.shape)

flower_labels = ["iris setosa", "iris virginica", "iris versicolor"]
print(flower_labels)

"""
One-hot encode our labels.
The np.eye returns a two-dimensional array with ones on a diagonal,
defaulting to the main diagonal.
Indexing with y then gives us the required one-hot encoding of y.
"""
#one hot encoding, another method
y = np.eye(len(set(y)))[y]
print(y.shape)

# Normalize our features to be in the range of zero to one
x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
print(x.shape)

# create indices for the train-test split
np.random.seed(42)
split = 0.8 # this makes 120 train and 30 test features
train_indices = np.random.choice(len(x), round(len(x) * split), replace=False)
test_indices =np.array(list(set(range(len(x))) - set(train_indices)))

# the train-test split
train_x = x[train_indices]
test_x = x[test_indices]
train_y = y[train_indices]
test_y = y[test_indices]
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

k = 10

def prediction(train_x, test_x, train_y, k):
    distances = tf.reduce_sum(tf.abs(tf.subtract(train_x, tf.expand_dims(test_x, axis =1))), axis=2)
    _, top_k_indices = tf.nn.top_k(tf.negative(distances), k=k)
    top_k_labels = tf.gather(train_y, top_k_indices)
    predictions_sum = tf.reduce_sum(top_k_labels, axis=1)
    pred = tf.argmax(predictions_sum, axis=1)
    return pred

i, total = 0 , 0
results = zip(prediction(train_x, test_x, train_y, k), test_y) #concatenate predicted label with actual label
print("Predicted Actual")
print("--------- ------")
for pred, actual in results:
    print(i, flower_labels[pred.numpy()],"\t",flower_labels[np.argmax(actual)] )
    if pred.numpy() == np.argmax(actual):
        total += 1
    i += 1
accuracy = round(total/len(test_x),3)*100
print("Accuracy = ",accuracy,"%")

