'''
First attemtpt to implement a perceptron from scratch
Author: Yiyang
'''

import numpy as np



class Naive_Perceptron:

    def __init__(self, learning_rate = 0.01, epochs = 50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss = []


    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                prediction = self.predict(x_i)
                update = self.learning_rate * (y[idx] - prediction)
                self.weights += update * x_i
                self.bias += update
                self.loss.append(np.sum((y - self.predict(X)) ** 2))


    def predict(self, X):

        return np.where(X @ self.weights + self.bias > 0, 1, 0)
    
    def getParams(self):
        
        return (self.weights, self.bias)



if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from matplotlib import pyplot as plt

    # Load dataset
    X, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train perceptron
    p = Naive_Perceptron(learning_rate=0.1, epochs=100)
    p.fit(X_train, y_train)

    # Make predictions
    predictions = p.predict(X_test)

    # Evaluate accuracy
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc * 100:.2f}%")

    # Plot loss over epochs
    plt.plot(p.loss)
    plt.title("Loss over epochs")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()


