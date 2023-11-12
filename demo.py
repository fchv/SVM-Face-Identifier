from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

#Gather Dataset
from sklearn.datasets import fetch_lfw_people
#Get identities that have at least 100 images in the dataset (there are 5 identities)
faces = fetch_lfw_people(min_faces_per_person=100)
X = faces.data
y = faces.target

# Resize the dataset to contain 105 images for each identity
c = len(y)
Colin_Powell_count = 0
Donald_Rumsfeld_count = 0
George_Bush_count = 0
Gerhard_Schroeder_count = 0
Tony_Blair_count = 0
i = -1
while i < c-1:
    i += 1
    if (y[i] == 0):
        Colin_Powell_count += 1
        if Colin_Powell_count > 109:
            X = np.delete(X, i, 0)
            y = np.delete(y, i)
            c -= 1
            i -= 1
    elif (y[i] == 1):
        Donald_Rumsfeld_count += 1
        if Donald_Rumsfeld_count > 109:
            X = np.delete(X, i, 0)
            y = np.delete(y, i)
            c -= 1
            i -= 1
    elif (y[i] == 2):
        George_Bush_count+= 1
        if George_Bush_count > 109:
            X = np.delete(X, i, 0)
            y = np.delete(y, i)
            c -= 1
            i -= 1
    elif (y[i] == 3):
        Gerhard_Schroeder_count += 1
        if Gerhard_Schroeder_count > 109:
            X = np.delete(X, i, 0)
            y = np.delete(y, i)
            c -= 1
            i -= 1
    elif (y[i] == 4):
        Tony_Blair_count += 1
        if Tony_Blair_count > 109:
            X = np.delete(X, i, 0)
            y = np.delete(y, i)
            c -= 1
            i -= 1

n_features = X.shape[1]
print('Total number of data samples: ', len(X))
print("n_features: %d" % n_features)
print("classes (identities): ", faces.target_names)

#Split into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print('Test true identities:')
print(y_test)



import numpy as np

class MultiClassSVM:

    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.C = lambda_param #C is the error term, also represented by lamda
        self.epochs = n_iters
        self.classifiers = []

    def fit(self, X, y):
        #Find how many unique possible classes (identities) there are
        classes = np.unique(y)
        num_classes = len(classes)
        #The features represents the size of each image (62 x 47 = 2914)
        num_samples, num_features = X.shape

        for i in range(num_classes):
            #Initialize weights and bias
            w = np.zeros(num_features)
            b = 0

            # Convert to binary classification problem
            binary_labels = np.where(y == classes[i], 1, -1)

            #Perform Gradient Descent
            for _ in range(self.epochs):
                #Find Hinge loss
                score = np.dot(X, w) - b #decision function
                loss = 1 - binary_labels * score
                
                #Calculate gradient
                gradient_w = np.zeros(num_features)
                gradient_b = 0

                for j in range(num_samples):
                    if loss[j] > 0:
                        gradient_w = gradient_w - ( binary_labels[j] * X[j] )
                        gradient_b = gradient_b - binary_labels[j]

                gradient_w = (gradient_w/num_samples) + (2 * self.C * w)
                gradient_b = (gradient_b/num_samples)

                #Update weights and bias
                w = w - self.learning_rate * gradient_w
                b = b - self.learning_rate * gradient_b

            #After optimizing, save the weights and bias for this class's classifier
            self.classifiers.append((w, b))

    def predict(self, X):
        #For given input, create a row containing a 0 for each possible classifier
        scores = np.zeros((len(X), len(self.classifiers)))

        #Calculate the decision function for each class
        for i, (w, b) in enumerate(self.classifiers):
            scores[:, i] = np.dot(X, w) - b

        #Select the class with the highest score
        #For each row in scores, we find the greatest positive or greatest negative, and round it to an int
        predictions = np.argmax(scores, axis=1)
        return predictions


svm = MultiClassSVM()
#Train
svm.fit(X_train, y_train)
#Predict
predictions = svm.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(predictions, y_test))


#Plot the some of the predictions of the dataset

def plot_gallery(images, titles, h, w, n_row=5, n_col=5):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=9)
        plt.xticks(())
        plt.yticks(())

import matplotlib.pyplot as plt
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(predictions, y_test, faces.target_names, i)
                     for i in range(predictions.shape[0])]

n_samples, h, w = faces.images.shape
plot_gallery(X_test, prediction_titles, h, w)

plt.show()
