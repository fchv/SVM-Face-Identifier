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



#Gather Dataset
from sklearn.datasets import fetch_lfw_people
#Get identities that have at least 100 images in the dataset (there are 5 identities)
#The original images are 250 x 250 pixels, but the default slice and resize arguments reduce them to 62 x 47
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


from sklearn.model_selection import train_test_split
#Split into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



svm = MultiClassSVM()
#Train
svm.fit(X_train, y_train)
#Predict
predictions = svm.predict(X_test)

import sklearn.metrics as metrics
print("Accuracy: ", metrics.accuracy_score(predictions, y_test))
confusionMatrix = metrics.confusion_matrix(y_test, predictions)
print("Confusion Matrix: ", confusionMatrix)




############################### UI ###############################

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from skimage import io
from skimage import color
from skimage import transform

def openImage():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    

    image = io.imread(file_path)
    image = color.rgb2gray(image)
    image = transform.resize(image, [47,62])
    input = np.ravel(image)

    prediction = svm.predict(input)
    predicted_identity = faces.target_names[prediction[0]]
    
    if file_path:
        displayImage(file_path, predicted_identity)

def displayImage(file_path, predicted_identity):
    image = Image.open(file_path)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.photo = photo
    status_label.config(text=f"File Name: {file_path} \nPredicted Identity: {predicted_identity}")
    
root = tk.Tk()
root.title("Face Identifier")
text_widget = tk.Text(root, wrap=tk.WORD, height=45, width=55)
open_button = tk.Button(root, text="Select Image", command=openImage)
open_button.pack(padx=40, pady=20)
image_label = tk.Label(root)
image_label.pack(padx=20, pady=20)
status_label = tk.Label(root, text="", padx=40, pady=20)
status_label.pack()
root.mainloop()
