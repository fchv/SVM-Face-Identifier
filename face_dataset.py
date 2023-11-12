from sklearn.datasets import fetch_lfw_people
import numpy as np
faces = fetch_lfw_people(min_faces_per_person=100)

x = faces.data
print("length of data: ", len(x))
print("Number of features in the data: ", x.shape[1])
print('')

#Note there are only 4 identities that contain at least 100 images
y = faces.target
print("length of target: ", len(y))
print("Each unique targets: ", np.unique(y)) 

print(faces.target_names)


print('')
Colin_Powell_count = 0
Donald_Rumsfeld_count = 0
George_Bush_count = 0
Gerhard_Schroeder_count = 0
Tony_Blair_count =0
for i in range(len(y)):
    if (y[i] == 0):
        Colin_Powell_count += 1
    if (y[i] == 1):
        Donald_Rumsfeld_count += 1
    if (y[i] == 2):
        George_Bush_count+= 1
    if (y[i] == 3):
        Gerhard_Schroeder_count += 1
    if (y[i] == 4):
        Tony_Blair_count += 1
print(Colin_Powell_count)
print(Donald_Rumsfeld_count)
print(George_Bush_count)
print(Gerhard_Schroeder_count)
print(Tony_Blair_count)