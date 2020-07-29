#Import Library
import numpy as np #to read data as array
import pandas as pd #to read from csv files
from sklearn import linear_model
import sklearn # to split data into training set and test set
from sklearn.utils import shuffle
import matplotlib.pyplot as plt# to plot graph
from matplotlib import style
import pickle  #to save model with best score or accuracy

#style.use("ggplot")

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())

predict = "G3"

data = data[ ["G1", "G2", "absences", "failures", "studytime", "traveltime", "G3"] ]

x = np.array(data.drop([predict], 1)) #drop g3 and include rest
y =np.array(data[predict]) #final grade
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)
print("Best Score:"+str(best*100))

# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])


# Drawing and plotting model
plot = "G1" \
       ""
plt.scatter(data[plot], data["G3"])
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()
