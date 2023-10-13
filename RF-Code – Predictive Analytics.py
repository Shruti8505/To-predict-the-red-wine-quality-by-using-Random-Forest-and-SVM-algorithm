# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:47:36 2023

@author: Rajdeep
"""
import pandas

b_data = pandas.read_excel("C:\MBA\MBA-V\BUSI-652\Final-Team-Report\winequality-red.xlsx")

y = b_data["quality"]
x = b_data[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]

import sklearn.model_selection

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)


import sklearn.ensemble

b_model = sklearn.ensemble.RandomForestRegressor(n_estimators=1000, max_features=10)

b_model.fit(x_train, y_train)

b_model.predict(x_test)

y_pred = b_model.predict(x_test)

ape = abs(y_test - y_pred)/y_test *100

mape = ape.mean()
mape
accuracy = 100-mape
accuracy

# Rationalization 

b_model = sklearn.ensemble.RandomForestRegressor(n_estimators=5000, max_features=10)

b_model.fit(x_train, y_train)

b_model.predict(x_test)

y_pred = b_model.predict(x_test)

ape = abs(y_test - y_pred)/y_test *100

mape = ape.mean()
mape
accuracy = 100-mape
accuracy


#Rationalization with 10000 estimators
b_model = sklearn.ensemble.RandomForestRegressor(n_estimators=10000, max_features=10)

b_model.fit(x_train, y_train)

b_model.predict(x_test)

y_pred = b_model.predict(x_test)

ape = abs(y_test - y_pred)/y_test *100

mape = ape.mean()
mape
accuracy = 100-mape
accuracy


# Final chosen model 
b_model = sklearn.ensemble.RandomForestRegressor(n_estimators=1000, max_features=10)


# Predicted Value

fixed_acidity_mean = x["fixed acidity"].mean()
volatile_acidity_mean = x["volatile acidity"].mean()
citric_acid_mean = x["citric acid"].mean() 
residual_sugar_mean = x["residual sugar"].mean() 
chlorides_mean = x["chlorides"].mean()
free_sulfur_mean = x["free sulfur dioxide"].mean() 
total_sulfur_mean = x["total sulfur dioxide"].mean() 
density_mean = x["density"].mean() 
pH_mean = x["pH"].mean() 
sulphates_mean = x["sulphates"].mean() 
alcohol_mean = x["alcohol"].mean()

b_model.fit(x_train, y_train)
b_model.predict( [[fixed_acidity_mean, volatile_acidity_mean, citric_acid_mean, residual_sugar_mean, chlorides_mean, free_sulfur_mean, total_sulfur_mean, density_mean, pH_mean, sulphates_mean, alcohol_mean ]])

#Simulation

# Keeping fixed acidity as variable

min(x["fixed acidity"])
max(x["fixed acidity"])

fixed_acidity_arr = [5,8,10,12,14,15]

PredictedValue = []
for element in fixed_acidity_arr:
    predVal= b_model.predict( [[element, volatile_acidity_mean, citric_acid_mean, residual_sugar_mean, chlorides_mean, free_sulfur_mean, total_sulfur_mean, density_mean, pH_mean, sulphates_mean, alcohol_mean]] )
    PredictedValue.append(predVal)

PredictedValue_list_1 = [item[0]for item in PredictedValue]


import matplotlib.pyplot as plt
a1 = fixed_acidity_arr
b1 = PredictedValue_list_1
labels = fixed_acidity_arr
plt.plot(a1, b1)
plt.xlabel("Fixed Acidity")
plt.ylabel("Quality")
plt.show()


Slope_1 = (PredictedValue_list_1[3] - PredictedValue_list_1[0])/(12-5)
Slope_1

# Keeping volatile acidity as variable

min(x["volatile acidity"])
max(x["volatile acidity"])

volatile_acidity_arr = [0.5, 0.8, 1, 1.2, 1.5]

PredictedValue = []
for element in volatile_acidity_arr:
    predVal= b_model.predict( [[fixed_acidity_mean, element, citric_acid_mean, residual_sugar_mean, chlorides_mean, free_sulfur_mean, total_sulfur_mean, density_mean, pH_mean, sulphates_mean, alcohol_mean]] )
    PredictedValue.append(predVal)

PredictedValue_list_2 = [item[0]for item in PredictedValue]


a2 = volatile_acidity_arr
b2 = PredictedValue_list_2
labels = volatile_acidity_arr
plt.plot(a2, b2)
plt.xlabel("Volatile Acidity")
plt.ylabel("Quality")
plt.show()

Slope_2 = (PredictedValue_list_2[3] - PredictedValue_list_2[0])/(1.2-0.5)
Slope_2

# Keeping citric acid as variable

min(x["citric acid"])
max(x["citric acid"])

citric_acid_arr = [0.2, 0.4, 0.6, 0.8, 1]

PredictedValue = []
for element in citric_acid_arr:
    predVal= b_model.predict( [[fixed_acidity_mean, volatile_acidity_mean, element, residual_sugar_mean, chlorides_mean, free_sulfur_mean, total_sulfur_mean, density_mean, pH_mean, sulphates_mean, alcohol_mean]] )
    PredictedValue.append(predVal)

PredictedValue_list_3 = [item[0]for item in PredictedValue]


a3 = citric_acid_arr
b3 = PredictedValue_list_3
labels = citric_acid_arr
plt.plot(a3, b3)
plt.xlabel("Citric Acid")
plt.ylabel("Quality")
plt.show()

Slope_3 = (PredictedValue_list_3[4] - PredictedValue_list_3[0])/(1-0.2)
Slope_3


# Keeping residual sugar as variable

min(x["residual sugar"])
max(x["residual sugar"])

residual_sugar_arr = [1, 4, 8, 10, 12, 14]

PredictedValue = []
for element in residual_sugar_arr:
    predVal= b_model.predict( [[fixed_acidity_mean, volatile_acidity_mean, citric_acid_mean, element, chlorides_mean, free_sulfur_mean, total_sulfur_mean, density_mean, pH_mean, sulphates_mean, alcohol_mean]] )
    PredictedValue.append(predVal)

PredictedValue_list_4 = [item[0]for item in PredictedValue]


a4 = residual_sugar_arr
b4 = PredictedValue_list_4
labels = residual_sugar_arr
plt.plot(a4, b4)
plt.xlabel("Residual Sugar")
plt.ylabel("Quality")
plt.show()

Slope_4 = (PredictedValue_list_4[2] - PredictedValue_list_4[0])/(8-1)
Slope_4

# Keeping chlorides as variable

min(x["chlorides"])
max(x["chlorides"])

chlorides_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

PredictedValue = []
for element in chlorides_arr:
    predVal= b_model.predict( [[fixed_acidity_mean, volatile_acidity_mean, citric_acid_mean, residual_sugar_mean, element, free_sulfur_mean, total_sulfur_mean, density_mean, pH_mean, sulphates_mean, alcohol_mean]] )
    PredictedValue.append(predVal)

PredictedValue_list_5 = [item[0]for item in PredictedValue]


a5 = chlorides_arr
b5 = PredictedValue_list_5
labels = chlorides_arr
plt.plot(a5, b5)
plt.xlabel("Chlorides")
plt.ylabel("Quality")
plt.show()

Slope_5 = (PredictedValue_list_5[5] - PredictedValue_list_5[1])/(0.6-0.2)
Slope_5

# Keeping free sulfur dioxide as variable

min(x["free sulfur dioxide"])
max(x["free sulfur dioxide"])

free_sulfur_arr = [10, 20, 30, 40, 50, 60, 70]

PredictedValue = []
for element in free_sulfur_arr:
    predVal= b_model.predict( [[fixed_acidity_mean, volatile_acidity_mean, citric_acid_mean, residual_sugar_mean, chlorides_mean, element, total_sulfur_mean, density_mean, pH_mean, sulphates_mean, alcohol_mean]] )
    PredictedValue.append(predVal)

PredictedValue_list_6 = [item[0]for item in PredictedValue]


a6 = free_sulfur_arr
b6 = PredictedValue_list_6
labels = free_sulfur_arr
plt.plot(a6, b6)
plt.xlabel("Free Sulfur Dioxide")
plt.ylabel("Quality")
plt.show()

Slope_6 = (PredictedValue_list_6[6] - PredictedValue_list_6[3])/(70-40)
Slope_6

# Keeping total sulfur dioxide as variable

min(x["total sulfur dioxide"])
max(x["total sulfur dioxide"])

total_sulfur_arr = [10, 40, 80, 120, 160, 200, 240, 280]

PredictedValue = []
for element in total_sulfur_arr:
    predVal= b_model.predict( [[fixed_acidity_mean, volatile_acidity_mean, citric_acid_mean, residual_sugar_mean, chlorides_mean, free_sulfur_mean, element, density_mean, pH_mean, sulphates_mean, alcohol_mean]] )
    PredictedValue.append(predVal)

PredictedValue_list_7 = [item[0]for item in PredictedValue]

a7 = total_sulfur_arr
b7 = PredictedValue_list_7
labels = total_sulfur_arr
plt.plot(a7, b7)
plt.xlabel("Total Sulfur Dioxide")
plt.ylabel("Quality")
plt.show()

Slope_7 = (PredictedValue_list_7[7] - PredictedValue_list_7[3])/(280-120)
Slope_7

# Keeping density as variable

min(x["density"])
max(x["density"])

density_arr = [0.99, 1]

PredictedValue = []
for element in density_arr:
    predVal= b_model.predict( [[fixed_acidity_mean, volatile_acidity_mean, citric_acid_mean, residual_sugar_mean, chlorides_mean, free_sulfur_mean, total_sulfur_mean, element, pH_mean, sulphates_mean, alcohol_mean]] )
    PredictedValue.append(predVal)

PredictedValue_list_8 = [item[0]for item in PredictedValue]


a8 = density_arr
b8 = PredictedValue_list_8
labels = density_arr
plt.plot(a8, b8)
plt.xlabel("Density")
plt.ylabel("Quality")
plt.show()

Slope_8 = (PredictedValue_list_8[1] - PredictedValue_list_8[0])/(1-0.99)
Slope_8

# Keeping pH as variable

min(x["pH"])
max(x["pH"])

pH_arr = [3, 3.2, 3.5, 3.8, 4]

PredictedValue = []
for element in pH_arr:
    predVal= b_model.predict( [[fixed_acidity_mean, volatile_acidity_mean, citric_acid_mean, residual_sugar_mean, chlorides_mean, free_sulfur_mean, total_sulfur_mean, density_mean, element, sulphates_mean, alcohol_mean]] )
    PredictedValue.append(predVal)

PredictedValue_list_9 = [item[0]for item in PredictedValue]


a9 = pH_arr
b9 = PredictedValue_list_9
labels = pH_arr
plt.plot(a9, b9)
plt.xlabel("pH")
plt.ylabel("Quality")
plt.show()

Slope_9 = (PredictedValue_list_9[3] - PredictedValue_list_9[1])/(3.8-3.2)
Slope_9


# Keeping sulphates as variable

min(x["sulphates"])
max(x["sulphates"])

sulphates_arr = [ 0.5, 0.8, 1.2, 1.5, 1.8]

PredictedValue = []
for element in sulphates_arr:
    predVal= b_model.predict( [[fixed_acidity_mean, volatile_acidity_mean, citric_acid_mean, residual_sugar_mean, chlorides_mean, free_sulfur_mean, total_sulfur_mean, density_mean, pH_mean, element, alcohol_mean]] )
    PredictedValue.append(predVal)

PredictedValue_list_10 = [item[0]for item in PredictedValue]


a10 = sulphates_arr
b10 = PredictedValue_list_10
labels = sulphates_arr
plt.plot(a10, b10)
plt.xlabel("Sulphates")
plt.ylabel("Quality")
plt.show()

Slope_10 = (PredictedValue_list_10[4] - PredictedValue_list_10[0])/(1.8-0.5)
Slope_10

# Keeping alcohol as variable

min(x["alcohol"])
max(x["alcohol"])

alcohol_arr = [9, 10, 11, 12, 13, 14]

PredictedValue = []
for element in alcohol_arr:
    predVal= b_model.predict( [[fixed_acidity_mean, volatile_acidity_mean, citric_acid_mean, residual_sugar_mean, chlorides_mean, free_sulfur_mean, total_sulfur_mean, density_mean, pH_mean, sulphates_mean, element]] )
    PredictedValue.append(predVal)

PredictedValue_list_11 = [item[0]for item in PredictedValue]


a11 = alcohol_arr
b11 = PredictedValue_list_11
labels = alcohol_arr
plt.plot(a11, b11)
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.show()

Slope_11 = (PredictedValue_list_11[3] - PredictedValue_list_11[0])/(12-9)
Slope_11