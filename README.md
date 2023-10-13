# To-predict-the-red-wine-quality-by-using-Random-Forest-and-SVM-algorithm
This investigation aims to develop a predictive model for assessing the quality of red wine using Random Forest and Support vector machines (SVMs) prediction model techniques. The study seeks to contribute new understanding and valuable recommendations for wine manufacturers, retailers, and consumers by investigating this issue.

# Scope
The primary goal of this study is to evaluate the factors given in the dataset to conclude the quality of red wine. The following phases, which are listed below, will provide an overview of the scope of this project:

1. Analyze the dataset comprehensively and learn about all the parameters that might impact the quality of red wine.
2. To predict the red wine quality by using Random Forest and SVM algorithm.
3. Perform rationalization for the two models and compare them based on accuracy.
4. Select the model with the highest performance that accurately predicts the wine's quality.
5. Perform a simulation to learn about the impact of each parameter by visualizing them using Python.
6. Perform a conclusive analysis of the impact of variables and their slopes obtained during each simulation process.
# Objective
By examining the dataset, training different models, and conducting comparison, simulation, and Rationalization, this study seeks to construct a prediction model based on the dataset that delivers accurate predictions and insights into the input variables that determine the quality of red wine.
# Tools/Analytics
This section illustrates the procedural processes used in developing the model and performing the evaluation.

1. Data Preparation • Import the Python library: Pandas. • Reading the dataset from a .xls file using the Pandas library. • Assigning the input features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulfates, and alcohol) to a variable ('x') and the output feature (Quality) to another variable ('y').

2. Data Split • Splitting the dataset into training and testing using sklearn.model_selection.train_test_split ( ) function and keeping the test_size = 0.2 where the training and test data is divided in 80-20 ratio. • Giving 'x_train' the training input data and 'y_train' the associated output data. • Assigning the test's input data to the variable "x_test" and the outcomes of the test to the variable "y_test".

3. Model Development – Random Forest • The RandomForestRegressor function is used and imported from the sklearn.ensemble library. • To train the model, the value of n_estimator varies from 1000 to 10000 and as our total input is 11, max_feature = 10. • Here, the Output of the RandomForestRegressor function is assigned to b_model. • In .fit ( ), the x_train and y_train are passed as the parameters. • The mean of all the input variables is calculated by .mean ( ) • The accuracy is calculated by mape = ape.mean (), where accuracy = 100 – mape.

4. Model Development – Support vector machines (SVMs) • The sklearn.svm.SVR function is used and imported from the sklearn.svm library. • To train the model, the kernel is set as linear, rbf and poly. • Here, the Output of the sklearn.svm.SVR function is assigned to b_model. • In .fit ( ), the x_train and y_train are passed as the parameters. • The mean of all the input variables is calculated by .mean ( ) • The statistics library is imported, and the accuracy is calculated by mape = statistics.mean(ape), where accuracy = 100 – mape.

# Analysis
In the given case, the analysis is done by considering the Random Forest algorithm and Support vector machines (SVMs) method. The rationalization is done for both methods, and here starting off with the RF method, the rationalization technique is conducted by first calculating the mean of the eleven input variables: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulphur dioxide, total sulphur dioxide, density, pH, sulphates, alcohol, and quality. Further, for the development of the model, the value of n_estimators is taken as 1000, 5000, and 10000 and max_feature = 10. Max_feature is ten since we have eleven total input variables; hence, max_feature is n-1, where n=11, so max_feature=10. After studying Figures 1, 2, and 3, it is clearly observed that with n_estimators = 10000, the accuracy of the Random Forest (RF) Prediction Model is 91.6329, which is slightly better than the other two models.

Similarly, the rationalization technique is also conducted for the SVM method by first calculating the mean of the eleven input variables: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulphur dioxide, total sulphur dioxide, density, pH, sulphates, alcohol, and quality. Further, the kernel's value is taken as linear, rbf, and poly for the model's development. After studying Figures 4, 5, and 6, it is clearly observed that with kernel = linear, the accuracy of the SVM Prediction Model is 90.580, which is slightly better than the other two models.

Upon comparing the prediction models of Random Forest (RF) and Support Vector Machine (SVM), it becomes apparent that the RF prediction model has the best degree of accuracy in comparison to all other produced prediction models. Specifically, the RF prediction model achieves an accuracy rate of 91.6329%.

![image](https://github.com/Shruti8505/To-predict-the-red-wine-quality-by-using-Random-Forest-and-SVM-algorithm/assets/145620350/262f0e6d-5128-4b05-a373-d80a8fb09a9a)
![image](https://github.com/Shruti8505/To-predict-the-red-wine-quality-by-using-Random-Forest-and-SVM-algorithm/assets/145620350/bc82798e-68a4-4a2a-8aa5-f3f387e77170)
# Impact of Variables with Visualization
The impact of each variable on the wine's quality is explained in this part using graphs to illustrate the discussion taking the Random Forest and SVM methods into consideration. In this case, one of the inputs is kept as a variable to be compared to the wine's quality, while the other inputs are kept constant at their mean values. Spyder IDE is used for doing all the aforementioned techniques, including Visualization. 

![image](https://github.com/Shruti8505/To-predict-the-red-wine-quality-by-using-Random-Forest-and-SVM-algorithm/assets/145620350/cfaf308b-1f25-4426-aa92-ced3f30c495a)
![image](https://github.com/Shruti8505/To-predict-the-red-wine-quality-by-using-Random-Forest-and-SVM-algorithm/assets/145620350/d94d63fd-33ac-4e97-ac49-97e35c325598)
![image](https://github.com/Shruti8505/To-predict-the-red-wine-quality-by-using-Random-Forest-and-SVM-algorithm/assets/145620350/edc5d5c8-018f-4856-921c-b4987a20c9bd)
![image](https://github.com/Shruti8505/To-predict-the-red-wine-quality-by-using-Random-Forest-and-SVM-algorithm/assets/145620350/f74486cc-732e-44fd-84e2-965c9f9cb29e)
![image](https://github.com/Shruti8505/To-predict-the-red-wine-quality-by-using-Random-Forest-and-SVM-algorithm/assets/145620350/39582b92-06b1-442b-b8d2-8c43fdd34cd6)

Based on the analysis of Figures 7 to 28, it can be noted that Figures 7, 13, 25, and 27 depict instances of an overfitted model. This observation suggests the presence of overfitting, as seen by the substantial fluctuations observed in the graph. These fluctuations indicate that the model has captured all the noise and variability present in the training data. Overfitting occurs when a model becomes too complex and tends to closely mimic the training data, leading to suboptimal performance in predicting unknown data. Figures 8, 10, 12, 14, 16, 18, 20, 21, 22, 24, 26, and 28 depict the under-fitted model characterized by a relatively smooth graph with little volatility. When a model is too simplistic, it needs to have the ability to accurately represent the fundamental patterns and interrelationships present within the data. Finally, the balanced model is represented by figures 9, 11, 15, 17, 19, and 23.

Additionally, as per the above figures, Residual Sugar, Sulphates, and Alcohol significantly influence the quality of the wine in both the Random Forest (RF) and Support Vector Machine (SVM) prediction models. The slope values for the variables Residual Sugar, Sulphates, and Alcohol in the RF prediction models are 0.0315, 0.4230, and 0.1396, respectively. The SVM prediction models exhibit slope values of 0.0164, 0.9009, and 0.3629 for Residual Sugar, Sulphates, and Alcohol, respectively.
# Conclusion
With the help of Random Forest and SVM models, this study set out on a long trip to create a forecasting model for determining the quality of red wine. In response to heightened competition and changing customer expectations, the wine industry is progressively embracing data-driven methodologies to ensure the constant production of wines of exceptional quality. The research conducted in this study has implications that transcend beyond the wine sector, as it demonstrates the potential of predictive analytics in enhancing decision-making processes and attaining a competitive advantage.

This study subjected a comprehensive dataset of diverse input parameters and wine quality ratings to rigorous analysis. Two specific predictive model methods, namely Random Forest and Support Vector Machines (SVM), were used for this purpose. The evaluation and comparison of these models demonstrated that the Random Forest prediction model had superior performance compared to the SVM model, with a noticeable accuracy rate of 91.6329%. This discovery highlights the Random Forest algorithm's efficacy in accurately estimating wine quality by using the provided parameters. Furthermore, the visualization of the influence of each variable on wine quality provided valuable insights into their respective relevance. Significantly, Residual Sugar, Sulphates, and Alcohol were identified as pivotal elements influencing wine quality in the Random Forest and Support Vector Machine (SVM) prediction models. The significance of these factors in affecting wine quality was further demonstrated by the slope values.

The findings of this analysis not only provide significant information for winemakers, allowing them to improve production methods and better fulfil market needs, but it also illustrates the potential of predictive analytics in the wine business. This research makes a valuable contribution to the improvement of wine quality by integrating conventional winemaking processes with technological advancements. As a result, it offers benefits to makers, merchants, and consumers alike. In a more expansive framework, it demonstrates the efficacy of data-driven decision-making, a notion that has growing significance in the contemporary and fiercely competitive corporate environment.


