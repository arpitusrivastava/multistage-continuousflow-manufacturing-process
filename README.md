# multistage-continuousflow-manufacturing-process
Kaggle dataset : https://www.kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process
The data comes from a multi-stage continuous flow manufacturing process. In the first stage,
Machines 1, 2, and 3 operate in parallel, and feed their outputs into a step that combines the
flows. Output from the combiner is measured in 15 locations surrounding the outer surface of
the material exiting the combiner.
Next, the output flows into a second stage, where Machines 4 and 5 process in series. After
Machine 5, measurements are made again in the same 15 locations surrounding the outer
surface of the material exiting Machine 5.
In this model we are trying to measure how accurately we are able to predict the values in the
15 locations using machine learning models for any given set of inputs.
The above problem is a supervised learning problem. Here we are trying to find
correlation between the variables of inputs here we are implementing a multi-stage
machine learning pipeline.
In the first stage feature selection is done using backward elimination. Then, it
splits the data into training and testing datasets and applies a Support Vector
Machine (SVM) with a polynomial kernel for prediction. Finally, it evaluates the
model by calculating its score on the test data-set.
The second stage of the code goes through a similar process of fitting an OLS
regression model, performing backward elimination to select significant features,
and then training a SVM-Poly model on the selected features. The score of the
model on the test data is then printed.
Following the above stages, for each column in the output (target) data, we are
performing backward elimination to select the most significant features from the input
data. This is done by fitting an OLS model, then eliminating the feature with the
highest p-value if it's greater than 0.05. Once all insignificant features are removed,
we train an SVM with a Polynomial kernel using the selected features, and print the
score of the model.
