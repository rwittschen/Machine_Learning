# Machine Learning Project
Lee Wittschen  
March 2, 2018  


# Introduction

The purpose of this analysis is to use data from personal activity measurement devices (i.e. Fitbit, Nike FuelBand, etc.) on 6 participants to model and predict correct and incorrect barbell lifts. The lifts were completed in 5 different ways and the data comes from: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

# Data Exploration & Processing

The data for this analysis is split into training and testing data sets. Given that we want to compute out-of-sample errors, we will split the training set into training and testing sets while using the testing data set for final model validation. We load the data as follows:

```r
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", header=TRUE)
dim(training)
```

```
## [1] 19622   160
```

```r
validation <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", header=TRUE)
dim(validation)
```

```
## [1]  20 160
```
The data sets have a consistent structure, as expected, and we complete initial data exploration on the larger data set, training.
A review of the data set structure shows that some data attributes contain a large number of NA values and some columns contain activity metdata (i.e., user name). We will need to address these attributes prior to building the model.


## Data Cleanse

We see that some columns are metadata (i.e. user name) which are not predictive and that some attributes contain a lot of NA values, which we want to remove from the data sets. We also want to remove any attributes that have one or very few unique values relative to the number of sample. These attributes have near zero variance and will also be removed from the data sets as they should not be strong predictors. This is done as follows:

```r
set.seed(4567)
# Identify low variance predictors using the NearZeroVar function
lo_var <- nearZeroVar(training)
# Remove these predictors from the data sets
training <- training[, -lo_var]
validation <- validation[, -lo_var]
# Identify predictors with a avg number of NA values greater than 95%
NAs <- sapply(training, function(x) mean(is.na(x))) >0.99
# Remove these predictors from the data sets
training <- training[, NAs==FALSE]
validation <- validation[, NAs==FALSE]
# Remove the metadata attributes
training <- training[, -(1:6)]
validation <- validation[, -(1:6)]
dim(training)
```

```
## [1] 19622    94
```

```r
dim(validation)
```

```
## [1] 20 94
```

## Data Partition

As mentioned previously, we split the training data set into training (65%) and testing (35%) to compute out-of-sample error. This allows us to train on one data set and apply it to the training set to get the model accuracy on a new data set. We partition the data as follows:

```r
train_split <- createDataPartition(y=training$classe, p=0.65, list=FALSE)
training <- training[train_split, ]
testing <- training[-train_split, ]
```
Now that we have our training, testing and validation data sets and can begin modeling.

# Machine Learning Algorithms

We will build three models using the training data; decision tree, random forest and gradient boosting method. The decision tree model will be used since it can handle both linear and non-linear relationships. The random forest provides reduced variance over decision trees. The gradient boosting method within the caret package shall be used as it should provide improved accuracy over decision trees.

## Decision Tree Model

We run the decision tree model as follows:

```r
model_DT <- train(classe ~., data=training, method="rpart", trControl=trainControl(method="cv", number=3))
# Use the trainControl function to select the cross-validation resampling method and to set the number of folds in the K-fold cross-validation
```

Next, we evalute the decision tree model on the test data:

```r
# Predict results using the testing data set
predictions_DT <- predict(model_DT, testing)
# Check confusion matrix
cm_DT <- confusionMatrix(predictions_DT, testing$classe)
cm_DT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1145  358  352  313  118
##          B   21  280   20  130  122
##          C   96  224  406  306  200
##          D    0    0    0    0    0
##          E    3    0    0    0  371
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4932          
##                  95% CI : (0.4784, 0.5079)
##     No Information Rate : 0.2833          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3391          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9051  0.32483  0.52185   0.0000  0.45746
## Specificity            0.6434  0.91868  0.77597   1.0000  0.99918
## Pos Pred Value         0.5009  0.48866  0.32955      NaN  0.99198
## Neg Pred Value         0.9449  0.85046  0.88494   0.8323  0.89245
## Prevalence             0.2833  0.19306  0.17424   0.1677  0.18163
## Detection Rate         0.2564  0.06271  0.09093   0.0000  0.08309
## Detection Prevalence   0.5120  0.12833  0.27592   0.0000  0.08376
## Balanced Accuracy      0.7743  0.62175  0.64891   0.5000  0.72832
```

## Random Forest

We run the random forest model as follows:

```r
# Create the model
model_RF <- train(classe~., data=training, method="rf", trControl=trainControl(method="cv", number=3))
# Use the trainControl function to select the cross-validation resampling method and to set the number of folds in the K-fold cross-validation
```
Next, we evaluate the radom forest moddel on the test data:

```r
# Predict the results using the testing data set
predictions_RF <- predict(model_RF, testing)
# Check confusion matrix
cm_RF <- confusionMatrix(predictions_RF, testing$classe)
cm_RF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1265    0    0    0    0
##          B    0  862    0    0    0
##          C    0    0  778    0    0
##          D    0    0    0  749    0
##          E    0    0    0    0  811
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9992, 1)
##     No Information Rate : 0.2833     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2833   0.1931   0.1742   0.1677   0.1816
## Detection Rate         0.2833   0.1931   0.1742   0.1677   0.1816
## Detection Prevalence   0.2833   0.1931   0.1742   0.1677   0.1816
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

## Gradient Boosting

Finally, we run the gradient boosting model:

```r
model_GB <- train(classe~., data=training, method="gbm", trControl=trainControl(method="cv", number=3), verbose=FALSE)
# Use the trainControl function to select the cross-validation resampling method and to set the number of folds in the K-fold cross-validation
```
Next, we evaluate the gradient boosting model:

```r
# Predict the results using the testing data set
predictions_GB <- predict(model_GB, testing)
# Check confusion matrix
cm_GB <- confusionMatrix(predictions_GB, testing$classe)
cm_GB
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1254   22    0    0    2
##          B    8  827   17    1    8
##          C    1   12  754   18    3
##          D    2    1    7  729    8
##          E    0    0    0    1  790
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9751          
##                  95% CI : (0.9701, 0.9795)
##     No Information Rate : 0.2833          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9686          
##  Mcnemar's Test P-Value : 0.0002092       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9913   0.9594   0.9692   0.9733   0.9741
## Specificity            0.9925   0.9906   0.9908   0.9952   0.9997
## Pos Pred Value         0.9812   0.9605   0.9569   0.9759   0.9987
## Neg Pred Value         0.9965   0.9903   0.9935   0.9946   0.9943
## Prevalence             0.2833   0.1931   0.1742   0.1677   0.1816
## Detection Rate         0.2809   0.1852   0.1689   0.1633   0.1769
## Detection Prevalence   0.2862   0.1928   0.1765   0.1673   0.1772
## Balanced Accuracy      0.9919   0.9750   0.9800   0.9842   0.9869
```

## Model Evaluation

Looking at the three models, we see the following for accuracy and out-of-sample error:

```
##            Model  Accuracy Accuracy.1
## 1  Decision Tree 0.4931691 0.50683091
## 2  Random Forest 1.0000000 0.00000000
## 3 Gradient Boost 0.9751400 0.02486002
```
The results show that the Random Forest model provides the best accuracy at 100%, with Gradient Boosting providing 98% accuracy and Decision Trees providing a rather poor 49% accuracy. 

Let's look at Random Forest model:

```r
model_RF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.66%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3622    2    2    0    1 0.001378550
## B   20 2445    4    0    0 0.009720535
## C    0    9 2207    9    0 0.008089888
## D    0    1   22 2067    1 0.011477762
## E    0    2    3    8 2332 0.005543710
```

```r
# Look at the importance of the predictors
varImp(model_RF)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt             100.00
## pitch_forearm          59.89
## yaw_belt               54.49
## pitch_belt             44.28
## roll_forearm           43.66
## magnet_dumbbell_y      43.57
## magnet_dumbbell_z      43.01
## accel_dumbbell_y       21.06
## accel_forearm_x        18.60
## roll_dumbbell          18.00
## magnet_dumbbell_x      16.92
## magnet_belt_z          16.34
## accel_dumbbell_z       15.10
## magnet_forearm_z       14.61
## accel_belt_z           13.73
## total_accel_dumbbell   13.46
## magnet_belt_y          12.12
## gyros_belt_z           10.96
## yaw_arm                10.74
## magnet_belt_x           9.95
```

# Conclusion

The 100% accuracy of the Random Forest model is somewhat troubling. Perhaps the data cleansing removed variables that would impact the decision tree and random forest models. The decision to remove the low variance variables and predominantly NA variables seems prudent, but this may have resulted in a very high accuracy for the random forest model.
