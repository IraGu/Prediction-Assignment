---
title: "Prediction Assignment Writeup"
author: "Ira Gu"
date: "June 9, 2018"
output: 
  html_document: 
    keep_md: yes
---
# Background and Objective

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we will be to useing data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).
All data were also taken from the website. The goal is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.

### Cleaning the Data
We read the data and remove any columns without values, then we remove any variables that are unrelated to movement since measurements in movement (gyroscope, magnetometer, acceleration) is the best indicator on whether an action (in this case a dumbell curl) is done correctly or not. We also removed total acceleration since x,y,z acceleration is already captured. This may speed up the training model while not affecting accuracy. 

```r
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",na.strings = c("","#DIV0!","NA"))
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",na.strings = c("","#DIV0!","NA"))
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
trainr <- training %>% select_if(~ !any(is.na(.)))
testr <- testing %>% select_if(~ !any(is.na(.)))
trainr<- trainr[, -grep("total", names(trainr))]
testr<- testr[, -grep("total", names(testr))]
trainr <- trainr[,8:length(trainr)]
testr <- testr[,8:length(testr)]
dim(trainr)
```

```
## [1] 19622    49
```

```r
dim(testr)
```

```
## [1] 20 49
```

```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

### Cross validation 
We split the data  into 2 training sets, testing and training, we will train the model on the training set and then only run the final model on the testing set once. We will be using caret to partition the data and train the models. We expect the out of sample error to be larger than the in sample error because we did not use any of the new data (test set) to train our models.This means its likely that we did not overfit our model. Small training set is used due to speed. 

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
inTrain <- createDataPartition(trainr$classe, p = 0.5, list = FALSE)
trainp <- trainr[inTrain,]
testp <- trainr[-inTrain,]
dim(trainp)
```

```
## [1] 9812   49
```

```r
dim(testp)
```

```
## [1] 9810   49
```

### Running the training Random Forest Model (rf)

```r
mod1 <- train(classe ~., method ="rf", data = trainp, trControl = trainControl(method = "cv"), number = 3)
pred <- predict(mod1, newdata = testp)
confusionMatrix(pred, testp$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2784   11    0    0    0
##          B    4 1873    6    0    1
##          C    2    6 1693   22    2
##          D    0    8   12 1584   10
##          E    0    0    0    2 1790
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9912         
##                  95% CI : (0.9892, 0.993)
##     No Information Rate : 0.2844         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9889         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9868   0.9895   0.9851   0.9928
## Specificity            0.9984   0.9986   0.9960   0.9963   0.9998
## Pos Pred Value         0.9961   0.9942   0.9814   0.9814   0.9989
## Neg Pred Value         0.9991   0.9968   0.9978   0.9971   0.9984
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2838   0.1909   0.1726   0.1615   0.1825
## Detection Prevalence   0.2849   0.1920   0.1758   0.1645   0.1827
## Balanced Accuracy      0.9981   0.9927   0.9928   0.9907   0.9963
```

### Running the Boosted Trees/Gradient Boosted Machine (gbm)

```r
mod2 <- train(classe~. , data = trainp, method = "gbm", verbose = FALSE)
pred <- predict(mod1, newdata = testp)
confusionMatrix(pred, testp$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2784   11    0    0    0
##          B    4 1873    6    0    1
##          C    2    6 1693   22    2
##          D    0    8   12 1584   10
##          E    0    0    0    2 1790
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9912         
##                  95% CI : (0.9892, 0.993)
##     No Information Rate : 0.2844         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9889         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9868   0.9895   0.9851   0.9928
## Specificity            0.9984   0.9986   0.9960   0.9963   0.9998
## Pos Pred Value         0.9961   0.9942   0.9814   0.9814   0.9989
## Neg Pred Value         0.9991   0.9968   0.9978   0.9971   0.9984
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2838   0.1909   0.1726   0.1615   0.1825
## Detection Prevalence   0.2849   0.1920   0.1758   0.1645   0.1827
## Balanced Accuracy      0.9981   0.9927   0.9928   0.9907   0.9963
```

### Conclusion
Looking at the results we can see that the performances of both 2 models are highly accurate. We will use gbm since it was slightly faster to calculate.

We will use the gbm model for our final prediction.

```r
predict(mod2, newdata=testr)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
