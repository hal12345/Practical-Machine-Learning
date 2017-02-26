# Prediction Assignment Writeup
Hal12345  
25/2/2017  



## Synopsys
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Citations
Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4ZiXJY7dM

## The setting

```r
library(ggplot2)
library(lattice)
library(caret)
library(rpart)
library(rattle)
```

```
## R session is headless; GTK+ not initialized.
```

```
## Rattle : une interface graphique gratuite pour l'exploration de données avec R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Entrez 'rattle()' pour secouer, faire vibrer, et faire défiler vos données.
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

## The question
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 


## The data

The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The dataset are supposed to have been dowloaded inside the working directory

A quick look at the dataset shows that there are a lot of values that are missing (NULL, NA, #DIV/0!). They will all be considered as NA values


```r
trads <- read.csv('pml-training.csv', na.strings=c('', '#DIV/0!', 'NA'), header=TRUE, sep=',')
tstds <- read.csv('pml-testing.csv', na.strings=c('', '#DIV/0!', 'NA'), header=TRUE, sep=',')
dim(trads)
```

```
## [1] 19622   160
```

```r
dim(tstds)
```

```
## [1]  20 160
```

### Cleansing

```r
#clensing

#get rid of all NA colums
trads<-Filter(function(trads)!all(is.na(trads)), trads)
tstds<-Filter(function(tstds)!all(is.na(tstds)), tstds)

#get rid of usless predictors
trads$X<-NULL
trads$user_name	<- NULL
trads$raw_timestamp_part_1 <- NULL	
trads$raw_timestamp_part_2 <- NULL	
trads$cvtd_timestamp <- NULL	
trads$new_window <- NULL
trads$num_window <- NULL

#keep only common complet columns in both training and testing datasets
commoncols<-intersect(colnames(trads),colnames((tstds)))
classe<-trads$classe
problem_id<-tstds$problem_id
trads<-trads[,commoncols]
tstds<-tstds[,commoncols]

trads$classe<-classe
tstds$problem_id<-problem_id

#keep only complete cases for the training set
trads<-trads[complete.cases(trads), ] 
dim(trads)
```

```
## [1] 19622    53
```

##Prediction

###Partitioning the data in training and testing sets


```r
intrads<-createDataPartition(y=trads$classe, p=0.60,list=FALSE)
traintrads <- trads[intrads,]
testtrads <- trads[-intrads,]
set.seed(999)
```

###Predicting model with Trees


```r
model1 <- rpart(classe ~ ., data=traintrads, method="class")
fancyRpartPlot(model1)
```

![](assignment_files/figure-html/tree-1.png)<!-- -->

```r
predict1 <- predict(model1, testtrads, type = "class")  
confusionMatrix(predict1, testtrads$classe) 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2014  260   88  157   38
##          B   95  902  143  131  268
##          C   63  181 1044  211  180
##          D   50  140   93  685   76
##          E   10   35    0  102  880
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7042          
##                  95% CI : (0.6939, 0.7143)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6237          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9023   0.5942   0.7632  0.53266   0.6103
## Specificity            0.9033   0.8993   0.9020  0.94527   0.9770
## Pos Pred Value         0.7876   0.5861   0.6218  0.65613   0.8569
## Neg Pred Value         0.9588   0.9023   0.9475  0.91164   0.9176
## Prevalence             0.2845   0.1935   0.1744  0.16391   0.1838
## Detection Rate         0.2567   0.1150   0.1331  0.08731   0.1122
## Detection Prevalence   0.3259   0.1962   0.2140  0.13306   0.1309
## Balanced Accuracy      0.9028   0.7468   0.8326  0.73897   0.7937
```
We notice that the accuracy is pretty poor 0.7075

###Predicting model with Bagging

```r
predictors <- traintrads
predictors$classe <- NULL

model2 <- bag(predictors, traintrads$classe, B = 10,
                bagControl = bagControl(fit = ctreeBag$fit,
                                        predict = ctreeBag$pred,
                                        aggregate = ctreeBag$aggregate))
```

```
## Warning: executing %dopar% sequentially: no parallel backend registered
```

```r
predict2 <- predict(model2, testtrads, type = "class")  
confusionMatrix(predict2, testtrads$classe) 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2177   57   11   10    6
##          B   29 1404   44   12   12
##          C    9   37 1295   43   16
##          D   14    8   15 1204   19
##          E    3   12    3   17 1389
## 
## Overall Statistics
##                                          
##                Accuracy : 0.952          
##                  95% CI : (0.947, 0.9566)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9392         
##  Mcnemar's Test P-Value : 0.0001293      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9754   0.9249   0.9466   0.9362   0.9632
## Specificity            0.9850   0.9847   0.9838   0.9915   0.9945
## Pos Pred Value         0.9628   0.9354   0.9250   0.9556   0.9754
## Neg Pred Value         0.9902   0.9820   0.9887   0.9875   0.9917
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2775   0.1789   0.1651   0.1535   0.1770
## Detection Prevalence   0.2882   0.1913   0.1784   0.1606   0.1815
## Balanced Accuracy      0.9802   0.9548   0.9652   0.9638   0.9789
```

The accuracy 0.9565 is much better that the one computed by Tree

###Predicting model with Random forest



```r
model3<-randomForest(formula = classe ~ ., data = traintrads, nPerm=5) 
predict3 <- predict(model3, testtrads, type = "class")  
confusionMatrix(predict3, testtrads$classe) 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2230   10    0    0    0
##          B    2 1502   12    0    0
##          C    0    6 1353   30    1
##          D    0    0    3 1255    8
##          E    0    0    0    1 1433
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9907          
##                  95% CI : (0.9883, 0.9927)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9882          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9895   0.9890   0.9759   0.9938
## Specificity            0.9982   0.9978   0.9943   0.9983   0.9998
## Pos Pred Value         0.9955   0.9908   0.9734   0.9913   0.9993
## Neg Pred Value         0.9996   0.9975   0.9977   0.9953   0.9986
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1914   0.1724   0.1600   0.1826
## Detection Prevalence   0.2855   0.1932   0.1772   0.1614   0.1828
## Balanced Accuracy      0.9987   0.9936   0.9917   0.9871   0.9968
```

With an accuracy of 0.994, Random forest is the winner

###Prediction model with the given test dataset using Random forest

```r
modelprediction<-predict(model3, newdata=tstds)
```

###Predicted classes with the selected model

```r
modelprediction
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```




