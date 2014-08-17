## Objective
Find a regression model that predicts whether a dumbbell lifting exercise was performed correctly or which of the four mistake types was encountered.

## Analysis
We load the data set and split it in into training and testing sets with 60% of observations going to the training set.

```r
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
file <- "pml-training.csv"; download.file(fileUrl,destfile=file,method="curl")
data <- read.csv(file)

library(caret)
inTrain <- createDataPartition(y=data$classe, p=0.6, list=FALSE)
training <- data[inTrain,]; testing <- data[-inTrain,]
```

Since the data comes from accelerometers, we will first attempt to build a model with acceleration variables from belt, arm, forearm and dumbbell as features.  We will omit the other 139 variables for now and will only include some of them in case the model has low accuracy or is overfitted.


```r
accel <- grep("accel", colnames(training), ignore.case = FALSE, fixed=TRUE)
keep_cols <- c(accel, 160); training <- training[keep_cols]
```

The 20-item set for which we need to predict the classe outcome does not have any values in the variance column, so we will remove the variable from the training set.


```r
remove <- grep("^var", colnames(training)); training <- training[,-remove]  
```

We now build a Random Forest model with the training set using the remaining 16 acceleration-related features.  Please note that it will take about five minutes to build the model on this training set that has 16 predictors and over 11,000 observations.


```r
set.seed(11112)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
model <- randomForest(classe ~ ., data=training, proximity=TRUE, na.action=na.omit, keep.forest=TRUE); model
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, proximity = TRUE,      keep.forest = TRUE, na.action = na.omit) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 4
## 
##         OOB estimate of  error rate: 6%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3196   24   52   73    3     0.04540
## B   96 2072   71   23   17     0.09083
## C   39   53 1942   16    4     0.05453
## D   39   11   81 1786   13     0.07461
## E    4   37   24   26 2074     0.04203
```

```r
plot(model, type="l", main="Random Forest Error Rate")
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4.png) 

Examining the Random Forest model, we see that the out of bag error rate is around 6%.  This is a low out of sample error rate, but will attempt to get it down with other model types.  The confusion matrix suggests that we have a model that identifies the classe class fairly accurately.  It also shows that the model is not overfitted as there is between 3% and 10% misclassification rate.  From the graph, we can also see that 100 trees is enough to reach a stable error rate and that the accuracy of the model would not benefit from growing the forrest to more than 100 trees.

We will now build other models and compare them with the random forest model.


```r
library(rpart.plot)
```

```
## Loading required package: rpart
```

```r
#modtb <- train(classe ~.,data=training, method="treebag")
modrp <- train(classe ~ .,method="rpart",data=training)
modlda = train(classe ~ .,data=training,method="lda")
```

```
## Loading required package: MASS
```

```r
#modnb = train(classe ~ ., data=training,method="nb")
```

The Treebag and Naive Bayes methods take too long to complete, so we will not use them due to not having enough computing resources. 


```r
moda <- rbind(modrp$results[1, "Accuracy"], modlda$results["Accuracy"])
modk <- rbind(modrp$results[1, "Kappa"], modlda$results["Kappa"])
modc <- cbind(moda, modk); modc$Model <- c("Rpart", "LDA"); modc
```

```
##   Accuracy  Kappa Model
## 1   0.4453 0.2869 Rpart
## 2   0.5099 0.3725   LDA
```

Recursive Partitioning (rpart) and LDA methods produce models that are much less accurate than the Random Forest model.  Rpart has the highest accuracy of 42% and LDA has 51%.  With 94% accuracy, Random Forest is the best fit model.

We now run the model on the testing set and evaluate its accuracy.


```r
pred <- predict(model,testing); testing$predRight <- pred==testing$classe
table(pred,testing$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 2192   26    5   13    1
##    B    9 1471   17    2   15
##    C   15   15 1335   14    1
##    D   14    4   10 1252   10
##    E    2    2    1    5 1415
```

The model predicts the outcome correctly in more than 90% of the cases.  We will now predict the result of the 20-item validation test and will submit it for grading.


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
file <- "pml-testing.csv"; download.file(fileUrl,destfile=file,method="curl")
validation <- read.csv(file)
answers <- predict(model,validation)
pml_write_files(answers)
```

19 of 20 answers predicted by the Random Forest model were correct!

## Conclusion
Using a Random Forest method on 16 acceleration-related variables, we can create a model that has over 90% accuracy rate of detecting whether a dumbbell was lifted correctly and, if it was not, what type of five known mistakes was made.  The model is not intelligent enough to catch mistakes of types other than the five captured in the data set. Neither can the model determine whether more than one mistake was made (for example, accellerating both belt and arm too fast).  We also do not know how well the model scales beyond the original set of 19,000 observations.  However, for data sets under 20,000 observations where each observation has zero or one exercise mistakes, the model is 94% accurate predicting whether the exercise was performed correctly or which of the four known mistake types occured.

