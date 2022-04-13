
# This is a baseline script for implementation of xgboost binary classification model

# Note modify data import to read column names, they currently require manual input.


#Clear out all global variables.
# rm(list = ls())


# set working directory
# path <- "/Users/toddhoffman/Documents/R_Code"
# setwd(path)


#load all the required libraries
require(xgboost)
library(data.table)
library(mlr)
library(ggplot2)
library(caret)
library(e1071)
library(DiagrammeR)
library(pROC)
require(randomForest)

#set variable names, as test input file does not have header.
setcol <- c("age", "workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","target")


#load data, update with file names for each specific project.
#Note: this assumes training and test sets we're split in SQL.
train <- read.table("adultdata.txt", header = F, sep = ",", col.names = setcol, na.strings = c(" ?"), stringsAsFactors = F,fill = TRUE)
test <- read.table("adulttest.txt",header = F,sep = ",",col.names = setcol,skip = 1, na.strings = c(" ?"),stringsAsFactors = F, fill = TRUE)

#Set table for more efficient usage of RAM.
setDT(train) 
setDT(test)



#Clean up data. This code is specific to this test dataset only.

#check missing values 
table(is.na(train))
sapply(train, function(x) sum(is.na(x))/length(x))*100

table(is.na(test))
sapply(test, function(x) sum(is.na(x))/length(x))*100

#print(train)
#print(test)

#remove extra character from target variable
library(stringr)
test [,target := substr(target,start = 1,stop = nchar(target)-1)]
print(test)

#remove leading whitespaces
char_col <- colnames(train)[ sapply (test,is.character)]
for(i in char_col) set(train,j=i,value = str_trim(train[[i]],side = "left"))
for(i in char_col) set(test,j=i,value = str_trim(test[[i]],side = "left"))

#set all missing value as "Missing" 
train[is.na(train)] <- "Missing" 
test[is.na(test)] <- "Missing"

#print(train)
#print(test)


#Transform data

#Define label names
labels_train <- train$target 
labels_test <- test$target


#Create new training and test set, with one hot ecoding
#NOTE: This only transforms categorical variables, leaves continuous as continuous (numeric).
new_tr <- model.matrix(~.+0,data = train[,-c("target"),with=F]) #not quite sure what the selections are doing here
new_ts <- model.matrix(~.+0,data = test[,-c("target"),with=F])


#Output sample of file to check format.
#write.csv (new_tr, file="new_tr_02.csv", row.names = TRUE)





#Convert target values to binary, needed for this dataset only (as you can't have both char and num in a single field).
labels_train <-gsub(">50K", 1, labels_train) #gsub is global substitution function.
labels_train <-gsub("<=50K", 0, labels_train)

labels_test <-gsub(">50K", 1, labels_test)
labels_test <-gsub("<=50K", 0, labels_test)




#convert factor to numeric 
labels_train <- as.character(labels_train) #Needs to be converted to char before converting to numeric - quirky R thing.
labels_train <- as.numeric(labels_train) #Needs to be converted to numeric before it can be transformed into xgb.DMatrix - quirky R thing.

labels_test <- as.character(labels_test) 
labels_test <- as.numeric(labels_test)

#Create dtrain and dtest datasets as xgb.DMatrix objects, this is the object type (all numeric) required for model input.
dtrain <- xgb.DMatrix(data = new_tr,label = labels_train) 
dtest <- xgb.DMatrix(data = new_ts,label = labels_test)


#Modelling

#Train models using simple method xgboost function, with manual parameters.
xgb_1 <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 10, objective = "binary:logistic", verbose = 2)


#Train model using advanced method xgb.train
#Note that this method (watchlist) references the test set when training the model to mitigate overfitting.
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
watchlist <- list(train=dtrain, test=dtest)
xgb_2 <- xgb.train (params = params, data = dtrain, nrounds = 79, watchlist = list(val=dtest,train=dtrain), print_every_n = 10, early_stop_round = 10, maximize = F , eval_metric = "error")


# Determines best round for model (xgb.cv function)
#xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stop_round = 20, maximize = F)
#min(xgbcv$test.error.mean)


#Scores test data set with for each model.
pred_xgb_1 <- predict (xgb_1,dtest)
pred_xgb_2 <- predict (xgb_2,dtest)


#Calculate area under ROC curve (AUC)
auc(labels_test,pred_xgb_1)
auc(labels_test,pred_xgb_2)


#Calculate average error of models.
err_pred_xgb_1 <- mean(as.numeric(pred_xgb_1 >.5)!=labels_test)
print(paste("test-error pred_xgb_1=", err_pred_xgb_1))

err_pred_xgb_2 <- mean(as.numeric(pred_xgb_2 >.5)!=labels_test)
print(paste("test-error pred_xgb_2=", err_pred_xgb_2))


#Remove any old result sets.
#rm(test_scored_comb)


#Creates dataframe to capture actual and predicted values (probabilties) from test sets, outputs as file. 
test_scored_comb <- data.frame(labels_test, pred_xgb_1, pred_xgb_2)
write.csv (test_scored_comb, file="test_scored_comb.csv", row.names = TRUE)


#ROC and AUC Plot
#NOTE Do not apply threshold cut off and convert predicted probabilities to integers before running creating a ROC
#NOTE The above code converts model 1 to binary, ROC is run on model 2 which has NOT YET BEEN cut off.
#NOTE 20171229 Disregard the above notes, currently trying to rewrite this to create ROC curves from a separate dataset.

#plot(roc(test_set$bad_widget, glm_response_scores, direction="<"),
#     col="yellow", lwd=3, main="The turtle finds its way")

roc1 <- roc(test_scored_comb$labels_test,
            test_scored_comb$pred_xgb_1, percent=TRUE,
            
            # arguments for auc
            #partial.auc=c(100, 90), partial.auc.correct=TRUE,
            #partial.auc.focus="sens",
            
            # arguments for ci
            #ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
            
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE, main="Insert Title")


# Add ROC Curve for second model
roc2 <- roc(test_scored_comb$labels_test,
            test_scored_comb$pred_xgb_2,
            plot=TRUE,
            col="blue",add=TRUE, percent=roc1$percent)


#Text version of tree plot
#xgb.dump(xgb_1, with_stats = T)


#Converts predicted probabilities to binary based on .5 cut off value.
pred_xgb_1 <- ifelse (pred_xgb_1 > 0.5,1,0)
pred_xgb_2 <- ifelse (pred_xgb_2 > 0.5,1,0)

#Move above code to here, to avoid overwriting predicted probabilities.
#confusion matrix
confusionMatrix (pred_xgb_1, labels_test)
confusionMatrix (pred_xgb_2, labels_test)


#Feature importance plot
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb_1)
xgb.plot.importance (importance_matrix = mat[1:20])


#Tree plot
#xgb.plot.tree(model = bstDMatrix)
