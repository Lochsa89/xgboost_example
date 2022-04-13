# Todd Hoffman - Kochava Test Exercise for Churn Prediction.
# Recommended to run using R Studio.
# Copyright Todd Hoffman 2020

# Overview:
# This script allows multiple versions of a training set to be modelled, which is
# useful for understanding the impact of holding out, or adding in, specific predictor variables.
# The specifc predictor variables used in each training set version are automatically logged in 
# the vars dataframe.

# The script also performs brute force hyper parameter optimization. While Packages such as Caret
# can perform hyperparametr optimization in a more automated way, unfortunately, Caret doesn't 
# currenntly support all available XGBoost hyperparamters. The script also calculates the performance 
# details of each model iteration / hyper parameter set, which is useful for logging and comparing peformance.

# Performance measures are output in the plots window, scroll through to see performance of 
# each training set version and algorithm tested.

# Load required libraries (NOTE: You may need to install them).
require(woeBinning)
require(xgboost)
library(data.table)
library(pROC)
require(randomForest)
library(caret)
library(ROSE)
library(gains)
library(sqldf)


# Clean out all variables (if needed). 
# Not necessarily a best practice, but used here to avoid mixing with other projects.
rm(list = ls())


# Set working directory (changed to your own directory to run).
setwd("/Users/thoffman/Documents/Personal/Kochava/")


# 1.00 This section loads and prepares data.

# 1.01 Read train .csv files
churn_demo_test <- read.csv("churn_demo_test.csv",header=TRUE)
churn_metric_test <- read.csv("churn_metric_test.csv",header=TRUE)
churn_metric_train <- read.csv("churn_metric_train.csv",header=TRUE)
churn_demo_train <- read.csv("churn_demo_train.csv",header=TRUE)


# 1.02 join the two data sets into a single data frame, add a few features.
churn_train <- sqldf('
  select tr_m.*
    , length(tr_d.name) as name_len
    , tr_d.sex
    , tr_d.age
    , leftstr(tr_m.CategoryID,2) || "-" || length(tr_m.CategoryID) as cat_l2_len

  from churn_metric_train as tr_m
    join churn_demo_train as tr_d
      on tr_m.DeviceID = tr_d.DeviceID
  ')


churn_test <- sqldf('
  select 
    te_m.*         
    , length(te_d.name) as name_len
    , te_d.sex
    , te_d.age
    , leftstr(te_m.CategoryID,2) || "-" || length(te_m.CategoryID) as cat_l2_len
                     
    from churn_metric_test as te_m
      join churn_demo_test as te_d
        on te_m.DeviceID = te_d.DeviceID
    ')


# 1.03 Change a character features to factor (needed to run WOE Binning).
churn_train <- as.data.frame(unclass(churn_train))
churn_test <- as.data.frame(unclass(churn_test))


# 1.04 Double check structure of dataframes.
str(churn_train)
str(churn_test)


# 1.05 Run Weight of Evidence Binning (WOE) function (creates optimized bins for numeric and categorical variables).
# More information on WOE here https://cran.r-project.org/web/packages/woeBinning/woeBinning.pdf

bin_test <- woe.binning(churn_train
            , "Churn"
            , churn_train
            , min.perc.total=0.01
            , min.perc.class=0.01
            , stop.limit=0.01
            , abbrev.fact.levels=0
            , event.class=1
            )

# 1.06 Uncomment this to see individual WOE measures (WARNING: lots of messy pop outs get generated).
#woe.binning.plot(bin_test,multiple.plots = TRUE)
#woe.binning.table(bin_test)


# 1.07 Apply WOE bins to train and test dataframes.
churn_train <- woe.binning.deploy(churn_train,bin_test)
churn_test <- woe.binning.deploy(churn_test,bin_test)


#1.08 Create output file for additional EDA and visualizationn w/Tableau.
# I have found it faster to do visual, exploratory EDA in Tableau. Plus, it makes
# it easier to crowd source interacrtive EDA to other less technical users.
#write.csv(churn_train,'churn_train.csv')


#1.09 Assign variables generic name to work with generic portions of script.
raw_data <- churn_train
raw_data_test <- churn_test


# 1.10 Specify the names of the training set versions to model here.
tr_name <- data.frame(model=c("tr_1","tr_2"))


# 1.11 List of colors to use for each training set/algorithm iteration. Used in performance measures plots.
col_scheme <- c("blue3","green3","yellow3","red3","darkmagenta") 


# 1.12 Counts the number of training sets to know how many times to run loop.
tr_count <- nrow(tr_name)


# 1.13 Specify subsets of predictor variables to use in training sets.
# NOTE: List the variables to EXCLUDE from each training set.

tr_1 <- c(
   "CategoryID"
   , "DeviceID"
   , "DeviceID.binnned"

)

# Features removed here produced overfitting in first iteration of model.
# Interesting that there was a strong correlation between name length and churn rate.
# Automated binning of category based feature also seemed to produce 
# poorly calibrated models for some features, as determined using 
# figure 4). Calibration Plot. For more info on Calibration Plots I recommend reading:
# Kuhn, M., & Johnson, K. (2013). Applied predictive modeling. New York: Springer
# In my experience a well calibrated model generalizes best.

tr_2 <- c(
  "CategoryID"
  , "DeviceID"
  , "DeviceID.binned"
  , "AttributionSite"
  , "AttributionNetwork"
  , "AttributionSite.binned"
  , "Age"
  , "Spend"
  , "Sex"
  , "QualityScore.binned"
  , "name_len"
  , "name_len.binned"
  , "cat_l2_len"
  , "cat_l2_len.binned"
  , "CategoryID.binned"
  , "QualityScore"
  , "Event1Count"
  , "Event2Count"
)




# 2.0 Start of i loop. Sets variable names to current training set details, splits training/test, oversamples.


# 2.01 Start the loop 
for (i in 1:nrow(tr_name)){


# 2.02 Set the time stamp for this model iteration (time stamp gets appended to model name and logged).
mod_date <- Sys.time()
  

# 2.03 Sets the current training set key details.
tr_cur_set <- get(as.character(tr_name[i,]))
tr_cur_name <- tr_name[i,]
tr_cur_col <- col_scheme[i]


# 2.04 Creates base_data for current iteration by removing unneeded columns.
base_data <- raw_data[,!colnames(raw_data) %in% tr_cur_set]


# 2.05 Creates a data frame that captures the variables used in the model being trained. 
# Useful for documenting which variables were used in which version of model, etc.
# Especially useful in case model perofrmance "breaks" in the experimentation process.

if (i==1) {
    # On first iteration of loop, create vars dataframe  
    vars <- data.frame(vars = names(base_data)
                     , model = tr_cur_name
                     , mod_date) 
  } else 
  
  {
    # On subsequent iterations of loop, append new records to existing vars dataframe.
    vars <- rbind(vars, data.frame(vars = names(base_data)
                             , model = tr_cur_name
                             , mod_date))
  }


# 2.06 Calculates 75% of the sample size.
smp_size <- floor(0.75 * nrow(base_data))


# 2.07  Sets the seed to make the partition reproducible.
set.seed(123)


# 2.08 Creates an random index used to split training versus test.
train_ind <- sample(seq_len(nrow(base_data)), size = smp_size)


# 2.09 Splits the data set based on the random index from above.
base_train <- base_data[train_ind, ]
test <- base_data[-train_ind, ]


# 2.10 Checks the resulting numbers of rows after the split.
nrow(base_train)
nrow(test)


# 2.11 Oversamples postive cases to roughly 50/50 prevalence.
train <- ovun.sample(Churn~.
                     , data = base_train
                     , p = 0.5
                     #, N = nrow(train)
                     , seed = 1
                     , method = "over")$data


# 2.11 Checks total number of records in training set, and totals by class after over sampling.
nrow(train)
sqldf("select Churn, count (*)
      from train
      group by Churn")
  

# 2.12 Uses set table operation for more efficient usage of RAM.
setDT(train) 
setDT(test)


# 2.13 Sets label names.
labels_train <- train$Churn
labels_test <- test$Churn


# 2.14 Converts N/A's to zeros, needed or N/A rows get truncated when converting to model.matrix.
options(na.action="na.pass")


# 2.15 Converts dataframe to matrix.
# This matrix gets used below to create a xgb input objects (dtrain and dtest)
new_tr <- model.matrix(~.+0,data = train[,-c("Churn"),],with=F)
new_ts <- model.matrix(~.+0,data = test[,-c("Churn"),],with=F)


# 2.16 Check that number of rows match in input and output data after conversion to matrix.
nrow(new_tr)
nrow(train)
nrow(new_ts)
nrow(test)


#Output sample of file to test in other tools if needed.
#write.csv (new_tr, file="new_tr_02.csv", row.names = TRUE)


# 2.17 Convert factor to numeric 
labels_train <- as.character(labels_train) #Needs to be converted to char before converting to numeric - quirky R thing.
labels_train <- as.numeric(labels_train) #Needs to be converted to numeric before it can be transformed into xgb.DMatrix - quirky R thing.
labels_test <- as.character(labels_test) 
labels_test <- as.numeric(labels_test)


# 2.18 Create dtrain and dtest datasets as xgb.DMatrix objects, this is the object type (all numeric) required for model input.
dtrain <- xgb.DMatrix(data = new_tr, label = labels_train)
dtest <- xgb.DMatrix(data = new_ts,label = labels_test)


# 2.19 Define hyperparameters ranges used to train model annd to optimize using method xgb.train.
hyper_grid <- expand.grid(booster = "gbtree" 
                        , objective = "binary:logistic"
                        , scale_neg_weight = 1
                        , eta=c(0.01,0.1,0.3)
                        , gamma=0
                        , max_depth=c(5,10,15)
                        , min_child_weight=1
                        , subsample=1
                        , nrounds=25
                        , eval_metric="logloss"
                        , colsample_bytree=1
                        , optimal_trees = 0
                        , min_val_logloss = 0
                        )


# 3.00 Train model iterating through each hyper parameter combination for XGBoost Models.

# 3.01 Start of loop, iterates through values in hyperparameter grid for XGBoost Models.
for (j in 1:nrow(hyper_grid)) {


# 3.02 Grab the current combination of hyper parameters to use.    
xgb_grid_cur <- list(
      booster = as.character(hyper_grid$booster[j]),###### Change this to as.character(). To fix bug in predict() at section 5.01
      eta = hyper_grid$eta[j],
      max_depth = hyper_grid$max_depth[j],
      min_child_weight = hyper_grid$min_child_weight[j],
      subsample = hyper_grid$subsample[j],
      colsample_bytree = hyper_grid$colsample_bytree[j]
      )  


# 3.03 Grab the current round number and eval metric.  
xgb_grid_nrounds_cur <- hyper_grid$nrounds[j]
xgb_grid_eval_metric_cur <- hyper_grid$eval_metric[j]


# 3.04 Print label for each round: name, iteration number, eta, depth, eval metric.
print(paste("model="
                , tr_cur_name
                , ", iteration="
                , j
                ," of "
                , nrow(hyper_grid)
                , ", eta="
                , xgb_grid_cur$eta
                , ", max_depth="
                , xgb_grid_cur$max_depth
                ,", eval_metric="
                , xgb_grid_eval_metric_cur
                , sep = ""))

        
# 3.05 Train model with current parameter set.
mod_obj <- xgb.train (params = xgb_grid_cur
                    , data = dtrain
                    , nrounds = xgb_grid_nrounds_cur
                    , watchlist = list(val=dtest,train=dtrain)
                    , print_every_n = 1
                    , early_stop_round = 3
                    , maximize = F
                    , eval_metric = xgb_grid_eval_metric_cur)


# 3.06 Update hyper_grid dataframe with best performing iteration (number of trees and value).
hyper_grid$optimal_trees[j] <- which.min(mod_obj$evaluation_log$val_logloss)
hyper_grid$min_val_logloss[j] <- min(mod_obj$evaluation_log$val_logloss)


# End of j loop.
########
}




# 4.00 Train the optimal model for the current training set.

# 4.01 Create opt_mod dataframe to log best model for each data set and parameter combination tested. This is used later.
if (i==1){
  opt_mod <-  data.frame(model_name = tr_cur_name, algorithm ="xgboost", model_obj = paste(tr_cur_name,"_mod_obj", sep=""), model_date = mod_date,hyper_grid[which.min(hyper_grid$min_val_logloss),])
}else
{
  opt_mod <- rbind(opt_mod, data.frame(model_name = tr_cur_name, algorithm ="xgboost", model_obj = paste(tr_cur_name,"_mod_obj", sep=""), model_date = mod_date,hyper_grid[which.min(hyper_grid$min_val_logloss),]))
}


# 4.02 Get the row of the very best model for this training set.
opt_row <- opt_mod[opt_mod$model_name == tr_cur_name,]


# 4.03 Create a parameter set based on the best model for this training set.
opt_params <- list(
  booster = as.character(opt_row$booster),
  objective = opt_row$objective,
  scale_neg_weight = opt_row$scale_neg_weight,
  eta = opt_row$eta,
  gamma = opt_row$gamma,
  max_depth = opt_row$max_depth,
  min_child_weight = opt_row$min_child_weight,
  subsample = opt_row$subsample,
  colsample_bytree = opt_row$colsample_bytree,
  optimal_trees = opt_row$optimal_trees,
  min_val_logloss = opt_row$min_val_logloss
)


# 4.04 Print label for the best fit model for the training set.
print(paste("Model:",tr_cur_name," - Best Fit",sep = ""))


# 4.05 Train final model using optimal parameters.
opt_mod_obj <- xgb.train (params = opt_params
                      , data = dtrain
                      , nrounds = opt_mod[opt_mod$model_name == tr_cur_name,"optimal_trees"]
                      , watchlist = list(val=dtest,train=dtrain)
                      , print_every_n = 1
                      , early_stop_round = 3
                      , maximize = F
                      , eval_metric = "logloss")

opt_mod_obj$params$booster
mod_obj$params$booster

# Note to self: Keeping this here for now for QA purposes, makes sure opt rounds is pulling correctly.
opt_trees_cur_mod <- opt_mod[opt_mod$model_name == tr_cur_name,"optimal_trees"]


# 4.06 Create new dynamic variable with the current (optimal) model name for the training set, assign model to it.
# Some people might say dynamic variables are a bad practice, but in this case it works well.
assign(paste(tr_cur_name,"_opt_mod_obj",sep=""), get("opt_mod_obj"))



# 5.00 Score test data using the optimal model.

# 5.01 Scores test data set, using optimal model, for current training set. 
# NOTE: Two versions of predictions are available, one with SHAP and one without as
# SHAP runs very slow with very large data sets.
pred <- predict(opt_mod_obj,dtest)
pred_shap <- predict(mod_obj,dtest, predcontrib = TRUE)


# TEST predict with SHAP values generated, approxcontrib parameter added.
#pred_shap <- predict (opt_mod_obj,dtest, predcontrib = TRUE, approxcontrib = F)

# 5.02 Create a variable dynamically, based on current optimal model name + _pred, assign predictions to it.
assign(paste(tr_cur_name,"_pred", sep=""), pred)


# 5.03 Calculate average error of models.
#err_pred <- mean(as.numeric(pred >.5)!=labels_test)


# 5.04 Create a combined data set of scored test set data.
if (i==1) 
  {
  test_scored_comb <- data.frame(name_key = paste(tr_cur_name,"xgb", mod_date, sep = "-"), model = tr_cur_name, algorithm  = "xgb", mod_date, labels_test, pred, class_pred = ifelse (pred > 0.5,1,0) )
  } else 
  {
  test_scored_comb <- rbind(test_scored_comb
                            , data.frame(name_key = paste(tr_cur_name,"xgb", mod_date, sep = "-"), model = tr_cur_name,  algorithm  = "xgb",mod_date, labels_test, pred, class_pred = ifelse (pred > 0.5,1,0) )
                            )
  }


# Random Forest model.
rf_model <- randomForest(x=na.roughfix(new_tr),y=as.factor(labels_train),ntree=10, maxnodes=4)



rf_prob <- data.frame(predict(rf_model,na.roughfix(new_ts),type = "prob"))


rf_scores <- data.frame( name_key = paste(tr_cur_name,"rf",mod_date,sep="-")
                        , model = tr_cur_name
                        , algorithm = "rf"
                        , mod_date = mod_date
                        , labels_test
                        , pred = rf_prob$X1
                        , class_pred = ifelse (rf_prob$X1 > 0.5,1,0)
                        )

test_scored_comb <- rbind(test_scored_comb,rf_scores)


}
# End of loop i
#############


#### TO DO: Add a random forest model here 

# Add Random Forest Model(s)

# 6.0 Model performance measures.





# 6.01 Create list of unique models that have been fit so far
tbl_name_keys <- sqldf("select name_key, model, algorithm, mod_date from test_scored_comb group by name_key")
 
for (k in 1:nrow(tbl_name_keys)){

tr_cur_name_key <- tbl_name_keys[k,"name_key"]



class_pred <- test_scored_comb[which(test_scored_comb$name_key == tr_cur_name_key ),"class_pred"]
labels_test <- test_scored_comb[which(test_scored_comb$name_key == tr_cur_name_key ),"labels_test"]
mod_date <- tbl_name_keys[k,"mod_date"]
pred <- test_scored_comb[which(test_scored_comb$name_key == tr_cur_name_key  ),"pred"]
tr_cur_col <- col_scheme[k]
algorithm <- tbl_name_keys[k,"algorithm"]

# 6.02 Run ROC on current opt model, extract AUC score from ROC object, append to measures object
auc_opt_mod <- roc(labels_test,class_pred)
auc_label <- "AUC"
measures_opt <- data.frame(model = tr_cur_name_key, model_date=mod_date, measure = auc_label, value = as.numeric(auc_opt_mod$auc))

if (k==1){
  measures <- measures_opt
}else{
  measures <- rbind(measures, measures_opt)
}


# 6.03 Create confusion matrices for current opt model, extract contingency table, create contingency table data frame.
conf_mat_opt <- confusionMatrix(table(class_pred, labels_test), positive="1") 
conf_mat_opt_ct <- data.frame(model = tr_cur_name_key, model_date = mod_date
                        , true_neg = conf_mat_opt$table[1,1]
                        , true_pos = conf_mat_opt$table[2,2]
                        , false_neg = conf_mat_opt$table[1,2]
                        , false_pos = conf_mat_opt$table[2,1]
                        )


if (k==1){
  conf_mat_ct <-conf_mat_opt_ct
}else{
  conf_mat_ct <- rbind(conf_mat_ct, conf_mat_opt_ct)
}



# 6.04 Extract and combine byClass measures from confusion matrix for current opt model.
conf_mat_opt_byClass <- data.frame(model = tr_cur_name_key, 
                                   model_date = mod_date, 
                                   measure = row.names(data.frame(conf_mat_opt$byClass)),
                                   value = conf_mat_opt$byClass,
                                   row.names = NULL)


# 6.05 Append confusion matrix by class measures to measures object.
measures <- rbind(measures,conf_mat_opt_byClass)


# 6.06 Extract overall measures from confusion matrix.
conf_mat_opt_overall <- data.frame(model = tr_cur_name_key, 
                                   model_date = mod_date, 
                                   measure = row.names(data.frame(conf_mat_opt$overall)),
                                   value = conf_mat_opt$overall,
                                   row.names = NULL)


# 6.07 Append overall measures to measures object
measures <- rbind(measures,conf_mat_opt_overall)




# 7.00 Calibration and Gains Analysis

# 7.01 Set the matrix layout used to arrange the various plots.
layout(test <- matrix(c(1,2,3, 1,2,3, 4,5,6,  4,5,6), nrow = 4, ncol = 3, byrow = TRUE))


# 7.02 Figure 1). Histograms of predicted probabilities. Note: par margin parameters (B,L,T,R)
par(#mfrow=c(2,3),
    mar=c(5,3,3,3)) 


hist(test_scored_comb[test_scored_comb$name_key == tr_cur_name_key,"pred"]
     , breaks=20
     , col=tr_cur_col
     , main=paste("1). Freq: ",tr_cur_name_key, sep="")
     , xlab="Predicted Probability"
     , ylim=c(1,150))


# 7.03 Run gains function for each model.
gains_opt_obj <-gains(actual=labels_test, predicted=pred, groups=10,
                    ties.method=c("max","min","first","average","random"),
                    conf=c("none","normal","t","boot"), boot.reps=1000, conf.level=0.95,
                    optimal=TRUE,percents=TRUE)



# 7.04 Perform some additional calcs need to create custom cumulative gains chart.
gains_opt <- data.frame(model = tr_cur_name_key, model_date = mod_date,  gains_opt_obj[1:11])
gains_opt$pct_cum_obs <- gains_opt$cume.obs / max(gains_opt$cume.obs)
gains_opt$pos_cases <- min(gains_opt$cume.mean.resp)*max(gains_opt$cume.obs)
gains_opt$base_pos_rate <-min(gains_opt$cume.mean.resp)
gains_opt$cum_pct_pos_cases_rand <- (gains_opt$cume.obs * gains_opt$base_pos_rate) / gains_opt$pos_cases
gains_opt$dif <- ((gains_opt$obs * gains_opt$mean.prediction)-(gains_opt$obs * gains_opt$mean.resp))/(gains_opt$obs * gains_opt$mean.prediction)
gains_opt$lift <- gains_opt$cume.pct.of.total / gains_opt$cum_pct_pos_cases_rand
gains_opt <- gains_opt[,c(1:5,14,6,11,8,17,18,9)]


# 7.05 Write gains_opt to gains_data
if (k==1){
gains_data <- gains_opt  
}else{
gains_data <- rbind(gains_data,gains_opt)    
}


# 7.06  Figure 2). Performance by Bin (equal number of obs in each bin).
plot(gains_opt$pct_cum_obs, gains_opt$mean.prediction
     , xlab="Depth of File"
     #, ylab="Value"
     , main=paste("2). Perf by Bin: ",tr_cur_name_key,sep="")
     , type="b"
     , col=tr_cur_col
     , pch=c("o")
     , las=1
     , lty=c(1)

     , ylim=c(0,1)
     #,cex.lab=.75, cex.axis=.75, cex.main=.85, cex.sub=.5
    )
  text(gains_opt$pct_cum_obs, gains_opt$mean.prediction, labels=round(gains_opt$mean.prediction,2), cex=.9, pos=3)

  points(gains_opt$pct_cum_obs, gains_opt$mean.resp, col=tr_cur_col, pch=c(19))
  lines(gains_opt$pct_cum_obs, gains_opt$mean.resp, col=tr_cur_col,lty=c(1), type="b")
  text(gains_opt$pct_cum_obs, gains_opt$mean.resp, labels=round(gains_opt$mean.resp,2), cex=.9, pos=1)

  
# 7.07 Figure 3). Cumulative Gains
plot(gains_opt$pct_cum_obs, gains_opt$cume.pct.of.total
       , xlab="Depth of File"
       #, ylab="Value"
       , main=paste("3). Cum Gains: ",tr_cur_name_key,sep="")
       , type="b"
       , col=tr_cur_col
       , pch=c(19)
       , las=1
       , lty=c(1)
       , ylim=c(0,1.1)

       #,cex.lab=.75, cex.axis=.75, cex.main=.85, cex.sub=.5
  )
  text(gains_opt$pct_cum_obs, gains_opt$cume.pct.of.total, labels=round(gains_opt$cume.pct.of.total,2), cex=.9, pos=3)
  
  points(gains_opt$pct_cum_obs, gains_opt$cum_pct_pos_cases_rand, col="bisque4", pch=c("o"))
  lines(gains_opt$pct_cum_obs, gains_opt$cum_pct_pos_cases_rand, col="bisque4",lty=c(1), type="b")
  text(gains_opt$pct_cum_obs, gains_opt$cum_pct_pos_cases_rand, labels=round(gains_opt$cum_pct_pos_cases_rand,2), cex=.9, pos=1)
  
  #  points(pct_cum_obs, cum_pct_pos_cases_xgb_1, col="bisque4", pch=19)
  #lines(gains_opt$pct_cum_obs, gains_opt$cume.pct.of.total, col="bisque4",lty=1)
  #  text(pct_cum_obs, cum_pct_pos_cases_xgb_1, labels=round(cum_pct_pos_cases_xgb_1,2), cex=.9, pos=1)
  

# 7.08 Run calibration function for each model 
cal_obj <- calibration(factor(labels_test) ~ pred
                       , data = test_scored_comb[test_scored_comb$name_key == tr_cur_name_key,]
                       , class = 1
                       , cuts = 10
)

# 7.09 Extract needed data from calibration object.
cal_data <- cal_obj$data


#7.10 Simulate pct events for random distribution.
cal_data$pct <- seq(from=0.1,to=1,by=.1)


cal_data$model <- tr_cur_name_key
cal_data$mod_date <- mod_date

#7.10.01 Create a data frame of combined calibrations statistics for all models.

#if (i==1){
  #cal_data_comb <- cal_data
#}else
  
#{
 # cal_data_comb <- rbind(cal_data_comb,cal_data)
#}






# 7.11 Calibration Plot - figure 4).
plot(cal_data$midpoint
     , cal_data$Percent /100
     , xlab="Bin Midpoint %"
     , ylab="Observed Percentage"
     , las=1
     , main=paste("4). Cal: ",tr_cur_name_key,sep="")
     , type="b"
     , col=tr_cur_col
     , pch=c(19)
     , xaxp  = c(min(cal_data$midpoint), max(cal_data$midpoint),9)
     #, lty=c(1)
     , ylim=c(0,1)
     #,cex.lab=.75, cex.axis=.75, cex.main=.85, cex.sub=.5

)
text(cal_data$midpoint, cal_data$Percent/100, labels=round(cal_data$Percent/100,2), cex=.9, pos=3)

points(cal_data$midpoint, cal_data$midpoint/100, col="bisque4", pch=c("o"))
lines(cal_data$midpoint, cal_data$midpoint/100, col="bisque4",lty=c(1), type="b")
text(cal_data$midpoint, cal_data$midpoint/100, labels=round(cal_data$midpoint/100,2), cex=.9, pos=3)


# 7.12 Confusion Matrix plot.
ct_val <- c(conf_mat_opt_ct$true_pos
            , conf_mat_opt_ct$true_neg
            , conf_mat_opt_ct$false_pos
            , conf_mat_opt_ct$false_neg)

plot(x=c(1,0,0,1)
     , y=c(1,0,1,0)
     , xlim = c(-.5,1.5)
     , ylim = c(-.5,1.5)
     , xlab="Actual Class"
     , ylab="Predicted Class"
     , axes=FALSE
     , bty = 'n' # Removes box around plot.
     , pch=""
     )

      # Custom axes
      axis(1,col="white", at=c(0,1), col.axis="black")
      axis(2,col="white", at=c(0,1), col.axis="black",lty=2)

      title(paste("5). ",tr_cur_name_key,sep=""), cex.main=1.2)

      rect(xleft = -.5, ybottom =-.5, xright=.5, ytop=.5)
      rect(xleft = .5, ybottom =-.5, xright=1.5, ytop=.5)
      rect(xleft = .5, ybottom =.5, xright=1.5, ytop=1.5)
      rect(xleft = -.5, ybottom =.5, xright=.5, ytop=1.5)
      
      text(x=c(1,0,0,1), y=c(1,0,1,0), labels=format(ct_val,big.mark = ","), pos=1)
      text(x=c(1,0,0,1)
     , y=c(1,0,1,0)
     , labels=c("True Positives","True Negatives","False Positives","False Negatives")
     , font=2
     , col= c("green4","green4","red4","red4")
     , pos=3
     #, cex=1.2
)


# 7.13 Filter measures data to needed measures only for current model.
perf_data <- data.frame(measures[measures$model == tr_cur_name_key 
                      
                      & measures$measure  
                      %in% c( "Neg Pred Value"
                             , "Pos Pred Value"
                             , "Sensitivity"
                             , "Specificity"
                             , "Accuracy"
                             , "AUC"
                             , "Kappa"
                             ,"Prevalence"
                             )
                      ,3:4]
                      )


# 7.14 Reorder the measures to plot in correct order, transpose to matrix.
perf_data<- perf_data[cbind(5,4,3,2,8,7,1,6),]
perf_mat <- t(perf_data[2])


# 7.15 Figure 6). Bar plot of opt model performance measures. 
par(mar=c(5,8,3,3)) #(B,L,T,R)
bp <- barplot(perf_mat
        #, beside=TRUE
        , col = tr_cur_col
        
        , horiz = TRUE
        , xlim = c(0,1.2)
        #, xlim = c(0,40000)
        , names.arg = perf_data$measure
        , las=1
        , main = paste("6). ",tr_cur_name_key,sep="")
        , space = c(2,1)
        #, cex.axis = .8
        #, cex.names = .9
)
text(perf_mat, bp,labels = round(perf_data$value,2) ,pos = 4, cex = 0.7, col = "black")


}
# End of loop k


# Fit best model
base_data_val <- raw_data_test[,!colnames(raw_data_test) %in% tr_2]
new_val <- model.matrix(~.+0,data = base_data_val,with=F)
dval <- xgb.DMatrix(data = new_val)
val_pred <- predict(tr_2_opt_mod_obj,dval)
val_sub <- data.frame(DeviceID = churn_test$DeviceID
                      , predicted_prob = val_pred
                      , predicted_class = ifelse (val_pred > 0.5,1,0))

#write.csv (val_sub, file="churn_test_scored_Hoffman_20200508.csv", row.names = FALSE)
