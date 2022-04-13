require(xgboost)

test <- agaricus.train

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
data(new_tr, package='xgboost')

bst <- xgboost(agaricus.train$data, agaricus.train$label, nrounds = 50,
               eta = 0.1, max_depth = 3, subsample = .5,
               method = "hist", objective = "binary:logistic", nthread = 2, verbose = 0)

xgb.plot.shap(agaricus.test$data, model = bst
              
              #, features = "odor=none"
              
              )
contr <- predict(bst, agaricus.test$data, predcontrib = TRUE)
xgb.plot.shap(agaricus.test$data, contr, model = bst, top_n = 12, n_col = 3)