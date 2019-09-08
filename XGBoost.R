#Fast and Accurate
#Requires Numerical data output
#Load Libraries
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)

#Load Data
data <- read.csv("binary.csv")
str(data)
data$rank <- as.factor(data$rank)

#Partion Data
set.seed(1234)
ind <- sample(2,nrow(data),replace=T,prob=c(0.8,0.2))
train <- data[ind==1,]
test <- data[ind==2,]

#one hot encoding for matrix
train_m <- sparse.model.matrix(admit~.-1,data=train)
head(train_m)
train_l <- train[,"admit"]
train_matrix <- xgb.DMatrix(data=as.matrix(train_m),label=train_l)
test_m <- sparse.model.matrix(admit~.-1,data=test)
test_l <- test[,"admit"]
test_matrix <- xgb.DMatrix(data=as.matrix(test_m),label=test_l)

#Parameters
nc <- length(unique(train_l))
xgb_para <-list("objective"="multi:softprob",
                "eval_metric"="mlogloss",
                "num_class"=nc)
watchlist <-list(train=train_matrix,test=test_matrix)


#eXtreme Gradient boosting
bst_model <- xgb.train(params = xgb_para,
                       data=train_matrix,
                       nrounds=90,
                       watchlist = watchlist)
bst_model


#Training & Test error plot
e <- data.frame(bst_model$evaluation_log)
plot(e$iter,e$train_mlogloss,col="blue",pch=20)
lines(e$iter,e$test_mlogloss,col="red")


#min value of test error
min(e$test_mlogloss)
e[e$test_mlogloss==0.595096,]


#Improving
#eXtreme Gradient boosting#eta=low<-more robust to over fitting
bst_model <- xgb.train(params = xgb_para,
                       data=train_matrix,
                       nrounds=90,
                       watchlist = watchlist,
                       eta=0.1)
bst_model
#Training & Test error plot
e <- data.frame(bst_model$evaluation_log)
plot(e$iter,e$train_mlogloss,col="blue",pch=20)
lines(e$iter,e$test_mlogloss,col="red")


#Feature Importance
imp <- xgb.importance(colnames(train_matrix),model=bst_model)
print(imp)
xgb.plot.importance(imp)


#Prediction and Confusion Matrix
p <- predict(bst_model,test_matrix)
pred <- matrix(p,nrow=nc,ncol=length(p)/nc) %>%
  t() %>%
  data.frame() %>%
  mutate(label=test_l,max_prob=max.col(.,"last")-1)
head(pred)
table(predicted=pred$max_prob,actual=pred$label)


#more xgboost parameters
#max.depth=6 tree depth
#gamma  =(1-inf) any number :slowly increse to avoid overfitting
#subsample=1 decrese to avoid overfitting
#colsample_bytree=1
bst_model <- xgb.train(params = xgb_para,
                       data=train_matrix,
                       nrounds=100,
                       watchlist = watchlist,
                       eta=0.02,
                       max.depth=3,
                       gamma=0.3,
                       subsample=0.4,
                       colsample_bytree=1,
                       missing=NA,
                       seed=333)
bst_model


#Training & Test error plot
e <- data.frame(bst_model$evaluation_log)
plot(e$iter,e$train_mlogloss,col="blue",pch=20)
lines(e$iter,e$test_mlogloss,col="red")

#better model is achieved