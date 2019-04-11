library(lattice)
library(ggplot2)
library(caret)
library(xgboost)
library(magrittr)
library(rpart)
#install.packages("party")
library(party)
train.dr.new <- read.csv("F:/7390/mid-term project/train_after_dimension_reduction.csv")
# Distribution of target variable
barplot(prop.table(table(train.dr.new$TARGET)),
        col = c("pink", "light green"),
        ylim = c(0,1),
        main = "Class Distribution")
# Data partitioning for train and test
train.dr.new$TARGET <- as.factor(train.dr.new$TARGET)
set.seed(1111)
index <- sample(2,nrow(train.dr.new),replace = TRUE,prob = c(0.9,0.1))
final.train <- train.dr.new[index==1,]
final.test <- train.dr.new[index==2,]
#install.packages("corrplot")
library(corrplot)
final.train$TARGET <- as.numeric(final.train$TARGET)
set.seed(1111)
index <- sample(2:ncol(final.train), 20, replace = FALSE)
final.co <- final.train[,index]
final.co <- data.frame(final.train$TARGET, final.co)
corr<-cor(final.co)
corrplot(corr, method="pie")
#install.packages("ROSE")
final.train$TARGET <- as.factor(final.train$TARGET)
final.train$TARGET
library(ROSE)
final.train <- ovun.sample(TARGET~., data=final.train[, -1], method="both", N=300000)$data
barplot(prop.table(table(final.train$TARGET)),
        col = c("pink", "light green"),
        ylim = c(0,1),
        main = "Class Distribution")
lrm <- glm(TARGET~., family = binomial(link = "logit"), data = final.train)
pt <- predict(lrm, final.test, type="response")
pt.mat <- as.factor(ifelse(pt>0.5, 1, 0))
#summary( pt.mat)
confusionMatrix(pt.mat, final.test$TARGET, positive='1')
#str(final.train)
auc(final.test$TARGET,pt)
print(pt.mat)
#plot(pt.mat)
# Distribution of target variable of prediction
barplot(prop.table(table(pt.mat)),
        col = c("pink", "light green"),
        ylim = c(0,1),
        main = "Class Distribution")
#-----XGBoost------------
cat("Train Model \n")

xgb_params = list(
  eta = 0.3,
  objective = 'binary:logistic',
  eval_metric='auc',
  colsample_bytree=0.7,
  subsample=0.7,
  min_child_weight=10
)

features<-setdiff(names(final.train),'TARGET')
dtrainmat = xgb.DMatrix(as.matrix(final.train[,features]), label=as.numeric(final.train$TARGET)-1)
dtestmat = xgb.DMatrix(as.matrix(final.test[,features]))
#summary(final.train$TARGET)
xgbmodel<-xgb.train(xgb_params,dtrainmat,nrounds=125,verbose=2)
pred<-predict(xgbmodel,dtestmat)

cols <- colnames(final.test)
xgb.importance(cols, model=xgbmodel) %>% 
  xgb.plot.importance(top_n = 30)

cat("Submit Predictions\n")
sub<-data.frame(SK_ID_CURR = final.test$SK_ID_CURR,TARGET=pred)
write.csv(sub,'xgb_baseline.csv',row.names = F)

#-----evaluation---------
print(xgbmodel)
#plot(xgbmodel)
print(pred)
plot(pred)
#------calculate the AUC value and Confusion Matrix-----
pred.mat <- as.factor(ifelse(pred>0.5, 1, 0))
confusionMatrix(pred.mat, final.test$TARGET, positive='1')
auc(final.test$TARGET,pred)

##----------Decision Tree Classification--------
nn <- NULL
for (i in 1:length(final.train)){
  nn <<- paste(nn, "+", names(final.train[i]))
}
nn
set.seed(1111)
index <- sample(1:nrow(final.train), 4000, replace = FALSE)
final.sam <- final.train[index,]
dtmodel <- rpart(TARGET~.,method="class",data = final.sam)
printcp(dtmodel)
plotcp(dtmodel)
plot(dtmodel, uniform=TRUE, main="Classification Tree for MidTerm Project")
text(dtmodel, use.n=TRUE, all=TRUE, cex=0.7)
#----------pruning the decision tree in case of overfitting-----
pfit<- prune(dtmodel, cp= 0.02)
summary(pfit)
plot(pfit, uniform=TRUE, main="pruned Classification Tree for MidTerm Project")
text(pfit, use.n=TRUE, all=TRUE, cex=0.7)
printcp(pfit)
plotcp(pfit)

dpred<-predict(pfit,final.test, type="prob")
print(dpred)
dpred.t <- ifelse(dpred[,2]>dpred[,1], 1, 0)
print(dpred.t)
##---------------##After pruning, the accuracy is bigger
confusionMatrix(as.factor(dpred.t), final.test$TARGET, positive='1')
##Random Forest
library(randomForest)
set.seed(222)
rf <- randomForest(TARGET~., data=final.sam, ntree=200, mtry=6, importance=TRUE, PROXIMITY=TRUE)
ncol(final.sam)
nrow(final.sam)
print(rf)
attributes(rf)
#head(rf)
##Prediction and Confusion Matrix with train data
rpred1 <- predict(rf, final.train)
head(rpred1)
#library(caret)
#library(e1071)
confusionMatrix(rpred1, final.train$TARGET, positive='1')

##Prediction and Confusion matrix with test data
rpred2 <- predict(rf, final.test)
head(rpred2)
#library(caret)
#library(e1071)
confusionMatrix(rpred2, final.test$TARGET, positive='1')

##Error rate for our random forest model
plot(rf)

##tune mtry
#tuneRF(final.train[,-2], final.train[,2], stepFactor=0.5, plot=TRUE, trace=TRUE, ntreeTry= 200, improve=0.05) 
#str(final.train)

##No. of nodes for the trees
hist(treesize(rf), main="No. of Nodes for the trees", col="green")

#Variable Importance
varImpPlot(rf)
importance(rf)
varUsed(rf)

##Partial Dependence Plot
#partialPlot(rf, final.train, AMT_CREDIT.x, "1")

##Extract Single Tree
getTree(rf, 1)

#------ROC----
#install.packages("ROCR")
#install.packages("dplyr")
library(ROCR)
library(dplyr)
str(preds_list)
preds_list <- list(pt, pred, as.numeric(dpred[,2]), rpred2)
m <- length(preds_list)
actuals_list <- rep(list(final.test$TARGET), m)
rp <- prediction(preds_list, actuals_list)
rocs <-performance(rp, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Data ROC Curves")
auc3 <-performance(rp, "auc")
auc3 <-unlist(slot(auc3, "y.values"))
auc3 <-round(auc3, 5)
print(auc3)
legend(x = "bottomright",
       legend = c("Logistic Regression", "XGBoost", "Decison Tree", "Random Forest"),
       fill = 1:m)
legend(.85, .6, auc3, title = "AUC",
       fill = 1:m, cex = .85)
