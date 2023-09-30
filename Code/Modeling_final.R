## ROS final data
bankrupt_ROS <- read.csv("bankrupt_ROS.csv", na.strings = "", header = T)
bankrupt_ROS_final <- bankrupt_ROS[,c(1,2,11,23,24,31,36,37,39,52,70,86,90)]
dim(bankrupt_ROS_final)
head(bankrupt_ROS_final)
bankrupt_ROS_final$Y.Bankrupt. <- factor(bankrupt_ROS_final$Y.Bankrupt.)


## SMOTE final data
bankrupt_SMOTE <- read.csv("bankrupt_SMOTE.csv", na.strings = "", header = T)
bankrupt_SMOTE_final <- bankrupt_SMOTE[,c(1,2,11,23,24,31,36,37,39,52,70,86,90)]
dim(bankrupt_SMOTE_final)
head(bankrupt_SMOTE_final)
bankrupt_SMOTE_final$Y.Bankrupt. <- factor(bankrupt_SMOTE_final$Y.Bankrupt.)


## modeling - ROS
set.seed(99)
spl <- sample(c(1:3), size=nrow(bankrupt_ROS_final), replace=TRUE, prob=c(0.6,0.2,0.2))
train.ros <- bankrupt_ROS_final[spl==1,]
train.index.ros <- rownames(bankrupt_ROS_final)[spl==1]
valid.ros <- bankrupt_ROS_final[spl==2,]
valid.index.ros <- rownames(bankrupt_ROS_final)[spl==2]
test.ros <- bankrupt_ROS_final[spl==3,]
test.index.ros <- rownames(bankrupt_ROS_final)[spl==3]

dim(train.ros)  # 7974, 13
dim(valid.ros) # 2639, 13
dim(test.ros) # 2585, 13

# Normalization
train.ros.norm <- train.ros
valid.ros.norm <- valid.ros
test.ros.norm <- test.ros

library(caret)
head(train.ros)
norm.values <- preProcess(train.ros[,c(2:13)], method=c("center","scale"))

train.ros.norm[,c(2:13)] <- predict(norm.values, train.ros[,c(2:13)])
valid.ros.norm[,c(2:13)] <- predict(norm.values, valid.ros[,c(2:13)])
test.ros.norm[,c(2:13)] <- predict(norm.values, test.ros[,c(2:13)])

# KNN
library(FNN)

accuracy.df <- data.frame(k = seq(1, 12, 1), accuracy = rep(0, 12))
for (i in 1:12){
  knn.pred <- FNN::knn(train = train.ros.norm[, 2:13], test = valid.ros.norm[, 2:13],
                       cl = train.ros.norm[,1], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, valid.ros.norm[,1])$overall[1]
} 
accuracy.df

knn.pred2 <- FNN::knn(train = train.ros.norm[,2:13], test = test.ros.norm[,2:13],
                      cl = train.ros.norm[,1], k = 2)

res.df <- data.frame(test.ros, knn.pred2)
head(res.df, n = 5)
confusionMatrix(knn.pred2, test.ros.norm[,1])

# Default tree
library(rpart)
library(rpart.plot)
default.ct <- rpart(Y.Bankrupt. ~ .,data = train.ros.norm, method = "class")
print(default.ct)
prp(default.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,
    box.col = ifelse(default.ct$frame$var == "<leaf>", 'gray', 'white'))

default.pred <- predict(default.ct, test.ros.norm, type = "class")
head(default.pred, n = 30)

confusionMatrix(default.pred, test.ros.norm$Y.Bankrupt.)

# Full Tree
library(rpart)
library(rpart.plot)
deeper.ct <- rpart(Y.Bankrupt. ~ ., data = train.ros.norm, method = "class",
                   cp = 0, minsplit = 1)
prp(deeper.ct, type = 1, extra = 1, under = TRUE, split.font = 1,
    varlen = -10, box.col = ifelse(deeper.ct$frame$var == "<leaf>",
                                   'gray', 'white'))
deeper.pred <- predict(deeper.ct, test.ros.norm, type = "class")

confusionMatrix(deeper.pred, test.ros.norm$Y.Bankrupt.)

# Random Forest
library(randomForest)
rf <-  randomForest(Y.Bankrupt. ~., data = train.ros.norm, 
                    importance = TRUE)
rf.pred <- predict(rf, test.ros.norm)
head(rf.pred, n = 30)

confusionMatrix(rf.pred, test.ros.norm$Y.Bankrupt.)

# final*****
library(randomForest)
rf1 <-  randomForest(Y.Bankrupt. ~., data = train.ros.norm, ntree = 3000,  mtry= 2,
                    importance = TRUE)
rf1.pred <- predict(rf1, test.ros.norm)
head(rf1.pred, n = 30)

confusionMatrix(rf1.pred, test.ros.norm$Y.Bankrupt.)

#
library(randomForest)
rf <-  randomForest(Y.Bankrupt. ~., data = train.ros.norm, ntree = 2000,  mtry= 2,
                    importance = TRUE)
rf.pred <- predict(rf, test.ros.norm)
head(rf.pred, n = 30)

confusionMatrix(rf.pred, test.ros.norm$Y.Bankrupt.)


# boosting
library(adabag)

boost <- boosting(Y.Bankrupt. ~ ., data = train.ros.norm)
bt.pred <- predict(boost, test.ros.norm)
head(bt.pred$class, n =20)
class(bt.pred$class)

confusionMatrix(factor(bt.pred$class), test.ros.norm$Y.Bankrupt.)

# Neural Nets
scale.values <- caret::preProcess(bankrupt_ROS_final[,c(2:13)], rangeBounds = c(0,1), methods = "range")
scaled = predict(scale.values, bankrupt_ROS_final[,c(2:13)])

bankrupt_ROS_scaled <- cbind(Y.Bankrupt. = bankrupt_ROS_final$Y.Bankrupt., scaled)

bankrupt_ROS_scaled[1] <- as.numeric(bankrupt_ROS_scaled$Y.Bankrupt.)-1
head(bankrupt_ROS_scaled[1])

var <- colnames(bankrupt_ROS_scaled)[-1]
var

library(nnet)
head(class.ind(bankrupt_ROS_scaled[train.index.ros,]$Y.Bankrupt.))

train.nn.ros <- cbind(bankrupt_ROS_scaled[train.index.ros,var], class.ind(bankrupt_ROS_scaled[train.index.ros,]$Y.Bankrupt.))
names(train.nn.ros) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(train.nn.ros)

valid.nn.ros <- cbind(bankrupt_ROS_scaled[valid.index.ros,var], class.ind(bankrupt_ROS_scaled[valid.index.ros,]$Y.Bankrupt.))
names(valid.nn.ros) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(valid.nn.ros)

test.nn.ros <- cbind(bankrupt_ROS_scaled[test.index.ros,var], class.ind(bankrupt_ROS_scaled[test.index.ros,]$Y.Bankrupt.))
names(test.nn.ros) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(test.nn.ros)

library(neuralnet)
nn1 <- neuralnet(Y.Bankrupt._0 + Y.Bankrupt._1 ~.,data = train.nn.ros, hidden = 6, stepmax = 1e+06)
plot(nn1)

head(train.nn.ros)
test.prediction <- compute(nn1, test.nn.ros[,-c(13:14)])
head(test.prediction)
test.class <- apply(test.prediction$net.result, 1, which.max) -1
head(test.class, n = 50)
class(test.class)

head(test.ros)
confusionMatrix(factor(test.class), factor(ifelse(test.nn.ros$Y.Bankrupt._0 == 1, 0, 1)))



## modeling - SMOTE
set.seed(99)
spl <- sample(c(1:3), size=nrow(bankrupt_SMOTE_final), replace=TRUE, prob=c(0.6,0.2,0.2))

train.sm <- bankrupt_SMOTE_final[spl==1,]
valid.sm <- bankrupt_SMOTE_final[spl==2,]
test.sm <- bankrupt_SMOTE_final[spl==3,]

dim(train.sm)  # 8113, 13
dim(valid.sm) # 2674, 13
dim(test.sm) # 2633, 13

# Normalization
train.sm.norm <- train.sm
valid.sm.norm <- valid.sm
test.sm.norm <- test.sm

library(caret)
head(train.sm)
norm.values <- preProcess(train.sm[,c(2:13)], method=c("center","scale"))

train.sm.norm[,c(2:13)] <- predict(norm.values, train.sm[,c(2:13)])
train.index.sm <- rownames(bankrupt_SMOTE_final)[spl==1]
valid.sm.norm[,c(2:13)] <- predict(norm.values, valid.sm[,c(2:13)])
valid.index.sm <- rownames(bankrupt_SMOTE_final)[spl==2]
test.sm.norm[,c(2:13)] <- predict(norm.values, test.sm[,c(2:13)])
test.index.sm <- rownames(bankrupt_SMOTE_final)[spl==3]

# KNN
library(FNN)

accuracy.df <- data.frame(k = seq(1, 12, 1), accuracy = rep(0, 12))
for (i in 1:12){
  knn.pred <- FNN::knn(train = train.sm.norm[, 2:13], test = valid.sm.norm[, 2:13],
                       cl = train.sm.norm[,1], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, valid.sm.norm[,1])$overall[1]
} 
accuracy.df

knn.pred1 <- FNN::knn(train = train.sm.norm[,2:13], test = test.sm.norm[,2:13],
                      cl = train.sm.norm[,1], k = 1)

res.df <- data.frame(test.sm, knn.pred1)
head(res.df, n = 5)

confusionMatrix(knn.pred1, test.sm.norm[,1])

# Default tree
library(rpart)
library(rpart.plot)
default.ct <- rpart(Y.Bankrupt. ~ .,data = train.sm.norm, method = "class")
print(default.ct)
prp(default.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,
    box.col = ifelse(default.ct$frame$var == "<leaf>", 'gray', 'white'))

default.pred <- predict(default.ct, test.sm.norm, type = "class")
head(default.pred, n = 30)

confusionMatrix(default.pred, test.sm.norm$Y.Bankrupt)

# Full Tree
library(rpart)
library(rpart.plot)
deeper.ct <- rpart(Y.Bankrupt. ~ ., data = train.sm.norm, method = "class",
                   cp = 0, minsplit = 1)
prp(deeper.ct, type = 1, extra = 1, under = TRUE, split.font = 1,
    varlen = -10, box.col = ifelse(deeper.ct$frame$var == "<leaf>",
                                   'gray', 'white'))
deeper.pred <- predict(deeper.ct, test.sm.norm, type = "class")

confusionMatrix(deeper.pred, test.sm.norm$Y.Bankrupt.)

# Random Forest
library(randomForest)
rf <-  randomForest(Y.Bankrupt. ~., data = train.sm.norm, 
                    importance = TRUE)
rf.pred <- predict(rf, test.sm.norm)
head(rf.pred, n = 30)

confusionMatrix(rf.pred, test.sm.norm$Y.Bankrupt.)

# boosting
library(adabag)

boost <- boosting(Y.Bankrupt. ~ ., data = train.sm.norm)
bt.pred <- predict(boost, test.sm.norm)
head(bt.pred$class, n =20)
class(bt.pred$class)

confusionMatrix(factor(bt.pred$class), test.sm.norm$Y.Bankrupt.)

# Neural Nets
scale.values <- caret::preProcess(bankrupt_SMOTE_final[,c(2:13)], rangeBounds = c(0,1), methods = "range")
scaled = predict(scale.values, bankrupt_SMOTE_final[,c(2:13)])

bankrupt_SMOTE_scaled <- cbind(Y.Bankrupt. = bankrupt_SMOTE_final$Y.Bankrupt., scaled)

bankrupt_SMOTE_scaled[1] <- as.numeric(bankrupt_SMOTE_scaled$Y.Bankrupt.)-1
head(bankrupt_SMOTE_scaled[1])

var <- colnames(bankrupt_SMOTE_scaled)[-1]
var

library(nnet)
head(class.ind(bankrupt_SMOTE_scaled[train.index.sm,]$Y.Bankrupt.))

train.nn.sm <- cbind(bankrupt_SMOTE_scaled[train.index.sm,var], class.ind(bankrupt_SMOTE_scaled[train.index.sm,]$Y.Bankrupt.))
names(train.nn.sm) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(train.nn.sm)

valid.nn.sm <- cbind(bankrupt_SMOTE_scaled[valid.index.sm,var], class.ind(bankrupt_SMOTE_scaled[valid.index.sm,]$Y.Bankrupt.))
names(valid.nn.sm) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(valid.nn.sm)

test.nn.sm <- cbind(bankrupt_SMOTE_scaled[test.index.sm,var], class.ind(bankrupt_SMOTE_scaled[test.index.sm,]$Y.Bankrupt.))
names(test.nn.sm) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(test.nn.sm)

library(neuralnet)
nn1 <- neuralnet(Y.Bankrupt._0 + Y.Bankrupt._1 ~.,data = train.nn.sm, hidden = 6)
plot(nn1)

head(train.nn.ros)
test.prediction <- compute(nn1, test.nn.sm[,-c(13:14)])
head(test.prediction)
test.class <- apply(test.prediction$net.result, 1, which.max) -1
head(test.class, n = 50)
class(test.class)

confusionMatrix(factor(test.class), factor(ifelse(test.nn.sm$Y.Bankrupt._0 == 1, 0, 1)))
