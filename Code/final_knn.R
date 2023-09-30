# 1
noshow <- read.csv("noshow_21fall.csv")
head(noshow)
#table(noshow$No.show) # noshow:5250, show:9750 

# 2
noshow <- noshow[,-c(1,2)]

# 3
noshow$Gender <- factor(noshow$Gender)
noshow$Scholarship <- factor(noshow$Scholarship)
noshow$Hipertension <- factor(noshow$Hipertension)
noshow$Diabetes <- factor(noshow$Diabetes)
noshow$Alcoholism <- factor(noshow$Alcoholism)
noshow$Handcap <- factor(noshow$Handcap)
noshow$SMS_received <- factor(noshow$SMS_received)
noshow$No.show <- factor(noshow$No.show)

str(noshow)

# 6
set.seed(99)

train.index <- sample(c(1:dim(noshow)[1]), dim(noshow)[1]*0.6)

train.df <- noshow[train.index, ]
valid.df <- noshow[-train.index, ]


dim(train.df) # 9000, 10
dim(valid.df) # 6000, 10

# Normalization
train.norm <- train.df
valid.norm <- valid.df

library(caret)
head(train.df)
norm.values <- preProcess(train.df[,c(2, 9)], method=c("center","scale"))

train.norm[,c(2, 9)] <- predict(norm.values, train.df[,c(2, 9)])
valid.norm[,c(2, 9)] <- predict(norm.values, valid.df[,c(2, 9)])

head(train.norm)
head(valid.norm)

# KNN
library(FNN)

accuracy.df <- data.frame(k = seq(1, 12, 1), accuracy = rep(0, 12))
for (i in 1:12){
  knn.pred <- FNN::knn(train = train.norm[, 1:9], test = valid.norm[,1:9],
                       cl = train.norm[,10], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, valid.norm[,1:9])$overall[1]
} 
accuracy.df