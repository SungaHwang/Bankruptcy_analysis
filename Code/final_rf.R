# (3) -1
noshow <- read.csv("noshow_21fall.csv", na.strings = "")
# head(noshow)
# summary(noshow)
# str(noshow)


# (3) -2
noshow <- noshow[,-c(1,2)]

# (3) -3
noshow$Gender <- factor(noshow$Gender)
noshow$Scholarship <- factor(noshow$Scholarship)
noshow$Hipertension <- factor(noshow$Hipertension)
noshow$Diabetes <- factor(noshow$Diabetes)
noshow$Alcoholism <- factor(noshow$Alcoholism)
noshow$Handcap <- factor(noshow$Handcap)
noshow$SMS_received <- factor(noshow$SMS_received)
noshow$No.show <- factor(noshow$No.show)

# (3) -4
set.seed(99)

train.index <- sample(c(1:dim(noshow)[1]), dim(noshow)[1]*0.6)

train.df <- noshow[train.index, ]
valid.df <- noshow[-train.index, ]

# dim(train.df) # 9000, 10
# dim(valid.df) # 6000, 10

# (3) -5
library(randomForest)
rf <- randomForest(No.show ~., data = train.df, ntree = 500, importance = TRUE)

# (3) -6
varImpPlot(rf, type =1)

# (3) -7
library(caret)
rf.pred.train <- predict(rf, data = train.df)
confusionMatrix(rf.pred.train, train.df$No.show)

rf.pred.valid <- predict(rf, newdata = valid.df)
confusionMatrix(rf.pred.valid, valid.df$No.show)

