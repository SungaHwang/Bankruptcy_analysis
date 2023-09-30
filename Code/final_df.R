# (2) -1
noshow <- read.csv("noshow_21fall.csv", na.strings = "")
# head(noshow)
# summary(noshow)
# str(noshow)


# (2) -2
noshow <- noshow[,-c(1,2)]

# (2) -3
noshow$Gender <- factor(noshow$Gender)
noshow$Scholarship <- factor(noshow$Scholarship)
noshow$Hipertension <- factor(noshow$Hipertension)
noshow$Diabetes <- factor(noshow$Diabetes)
noshow$Alcoholism <- factor(noshow$Alcoholism)
noshow$Handcap <- factor(noshow$Handcap)
noshow$SMS_received <- factor(noshow$SMS_received)
noshow$No.show <- factor(noshow$No.show)

# (2) -4
set.seed(99)

train.index <- sample(c(1:dim(noshow)[1]), dim(noshow)[1]*0.6)

train.df <- noshow[train.index, ]
valid.df <- noshow[-train.index, ]

# dim(train.df) # 9000, 10
# dim(valid.df) # 6000, 10

# (2) -5
library(rpart)
library(rpart.plot)
default.ct <- rpart(No.show ~., data = train.df, method = "class", cp =0.002)
prp(default.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,
    box.col = ifelse(default.ct$frame$var == "<leaf>", 'gray', 'white'))

# (2) -6
library(caret)
default.pred.train <- predict(default.ct, data = train.df, type = "class")
confusionMatrix(default.pred.train, train.df$No.show)

default.pred.valid <- predict(default.ct, newdata = valid.df, type = "class")
confusionMatrix(default.pred.valid, valid.df$No.show)