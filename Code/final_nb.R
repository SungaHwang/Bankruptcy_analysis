# (1) -1
noshow <- read.csv("noshow_21fall.csv", na.strings = "")
# head(noshow)
# summary(noshow)
# str(noshow)

# (1) -2
noshow <- noshow[,-c(1,2)]

# (1) -3
noshow$Gender <- factor(noshow$Gender)
noshow$Scholarship <- factor(noshow$Scholarship)
noshow$Hipertension <- factor(noshow$Hipertension)
noshow$Diabetes <- factor(noshow$Diabetes)
noshow$Alcoholism <- factor(noshow$Alcoholism)
noshow$Handcap <- factor(noshow$Handcap)
noshow$SMS_received <- factor(noshow$SMS_received)
noshow$No.show <- factor(noshow$No.show)

# (1) -4
#hist(as.numeric(noshow$Age))
noshow$Age <- factor(floor(noshow$Age/10))

# (1) -5
#hist(as.numeric(noshow$Difftime))
noshow$Difftime <- factor(floor(ifelse(noshow$Difftime<2, 0,
                                       log2(noshow$Difftime))))

# (1) -6
set.seed(99)

train.index <- sample(c(1:dim(noshow)[1]), dim(noshow)[1]*0.6)
train.df <- noshow[train.index, ]
valid.df <- noshow[-train.index, ]

# dim(train.df) # 9000, 10
# dim(valid.df) # 6000, 10

# (1) -7
library(e1071)
noshow.nb <- naiveBayes(No.show ~., data = train.df)

# (1) -8
library(caret)
train.pred <- predict(noshow.nb, newdata = train.df)
confusionMatrix(train.pred, train.df$No.show)

valid.pred <- predict(noshow.nb, newdata = valid.df)
confusionMatrix(valid.pred, valid.df$No.show)
