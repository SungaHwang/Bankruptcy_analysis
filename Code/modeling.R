##이진 분류
# 파산을 최대한 예방하는게 중요하므로 재현율이 높은 것이 우선. (실제 값이 positive 인 대상 중 예측을 positive로 일치한 데이터 비율)
# + accuracy

# 파산 여부 분류
bankrupt.df <- read.csv("data.csv", na.strings = "", header = T)
dim(bankrupt.df) # 칼럼이 96개로 매우 많음
head(bankrupt.df)
str(bankrupt.df) # Y, X94: int(정수), 나머지 num(정수 또는 소수)

#bankrupt.df$Bankrupt. <- factor(bankrupt.df$Bankrupt.)

# 결측값 확인- 없음
sum(is.na(bankrupt.df))

# Net Income Flag는 값이 모두 1이라 분석에서 제외
bankrupt.df<-subset(bankrupt.df, select=-Net.Income.Flag)
dim(bankrupt.df)

# 타깃값 분포 확인 - 심한 불균형 -> oversampling/ undersampling필요
bankrupt.df$Bankrupt.[bankrupt.df$Bankrupt.==1]<-"Yes"
bankrupt.df$Bankrupt.[bankrupt.df$Bankrupt.==0]<-"No"

barplot(table(bankrupt.df$Bankrupt.),col=rainbow(5),
        main="Frequency of Bankrutcy",
        xlab="Bankruptcy Tag",
        ylab="Number of Companies")
box()

print(c((paste0("Bankrupt:",nrow(bankrupt.df[bankrupt.df$Bankrupt.=="1",]))),(paste0("Surivive:",nrow(bankrupt.df[bankrupt.df$Bankrupt.=="0",])))))

# 분포확인
library(psych)
library(dplyr)
bankrupt.df %>%
  select(where(is.numeric)) %>%
  multi.hist()

# 변수 선택 (R^2값이 너무 작아서 보류)
fit <- lm(Bankrupt.~., data = bankrupt.df)
summary(fit)
fit.both <- step(fit, direction = "both")
summary(fit.both)

# PCA
X <- bankrupt.df[, -1]

fit_pca <- princomp(X, cor = TRUE)
fit_pca$sdev^2
fit_pca$loadings

A <- cor(X)
eigen_A <- eigen(A)
eigen_A

fit_pca$sdev^2
fit_pca$loadings

vec1 <- eigen_A$vectors[, 1]
t(vec1) %*% vec1


summary(fit_pca)
screeplot(fit_pca, npcs = 5, type = "lines", main = "scree plot")

library(factoextra)
fviz_eig(fit_pca)
summary(fit_pca)

fviz_contrib(fit_pca, choice = "var", axes = 1)
fviz_contrib(fit_pca, choice = "var", axes = 2)
fviz_contrib(fit_pca, choice = "var", axes = 3)

fviz_pca_ind(fit_pca, col.ind = "cos2", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE)

fviz_pca_var(fit_pca, col.ind = "cos2", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE)

fviz_pca_biplot(fit_pca, geom.ind = "point",
                fill.ind = dat$survival2,
                col.ind = "black", pointshape = 21, pointsize = 2, label = "var",
                alpha.var = "contrib", col.var = "contrib", gradient.cols = "RdGy")

head(X)

#Dim1

#Dim2

#Dim3

# factor analysis
X <- bankrupt.df[, -c(1)]
X

cor(X)

# bartlett sphericity test
library(psych)
cortest.bartlett(cor(X), n = nrow(X))

# reject null (null: equal to identity metrics)

fit <- principal(X, cor = "cor", nfactors = 4, rotate = "none")
fit

fit_varimax <- principal(X, cor = "cor", nfactors = 4, rotate = "varimax")
fit_varimax

fa.diagram(fit_varimax, simple = FALSE, cut = 0.7, digit = 3)
fa.diagram(fit_varimax, simple = FALSE, cut = 0.5, digit = 3)



# Modeling
#library(caret)
#set.seed(1)
#train.index <- sample(c(1:dim(bankrupt.df)[1]), dim(bankrupt.df)[1]*0.6)
#train.df <- bankrupt.df[train.index,]
#dim(train.df)
#valid.df <- bankrupt.df[-train.index,]
#dim(valid.df)
#head(train.df)

library(caret)
set.seed(12)
spl <- sample(c(1:3), size = nrow(bankrupt.df), replace = TRUE, prob = c(0.6, 0.2, 0.2))

train.df <- bankrupt.df[spl == 1,]
valid.df <- bankrupt.df[spl == 2,]
test.df <- bankrupt.df[spl == 3,]

dim(train.df)
dim(valid.df)
dim(test.df)

train.norm.df <- train.df
valid.norm.df <- valid.df
test.norm.df <- test.df

norm.values <- preProcess(train.df[ , 1:4], method = c("center", "scale"))

train.norm.df[, 1:4] <- predict(norm.values, train.df[, 1:4])
head(train.norm.df, 3)

valid.norm.df[, 1:4] <- predict(norm.values, valid.df[, 1:4])
head(valid.norm.df, 3)

test.norm.df[, 1:4] <- predict(norm.values, test.df[, 1:4])
head(test.norm.df, 3)

## KNN
library(FNN)

accuracy.df <- data.frame(k = seq(1, 12, 1), accuracy = rep(0, 12))
for (i in 1:12){
  knn.pred <- FNN::knn(train = train.norm.df[, 2:95], test = valid.norm.df[, 2:95],
                       cl = train.norm.df[,1], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, valid.norm.df[,1])$overall[1]
} 
options(digits =2)
accuracy.df

knn.pred5 <- FNN::knn(train = train.norm.df[,2:95], test = test.norm.df[,2:95],
                      cl = train.norm.df[,1], k = 5)

res.df <- data.frame(test.df, knn.pred5)
head(res.df, n = 5)
confusionMatrix(knn.pred5, test.norm.df[,1])


## Decisiontree
library(rpart)
library(rpart.plot)
default.ct <- rpart(Bankrupt. ~ .,data = train.df, method = "class")
print(default.ct)
prp(default.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,
    box.col = ifelse(default.ct$frame$var == "<leaf>", 'gray', 'white'))

dt.pred <- predict(default.ct, test.df, type = "class")
head(dt.pred, n = 30)
confusionMatrix(dt.pred, test.df$Bankrupt.)


## randomforest
library(randomForest)
rf <-  randomForest(Bankrupt. ~., data = train.df, ntree = 500, 
                    importance = TRUE)

varImpPlot(rf)
varImpPlot(rf, type = 1)
varImpPlot(rf, type = 2)

rf.pred <- predict(rf, test.df)
head(rf.pred, n = 30)
confusionMatrix(rf.pred, test.df$Bankrupt.)

## Neural Nets
head(train.df)
train.df$Yes <- train.df$Bankrupt. == "Yes"
train.df$No <- train.df$ABankrupt. == "No"

library(neuralnet)
nn1 <- neuralnet(Bankrupt. ~., data = train.df, hidden = 2)
plot(nn1)

library(caret)
nn1_predict <- compute(nn1, train.df[,-c(1)])

confusionMatrix(factor(ifelse(nn1_predict == "Yes", "Bankrupt", "Surivive")),
                factor(train.df$Bankrupt.))
