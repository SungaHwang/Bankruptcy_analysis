############### 전처리 + Oversampling ###############
rm(list=ls())



##### 데이터 #####

# 읽어오기 
bk <- read.csv("bankrupcy.csv")
head(bk)
str(bk)
summary(bk)

# net income flag 삭제
bk <- bk[,-c(86,95)]
head(bk)
dim(bk)

# 범주형 변수 변환 (분류대상)
bk$Bankrupt. <- factor(bk$Bankrupt.)
table(bk$Bankrupt.)

# 변수명 변경 x1.aaa
colnames(bk)[1] <- paste("Y.", colnames(bk)[1], sep="")
class(colnames(bk)[1])
head(colnames(bk[1]))

for (i in 2:dim(bk)[2]){
  colnames(bk)[i] <- paste("X", i-1, ".", colnames(bk[i]), sep="")
}

colnames(bk)


##### oversampling #####
table(bk$Y.Bankrupt.) # 약 30배

# caret upsampling - 개수 똑같이 + record 중복
library(caret)
bk.os.up <- upSample(subset(bk, select=-Y.Bankrupt.), bk$Y.Bankrupt.)

table(bk.os.up$Class)
colnames(bk.os.up)[94] <- "Y.Bankrupt."
str(bk.os.up)
bk.os.up <- bk.os.up[c(94,1:93)]
head(bk.os.up)

# SMOTE - KNN=5 기반
?smote
install.packages("performanceEstimation")
library(performanceEstimation)

bk.os.sm <- smote(Y.Bankrupt. ~ ., bk, perc.over=30, perc.under=1)
table(bk.os.sm$Y.Bankrupt.)
head(bk.os.sm)



# csv 파일로 저장
os.df <- bk.os.up[,c(1,2,11,23,24,31,36,37,39,52,70,86,90)]
sm.df <- bk.os.sm[,c(1,2,11,23,24,31,36,37,39,52,70,86,90)]

head(os.df)
head(sm.df)

write.csv(bk.os.up, file="bankrupt_oversampling.csv", row.names=FALSE)
write.csv(bk.os.sm, file="bankrupt_SMOTE.csv", row.names=FALSE)



##### Model - KNN (upsampling) #####

# train / valid / test
set.seed(99)
spl <- sample(c(1:3), size=nrow(bk.os.up), replace=TRUE, prob=c(0.6,0.2,0.2))

train.knn.up <- bk.os.up[spl==1,]
valid.knn.up <- bk.os.up[spl==2,]
test.knn.up <- bk.os.up[spl==3,]

dim(train.knn.up) # 7974, 94
dim(valid.knn.up) # 2639, 94
dim(test.knn.up) # 2585, 94

# Normalization
train.knn.up.norm <- train.knn.up
valid.knn.up.norm <- valid.knn.up
test.knn.up.norm <- test.knn.up

norm.values <- preProcess(train.knn.up[,2:94], method=c("center","scale"))

train.knn.up.norm[,2:94] <- predict(norm.values, train.knn.up[,2:94])
valid.knn.up.norm[,2:94] <- predict(norm.values, valid.knn.up[,2:94])
test.knn.up.norm[,2:94] <- predict(norm.values, test.knn.up[,2:94])



##### Model - KNN (SMOTE) #####

# train / valid / test
set.seed(99)
spl <- sample(c(1:3), size=nrow(bk.os.sm), replace=TRUE, prob=c(0.6,0.2,0.2))

train.knn.sm <- bk.os.sm[spl==1,]
valid.knn.sm <- bk.os.sm[spl==2,]
test.knn.sm <- bk.os.sm[spl==3,]

dim(train.knn.sm) # 8113, 94
dim(valid.knn.sm) # 2674, 94
dim(test.knn.sm) # 2633, 94

# Normalization
train.knn.sm.norm <- train.knn.sm
valid.knn.sm.norm <- valid.knn.sm
test.knn.sm.norm <- test.knn.sm

norm.values <- preProcess(train.knn.sm[,2:94], method=c("center","scale"))

train.knn.sm.norm[,2:94] <- predict(norm.values, train.knn.sm[,2:94])
valid.knn.sm.norm[,2:94] <- predict(norm.values, valid.knn.sm[,2:94])
test.knn.sm.norm[,2:94] <- predict(norm.values, test.knn.sm[,2:94])



##### MODEL - DT (upsampling) #####

# train / valid / test
set.seed(99)
spl <- sample(c(1:3), size=nrow(bk.os.up), replace=TRUE, prob=c(0.6,0.2,0.2))

train.dt.up <- bk.os.up[spl==1,]
valid.dt.up <- bk.os.up[spl==2,]
test.dt.up <- bk.os.up[spl==3,]

dim(train.dt.up) # 7974, 94
dim(valid.dt.up) # 2639, 94
dim(test.dt.up) # 2585, 94



##### MODEL - DT (SMOTE) #####

# train / valid / test
set.seed(99)
spl <- sample(c(1:3), size=nrow(bk.os.sm), replace=TRUE, prob=c(0.6,0.2,0.2))

train.dt.sm <- bk.os.sm[spl==1,]
valid.dt.sm <- bk.os.sm[spl==2,]
test.dt.sm <- bk.os.sm[spl==3,]

dim(train.dt.sm) # 8113, 94
dim(valid.dt.sm) # 2674, 94
dim(test.dt.sm) # 2633, 94



##### NN (upsampling) ##### 
bk.nn.up <- bk.os.up

# 범위 수정 (0 ~ 1) - x14
colnames(bk.nn.up)[15]
bk.nn.up[15] <- (bk.nn.up[15] - min(bk.nn.up[15]))/(max(bk.nn.up[15])-min(bk.nn.up[15]))

str(bk.nn.up)
summary(bk.nn.up[15])

# 자료형 수정
bk.nn.up[1] <- as.numeric(bk.nn.up$Y.Bankrupt.)-1
head(bk.nn.up[1])

# train / valid / test
set.seed(99)
spl <- sample(c(1:3), size=nrow(bk.os.up), replace=TRUE, prob=c(0.6,0.2,0.2))

train.nn.up <- bk.os.up[spl==1,]
train.index.up <- rownames(bk)[spl==1]
length(train.index.up)

valid.nn.up <- bk.os.up[spl==2,]
valid.index.up <- rownames(bk)[spl==2]
length(valid.index.up)

test.nn.up <- bk.os.up[spl==3,]
test.index.up <- rownames(bk)[spl==3]
length(test.index.up)

dim(train.nn.up) # 7974, 94
dim(valid.nn.up) # 2639, 94
dim(test.nn.up) # 2585, 94

var <- colnames(bk.nn.up)[-1]
var

# dummy variable
library(nnet)

head(class.ind(bk.nn.up[train.index.up,]$Y.Bankrupt.))

train.nn.up <- cbind(bk.nn.up[train.index.up,var], class.ind(bk.nn.up[train.index.up,]$Y.Bankrupt.))
names(train.nn.up) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(train.nn.up)

valid.nn.up <- cbind(bk.nn.up[valid.index.up,var], class.ind(bk.nn.up[valid.index.up,]$Y.Bankrupt.))
names(valid.nn.up) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(valid.nn.up)

test.nn.up <- cbind(bk.nn.up[test.index.up,var], class.ind(bk.nn.up[test.index.up,]$Y.Bankrupt.))
names(test.nn.up) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(test.nn.up)



##### NN (SMOTE) ##### 
bk.nn.sm <- bk.os.sm

# 범위 수정 (0 ~ 1) - x14
colnames(bk.nn.sm)[15]
bk.nn.sm[15] <- (bk.nn.sm[15] - min(bk.nn.sm[15]))/(max(bk.nn.sm[15])-min(bk.nn.sm[15]))

str(bk.nn.sm)
summary(bk.nn.sm[15])

# 자료형 수정
bk.nn.sm[1] <- as.numeric(bk.nn.sm$Y.Bankrupt.)-1
head(bk.nn.sm[1])

# train / valid / test
set.seed(99)
spl <- sample(c(1:3), size=nrow(bk.os.sm), replace=TRUE, prob=c(0.6,0.2,0.2))

train.nn.sm <- bk.os.sm[spl==1,]
train.index.sm <- rownames(bk)[spl==1]
length(train.index.sm)

valid.nn.sm <- bk.os.sm[spl==2,]
valid.index.sm <- rownames(bk)[spl==2]
length(valid.index.sm)

test.nn.sm <- bk.os.sm[spl==3,]
test.index.sm <- rownames(bk)[spl==3]
length(test.index.sm)

dim(train.nn.sm) # 7974, 94
dim(valid.nn.sm) # 2639, 94
dim(test.nn.sm) # 2585, 94

var <- colnames(bk.nn.sm)[-1]
var

# dummy variable
library(nnet)

head(class.ind(bk.nn.sm[train.index.sm,]$Y.Bankrupt.))

train.nn.sm <- cbind(bk.nn.sm[train.index.sm,var], class.ind(bk.nn.sm[train.index.sm,]$Y.Bankrupt.))
names(train.nn.sm) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(train.nn.sm)

valid.nn.sm <- cbind(bk.nn.sm[valid.index.sm,var], class.ind(bk.nn.sm[valid.index.sm,]$Y.Bankrupt.))
names(valid.nn.sm) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(valid.nn.sm)

test.nn.sm <- cbind(bk.nn.sm[test.index.sm,var], class.ind(bk.nn.sm[test.index.sm,]$Y.Bankrupt.))
names(test.nn.sm) <- c(var, paste("Y.Bankrupt._", c(0,1), sep=""))
head(test.nn.sm)


