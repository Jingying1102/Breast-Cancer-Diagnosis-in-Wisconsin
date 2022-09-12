bcancer.raw <- read.csv("D:/STAT5330/Final Proj/breast cancer data.csv")
bcancer<-bcancer.raw[,2:32]
bcancer$diagnosis<-ifelse(bcancer$diagnosis=="M",1,0)
bcancer<-bcancer[,c(1:3,6:13,16:23,26:31)]
train<-bcancer[1:500,]
test<-bcancer[501:569,]
library(MASS)
library(caret)
library(ggplot2)
library(corrplot)
library(GGally)

#exploratory data analysis
table(bcancer.raw$diagnosis)
lbl <- c("malignant", "benign")
pie(table(bcancer.raw$diagnosis),radius=1, labels=paste(lbl,":",round(table(bcancer.raw$diagnosis)/length(bcancer.raw$diagnosis)*100),"%"),
    col=c(4,2), main="Pie Chart with Percentage of response")
png("2.png",width=400,height=400)
pairs(train[,c(2,3,6:11)],pch=".")
pairs(train[,c(12,13,16:21)],pch=".")
pairs(train[,c(22,23,26:31)],pch=".")
correlation<-cor(train[,-1])
corrplot::corrplot(correlation,order="hclust",tl.cex = 0.7,addrect = 8,mar=c(0,0,0,0),tl.col=1)
ggpairs(train[,1:4],aes(col=as.factor(diagnosis),pch="."),lower=list(continuous="smooth",mapping=aes()),upper=list(mapping=aes(cex=0.1)))+
  theme_bw()+
  labs(title="Cancer Mean")+
  theme(plot.title = element_text(face = "bold",color = "black",hjust = 0.5,size = 12))

boxplot(train[,2:11])

#regression

#LDA
modelFitLDA<- train(train[,2:25],as.factor(train[,1]), method='lda',preProcess=c('scale', 'center'), data=train)
LDAconfusionM<-confusionMatrix(as.factor(test[,1]), predict(modelFitLDA, test[,2:25]))
LDAconfusionM

#QDA
modelFitQDA<- train(train[,2:25],as.factor(train[,1]), method='qda',preProcess=c('scale', 'center'), data=train)
QDAconfusionM<-confusionMatrix(as.factor(test[,1]), predict(modelFitQDA, test[,2:25]))
QDAconfusionM

#logistic
logmodel<-glm(diagnosis~.,data=train, family=binomial(link="logit"))
predlog<-predict(logmodel,test,type='response')
threshold<-0.5
logpredvalues<-ifelse(predlog>threshold,1,0)
confusionMatrix(as.factor(test[,1]), as.factor(logpredvalues))

#linear svm
library(gbm)
library(kernlab)
cost = 10
svm.fitlinear <- ksvm(as.matrix(train[,2:25]),as.factor(train[,1]),type="C-svc",kernel='vanilladot',C=cost,scaled=c())
predlinearsvm<-predict(svm.fitlinear,test[,2:25],type='response')
confusionMatrix(as.factor(test[,1]), as.factor(predlinearsvm))

#non-linear svm
svm.fitnonlinear <- ksvm(as.matrix(train[,2:25]),as.factor(train[,1]),type="C-svc",kernel='rbfdot',C=cost,scaled=c())
prednonlinearsvm<-predict(svm.fitnonlinear,test[,2:25],type='response')
confusionMatrix(as.factor(test[,1]), as.factor(prednonlinearsvm))

#randomforest
library(randomForest)
nfold = 5
infold = sample(rep(1:nfold, length.out=length(train$diagnosis)))
seqnodesize<-seq(1,10,1)
accuracyN<-c()
accuracyCV<-c()
for(i in 1:10){
  
  for(l in 1:nfold){
    trainCV=train[infold !=l,]
    testCV=train[which(infold==l),]
    rf.fit = randomForest(trainCV[,2:25], as.factor(trainCV[,1]), 
                          ntree = 500, mtry = 7, nodesize = seqnodesize[i])
    predrf<-predict(rf.fit,testCV[,2:25],type='response')
    a<-confusionMatrix(as.factor(testCV[,1]), as.factor(predrf))
    accuracyCV[l]<-a$overall[1]
  }
  accuracyN[i]<-mean(accuracyCV)
}
accuracyN
plot(seqnodesize,accuracyN)


rf.fitimport = randomForest(train[,2:25], as.factor(train[,1]), ntree = 1000, mtry = 7, nodesize = 4, importance = TRUE)
par(mar=rep(2,4))
importance(rf.fitimport)[,4]
barplot(importance(rf.fitimport)[,4])
predict.tree<-predict(test,rf.fitimport)


#boosting
library(gbm)

gbm.fit = gbm(diagnosis~.,data = train,distribution="adaboost", n.trees=2000, shrinkage=0.01, bag.fraction=0.8, cv.folds=5)
usetree = gbm.perf(gbm.fit, method="cv")
usetree
hyper_grid <- expand.grid(
  learning_rate = seq(0.01,0.1,0.01),
  RMSE = NA,  trees = NA,  time = NA)
for(i in seq_len(nrow(hyper_grid))) {
  train_time <- system.time({
    m <- gbm(
      diagnosis~.,data = train,distribution = "adaboost",
      n.trees = 2000, 
      shrinkage = hyper_grid$learning_rate[i], 
      bag.fraction=0.8,cv.folds = 5 
    )
  })
  hyper_grid$RMSE[i]  <- sqrt(min(m$cv.error))
  hyper_grid$trees[i] <- which.min(m$cv.error)
  hyper_grid$time[i]  <- train_time[["elapsed"]]
}
hyper_grid
predgbm<-predict(gbm.fit, test, n.trees=608, type = "response",
                 single.tree = FALSE)
threshold<-0.5
gbmpredvalues<-ifelse(predgbm>threshold,1,0)
confusionMatrix(as.factor(test[,1]), as.factor(gbmpredvalues))
