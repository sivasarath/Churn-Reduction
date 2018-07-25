rm(list=ls(all=T))
setwd("C:/Users/sarath chandra/Desktop/data science/new project")

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

install.packages(x)
lapply(x, require, character.only = TRUE)


## Read the data
marketing_train = read.csv("Train_data.csv", header = F, na.strings = c(" ", "", "NA"))
test=read.csv("Test_data.csv", header = F, na.strings = c(" ", "", "NA"))

plot(x=marketing_train$V21)
transform(marketing_train, new=as.numeric(marketing_train$V4))
hist(new)

numeric_index = sapply(marketing_train,is.numeric) 

numeric_data = marketing_train[,numeric_index]

cnames = colnames(numeric_data)

for (i in 1:length(cnames))
   {
     assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "responded"), data = subset(marketing_train))+ 
              stat_boxplot(geom = "errorbar", width = 0.5) +
              geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                           outlier.size=1, notch=FALSE) +
              theme(legend.position="bottom")+
              labs(y=cnames[i],x="responded")+
              ggtitle(paste("Box plot of responded for",cnames[i])))
}

for(i in cnames){
     val = marketing_train[,i][marketing_train[,i] %in% boxplot.stats(marketing_train[,i])$out]
     #print(length(val))
     marketing_train[,i][marketing_train[,i] %in% val] = NA
  }
   
   marketing_train = knnImputation(marketing_train, k = 3)
train=marketing_train 
C50_model = C5.0(responded ~., train, trials = 100, rules = TRUE)
C50_Predictions = predict(C50_model, test[,-21], type = "class")

ConfMatrix_C50 = table(test$responded, C50_Predictions)
confusionMatrix(ConfMatrix_C50)

RF_model = randomForest(responded ~ ., train, importance = TRUE, ntree = 500)

treeList = RF2List(RF_model)
exec = extractRules(treeList, train[,-17])
RF_Predictions = predict(RF_model, test[,-17])

NB_model = naiveBayes(responded ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:20], type = 'class')

#Look at confusion matrix
Conf_matrix = table(observed = test[,21], predicted = NB_Predictions)
confusionMatrix(Conf_matrix)