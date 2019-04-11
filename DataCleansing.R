library(lattice)
library(ggplot2)
library(caret)
# Read train data
data.train <- read.csv("F:/7390/mid-term project/application_train.csv/application_train.csv")
# Delete cols which have more than 10% missing values
fMiss <- function(x) {sum(is.na(x))/length(x)*100}
miss.table.train <- apply(data.train,2,fMiss)
miss.table.train
missnames.train <- which(miss.table.train >= 10)
df <- data.frame(missnames.train)
head(df)
nrow(df)
hist(df)
data.train.new <- data.train[,-which(names(data.train)%in%names(missnames.train))]
# Seperate categorical and numerical variables
train.f <- data.frame(0)
train.f <- train.f[-1]
train.n <- data.frame(0)
train.n <- train.n[-1]
for (i in 1:length(data.train.new)){
  if (is.factor(data.train.new[,i])){
    train.f <<- data.frame(train.f,data.train.new[i])
  }
  else{
    train.n <<- data.frame(train.n,data.train.new[i])
  }
}
str(train.f)
str(train.n)
# Fill missing values with central imputation
install.packages("DMwR2")
library(DMwR2)
train.n.fill <- centralImputation(train.n)
# Transfer categorical values into numerical
install.packages("dummies")
library(dummies)
train.f.dummy <- dummy.data.frame(train.f)
# Combine numerical and dummy train data
train.co <- data.frame(train.n.fill, train.f.dummy)
# Write to csv
write.csv(train.co,"D:/trainpre.csv",row.names = F)
# Read from csv
train.co <- read.csv("D:/trainpre.csv")
gc()

# Read bureau data
bureau <- read.csv("D:/all/bureau.csv")
str(bureau)
# Delete cols which have more than 10% missing values
miss.table.bureau <- apply(bureau,2,fMiss)
missnames.bureau <- which(miss.table.bureau >= 10)
bureau.new <- bureau[,-which(names(bureau)%in%names(missnames.bureau))]
str(bureau.new)
# Seperate categorical and numerical variables
bureau.f <- data.frame(0)
bureau.f <- bureau.f[-1]
bureau.n <- data.frame(0)
bureau.n <- bureau.n[-1]
for (i in 1:length(bureau.new)){
  if (is.factor(bureau.new[,i])){
    bureau.f <<- data.frame(bureau.f,bureau.new[i])
  }
  else{
    bureau.n <<- data.frame(bureau.n,bureau.new[i])
  }
}

str(bureau.f)
str(bureau.n)
# Seperate variables of using sum or mean aggregation in numerical data
sumnames.bureau <- c('SK_ID_CURR','CNT_CREDIT_PROLONG','AMT_CREDIT_SUM')
bureau.n.sum <- bureau.n[,which(names(bureau.n)%in%sumnames.bureau)]
meannames.bureau <- c('SK_ID_CURR','DAYS_CREDIT','CREDIT_DAY_OVERDUE','DAYS_CREDIT_ENDDATE','AMT_CREDIT_SUM_OVERDUE','DAYS_CREDIT_UPDATE')
bureau.n.mean <- bureau.n[,which(names(bureau.n)%in%meannames.bureau)]
# Aggregate numerical data
bureau.sum.aggr <- aggregate(bureau.n.sum, by = list(SK_ID_CURR=bureau.n.sum$SK_ID_CURR), FUN = sum)[,-2]
bureau.mean.aggr <- aggregate(bureau.n.mean, by = list(SK_ID_CURR=bureau.n.mean$SK_ID_CURR), FUN = mean)[,-2]
# Fill missing values with central imputation
bureau.n.aggr <- centralImputation(data.frame(bureau.sum.aggr, bureau.mean.aggr[-1]))
# Transfer categorical values into numerical
bureau.f.dummy <- dummy.data.frame(bureau.f)
bureau.f.dummy <- data.frame(bureau[1], bureau.f.dummy)
str(bureau.f.dummy)
# Aggregate dummy data
bureau.dummy.aggr <- aggregate(bureau.f.dummy, by = list(SK_ID_CURR=bureau.f.dummy$SK_ID_CURR), FUN = max)[,-2]
# Combind bureau data
bureau.co <- data.frame(bureau.n.aggr, bureau.dummy.aggr[-1])
bureau.co <- bureau.co[order(bureau.co[,1]),]
head(bureau.co)
# Join bureau and train
train.b <- merge(train.co, bureau.co, by = "SK_ID_CURR", all.x = TRUE)
# Write to csv
write.csv(train.b,"D:/train+bureau.csv",row.names = F)
# Read from csv
train.b <- read.csv("D:/train+bureau.csv")

# Read previous_application data
previous <- read.csv("D:/all/previous_application.csv")
str(previous)
# Delete cols which have more than 10% missing values
miss.table.previous <- apply(previous,2,fMiss)
missnames.previous <- which(miss.table.previous >= 10)
previous.new <- previous[,-which(names(previous)%in%names(missnames.previous))]
str(previous.new)
# Seperate categorical and numerical variables
previous.f <- data.frame(0)
previous.f <- previous.f[-1]
previous.n <- data.frame(0)
previous.n <- previous.n[-1]
for (i in 1:length(previous.new)){
  if (is.factor(previous.new[,i])){
    previous.f <<- data.frame(previous.f,previous.new[i])
  }
  else{
    previous.n <<- data.frame(previous.n,previous.new[i])
  }
}

str(previous.f)
str(previous.n)
# Aggregate numerical data
previous.n.aggr <- aggregate(previous.n, by = list(SK_ID_CURR=previous.n$SK_ID_CURR), FUN = mean)[,-c(2,3)]
# Fill missing values with central imputation
previous.n.aggr <- centralImputation(previous.n.aggr)
# Transfer categorical values into numerical
previous.f.dummy <- dummy.data.frame(previous.f)
previous.f.dummy <- data.frame(previous[2], previous.f.dummy)
str(previous.f.dummy)
# Aggregate dummy data
previous.dummy.aggr <- aggregate(previous.f.dummy, by = list(SK_ID_CURR=previous.f.dummy$SK_ID_CURR), FUN = max)[,-2]
# Combind previous data
previous.co <- data.frame(previous.n.aggr, previous.dummy.aggr[-1])
previous.co <- previous.co[order(previous.co[,1]),]
str(previous.co)
# Join previous and train+brueau
train.bp <- merge(train.b, previous.co, by = "SK_ID_CURR", all.x = TRUE)
train.bp[is.na(train.bp)] <- 0
str(train.bp)
# Write to csv
write.csv(train.bp,"D:/train+bureau+previous.csv",row.names = F)
# Read from csv
train.bp <- read.csv("D:/train+bureau+previous.csv")
# Do random sampling to train data
set.seed(1111)
index <- sample(1:nrow(train.bp), 40000, replace = FALSE)
train.sam <- train.bp[index,]
options(max.print=1000000)

# Linear regression dimension reduction
train.sam$TARGET <- as.factor(train.sam$TARGET)
lrmodel <- glm(TARGET~., family = binomial(link = "logit"), data = train.sam[,-1])
summary(lrmodel)
red.names <- names(which(coef(summary(lrmodel))[,'Pr(>|z|)'] <= 0.05))
train.dr <- train.bp[,which(names(train.bp)%in%red.names)]
train.dr <- data.frame(train.bp[1],train.bp['TARGET'],train.dr)
train.dr$TARGET <- as.factor(train.dr$TARGET)
str(train.dr)
lrmodel2 <- glm(TARGET~., family = binomial(link = "logit"), data = train.dr[,-1])
summary(lrmodel2)
red.names2 <- names(which(coef(summary(lrmodel2))[,'Pr(>|z|)'] <= 0.01))
train.dr.new <- train.dr[,which(names(train.dr)%in%red.names2)]
train.dr.new <- data.frame(train.dr[1],train.dr['TARGET'],train.dr.new)
str(train.dr.new)
# Write to csv
write.csv(train.dr.new,"D:/train_after_dimension_reduction.csv",row.names = F)
# Read from csv
train.dr.new <- read.csv("D:/train_after_dimension_reduction.csv")
# Data partitioning for train and test
train.dr.new$TARGET <- as.factor(train.dr.new$TARGET)
set.seed(1111)
index <- sample(2,nrow(train.dr.new),replace = TRUE,prob = c(0.9,0.1))
final.train <- train.dr.new[index==1,]
final.test <- train.dr.new[index==2,]

lrm <- glm(TARGET~., family = binomial(link = "logit"), data = final.train)
pt <- predict(lrm, final.test)