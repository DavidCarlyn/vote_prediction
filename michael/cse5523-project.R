library(tidyverse)
library(gridExtra)
library(ISLR)
library(MASS)
library(boot)
library(glmnet)
library(tree)
library(randomForest)
library(ggplot2)
library(readr)

all_features <- read_csv("all_features.csv")
View(all_features)

hist(all_features$`1`)

all_features <- all_features %>% dplyr::select(-c(X1)) %>% mutate(DV = as.factor(DV))


set.seed(4620)
x <- model.matrix(DV ~ ., all_features)[,-1]
y <- all_features$DV
ix <-  sample(1:nrow(x),nrow(x)/5) # 80/20 split
x.train <- x[-ix,]
x.test <- x[ix,]
y.train <- y[-ix]
y.test <- y[ix]

x.train <- as.matrix(read_csv('X_train.csv', col_names = FALSE))
x.test <- as.matrix(read_csv('X_test.csv', col_names = FALSE))
y.train <- read_csv('y_train.csv', col_names = FALSE)
y.train <- as.factor(y.train$X1)
y.test <- read_csv('y_test.csv', col_names = FALSE)
y.test <- as.factor(y.test$X1)



lasso.cv.resp <- cv.glmnet(x.train, y.train, alpha = 1, family = "binomial")
plot(lasso.cv.resp)
lambda.cv.resp <- lasso.cv.resp$lambda.min
lambda.cv.resp


fit.lasso.resp <- glmnet(x.train, y.train, alpha = 1, family = "binomial", lambda = lambda.cv.resp)
pred.lasso <-  predict(fit.lasso.resp, newx = x.test, type = "class")
confusionMatrix(as.factor(pred.lasso), y.test)
coef(fit.lasso.resp)


all_features %>% ggplot() + geom_histogram(aes(x = `10`, y = ..density.., fill = DV))
all_features %>% ggplot() + geom_density(aes(x = `12`, fill = DV))
hist(all_features$`0`, probability = TRUE)

all_features %>% ggplot() + geom_boxplot(aes(x = `DV`, y = `1`))

all_features %>% filter(DV == 1) %>% summarise(avg = mean(`98`), n = n())
all_features %>% filter(DV == 0) %>% summarise(avg = mean(`98`), n = n())

all_features[5,]

logistic.fit <- cv.glmnet()
