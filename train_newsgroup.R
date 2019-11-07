# working directory
path <- dirname(rstudioapi::getSourceEditorContext()$path)
source(paste(path, "auto_lda.R", sep = "/"))

## load the text mining (tm) package
library(tm)
library(ldatuning)

## load dplyr and tidyr for data munging
library(dplyr)
library(tidyr)

## Metrics
library(ModelMetrics)

data.newsgroups <- read.csv("http://ssc.wisc.edu/~ahanna/20_newsgroups.csv", stringsAsFactors = FALSE)
data.newsgroups <- data.newsgroups[sample(nrow(data.newsgroups), nrow(data.newsgroups)), ]
data.newsgroups <- tbl_df(data.newsgroups)
names(data.newsgroups)
head(data.newsgroups$text, 2)
## pre-processing
data.newsgroups$text <- tolower(data.newsgroups$text)
data.newsgroups$text <- removePunctuation(data.newsgroups$text)
data.newsgroups$text <- removeNumbers(data.newsgroups$text)
data.newsgroups$text <- removeWords(data.newsgroups$text, stopwords("english"))
data.newsgroups$text <- stripWhitespace(data.newsgroups$text)
head(data.newsgroups$text, 2)

## sample 20% of the data
data <- split(data.newsgroups,cut(data.newsgroups$X,seq(0,nrow(data.newsgroups),length.out=21)))
# toy datasst (the first batch)
toy_data <- data[[1]][-1]
k_target <- length(unique(toy_data$target))

results.table <- data.frame(matrix(NA, nrow = 1, ncol = 9))
colnames(results.table) <- c("Labeled_K", "New_Method(Min)",
                             "New_Method(Mean)", "Griffiths2004","CaoJuan2009", "Arun2010", "Deveaud2014", 
                             "Running_Time_New", "Running_Time_Old")

# sparse document term matrix of the corpus
C <- myCorpus_this <- VCorpus(VectorSource(toy_data$text))
dtm <- DocumentTermMatrix(C, control = list(stemming = F, tolower = TRUE, removeNumbers = T, removePunctuation = TRUE, language = c("english"), stopwords = T, stemming = T))
dtm <- removeSparseTerms(dtm, .95)

# Iteration LDA
Starttime <- Sys.time()
upper_bound <- length(dtm$dimnames$Terms)
topics = seq(2, upper_bound, by=50)
# 
k_values_my <- train_Auto_LDA(dtm, topics, (upper_bound - 2), verbose.al.lt = TRUE)
Endtime <- Sys.time()

## Four Measures Result
Starttime.Four <- Sys.time()
topics = seq(2, upper_bound, by=1)
system.time({
  tunes_Four <- FindTopicsNumber(
    dtm,
    topics = topics,
    metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
    method = "Gibbs",
    control = list(seed = 77),
    mc.cores = 4L,
    verbose = TRUE
  )
})
Endtime.Four <- Sys.time()

Grif <- which.max(tunes_Four$Griffiths2004) + 1
Deve <- which.max(tunes_Four$Deveaud2014) + 1
CaoJ <- which.min(tunes_Four$CaoJuan2009) + 1
Arun <- which.min(tunes_Four$Arun2010) + 1

results.table$Labeled_K[1] <- k_target
results.table$`New_Method(Min)`[1] <- k_values_my[2]
results.table$`New_Method(Mean)`[1] <- k_values_my[1]
results.table$Griffiths2004[1] <- Grif
results.table$CaoJuan2009[1] <- CaoJ
results.table$Arun2010[1] <- Arun
results.table$Deveaud2014[1] <- Deve

## Evaluation - running time
# difftime(timeEnd, timeStart, units='mins')
results.table$Running_Time_New[1] <- difftime(Endtime, Starttime, units='mins')
results.table$Running_Time_Old[1] <- difftime(Endtime.Four, Starttime.Four, units='mins')
results.table$Running_Time_New <- round(as.numeric(results.table$Running_Time_New), 2)
results.table$Running_Time_Old <- round(as.numeric(results.table$Running_Time_Old), 2)

View(results.table)

## Evaluation - Mean absolute error
results.table_evaluation <- results.table[,1:(ncol(results.table)-2)]
evaluation_tbl <- data.frame(matrix(NA, ncol = 7, nrow = 1))

for(i in 2:ncol(results.table_evaluation)) {
  mae_error <- mae(results.table_evaluation$Labeled_K, results.table_evaluation[,i])
  evaluation_tbl[1, i] <- mae_error
}
colnames(evaluation_tbl) <- c("Label K", "New_Method(min)","New_Method(mean)","Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014")

evaluation_tbl$`Label K` <- results.table$Labeled_K

View(evaluation_tbl)

## Qualitative Inspection of Learned Representations on LDA model using the k we obtain from autoLDA
lda.model <- LDA(dtm, k = results.table$`New_Method(min)`, control = list(seed=seedNum))
beta_lda <- tidy(lda.model, matrix = "beta")
dff <- tidy(lda.model, matrix = "gamma")
ldaOut <- as.matrix(terms(lda.model, 20))
ldaOut.topics <- as.matrix(topicmodels::topics(lda.model))
