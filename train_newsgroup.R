# working directory
source("auto_LDA.R")
# if you use Rstudio
# path <- dirname(rstudioapi::getSourceEditorContext()$path)
# source(paste(path, "auto_lda.R", sep = "/"))



## load the text mining (tm) package
library(tm, verbose = FALSE)
library(ldatuning, verbose = FALSE)

## load dplyr and tidyr for data munging
library(dplyr, quietly = TRUE)
library(tidyr, verbose = FALSE)

## Metrics
library(ModelMetrics, quietly = TRUE)

data.newsgroups <- read.csv("http://ssc.wisc.edu/~ahanna/20_newsgroups.csv", stringsAsFactors = FALSE)
data.newsgroups <- data.newsgroups[sample(nrow(data.newsgroups), nrow(data.newsgroups)), ]
data.newsgroups <- tbl_df(data.newsgroups)
names(data.newsgroups)
# head(data.newsgroups$text, 2)
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
toy_data <- data[[1]]
toy_data <- split(toy_data,cut(toy_data$X,seq(0,nrow(toy_data),length.out=6)))
k_target <- length(unique(toy_data$target))

results.table <- data.frame(matrix(NA, nrow = length(toy_data), ncol = 10))
colnames(results.table) <- c("Batch", "Labeled_K", "New_Method(Min)",
                             "New_Method(Mean)", "Griffiths2004","CaoJuan2009", "Arun2010", "Deveaud2014", 
                             "Running_Time_New", "Running_Time_Old")

#======================
for (i in 1:length(toy_data)) {
  d <- toy_data[[i]][-1]
  
  # Labeled number of topics (labeled k)
  k_target <- length(unique(d$target))
  
  ## create a Document-Term Matrix and apply a number of preprocessing transformations
  ## many preprocessing transformations take place by default: 
  ## removing punctuation, lower casing, stripping whitespace
  C <- myCorpus_this <- VCorpus(VectorSource(d$text))
  dtm <- DocumentTermMatrix(C, control = list(stemming = F, tolower = TRUE, removeNumbers = T, removePunctuation = TRUE, language = c("english"), stopwords = T, stemming = T))
  dtm <- removeSparseTerms(dtm, .95)
  # l <- LDA(dtm, k = 20, control = list(seed=seedNum))
  # ldaOut <- as.matrix(terms(l, 10))
  
  # Iteration LDA
  Starttime <- Sys.time()
  upper_bound <- length(dtm$dimnames$Terms)
  topics = seq.int(2, upper_bound, by=50)
  topics = c(topics, upper_bound)
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
  
  # 
  
  #K values
  results.table$Labeled_K[i] <- k_target
  results.table$`New_Method(Min)`[i] <- k_values_my[2]
  results.table$`New_Method(Mean)`[i] <- k_values_my[1]
  results.table$Griffiths2004[i] <- Grif
  results.table$CaoJuan2009[i] <- CaoJ
  results.table$Arun2010[i] <- Arun
  results.table$Deveaud2014[i] <- Deve
  
  #Running Time
  # difftime(timeEnd, timeStart, units='mins')
  results.table$Running_Time_New[i] <- difftime(Endtime, Starttime, units='mins')
  results.table$Running_Time_Old[i] <- difftime(Endtime.Four, Starttime.Four, units='mins')
  results.table$Running_Time_New[i] <- round(as.numeric(results.table$Running_Time_New[i]), 2)
  results.table$Running_Time_Old[i] <- round(as.numeric(results.table$Running_Time_Old[i]), 2)
  
  cat("Batch ", i, " is finished.\n\n", sep = "")
  cat("===============================================\n")
  
  
} 

# Rstuio
# View(results.table) 
# Command Line
print(results.table)
# save table
write.csv(results.table, "./results/results_newsgroup.csv")


## Evaluation - Mean absolute error
results.table_evaluation <- results.table[,1:(ncol(results.table)-2)]
evaluation_tbl <- data.frame(matrix(NA, ncol = 6, nrow = 1))

for(i in 3:ncol(results.table_evaluation)) {
  mae_error <- mae(results.table_evaluation$Labeled_K, results.table_evaluation[,i])
  evaluation_tbl[1, i-2] <- mae_error
}
colnames(evaluation_tbl) <- c("New_Method(min)","New_Method(mean)","Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014")

# Rstudio
# View(evaluation_tbl)
# Command Line
print(evaluation_tbl)
# save table
write.csv(evaluation_tbl, "./results/evaluation_newsgroup.csv")

## Qualitative Inspection of Learned Representations on LDA model using the k we obtain from autoLDA
lda.model <- LDA(dtm, k = results.table$`New_Method(min)`, control = list(seed=seedNum))
beta_lda <- tidy(lda.model, matrix = "beta")
dff <- tidy(lda.model, matrix = "gamma")
ldaOut <- as.matrix(terms(lda.model, 20))
ldaOut.topics <- as.matrix(topicmodels::topics(lda.model))
print(ldaOut.topics)
