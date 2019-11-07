# packages
library(tm)
library(ldatuning)
library(quanteda)
library(rstudioapi)
library(stringr)
library(slam)
library(ModelMetrics)


#=============================================================================
# Load Dataset
#=============================================================================
current_path = rstudioapi::getActiveDocumentContext()$path 
# set current word dictinoary
setwd(dirname(current_path ))
reuters <- read.csv("./reuters/reutersCSV.csv", stringsAsFactors = FALSE)

#=============================================================================
# Clean Dataset
# combine document title & text into single string; extract company tags; clean data
# Reference:
#   Author: ClairBee
#   Website: https://github.com/ClairBee/cs909/blob/master/Text-classification-with-LDA.R
#=============================================================================
clean.data <- function(data) {
  
  #combine document text & title into single string
  data$doc <- paste(data$doc.title, data$doc.text, sep = " ")
  
  # extract company identifiers
  # correct < symbol from &lt; - allows us to pull out company identifiers
  data$doc <- gsub("&lt;", "<", data$doc)
  
  # replace non-apostrophe punctuation with spaces
  # (splits up eg. Jun/Jul - otherwise this would appear as a single word)
  data$doc <- gsub("[^[:alnum:][:space:]<>]", " ", data$doc)
  
  # convert all factors back into character strings
  data <- data.frame(lapply(data, as.character), stringsAsFactors = FALSE)
  
  # extract company tags as single unbroken strings
  companytags <- lapply(str_extract_all(data$doc, "(<.+?>)"), paste, collapse = " ")
  companytags <- gsub(" ", "", companytags)       # remove strings from company tags
  companytags <- gsub("><", "> <", companytags)   # replace spaces between company tags
  
  # remove company tags from main document (they are essentially a duplicate)
  data$doc <- str_replace_all(data$doc, "(<.+?>)", "")  
  
  # data cleaning
  # has to be done here instead of in DTM function - 
  # otherwise bigrams often include stopwords or whitespace
  # stemming is performed as part of DTM creation - not as effective at this point.
  
  # general text cleaning
  data$doc <- tolower(data$doc)
  data$doc <- removePunctuation(data$doc)
  data$doc <- removeNumbers(data$doc)
  data$doc <- removeWords(data$doc, stopwords("english"))
  data$doc <- stripWhitespace(data$doc)
  
  cbind.data.frame(data, companytags, stringsAsFactors = FALSE)
}

reuters <- clean.data(reuters)

# tag topics (this creates duplicates of some lines, one for each topic of interest - equally weighted)
get.topics <- function(data, topic.list) {
  
  # data frame to hold output
  tagged <- data.frame()
  
  # iterate over all columns, extracting documents with that tag applied
  for (i in 4:138) {
    
    # ignore any columns without tags
    if (sum(as.numeric(data[,i])) > 0) {
      
      # pick up topic name from column header
      topic.name <- gsub("topic.", "", colnames(data)[i])
      
      # only tag topics of interest
      if (topic.name %in% topic.list) {
        
        just.tagged <- cbind("pid" = data[data[,i] == 1, 1], 
                             "topic" = topic.name)
        
        tagged <- rbind(tagged, just.tagged)
      }
    }
  }
  
  # extract all documents that don't relate to topics on the list and tag as 'Other'
  untagged <- cbind("pid" = data[!data$pid %in% tagged$pid, 1],
                    "topic" = "_other")
  
  # output all data with topic tags assigned
  data.frame(lapply(rbind(tagged, untagged), as.character), stringsAsFactors = F)
}

#=============================================================================
# Load ground truth labels(topics)
#=============================================================================
path <- "./dataset/all-topics-strings.lc.txt"
print(path)
topics.of.interest <- NA
conn <- file(path,open="r")
lines <- readLines(conn)
for (i in 1:length(lines)){
  topics.of.interest <- c(topics.of.interest, lines[i])
}
close(conn)
topics.of.interest <- topics.of.interest[-1]
topic.list <- get.topics(reuters, topics.of.interest)


#=============================================================================
# CREATE DOCUMENT-TERM MATRIX
## create a Document-Term Matrix and apply a number of preprocessing transformations
## many preprocessing transformations take place by default: 
## removing punctuation, lower casing, stripping whitespace
#=============================================================================
# create a bigram tokenizer (will include unigrams & bigrams)
bigram_tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2))

# create a DTM of unigrams & bigrams
dtm.all <- DocumentTermMatrix(Corpus(VectorSource(reuters$doc)),
                              control = list(tokenize = scan_tokenizer,
                                             stemming = T))

# remove sparse terms (LDA does not work well with sparse terms)
dtm.all <- removeSparseTerms(dtm.all, .95)

# replace document number with PID
dtm.all$dimnames$Docs <- reuters$pid

# for training LDA, remove 'noise' documents from training set
lda.dtm <- reuters$pid[reuters$pid %in% c(reuters$pid[reuters$purpose %in% c("train", "test")],
                                          topic.list$pid[topic.list$topic %in% topics.of.interest])]

selected_doc <- lda.dtm

lda.dtm <- dtm.all[dtm.all$dimnames$Docs %in% lda.dtm,]

# remove empty rows
selected_doc <- as.vector(which(row_sums(lda.dtm) > 0))
lda.dtm <- lda.dtm[row_sums(lda.dtm) > 0, ]

#=============================================================================
# Train Auto LDA on Reusters dataset GENERATE TOPIC MODELS
#=============================================================================

## Upper bound of the topics
max_topics = 190


## training dataset to data frame
doc_top_matrix <- data.frame(reuters[,4:138])
cnames <- colnames(doc_top_matrix)

doc_topic_matrix <- sapply(doc_top_matrix, as.integer)

colnames(doc_topic_matrix) <- cnames

id <- c(1:nrow(doc_topic_matrix))
doc_topic_matrix <- cbind(doc_topic_matrix, id)

doc_topic_matrix <- data.frame(matrix(doc_topic_matrix, nrow = nrow(doc_topic_matrix), ncol = ncol(doc_topic_matrix)))

colnames(doc_topic_matrix) <- c(cnames, "id")

doc_topic_matrix <- doc_topic_matrix[selected_doc,]

id <- c(1:nrow(doc_topic_matrix))

doc_topic_matrix$id <- id

# Divide the dataset into batches
data <- split(doc_topic_matrix,cut(doc_topic_matrix$id,seq(0,nrow(doc_topic_matrix),length.out=21)))

labeled_y <- NA

for (i in 1:length(data)) {
  d <- data[[i]]
  d <- d[,-ncol(d)]
  label_k <- length(which(apply(d,2,sum)!=0))
  labeled_y <- c(labeled_y, label_k)
}

labeled_y <- labeled_y[-1]

# Results table for record the results
results.table_reuters <- data.frame(matrix(NA, nrow = 1, ncol = 9))
colnames(results.table_reuters) <- c("Labeled_K", "New_Method(Min)",
                                     "New_Method(Mean)", "Griffiths2004","CaoJuan2009", "Arun2010", "Deveaud2014", 
                                     "Running_Time_New", "Running_Time_Old")

results.table_reuters$Batch <- c(1:length(labeled_y))

source("auto_LDA.R")

toy_data <- data[[1]]

cat("\n\n===============================================\n")

# Labeled number of topics (labeled k)
k_target <- labeled_y[1]

dtm <- lda.dtm[toy_data$id,]

# Iteration auto_LDA
Starttime <- Sys.time()
topics = seq(2, max_topics, by=50)
topics <- c(topics, max_topics)
k_values_my <- train_auto_LDA(dtm, topics, max_topics - 2, verbose.al.lt = TRUE)
Endtime <- Sys.time()

## train Four Measures separately Result
Starttime.Four <- Sys.time()
topics = seq(2, max_topics, by=1)
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

## 
cat("\n Calculate Four Measures finished \n")

#K values
results.table_reuters$Labeled_K[1] <- k_target

k_min <- k_values_my[2]
k_mean <- k_values_my[1]

results.table_reuters$`New_Method(Min)`[1] <- k_min
results.table_reuters$`New_Method(Mean)`[1] <- k_mean


results.table_reuters$Griffiths2004[1] <- Grif
results.table_reuters$CaoJuan2009[1] <- CaoJ
results.table_reuters$Arun2010[1] <- Arun
results.table_reuters$Deveaud2014[1] <- Deve

#Running Time
results.table_reuters$Running_Time_New[1] <- difftime(Endtime, Starttime, units='mins')
results.table_reuters$Running_Time_Old[1] <- difftime(Endtime.Four, Starttime.Four, units='mins')

cat("===============================================\n")


#=============================================================================
# Evaluation
#=============================================================================
evaluation_tbl <- data.frame(matrix(NA, ncol = 7, nrow = 2))
results.table_evaluation <- results.table[,1:(ncol(results.table)-2)]


# Calculate the MAE for differnt methods
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

