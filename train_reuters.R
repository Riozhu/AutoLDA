# packages
library(tm, verbose = FALSE)
library(ldatuning, verbose = FALSE)
library(quanteda, quietly = TRUE)
library(rstudioapi, verbose = FALSE)
library(stringr, verbose = FALSE)
library(slam, verbose = FALSE)
library(ModelMetrics, verbose = FALSE)


#=============================================================================
# Set WORKING DIRECTORY to source file location
# Load Dataset
#=============================================================================
# if you use Rstudio
# current_path = rstudioapi::getActiveDocumentContext()$path 
# set current word dictinoary
# setwd(dirname(current_path ))
reuters <- read.csv("./reuters/reutersCSV.csv", stringsAsFactors = FALSE)
source("auto_LDA.R")
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
path <- "./reuters/all-topics-strings.lc.txt"
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
# Creat document term matrix
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
toy_data <- data[[1]]
toy_data <- split(toy_data,cut(toy_data$id,seq(0,nrow(toy_data),length.out=6)))

# Labeled number of topics (labeled k)
labeled_y <- NA
for (i in 1:length(toy_data)) {
  d <- toy_data[[i]]
  d <- d[,-ncol(d)]
  label_k <- length(which(apply(d,2,sum)!=0))
  labeled_y <- c(labeled_y, label_k)
}

labeled_y <- labeled_y[-1]

# Results table for record the results
results.table <- data.frame(matrix(NA, nrow = length(toy_data), ncol = 10))
colnames(results.table) <- c("Batch","Labeled_K", "New_Method(Min)",
                                     "New_Method(Mean)", "Griffiths2004","CaoJuan2009", "Arun2010", "Deveaud2014", 
                                     "Running_Time_New", "Running_Time_Old")

cat("\n\n===============================================\n")

for (i in 1:length(toy_data)) {
  cat("Batch: ", i)
  d <- toy_data[[i]]
  k_target <- labeled_y[1]
  # document term matrix of this batch
  dtm <- lda.dtm[d$id,]
  
  # Iteration LDA
  Starttime <- Sys.time()
  upper_bound <- length(dtm$dimnames$Terms)
  topics = seq.default(from = 2, to = upper_bound, by = 50)
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
write.csv(results.table, "./results/result_reuters.csv")

#=============================================================================
# Evaluation
#=============================================================================
evaluation_tbl <- data.frame(matrix(NA, ncol = 6, nrow = 1))
results.table_evaluation <- results.table[,1:(ncol(results.table)-2)]


# Calculate the MAE for differnt methods
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
write.csv(evaluation_tbl, "./results/evaluation_reuters.csv")

## Qualitative Inspection of Learned Representations on LDA model using the k we obtain from autoLDA
lda.model <- LDA(dtm, k = results.table$`New_Method(min)`, control = list(seed=seedNum))
beta_lda <- tidy(lda.model, matrix = "beta")
dff <- tidy(lda.model, matrix = "gamma")
ldaOut <- as.matrix(terms(lda.model, 20))
ldaOut.topics <- as.matrix(topicmodels::topics(lda.model))
print(ldaOut.topics)
