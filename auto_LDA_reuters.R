library(tm)
library(ldatuning)





train_auto_LDA <- function(dtm, topics, range.topics, seed = 77, mc.cores = 4L, verbose.al.al = TRUE, verbose.al.lt = FALSE) {
  Iter <- 0
  startTime <- Sys.time()
  
  k.train <- auto_lda(dtm, topics, range.topics, seed, mc.cores, verbose.al.al, verbose.al.lt, Iter)
  
  endTime <- Sys.time()
  
  time.diff <- difftime(endTime, startTime, units='mins')
  
  cat("Time Running:", time.diff, "\n\n", sep = " ")
  
  return(k.train)
}



auto_lda <- function(dtm, topics, range.topics, seed = 77, mc.cores = 4L, verbose.al.al = TRUE, verbose.al.lt = FALSE, Iter) {
  system.time({
    tunes <- FindTopicsNumber(
      dtm,
      topics = topics,
      metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
      method = "Gibbs",
      control = list(seed = 77),
      mc.cores = mc.cores,
      verbose = verbose.al.lt
    )
  })
  
  columns <- base::subset(tunes, select = 2:ncol(tunes))
  tunes <- base::data.frame(
    tunes["topics"],
    base::apply(columns, 2, function(column) {
      scales::rescale(column, to = c(0, 1), from = range(column))
    })
  )
  
  tunes <- tunes
  
  tunes$mean_max = rowMeans(tunes[, c("Griffiths2004", "Deveaud2014")])
  tunes$mean_min = rowMeans(tunes[, c("CaoJuan2009", "Arun2010")])
  
  min_range = tunes[which.min(tunes$mean_min), ]$topics
  max_range = tunes[which.max(tunes$mean_max), ]$topics
  
  if(min_range > max_range) {
    tx <- min_range
    min_range <- max_range
    max_range <- tx
  }
  
  period = topics[2] - topics[1]
  
  Iter <- Iter + 1
  
  if(verbose.al.al) {
    cat("Iteration: ", Iter, "\n")
    cat("Measure shows the range is from", min_range, "to", max_range, "\n", sep = " ")
    cat("The period is", period, "\n", sep = " ")
  }
  
  
  if(period == 1) {
    if(max_range - min_range > 20) {
      cat("Warning: Measures disagree each others much. \n")
      cat("THe best number of suggested topic is in the range of", min_range, "and", max_range, "\n",sep = " ")
      # cat("The best K is", floor(mean(as.vector(c(min_range,max_range)))), sep = " ")
      # result <- floor(mean(as.vector(c(min_range,max_range))))
      cat("The min best K is", min_range, sep = " ")
      cat("The mean best K is", floor(mean(as.vector(c(min_range,max_range)))), sep = " " )
      result <- c(floor(mean(as.vector(c(min_range,max_range)))),min_range)
      
      
    } else {
      result <- floor(mean(as.vector(c(min_range,max_range))))
      cat("THe best number of suggested topic is in the range of", min_range, "and", max_range, "\n", sep = " ")
      cat("The best K is", result, "\n", sep = " ")
      cat("The best K is", min_range, "\n", sep = " ")
      result <- c(result, min_range)
    }
  } else {
    if(max_range - min_range > 200) {
      cat("Warning: the measure shows that the range of k is too big \n")
      result <- NA
    } else {
      if(max_range - min_range > 100 && max_range - min_range <= 200) {
        topics = seq(min_range, max_range, by=20)
        range.topics <- max_range - min_range
        result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
      } else { 
        if(max_range - min_range <= 100 && max_range - min_range > 50) {
          if(range.topics > 100) {
            topics = seq(min_range, max_range, by=10)
            range.topics <- max_range - min_range
            result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
          } else if(range.topics <= 100 && period == 10) {
            topics = seq(min_range, max_range, by=5)
            range.topics <- max_range - min_range
            result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
          } else if(range.topics <= 100 && period == 5) {
            if(verbose.al.al) {
              cat("Warning: Measures disagree with the each other. Where the range is from", min_range, "to", max_range, ", and increasing period is", period, "\n", sep = " " )
            }
            
            topics = seq(min_range, max_range, by=2)
            range.topics <- max_range - min_range
            result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
          } else if(range.topics <= 100 && period == 2) {
            if(verbose.al.al) {
              cat("Warning: Measures disagree with the each other. Where the range is from", min_range, "to", max_range, ", and increasing period is", period, "\n", sep = " " )
            }
            topics = seq(min_range, max_range, by=1)
            range.topics <- max_range - min_range
            result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
          }
        } else {
          if(max_range - min_range > 20 && max_range - min_range <= 50) {
            if(range.topics > 50) {
              topics = seq(min_range, max_range, by=5)
              range.topics <- max_range - min_range
              result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
            } else if (range.topics <= 50 && period == 5) {
              topics = seq(min_range, max_range, by=2)
              range.topics <- max_range - min_range
              result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
            } else if (range.topics <= 50 && period == 2) {
              if(verbose.al.al) {
                cat("Warning: Measures disagree with the each other. Where the range is from", min_range, "to", max_range, ", and increasing period is", period, "\n", sep = " " )
              }
              topics = seq(min_range, max_range, by=1)
              range.topics <- max_range - min_range
              result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
            }
          } else {
            if(max_range - min_range > 10 && max_range - min_range <= 20) {
              if(range.topics > 20) {
                topics = seq(min_range, max_range, by=2)
                range.topics <- max_range - min_range
                result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
              } else {
                topics = seq(min_range, max_range, by=1)
                range.topics <- max_range - min_range
                result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
              }
            } else {
              if (max_range - min_range >= 1 && max_range - min_range <= 10) {
                topics = seq(min_range, max_range, by=1)
                range.topics <- max_range - min_range
                result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
              } else {
                if (max_range - min_range  == 0) {
                  topics = seq(min_range - period, max_range + period, by=1)
                  range.topics <- max_range - min_range
                  result <- auto_lda(dtm, topics, range.topics, Iter = Iter)
                }
              }
            }
          }
        }
      }
    }
  }
  cat("\n")
  
  # return.range <- max_range - min_range
  
  return(result)
}