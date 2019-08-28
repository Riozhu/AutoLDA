==================================================================
###AutoLDA 
An ensemble method for efficiently selecting the optimal number of topics in LDA topic models
==================================================================
Author: Timothy Graham, Guangnan Zhu, Paul Henman
Date: August 28th 2019

—————————————————————————————————————————————————————————
This code was tested to run successfully in Mac OS X Sierra and Nectar Server (Linux).To run the code, there are two approaches and need to set up following dependencies first:
1. R (version 3.5.0 (2018-04-23). Can be found in https://cran.r- project.org/mirrors.html
2. R studio(R version 3.5.0 Copyright (C) 2018 The R Foundation for Statistical Computing Platform: x86_64-apple-darwin15.6.0 (64-bit)
Can be found in https://www.rstudio.com/products/RStudio/

The dataset folder contains the Newsgroup dataset. Before you run, please install the R packages in load_dataset.R.

##
Run
—————————————————————————————————————————————————————————
The code in auto_LDA_reuters.R file is AutoLDA method. You can test it in load_dataset.R. Notice that since AutoLDA is developed from LDA method and ldatuning package, you must install packages "tm" and "ldatuning" before you use AutoLDA method.


Known issue
—————————————————————————————————————————————————————————
1. Install the packages may have occur some error in namespace. You can just copy
 the error and use stack overflow to check how to fix it. Because we are not sure
 which install may occur error, so in this readme file, we won’t list the errors.
2. The execution time could be long. Please just wait for it.