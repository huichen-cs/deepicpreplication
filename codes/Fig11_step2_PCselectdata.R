library(princurve)
library(stringr)
library(dplyr)

# get the SMOTE generated minority index which distance within the original average
files <- list.files(path="C://Users//ZhaoY//Downloads//results//question3//segments//defect", pattern="*.csv", full.names=TRUE, recursive=FALSE)
print(files)
filesavepath = "C://Users//ZhaoY//Downloads//results//question3//SMOTEresultfromR//index//"
smgenerateddefectpath = "C://Users//ZhaoY//Downloads//results//question3//smotegeneratedminoritybeforeR//"

lapply(files, function(x) {
  filename <- tools::file_path_sans_ext(basename(x))
  filename_sep = str_split(filename, "_")
  sourcepath <- paste0(smgenerateddefectpath, unlist(filename_sep)[1], "_smgeneratedmino_", unlist(filename_sep)[3], '.csv')
  filesavename <- paste0(filesavepath, unlist(filename_sep)[1], "_seg_", unlist(filename_sep)[3], '.csv')
  print(sourcepath)
  print(filesavename)
  
  smotedata <- read.csv(sourcepath, header =TRUE)
  mtrxdsmotedata <- matrix(unlist(smotedata), ncol = ncol(smotedata), nrow = nrow(smotedata))
  fitsmotedata <-principal_curve(mtrxdsmotedata)
  
  
  dfect <- read.csv(x, header =TRUE)
  typeof(dfect)
  print(dfect)
  mtrxdfect <- matrix(unlist(dfect), ncol = ncol(dfect), nrow = nrow(dfect))
  fitdfect <-principal_curve(mtrxdfect)
  proj <- project_to_curve(mtrxdfect, fitdfect$s[fitdfect$ord,])
  dfecttotaldist <- proj$dist
  dfectvaedist <- dfecttotaldist/nrow(dfect)
  
  projres <- project_to_curve(mtrxdsmotedata, fitdfect$s[fitdfect$ord,])
  my_list <- list() 
  val_list <- list() 
  j <- 0
  i <- 0
  for (d in projres$dist_ind) {
    if(d<dfectvaedist) {
      
      my_list[j] = i
      val_list[j] = d
      j <- j+1
    }
    i <- i+1
  }
  
  dfall = do.call(rbind, Map(data.frame, A=my_list, B=val_list))
  
  write.csv(dfall,filesavename, row.names = FALSE)
})


















































