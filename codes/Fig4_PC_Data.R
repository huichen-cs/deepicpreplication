library(tools)

library(princurve)
files <- list.files(path="C://Users//ZhaoY//Downloads//results//dropbox//Fig3_results//segments", pattern="*.csv", full.names=TRUE, recursive=FALSE)
print(files)
filepath = "C://Users//ZhaoY//Downloads//results//dropbox//pc_data//"
lapply(files, function(x) {
  print(x)
  filename <- tools::file_path_sans_ext(basename(x))
  print(filename)
  savepath <- paste0(filepath, filename, '.csv')
  #print(savepath)
  data <- read.csv(x, header=TRUE) # load file
  #print(names(data))
  # apply function
  mtrxdata <- matrix(unlist(data), ncol = ncol(data), nrow = nrow(data))
  fitdata <-principal_curve(mtrxdata)
  print(fitdata$s[fitdata$ord,])
  ret <- rbind(names(data), fitdata$s[fitdata$ord,])  
  write.table(ret, savepath, sep=",", row.names = FALSE, col.names=FALSE)

})




























