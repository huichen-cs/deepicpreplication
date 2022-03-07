library(princurve)
library(stringr)
library(dplyr)

# plot SMOTEnoPC balanced data
# run rq3pcsepsegdatawithcommitdateSMOTEnoselection.py get data first, then:
# get the SMOTE generated minority index which distance within the original average
files <- list.files(path="C://Users//ZhaoY//Downloads//results//dropbox//FIg11//SMOTEnoPCbalaceddata", pattern="*.csv", full.names=TRUE, recursive=FALSE)
filesavepath = "C://Users//ZhaoY//Downloads//results//dropbox//FIg11//resultsSMOTEnoPCfromR//"
lapply(files, function(x) {
  filename <- tools::file_path_sans_ext(basename(x))
  filename_sep = str_split(filename, "_")
  savepath <- paste0(filesavepath, unlist(filename_sep)[1], "_SMOTEnoPC_", "PCdata_",unlist(filename_sep)[3], '.csv')
  print(savepath)
  
  dori <- read.csv(x, header =TRUE)
  dori <- select(dori, -c('commitdate'))
  mtrxdori <- matrix(unlist(dori), ncol = ncol(dori), nrow = nrow(dori))
  fitdori <-principal_curve(mtrxdori)
  print(fitdori$s[fitdori$ord,])
  ret <- rbind(names(dori), fitdori$s[fitdori$ord,])  
  write.table(ret, savepath, sep=",", row.names = FALSE, col.names=FALSE)
})

















































