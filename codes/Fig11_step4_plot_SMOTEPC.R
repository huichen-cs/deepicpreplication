library(princurve)
library(stringr)
library(dplyr)



files <- list.files(path="C://Users//ZhaoY//Downloads//results//question3//SMOTEresultfromR//balanceddata", pattern="*.csv", full.names=TRUE, recursive=FALSE)
filesavepath = "C://Users//ZhaoY//Downloads//results//question3//resultswithSMOTEPC//bal//"
lapply(files, function(x) {
  filename <- tools::file_path_sans_ext(basename(x))
  filename_sep = str_split(filename, "_")
  sourcepath <- paste0(filesavepath, unlist(filename_sep)[1], "_combselsmdatabal_", unlist(filename_sep)[3], "_bal", '.pdf')
  print(sourcepath)
  
  dori <- read.csv(x, header =TRUE)
#  dori <- select(dori, -c('commitdate'))
  mtrxdori <- matrix(unlist(dori), ncol = ncol(dori), nrow = nrow(dori))
  fitdori <-principal_curve(mtrxdori)
  pdf(sourcepath)
  #jpeg(savepath)
  plot(fitdori)
  points(fitdori)
  dev.off()
})







































