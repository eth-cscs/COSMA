library(ggplot2)
library(ggrepel)
library(reshape2)
library(plyr)
#library("reshape2")


path = "C:/gk_pliki/uczelnia/doktorat/performance_modelling/repo/papers/MMM-paper/results" #getwd()
#exp_name = "weak48"
exp_filename = "result.csv"
variants = c("memory_p1","memory_p2", "strong")
sizes = c("square","tall")
variantPlots = c("FLOPS")
algorithms = c("CARMA [22]", "ScaLAPACK [52]","CTF","COSMM")
annotl = c("CARMA [21]","CTF [47]","COSMA (this work)", "ScaLAPACK [51]")

GFLOPSperCore = 1209/36


statistics = 0

annotCoord = list()
#annotCoordX[["square_memory_p1_time"]] = 


#exp_filename = paste(exp_name,'.csv',sep="")
#setwd(paste(path,exp_name,sep =""))
source(paste(path, "/SPCL_Stats.R", sep=""))


# prepare the data 
rawData <- read.csv(file=exp_filename, sep=",", stringsAsFactors=FALSE, header=TRUE)

for (i1 in 1:length(sizes)) {
  size = sizes[i1]
  if (size == 'square') {
    dataFirst = rawData[rawData$m == rawData$k,]
  } else {
    dataFirst = rawData[rawData$m < rawData$k,]
  }
  for (i2 in 1:length(variants)) {
    variant = variants[i2]
    data = dataFirst[dataFirst$case == variant,]
    
    if (nrow(data) == 0)
      next
    
      tscaling <- data[c("p", "algorithm", "time")]
        m = data$m
        tscaling$time = 200* data$m * data$n * data$k / (data$time * 1e6) / (GFLOPSperCore * data$p)
        ylabel = "% peak performance"
        yscale = scale_y_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
        
        name = paste(size, variant, sep="_")
   
      
      print(tscaling[c("algorithm","p","time")])
      
      aspRatio = 0.75
      w = 10
      textSize = 30
      pointSize = 5
      if (size == 'square' && variant == 'strong') {
        annotx = c(9000,712,1524,256)
        annoty = c(50,5,70,20)
        annotPointX1 = c(4096,1024,512,128)
        
        annotPointX2 = c(7000,800,700,150)
        annotPointY2 = c(47,8,67,23)
        limit = ylim(0, 90)
      } else if (size == 'tall' && variant == 'memory_p1')  {
        annotx = c(512,200,506,1224)
        annoty = c(42,5,74,32)
        annotPointX1 = c(1024,256,512,1024)
        
        annotPointX2 = c(900,200,400,900)
        annotPointY2 = c(45,8,70,28)
      }
      else if (size == 'square' && variant == 'memory_p2')  {
        annotx = c(180,200,4048,4396)
        annoty = c(22,5,77,36)
        annotPointX1 = c(128,433,4096,2048)
        
        annotPointX2 = c(150,335,3500,3050)
        annotPointY2 = c(26,5,73,32)
      }
      else {
        next
      }
      
      annotPointY1 = c(tscaling[tscaling$p == annotPointX1[1] & tscaling$algorithm == 'CARMA [21] ',]$time[1],
                      tscaling[tscaling$p == annotPointX1[2] & tscaling$algorithm == 'CTF [47] ',]$time[1],
                      tscaling[tscaling$p == annotPointX1[3] & tscaling$algorithm == 'COSMA (this work) ',]$time[1],
                      tscaling[tscaling$p == annotPointX1[4] & tscaling$algorithm == 'ScaLAPACK [51] ',]$time[1])
      
      # plot the timers
      pdf(file=paste("size_",size, "_var_", variant , "_", varPlot, ".pdf", sep=""),
          width = w, height = w*aspRatio)
      
      limit = yscale
      shapes = scale_shape_manual(values=c(15, 16, 17,18))
      shapesColors = scale_color_manual(values = c("#F8766D", "#7CAE00","#00BFC4",  "#C77CFF"))
      
      data3 = ddply(tscaling, ~ algorithm+p, summarize, min=min(time), max=max(time), mean=median(time))
      
      p <- ggplot(mapping=aes(x=p, y=mean, ymin=min, ymax=max, fill=algorithm, color=algorithm, shape=algorithm)) +
        geom_ribbon(data=data3[data3$algorithm != "CARMA [21] ",], alpha=0.3, show.legend=FALSE)+
        shapes + 
        geom_point(data=data3, size = 4, show.legend=FALSE) +
        geom_errorbar(data=data3[data3$algorithm == "CARMA [21] ",], width=0.1, size=1, show.legend=FALSE) +
        scale_x_continuous(trans='log2',labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
        #scale_x_log2("# of cores", breaks=c(128, 256, 512, 1024, 2048, 4096, 8192, 16384)) +
        # scale_y_log10(ylabel) +
        xlab("# of cores") +
        yscale +
        ylab(ylabel) +
        theme_bw(27) +
        annotate("text", x = annotx, y = annoty, label = annotl, size=textSize/3) +
        annotate("segment", x = annotPointX2[1], xend = annotPointX1[1],
                 y = annotPointY2[1], yend = annotPointY1[1]) +
        annotate("segment", x = annotPointX2[2], xend = annotPointX1[2],
                 y = annotPointY2[2], yend = annotPointY1[2]) +
        annotate("segment", x = annotPointX2[3], xend = annotPointX1[3],
                 y = annotPointY2[3], yend = annotPointY1[3]) +
        annotate("segment", x = annotPointX2[4], xend = annotPointX1[4],
                 y = annotPointY2[4], yend = annotPointY1[4]) 
      print(p)
      
      dev.off()
  }
}

