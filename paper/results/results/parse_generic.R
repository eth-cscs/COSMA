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
variantPlots = c("time","FLOPS")
algorithms = c("CARMA", "ScaLAPACK","CTF","COSMM")

GFLOPSperCore = 1209/36


statistics = 0

annotCoord = list()
#annotCoordX[["square_memory_p1_time"]] = 


#exp_filename = paste(exp_name,'.csv',sep="")
#setwd(paste(path,exp_name,sep =""))
source(paste(path, "/SPCL_Stats.R", sep=""))


# prepare the data 
rawData <- read.csv(file=exp_filename, sep=",", stringsAsFactors=FALSE, header=TRUE)


#data for the bar plot
barData <- data.frame(matrix(ncol = 4, nrow = 0))
x <- c("statistics", "alg", "experiment", "value")
#x <- c("algorithm","statistics","square_strong", "tall_strong", "square_memory_p1", "tall_memory_p1", "square_memory_p2", "tall_memory_p2")
#y <- c("COSMM", "CTF", "CTF_max", "ScaLAPACK_avg", "ScaLAPACK_max","CARMA_avg", "CARMA_max")
colnames(barData) <- x
#rownames(barData) <- y


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
    
    for (i3 in 1:length(variantPlots)) {
      tscaling <- data[c("p", "algorithm", "time")]
      varPlot = variantPlots[i3]
      if (varPlot == 'FLOPS') {
        m = data$m
        tscaling$time = 200* data$m * data$n * data$k / (data$time * 1e6) / (GFLOPSperCore * data$p)
        ylabel = "% peak performance"
        yscale = scale_y_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
        
        name = paste(size, variant, sep="_")
        for (i4 in 1:length(algorithms)){
          alg = algorithms[i4]
    
        #  x <- c("statistics", "alg", "experiment", "value")
          barData[nrow(barData) + 1,] = list("max", alg, name, max(tscaling[tscaling$algorithm == alg,]$time))
          barData[nrow(barData) + 1,] = list("avg", alg, name, gm_mean(tscaling[tscaling$algorithm == alg,]$time))
          #barData[paste(alg, "max", sep="_"),name] = max(tscaling[tscaling$algorithm == alg,]$time)
          #barData[paste(alg, "avg", sep="_"),name] = gm_mean(tscaling[tscaling$algorithm == alg,]$time)
        }
      }
      else {
        ylabel = "total time [ms]"
        yscale  = scale_y_continuous(trans='log2',labels=function(x) format(x, big.mark = ",", scientific = FALSE))

      }
      
      if (statistics == 1) {
        tscaling <- summarySE(tscaling, measurevar="time", groupvars=c("algorithm", "p"), conf.interval=.95)
        colnames(tscaling)[colnames(tscaling)=="CI.Norm.Norm"] <- "cil"
        print(tscaling[c("algorithm","p","median","cil","cih")])
        
        # plot the timers
        pdf(file=paste("size_",size, "_var_", variant , "_", varPlot, ".pdf", sep=""))
        p <- ggplot(tscaling, aes(x=p, y=median, colour=algorithm, shape=algorithm)) +
          geom_ribbon(aes(ymin=cil, ymax=cih), 
                      alpha=0.2,
                      #  linetype="blank",
                      show.legend=FALSE) +
          geom_line(show.legend=TRUE) +
          geom_point(size=2, show.legend=FALSE) +
          scale_x_continuous(trans='log2') +
          yscale +
          #  scale_y_continuous(trans='log2') +
          xlab("number of cores") +
          ylab(ylabel) +
          theme(legend.position="top", legend.direction="horizontal") +
          theme(legend.title=element_blank())
        print(p)
        dev.off()  
      }
      else {
        print(tscaling[c("algorithm","p","time")])
        
        aspRatio = 0.7
        w = 10
        textSize = 30
        pointSize = 5
        annotx = c(206,512,2024,1200)
        if (varPlot == 'FLOPS') {
          annoty = c(10,20,90,40)
        } else {
          annoty = c(20384,16000,4000,130000)
        }
        annotPointX = c(256,512,1024,8192)
        annotPointY = c(tscaling[tscaling$p == annotPointX[1] & tscaling$algorithm == 'CARMA',]$time,
                        tscaling[tscaling$p == annotPointX[2] & tscaling$algorithm == 'CTF',]$time,
                        tscaling[tscaling$p == annotPointX[3] & tscaling$algorithm == 'COSMM',]$time,
                        tscaling[tscaling$p == annotPointX[4] & tscaling$algorithm == 'ScaLAPACK',]$time)
        annotl = c("CARMA", "CTF", "COSMM","ScaLAPACK")
        # plot the timers
        pdf(file=paste("size_",size, "_var_", variant , "_", varPlot, ".pdf", sep=""),
            width = w, height = w*aspRatio)
        
        limit = yscale
        shapes = scale_shape_manual(values=c(15, 16, 17,18))
        shapesColors = scale_color_manual(values = c("#F8766D", "#7CAE00","#00BFC4",  "#C77CFF"))
        # if (size == "tall" && variant == "strong") {
        #   shapes = scale_shape_manual(values=c(16, 17,18))
        #   shapesColors = scale_color_manual(values = c("#7CAE00","#00BFC4",  "#C77CFF"))
        # }
         if (size == "square" && variant == "memory_p2" && varPlot == "FLOPS") {
           limit = ylim(0, 90)
         }
        
        data3 = ddply(tscaling, ~ algorithm+p, summarize, min=min(time), max=max(time), mean=median(time))

        p <- ggplot(mapping=aes(x=p, y=mean, ymin=min, ymax=max, fill=algorithm, color=algorithm, shape=algorithm)) +
          geom_ribbon(data=data3[data3$algorithm != "CARMA [21] ",], alpha=0.3)+
          geom_point(data=data3, size = 4) +
          geom_errorbar(data=data3[data3$algorithm == "CARMA [21] ",], width=0.1, size=1) +
          scale_x_continuous(trans='log2',labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
          #scale_x_log2("# of cores", breaks=c(128, 256, 512, 1024, 2048, 4096, 8192, 16384)) +
         # scale_y_log10(ylabel) +
          xlab("# of cores") +
          yscale +
          ylab(ylabel) +
          theme_bw(20) +
          theme(legend.position = c(0.45,0.98),
               legend.title=element_blank(),
               legend.text=element_text(size=22),
               legend.direction="horizontal",
               text = element_text(size=textSize),
               aspect.ratio=aspRatio
          )
        
          # legend(legend = c('a', 'b','c','d'),
          #        x = 0.7,
          #        y = 0.8,
          #        xpd = TRUE, 
          #        # inset = c(0,0), 
          #        # bty = "n", 
          #        # lty = c(1, 1, 1, 1),
          #        # lwd=2, 
          #        y.intersp=2)
       #   theme(legend.position=c(0.15, 0.8), legend.title=element_blank())
        print(p)

        dev.off()

   #      p <- ggplot(tscaling, aes(x=p, y=time, colour=algorithm, shape=algorithm)) +
   #        geom_point(size=pointSize, show.legend=TRUE) +
   #        shapes +
   #        shapesColors +
   #  #       scale_shape_manual(values=c(15, 16, 17,18))+
   # #       scale_fill_manual(labels = c("CARMA [22] ", "COSMM (our) ", "CTF [49] ", "ScaLAPACK [53] "))+
   #        yscale +
   #        limit +
   #        xlab("# of cores") +
   #        ylab(ylabel) +
   #        theme_bw(25) +
   #        #scale_fill_manual() +
   #        coord_fixed() +
   #       # geom_smooth(method="lm") +
   #        scale_x_continuous(trans='log2',labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
   #        theme(legend.position = c(0.45,0.96),
   #              legend.title=element_blank(),
   #              legend.text=element_text(size=25),
   #              legend.direction="horizontal",
   #              text = element_text(size=textSize),
   #              aspect.ratio=aspRatio
   #        ) +
   #        guides(col=guide_legend(nrow=2,byrow=FALSE,keyheight = 2),
   #               keywidth=19.5,
   #               keyheight=2.9,
   #               default.unit="inch")
   #       # guides(col = guide_legend(nrow = 2))
   #      #  legend(y.intersp=2)
   # 
   #      
   #        # annotate("text", x = annotx, y = annoty, label = annotl, size=textSize/3) +
   #        # annotate("segment", x = annotx[1], xend = annotPointX[1],
   #        #          y = annoty[1], yend = annotPointY[1]) +
   #        # annotate("segment", x = annotx[2], xend = annotPointX[2],
   #        #          y = annoty[2], yend = annotPointY[2]) +
   #        # annotate("segment", x = annotx[3], xend = annotPointX[3],
   #        #          y = annoty[3], yend = annotPointY[3]) +
   #        # annotate("segment", x = annotx[4], xend = annotPointX[4],
   #        #          y = annoty[4], yend = annotPointY[4]) +
   # 
   #      print(p)
   #      dev.off()
      }
      
    }
  }
}

