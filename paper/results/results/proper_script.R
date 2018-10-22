library(ggplot2)
library(reshape2)



aspectRatio = 3.0
hg = 4.4
pdf("barPlot2.pdf", height = hg, width = hg * aspectRatio)
data = read.table("result.csv", header = T, sep = ',')

GFLOPSperCore = 1209/36
data$FLOPS =  200* data$m * data$n * data$k / (data$time * 1e6) / (GFLOPSperCore * data$p)
data$case = paste(data$case,data$m == data$k,sep ="_")
#data = data[data$case != "memory_p2_FALSE",]
#data = data[data$case != "strong_TRUE",]
data[data$case == "memory_p1_FALSE",]$case = "m<k, lim. memory"
data[data$case == "memory_p1_TRUE",]$case = "m=k, lim. memory"
data[data$case == "memory_p2_TRUE",]$case = "m=k, extra memory"
data[data$case == "strong_FALSE",]$case = "m<k, strong scaling"
data[data$case == "memory_p2_FALSE",]$case = "m<k, extra memory"
data[data$case == "strong_TRUE",]$case = "m=k, strong scaling"

# 
# #find min, max, avg speedup
# data = read.table("result.csv", header = T, sep = ',')
# data$case = paste(data$case,data$m == data$k,sep ="_")
# data2 = reshape(data,timevar="algorithm",idvar=c("m","n","k","p","case"),direction="wide")
# data2[is.na(data2)] <- 99999999
# data2<-data2[!(data2$"time.COSMM (this work) "==99999999),]
# data2$minTime = apply(data2[, 6:8], 1, min)
# data2<-data2[!(data2$minTime==99999999),]
# data2$maxSpeedup = data2$minTime / data2$"time.COSMM (this work) "
# #data2<-data2[!(data2$maxSpeedup>6.4),]
# #data2 = data2[data2$m %% 36 == 0,]
# meanSpeedups = c(gm_mean(data2[data2$case == "strong_TRUE",]$maxSpeedup),
#                 gm_mean(data2[data2$case == "strong_FALSE",]$maxSpeedup),
#                 gm_mean(data2[data2$case == "memory_p1_TRUE",]$maxSpeedup),
#                 gm_mean(data2[data2$case == "memory_p1_FALSE",]$maxSpeedup),
#                 gm_mean(data2[data2$case == "memory_p2_TRUE",]$maxSpeedup),
#                 gm_mean(data2[data2$case == "memory_p2_FALSE",]$maxSpeedup))
# 
# maxSpeedups = c(max(data2[data2$case == "strong_TRUE",]$maxSpeedup),
#                 max(data2[data2$case == "strong_FALSE",]$maxSpeedup),
#                 max(data2[data2$case == "memory_p1_TRUE",]$maxSpeedup),
#                 max(data2[data2$case == "memory_p1_FALSE",]$maxSpeedup),
#                 max(data2[data2$case == "memory_p2_TRUE",]$maxSpeedup),
#                 max(data2[data2$case == "memory_p2_FALSE",]$maxSpeedup))
# 
# minSpeedups = c(min(data2[data2$case == "strong_TRUE",]$maxSpeedup),
#                 min(data2[data2$case == "strong_FALSE",]$maxSpeedup),
#                 min(data2[data2$case == "memory_p1_TRUE",]$maxSpeedup),
#                 min(data2[data2$case == "memory_p1_FALSE",]$maxSpeedup),
#                 min(data2[data2$case == "memory_p2_TRUE",]$maxSpeedup),
#                 min(data2[data2$case == "memory_p2_FALSE",]$maxSpeedup))
# 
# overalSpeedups = c(min(data2$maxSpeedup), gm_mean(data2$maxSpeedup), max(data2$maxSpeedup))



p = ggplot(data, aes(x = algorithm, y = FLOPS, fill = algorithm)) +
  geom_violin() +
  facet_grid(.~case) +
  scale_y_continuous("% peak performance", limits = c(0,100)) +
  scale_fill_discrete(labels = c("CARMA [21] ", "COSMA (this work) ", "CTF [47] ", "ScaLAPACK [51] "))+
  theme_bw(17) +
 # ylim(0, 110) +
theme(axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      legend.position = c(0.5,0.935),
      legend.title=element_blank(),
      legend.text=element_text(size=17)
      ) +
  guides(fill=guide_legend(nrow=1,byrow=TRUE),
         keywidth=19.5,
         keyheight=2.9,
         default.unit="inch")
print(p)
dev.off()