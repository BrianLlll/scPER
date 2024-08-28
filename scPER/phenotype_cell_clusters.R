###generate the plot to identify the phenotype-associated cell clusters

library(ggplot2)
library(viridis)
library(ggpubr)
library(ggthemes)
library(rstatix)

args = commandArgs(trailingOnly=TRUE)

if (length(args)<2) {
  stop("Both scPER estimated proportions and labels for bulk samples must be supplied (input files)", call.=FALSE)
} 

df<-read.csv(args[1],sep='\t',row.names = 1)

label<-read.csv(args[2],row.names = 1)

common<-intersect(rownames(df),rownames(label))

df<-df[common,]

df$group<-label[common,1]

all_df<-c()

for (i in (1:(length(colnames(df))-1))){
  df_t<-data.frame(df[,i],rep(colnames(df)[i],nrow(df)),df[,length(colnames(df))])
  colnames(df_t)<-c('Proption','Cell type','Group')
  all_df<-rbind(all_df,df_t)
}


p<-ggplot(all_df,aes(x=`Cell type`, y=Proption,fill=Group))+
  geom_boxplot(lwd = 0.5,fatten = 1) +
  scale_fill_viridis(discrete = TRUE, alpha=0.6)+
  theme(
    plot.title = element_text(size=11)
  ) +
  xlab("")+ylab("")+theme(axis.text= element_text(size = 14),axis.title.y = element_text(size = 14,hjust=0.5))+
  stat_compare_means(aes(group = Group),size=3,label = "p.signif", paired = FALSE)+
  scale_fill_manual(values=c("sienna3","turquoise4",'mediumpurple3'))+
  theme_few()+ 
  theme(axis.text.x=element_text(color="black", family="sans", size=14,angle=45,hjust=1))+
  theme(axis.text.y=element_text(color="black",family="sans",size=14))+
  theme(axis.title.x=element_text(color="black", family="sans", size=14))+
  theme(axis.title.y=element_text(color="black", family="sans", size=14))+
  theme(legend.text=element_text( family="sans",size=10))+ 
  theme(legend.title = element_text(family="sans",size=10))+
  #scale_y_continuous(breaks=seq(0, 1, 0.1),limits = c(0,1))+
  theme(axis.ticks=element_line(size=0.55,colour='black'))+
  theme(axis.ticks.length.y=unit(.35,"lines"))+
  theme(axis.ticks.length.x=unit(0,"lines"))+
  theme(panel.border = element_blank())+theme(axis.line = element_line(size=0.55, colour = "black"))+
  theme(plot.title=element_text(color="black",family="sans",size=rel(1.2),hjust=0.5))

pdf('phenotypes_associated_clusters.pdf',width =10,height = 5)
p
dev.off()