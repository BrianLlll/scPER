
##estimate the cell proportions based on the latent representations of scRNA-seq reference and bulk samples
#dependencies
library(xgboost)
library(Dict)
library(matrixStats)

args = commandArgs(trailingOnly=TRUE)

if (length(args)<3) {
  stop("Latent variables of both referece and mixture, and cell type label for each single cell must be supplied (input files)", call.=FALSE)
} 

meta<-read.csv(args[3])

data <- read.table(args[1],header=T,sep="\t",row.names=1,check.names=F)

rownames(data)<-gsub("-", "\\.", rownames(data))
meta[,1]<-gsub("-", "\\.",meta[,1])


data$celltype<-meta[,2][match(rownames(data),meta[,1])]


uni_cell<-unique(data$celltype)


all_df<-matrix(0, nrow = length(uni_cell), ncol = 100)

for (i in (1:length(uni_cell))){
  new_df<-colMedians(as.matrix(data[which(data$celltype==uni_cell[i]),(1:ncol(data)-1)]))
  all_df[i,]<-new_df
  
} 
rownames(all_df)<-uni_cell


de_algo<-function(X,y){
  # algorithm
  
  dtrain <- xgb.DMatrix(data = X, label = y)
  param <- list(booster = "gblinear"
                , objective = "reg:linear"
                , subsample = 0.7
                , max_depth = 10
                , colsample_bytree = 0.7
                , eta = 0.037
                , eval_metric = 'rmse'
                , base_score = 0.012 #average
                , min_child_weight = 100)
  xgb <- xgb.train(params = param
                   , data = dtrain
                   , watchlist = list(train = dtrain)
                   , nrounds = 50
                   , verbose = 1
                   , print_every_n = 1L
                   
  )
  
  
  
  feature_names <- colnames(dtrain)
  weight_matrix <- xgb.importance(feature_names,model=xgb)
  
  weight_matrix$Weight[which(weight_matrix$Weight<0)]<-0
  
  weight_matrix$relative <- (weight_matrix$Weight/sum(weight_matrix$Weight))
  
  weight_matrix
  
}


sig_matrix<-args[1]
mixture_file<-args[2]

#read in data
X<-t(all_df)

Y <- read.table(mixture_file, header=T, sep="\t",row.names=1)

  
X <- data.matrix(X)
Y <- data.matrix(Y)
  
header <- c('Mixture',colnames(X),"proportion")
  
output<-matrix()
itor <- 1
mixtures <- dim(Y)[2]

while(itor <= mixtures){
  y <- Y[,itor]
    
  #run xgboost core algorithm
    
  results<-data.frame(de_algo(X,y))
    
  results<-results[order(results$Feature),]
    
  #print output
    
  out<-data.frame(results[,3])

  rownames(out)<-results$Feature
    
  colnames(out)<-colnames(Y)[itor]
    
  output <- cbind(output, out)

  itor <- itor + 1
    
  }
  
#results
  
new_out<-output[,-1]
  
#save results

write.table(t(new_out),'./scPER_predicted_proportions.txt',sep='\t')





