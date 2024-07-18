library(Rmagic)

args = commandArgs(trailingOnly=TRUE)

if (length(args)<2) {
  stop("Both referece and mixture files must be supplied (input files)", call.=FALSE)
} 

sc_df<-read.csv(args[1],row.names = 1)

bulk<-read.csv(args[2],row.names = 1)

###identify the overlap genes between reference and bulk mixture

overlap<-intersect(rownames(sc_df),rownames(bulk))

sc_df<-sc_df[overlap,]

####imputation using MAGIC

MAGIC_data <- Rmagic::magic(t(sc_df))

sc_df_impute<-t(MAGIC_data$result)

##select top5000 most variable genes

apply(sc_df_impute, 1, var) -> rowVar

names(rowVar[order(rowVar,decreasing = TRUE)][1:5000])->mad.genes

mad5000= sc_df_impute[mad.genes,] ### gene by sample


write.csv(mad5000,'/example/reference_top5k_imputation.csv',quote=FALSE)
print('Output imputed reference data')

bulk_overlap<-bulk[mad.genes,]

write.csv(bulk_overlap,'/example/bulk_5k_genes_matched.csv',quote=FALSE)
print('Output simulated bulk data')





