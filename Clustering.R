packages<-c("ggplot2","dplyr","tidyr","GGally","e1071","fastDummies","corrplot","cluster","factoextra","caret","FactoMineR")
installed<-rownames(installed.packages())
for(p in packages){
  if(!(p %in% installed)) install.packages(p)
  library(p, character.only = TRUE)
}


url<-"https://drive.google.com/uc?export=download&id=1IWA5bJ8t7s0pJKCGah4Fg4xVEFWS1SD5"
data<-read.csv(url)
head(data)


cat("Rows:",nrow(data),"Columns:",ncol(data),"\n")


str(data)


summary(data)


num_cols<-names(data)[sapply(data,is.numeric)]
cat_cols<-names(data)[!sapply(data,is.numeric)]
cat("Numerical:",num_cols, "\n")
cat("Categorical:",cat_cols, "\n")




for(col in num_cols){
  print(ggplot(data,aes_string(x=col)) +
          geom_histogram(fill="steelblue") +
          labs(title=paste("Histogram of",col)))
}




for(col in num_cols){
  print(ggplot(data,aes_string(y=col)) +
          geom_boxplot(fill="orange") +
          labs(title=paste("Boxplot of", col)))
}



ggcorr(data,label=TRUE)+
  theme(
    axis.text.x=element_text(angle=45,hjust=1) 
  )



num_data <- data[sapply(data, is.numeric)]
num_data <- num_data[, sapply(num_data, function(x) length(unique(x)) < 0.9 * nrow(num_data))]
cor_mat <- cor(num_data, use = "complete.obs")
avg_cor <- apply(abs(cor_mat), 1, mean)
selected_vars <- names(sort(avg_cor, decreasing = TRUE))[1:4]
colors <- if ("Gender" %in% names(data)) {
  as.numeric(as.factor(data$Gender))
} else {
  "black"
}
pairs(data[, selected_vars],
      main = "Key Variable Relationships",
      col = colors)




#Handle missing value
data[data == ""] <- NA
print(colSums(is.na(data)))

num_cols <- sapply(data, is.numeric)
cat_cols <- sapply(data, is.character)
for(col in names(data)[num_cols]){
  data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
}

# Categorical → mode
get_mode <- function(x){
  ux <- na.omit(unique(x))
  ux[which.max(tabulate(match(x, ux)))]
}

for(col in names(data)[cat_cols]){
  data[[col]][is.na(data[[col]])] <- get_mode(data[[col]])
}


data <- data[!duplicated(data), ]

cat("\nMissing values AFTER imputation (before encoding):\n")
print(colSums(is.na(data)))


#encoding
data<-dummyVars(" ~ .",data) %>% predict(newdata=data) %>% as.data.frame()




data_scaled<-scale(data)



skewed<-apply(data_scaled,2,function(x) abs(e1071::skewness(x))>1)
data_scaled[, skewed] <- log1p(abs(data_scaled[, skewed]))




pca_res<-prcomp(data_scaled,scale.=TRUE)




fviz_eig(pca_res)



pc_data<-pca_res$x[, 1:4]


wss<-c()
for(k in 1:20){
  set.seed(123)
  km_model<-kmeans(pc_data,centers=k,nstart=25)
  wss[k]<-km_model$tot.withinss
}



plot(1:20,wss,type="b",pch=19,
     xlab="Number of clusters (k)",
     ylab="Total Within-Cluster Sum of Squares",
     main="Elbow Method for K-means")




sil_scores<-c()
for(k in 2:20){
  set.seed(123)
  km_model<-kmeans(pc_data,centers=k,nstart=25)
  sil<-silhouette(km_model$cluster,dist(pc_data))
  sil_scores[k]<-mean(sil[,3])
}

sil_scores<-sil_scores[2:20]
best_k<-which.max(sil_scores)+1
cat("Best K based on silhouette score:",best_k, "\n")

plot(2:20, sil_scores, type="b", xlab="Number of clusters (k)", 
     ylab="Average Silhouette Score", main="Silhouette Analysis")




set.seed(123)
final_km<-kmeans(pc_data,centers=best_k,nstart=50)




final_sil <- silhouette(final_km$cluster, dist(pc_data))
cat("Average silhouette score for final model:", mean(final_sil[,3]), "\n")





plot(pc_data[,1], pc_data[,2], col=final_km$cluster, pch=19,
     xlab="PC1", ylab="PC2", main=paste("K-means Clusters (k =", best_k, ")"))
legend("topright", legend=paste("Cluster", 1:best_k), col=1:best_k, pch=19)

  
  
  
  
  
  
  