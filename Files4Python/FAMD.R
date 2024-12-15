library(FactoMineR)
library(factoextra)
library(readr)

data <- read_csv("/Users/pietropante/Desktop/Python_assignment/Files4Python/post_preprocessingRS_df.csv")
data$gender <- as.factor(data$gender)
data$marital <- as.factor(data$marital)
data$education <- as.factor(data$education)

ncol <- ncol(data)
famd_result <- FAMD(data, ncp = ncol, graph = FALSE)

eigenvalues <- famd_result$eig[, 1]
explained_variance <- eigenvalues / sum(eigenvalues)
cumulative_variance <- cumsum(explained_variance)
print(cumulative_variance)

num_components <- which(cumulative_variance >= 0.80)[1]
cat("Number of components needed to explain at least 80% of the variance:", num_components, "\n")

plot(1:length(cumulative_variance), cumulative_variance, type="o", col="blue", 
     xlab="Number of Components", ylab="Cumulative Explained Variance", 
     main="FAMD_R - Cumulative Explained Variance")
abline(h = 0.80, col = "red", lty = 2)
legend("bottomright", legend="80% Explained Variance", col="red", lty=2, inset = 0.05)

df_famd <- famd_result$ind$coord
df_famd_80 <- df_famd[, 1:num_components]
head(df_famd_80)
# Converti df_famd_80 in un data frame
df_famd_80_df <- as.data.frame(df_famd_80)

write_csv(df_famd_80_df, "/Users/pietropante/Desktop/Python_assignment/Files4Python/famd_80_fromR.csv")
