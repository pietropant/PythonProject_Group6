library(PCAmixdata)
library(readr)

data <- read_csv("/Users/pietropante/Desktop/Python_assignment/Files4Python/post_preprocessingRS_df.csv")
data$gender <- as.factor(data$gender)
data$marital <- as.factor(data$marital)
data$education <- as.factor(data$education)

data_split <- splitmix(data)
X_quanti <- data_split$X.quanti
X_quali <- data_split$X.quali

#Calcolo dinamico del numero massimo di dimensioni per prevenzione dell'error "subscript out of bounds"
num_dummy_vars <- sum(sapply(X_quali, function(x) length(levels(x)) - 1))
ndimens <- ncol(X_quanti) + num_dummy_vars

pcamix_result <- PCAmix(X.quanti = X_quanti, X.quali = X_quali, ndim = ndimens, rename.level = TRUE, graph = FALSE)
eigenvalues <- pcamix_result$eig[, 1]
explained_variance <- eigenvalues / sum(eigenvalues)
cumulative_variance <- cumsum(explained_variance)

print(cumulative_variance)

# Determina quante dimensioni spiegano almeno l'80% della varianza
num_components <- which(cumulative_variance >= 0.80)[1]
cat("Numero di componenti necessarie per spiegare almeno l'80% della varianza:", num_components, "\n")

# Ottieni il numero massimo di dimensioni disponibili
max_dimensions <- ncol(pcamix_result$ind$coord)
cat("Numero massimo di dimensioni disponibili:", max_dimensions, "\n")

# Adatta num_components al numero massimo disponibile
num_components <- min(num_components, max_dimensions)

df_pcamix <- pcamix_result$ind$coord
df_pcamix_selected <- df_pcamix[, 1:num_components]#Adattamento dinamico del numero di componenti

df_pcamix_selected_df <- as.data.frame(df_pcamix_selected)

print(head(df_pcamix_selected_df))

# Grafico della varianza spiegata cumulativa
plot(1:length(cumulative_variance), cumulative_variance, type = "o", col = "blue",
     xlab = "Number of Components", ylab = "Cumulative Explained Variance",
     main = "PCAMIX_R - Cumulative Explained Variance")
abline(h = 0.80, col = "red", lty = 2)
legend("bottomright", legend = c("80% Explained Variance"), 
       col = c("red"), lty = 2, inset = 0.05)

write_csv(df_pcamix_selected_df, "/Users/pietropante/Desktop/Python_assignment/Files4Python/pcamix_fromR.csv")