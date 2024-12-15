df <- read.csv("/Users/pietropante/Desktop/Python_assignment/Files4Python/FAMD_perFisher.csv", stringsAsFactors = FALSE)

df$gender <- as.factor(df$gender)
df$education <- as.factor(df$education)
df$marital <- as.factor(df$marital)

# Funzione per eseguire il test di Fisher per tutte le variabili categoriali
run_pairwise_fisher <- function(df_categorical) {
  cluster_var <- "cluster"
  cluster_levels <- levels(df_categorical[[cluster_var]])
  variables <- names(df_categorical)[-which(names(df_categorical) == cluster_var)]
  pairwise_results <- list()
  
  # Esegui il test di Fisher per ciascuna variabile categoriale
  for (var in variables) {
    full_table <- table(df_categorical[[var]], df_categorical[[cluster_var]])
    results_var <- combn(cluster_levels, 2, function(pair) {
      subset_table <- full_table[, pair]
      fisher_test <- fisher.test(subset_table, simulate.p.value = TRUE)
      data.frame(
        Variable = var,
        Cluster_1 = pair[1],
        Cluster_2 = pair[2],
        Fisher_P = fisher_test$p.value
      )
    }, simplify = FALSE)
    pairwise_results[[var]] <- do.call(rbind, results_var)
  }

  all_results <- do.call(rbind, pairwise_results)

  m <- nrow(all_results)
  alpha <- 0.05
  all_results$Bonferroni_P <- p.adjust(all_results$Fisher_P, method = "bonferroni")

  all_results$Significance <- ifelse(all_results$Bonferroni_P < alpha, "Significativa", "Non significativa")
  
  return(all_results)
}

# Esegui il test di Fisher per il primo clustering (Agglomerative)
df$cluster <- as.factor(df$FAMD_Agglomerative_Cluster)
df_categorical <- df[, c("gender", "education", "marital", "cluster")]
pairwise_results_agg <- run_pairwise_fisher(df_categorical)

cat("Agglomerative Clustering - Test di Fisher:\n")
cat("   Variable Cluster_1 Cluster_2     Fisher_P\n")
apply(pairwise_results_agg, 1, function(row) {
  cat(sprintf("%10s %9s %9s %13.10f\n", row['Variable'], row['Cluster_1'], row['Cluster_2'], as.numeric(row['Fisher_P'])))
})
cat("  Bonferroni_P      Significance\n")
apply(pairwise_results_agg, 1, function(row) {
  cat(sprintf("%13.9f %s\n", as.numeric(row['Bonferroni_P']), row['Significance']))
})

# Esegui il test di Fisher per il secondo clustering (KMedoids)
df$cluster <- as.factor(df$FAMD_KMedoids_Cluster)
df_categorical <- df[, c("gender", "education", "marital", "cluster")]
pairwise_results_kmedoids <- run_pairwise_fisher(df_categorical)

cat("\nKMedoids Clustering - Test di Fisher:\n")
cat("   Variable Cluster_1 Cluster_2     Fisher_P\n")
apply(pairwise_results_kmedoids, 1, function(row) {
  cat(sprintf("%10s %9s %9s %13.10f\n", row['Variable'], row['Cluster_1'], row['Cluster_2'], as.numeric(row['Fisher_P'])))
})
cat("  Bonferroni_P      Significance\n")
apply(pairwise_results_kmedoids, 1, function(row) {
  cat(sprintf("%13.9f %s\n", as.numeric(row['Bonferroni_P']), row['Significance']))
})

# Funzione per eseguire il test di Fisher globale
run_fisher_global <- function(df_categorical) {
  cluster_var <- "cluster"
  variables <- names(df_categorical)[-which(names(df_categorical) == cluster_var)]
  results <- list()

  # Esegui il test di Fisher per ciascuna variabile categoriale
  for (var in variables) {
    # Crea la tabella di contingenza completa per la variabile rispetto ai cluster
    contingency_table <- table(df_categorical[[var]], df_categorical[[cluster_var]])

    fisher_test <- fisher.test(contingency_table, simulate.p.value = TRUE)

    results[[var]] <- data.frame(
      Variable = var,
      P_Value = fisher_test$p.value
    )
  }

  all_results <- do.call(rbind, results)

  all_results$Significance <- ifelse(all_results$P_Value < 0.05, "Significativa", "Non significativa")
  
  return(all_results)
}

# Esegui il test di Fisher globale per Agglomerative Clustering
df$cluster <- as.factor(df$FAMD_Agglomerative_Cluster)
df_categorical <- df[, c("gender", "education", "marital", "cluster")]
global_results_agg <- run_fisher_global(df_categorical)

cat("Agglomerative Clustering - Test di Fisher (Globale):\n")
print(global_results_agg)

# Esegui il test di Fisher globale per KMedoids Clustering
df$cluster <- as.factor(df$FAMD_KMedoids_Cluster)
df_categorical <- df[, c("gender", "education", "marital", "cluster")]
global_results_kmedoids <- run_fisher_global(df_categorical)

cat("\nKMedoids Clustering - Test di Fisher (Globale):\n")
print(global_results_kmedoids)

