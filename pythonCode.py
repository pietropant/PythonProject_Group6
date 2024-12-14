import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import prince
from scipy.stats import kstest, shapiro, mannwhitneyu, spearmanr, kendalltau, kruskal, chi2_contingency
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from pandas import ExcelWriter

filepath = '/Users/pietropante/Desktop/Files4Python/dataset_project_eHealth20242025.csv'
df = pd.read_csv(filepath)
categorical_columns = ['gender', 'marital', 'education']
for col in df.columns:
    if col in categorical_columns:
        df[col] = df[col].astype('Int64')
        df[col] = df[col].astype('object')
    else:
        df[col] = df[col].astype('float64')
#print(df.info())

total_missing = df.isnull().sum().sum()
rows_with_missing = df.isnull().any(axis=1).sum()
print(f"Total number of missing values in the DataFrame: {total_missing}")
print(f"# of rows containing at least one missing value: {rows_with_missing}")

categorical = df[categorical_columns]
print(categorical.shape)
numerical = df.drop(columns=categorical_columns)
print(numerical.shape)
print("Numerical columns:\n", numerical.columns.tolist())
print("Categorical columns :\n", categorical.columns.tolist())

knn_imputer = KNNImputer(n_neighbors=5)  # Considering 160 istanze METTIAMO 5 PERCHè LA LETTERATURA NON SI PRONUNCIA RELATIVAMENTE
numerical_imputed = np.round(
    knn_imputer.fit_transform(numerical))  # round all'intero più vicino per questione di sensatezza negli imputati
numerical = pd.DataFrame(numerical_imputed, columns=numerical.columns)

for col in categorical.columns:
    mode_value = categorical[col].mode()[0]
    categorical[col] = categorical[col].fillna(mode_value)
    categorical[col] = categorical[col].astype('object')  # Pandas cerca di mantenere il tipo più
    # "adatto" possibile per i dati. Se la colonna contiene principalmente stringhe e la moda è
    # una stringa, non ci sono problemi. Tuttavia, se la colonna contiene valori numerici e la
    # moda è numerica, pandas potrebbe inferire che il tipo di dato migliore per quella colonna
    # è numerico (ad esempio, int64 o float64)
df = pd.concat([numerical, categorical], axis=1)

# per la sensatezza nelle imputazioni
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
non_integers_info = []
for col in numerical_columns:
    non_integer_values = df[col][df[col] != df[col].astype('int64')]
    for idx, value in non_integer_values.items():
        non_integers_info.append((col, idx, value))
if non_integers_info:
    print("Non-integer values found and their positions (column, row, value):")
    for col, idx, value in non_integers_info:
        print(f"Column: {col}, Row: {idx}, Value: {value}")
else:
    print("No non-integer values found.")#perche nelle imputazioni numeriche si fa il knn rounded!!!

print("\nDataFrame after imputation:")
print(df.info())
total_missing = df.isnull().sum().sum()
rows_with_missing = df.isnull().any(axis=1).sum()
print(f"Total number of missing values in the DataFrame after MV filling usink KNN: {total_missing}")
print(f"Number of rows containing at least one missing value after MV filling using KNN: {rows_with_missing}")
df = df.drop_duplicates()
print("Shape after removing duplicate rows:", df.shape)

df['phq9_score'] = df[['phq_1', 'phq_2', 'phq_3', 'phq_4', 'phq_5', 'phq_6', 'phq_7', 'phq_8', 'phq_9']].sum(axis=1)
df = df.drop(columns=['phq_1', 'phq_2', 'phq_3', 'phq_4', 'phq_5', 'phq_6', 'phq_7', 'phq_8', 'phq_9'])
df['gad7_score'] = df[['gad_1', 'gad_2', 'gad_3', 'gad_4', 'gad_5', 'gad_6', 'gad_7']].sum(axis=1)
df = df.drop(columns=['gad_1', 'gad_2', 'gad_3', 'gad_4', 'gad_5', 'gad_6', 'gad_7'])

rule_1_columns = ['asq_1', 'asq_2', 'asq_4', 'asq_5', 'asq_6', 'asq_7', 'asq_9', 'asq_12',
                  'asq_13', 'asq_16', 'asq_18', 'asq_19', 'asq_20', 'asq_21', 'asq_22', 'asq_23',
                  'asq_26', 'asq_33', 'asq_35', 'asq_39', 'asq_41', 'asq_42', 'asq_43', 'asq_45',
                  'asq_46']
rule_2_columns = ['asq_3', 'asq_8', 'asq_10', 'asq_11', 'asq_14', 'asq_15', 'asq_17', 'asq_24',
                  'asq_25', 'asq_27', 'asq_28', 'asq_29', 'asq_30', 'asq_31', 'asq_32', 'asq_34',
                  'asq_36', 'asq_37', 'asq_38', 'asq_40', 'asq_44', 'asq_47', 'asq_48', 'asq_49', 'asq_50']

def calculate_score(row):
    score = 0
    for col in rule_1_columns:
        if row[col] in [2, 3]:
            score += 1
    for col in rule_2_columns:
        if row[col] in [0, 1]:
            score += 1
    return score

df['asq_score'] = df.apply(calculate_score, axis=1)
df = df.drop(columns=['asq_1', 'asq_2', 'asq_3', 'asq_4', 'asq_5', 'asq_6', 'asq_7', 'asq_8', 'asq_9', 'asq_10',
                      'asq_11', 'asq_12', 'asq_13', 'asq_14', 'asq_15', 'asq_16', 'asq_17', 'asq_18', 'asq_19',
                      'asq_20',
                      'asq_21', 'asq_22', 'asq_23', 'asq_24', 'asq_25', 'asq_26', 'asq_27', 'asq_28', 'asq_29',
                      'asq_30',
                      'asq_31', 'asq_32', 'asq_33', 'asq_34', 'asq_35', 'asq_36', 'asq_37', 'asq_38', 'asq_39',
                      'asq_40',
                      'asq_41', 'asq_42', 'asq_43', 'asq_44', 'asq_45', 'asq_46', 'asq_47', 'asq_48', 'asq_49',
                      'asq_50'])

rule_1_columns = ['asrs_1', 'asrs_2', 'asrs_3']
rule_2_columns = ['asrs_4', 'asrs_5', 'asrs_6']

def calculate_asrs_score(row):
    score = 0
    for col in rule_1_columns:
        if row[col] in [2, 3, 4]:
            score += 1
    for col in rule_2_columns:
        if row[col] in [3, 4]:
            score += 1
    return score

df['asrs_score'] = df.apply(calculate_asrs_score,
                            axis=1)  # la soglia per discriminare se hai l'adhd o no è da 4 in sù, 4 incluso
df = df.drop(columns=['asrs_1', 'asrs_2', 'asrs_3', 'asrs_4', 'asrs_5', 'asrs_6'])
df['ssba_alcohol_score'] = df[['ssba_alcohol_1', 'ssba_alcohol_2', 'ssba_alcohol_3', 'ssba_alcohol_4']].sum(axis=1)
df.drop(['ssba_alcohol_1', 'ssba_alcohol_2', 'ssba_alcohol_3', 'ssba_alcohol_4'], axis=1, inplace=True)
df['ssba_internet_score'] = df[['ssba_internet_1', 'ssba_internet_2', 'ssba_internet_3', 'ssba_internet_4']].sum(axis=1)
df.drop(['ssba_internet_1', 'ssba_internet_2', 'ssba_internet_3', 'ssba_internet_4'], axis=1, inplace=True)
df['ssba_drug_score'] = df[['ssba_drug_1', 'ssba_drug_2', 'ssba_drug_3', 'ssba_drug_4']].sum(axis=1)
df.drop(['ssba_drug_1', 'ssba_drug_2', 'ssba_drug_3', 'ssba_drug_4'], axis=1, inplace=True)
df['ssba_gambling_score'] = df[['ssba_gambling_1', 'ssba_gambling_2', 'ssba_gambling_3', 'ssba_gambling_4']].sum(axis=1)
df.drop(['ssba_gambling_1', 'ssba_gambling_2', 'ssba_gambling_3', 'ssba_gambling_4'], axis=1,
        inplace=True)  # su ognuna di queste addiction la soglia è 3 tranne che per il gambling che è 2
#print(df.info())

columns_to_select = ['age', 'gender', 'education', 'marital', 'income',
                     'phq9_score', 'gad7_score', 'asq_score', 'asrs_score', 'ssba_alcohol_score','ssba_internet_score','ssba_drug_score', 'ssba_gambling_score']
df_of_interest = df[columns_to_select]

# Define rarity threshold and do a kind of outlier detection for categorical columns
categorical = df_of_interest[['gender', 'marital', 'education']]
numerical = df_of_interest.drop(columns=categorical.columns)
print("Numerical columns:\n", numerical.columns.tolist())
print("Categorical columns :\n", categorical.columns.tolist())

normal_columns = []
non_normal_columns = []
for col in numerical.columns:
    stat, p_value = stats.shapiro(
        numerical[col].dropna())  # per piccole distribuzioni (la nostra è di 150) è meglio usare Shapiro_Wilk
    if p_value >= 0.05:
        normal_columns.append((col, p_value, 'original'))
    else:
        non_normal_columns.append((col, p_value, 'original'))
    numerical_log = np.log1p(numerical[col])  # Logaritmo naturale (ln(x)) + 1 per evitare log(0)
    stat_log, p_value_log = stats.shapiro(numerical_log.dropna())
    if p_value_log >= 0.05:
        normal_columns.append((col, p_value_log, 'log'))
    else:
        non_normal_columns.append((col, p_value_log, 'log'))
if normal_columns or non_normal_columns:
    print("The following variables are normally distributed:")
    for col, p_value, var_type in normal_columns:
        if var_type == 'original':
            print(f"Column: {col} (original), p-value: {p_value:.4f}")
        elif var_type == 'log':
            print(f"Column: {col} (logarithmic), p-value: {p_value:.4f}")
    print("\nThe following variables are not normally distributed:")
    for col, p_value, var_type in non_normal_columns:
        if var_type == 'original':
            print(f"Column: {col} (original), p-value: {p_value:.4f}")
        elif var_type == 'log':
            print(f"Column: {col} (logarithmic), p-value: {p_value:.4f}")
else:
    print("No numerical variables are present.")

# plots
for col, p_value, var_type in normal_columns + non_normal_columns:
    if var_type == 'original':
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(numerical[col], kde=True, bins=30, color='skyblue', stat="density", linewidth=0)
        plt.title(f"Original distribution of {col} (p-value: {p_value:.4f})", fontsize=12)
        plt.xlabel(col, fontsize=10)
        plt.ylabel('Density', fontsize=10)
        plt.subplot(1, 2, 2)
        numerical_log = np.log1p(numerical[col])
        sns.histplot(numerical_log, kde=True, bins=30, color='lightgreen', stat="density", linewidth=0)
        plt.title(f"Distribution log({col})", fontsize=12)
        plt.xlabel(f"log({col})", fontsize=10)
        plt.ylabel('Density', fontsize=10)
        plt.suptitle(f"Comparison of Original and Log Transformed Distribution for {col}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(numerical.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# Pairplots per le variabili numeriche per visualizzare distribuzioni e relazioni bivariate tra tutte le variabili numeriche.
pairplot_fig = sns.pairplot(numerical, diag_kind="kde", plot_kws={'alpha': 0.5})
pairplot_fig.fig.suptitle('Pairplots for Numerical Variables', y=1.02, fontsize=16)
plt.show()

independence_results = []
# Test per variabili numeriche vs numeriche
for i, num_col1 in enumerate(numerical.columns):
    for num_col2 in numerical.columns[i + 1:]:
        # Test di Spearman e Kendall
        spearman_stat, spearman_p = spearmanr(numerical[num_col1], numerical[num_col2])
        kendall_stat, kendall_p = kendalltau(numerical[num_col1], numerical[num_col2])
        independence_results.append({
            "Variable 1": num_col1,
            "Variable 2": num_col2,
            "Test": "Spearman",
            "Statistic": spearman_stat,
            "P-value": spearman_p,
            "Independent": spearman_p >= 0.05
        })
        independence_results.append({
            "Variable 1": num_col1,
            "Variable 2": num_col2,
            "Test": "Kendall",
            "Statistic": kendall_stat,
            "P-value": kendall_p,
            "Independent": kendall_p >= 0.05
        })

# Test per variabili numeriche vs categoriali
for cat_col in categorical.columns:
    for num_col in numerical.columns:
        # Kruskal-Wallis Test
        groups = [numerical[num_col][df[cat_col] == level] for level in df[cat_col].unique() if not pd.isna(level)]
        kruskal_stat, kruskal_p = kruskal(*groups) if len(groups) > 1 else (np.nan, np.nan)
        independence_results.append({
            "Variable 1": num_col,
            "Variable 2": cat_col,
            "Test": "Kruskal-Wallis",
            "Statistic": kruskal_stat,
            "P-value": kruskal_p,
            "Independent": kruskal_p >= 0.05
        })

# Test per variabili categoriali vs categoriali
for i, cat_col1 in enumerate(categorical.columns):
    for cat_col2 in categorical.columns[i + 1:]:
        # Chi-squared Test. (Fisher qui lo posso fare solo 2x2)
        contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
        chi2_stat, chi2_p, _, _ = chi2_contingency(contingency_table)
        independence_results.append({
            "Variable 1": cat_col1,
            "Variable 2": cat_col2,
            "Test": "Chi-Square",
            "Statistic": chi2_stat,
            "P-value": chi2_p,
            "Independent": chi2_p >= 0.05
        })
independence_results_df = pd.DataFrame(independence_results)
pd.set_option('display.max_columns', independence_results_df.shape[0] + 1)
pd.set_option('display.max_rows', independence_results_df.shape[0] + 1)
print(independence_results_df.drop("Statistic", axis=1))

# QUINDI LE VARIABILI SONO TUTTE NON NORMALI-> Robust SCALER
# qui prima di fare outlier detection riscalo i numerici.

scaler = RobustScaler()
numerical_scaled = scaler.fit_transform(numerical)
numerical_scaled = pd.DataFrame(numerical_scaled, columns=numerical.columns)
numerical_scaled.boxplot(figsize=(12, 6))
plt.xticks(rotation=90)
plt.title("Boxplot of Normalized Numeric Columns for Outlier Detection")
plt.show()
# A riconferma:
outlier_info = []
for col in numerical_scaled.columns:
    Q1 = numerical_scaled[col].quantile(0.25)
    Q3 = numerical_scaled[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = numerical_scaled[(numerical_scaled[col] < lower_bound) | (numerical_scaled[col] > upper_bound)]
    for index in outliers.index:
        outlier_info.append((index, col))
if outlier_info:
    print("Indices of outlier instances with corresponding columns:")
    for index, col in outlier_info:
        print(f"Index: {index}, Column: {col}")
else:
    print("No outliers detected, as evinced by the boxplot.")

columns_to_select = ['age', 'income', 'phq9_score', 'gad7_score']
numerical_scaled = numerical_scaled[columns_to_select]
df_postpp = pd.concat([numerical_scaled, categorical], axis=1)

# Salvare come CSV
df_postpp.to_csv("/Users/pietropante/Desktop/Files4Python/post_preprocessingRS_df.csv", index=False)  # 'index=False' evita di salvare gli indici

one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')# drop='first' (m-1) colonne per m categorie di una categorica
categorical_encoded = one_hot_encoder.fit_transform(categorical).astype(int)
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=one_hot_encoder.get_feature_names_out(categorical.columns))
df_encoded = pd.concat([numerical_scaled, categorical_encoded_df.reset_index(drop=True)], axis=1)

# PCA
pca = PCA()
df_pca = pca.fit_transform(df_encoded)
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

#voglio la varianza all'80%
n_components_80 = np.argmax(cumulative_explained_variance >= 0.80) + 1  #+1 perché l'indice parte da 0
print(f"# of components to have 80% of cumulative explained variance: {n_components_80}")
print(f"Cumulative explained variance for {n_components_80} components: {cumulative_explained_variance[n_components_80 - 1]:.4f}")

#creiamo un nuovo dataframe con solo le PC per avere l'80% della varianza
df_pca_selected = pd.DataFrame(df_pca[:, :n_components_80], columns=[f'PC{i+1}' for i in range(n_components_80)])
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance * 100, color='b', marker='o')
plt.axhline(y=80, color='r', linestyle='--', label="80% della varianza")
plt.xlabel('# of components')
plt.ylabel('Cumulative explained variance (%)')
plt.title('PCA - Cumulative explained variance')
plt.legend()
plt.show()

# FAMD
famd = prince.FAMD(n_components=df_postpp.shape[1], random_state=42)
famd = famd.fit(df_postpp)

eigenvalues = famd.eigenvalues_
explained_variance = eigenvalues / eigenvalues.sum()
cumulative_variance = explained_variance.cumsum()
num_components = (cumulative_variance <= 0.80).sum() + 1
print(f"Number of components needed to explain at least 80% of the variance: {num_components}")
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, color= 'b', marker='o', label='Cumulative Variance')
for i, var in enumerate(cumulative_variance):
    plt.text(i + 1, var, f"{var:.3f}", ha='center', va='bottom', fontsize=9)
plt.axhline(y=0.80, color='r', linestyle='--', label='80% Explained Variance')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("FAMD - Cumulative Explained Variance")
plt.legend()
plt.show()

df_famd = famd.transform(df_postpp)

df_famd_80 = pd.DataFrame(
    df_famd.iloc[:, :num_components].to_numpy(),
    columns=[f"FAMD_{i+1}" for i in range(num_components)],
    index=df_postpp.index
)
print("Reduced dataset with principal components explaining at least 80% of the variance:")
print(df_famd_80.head())

#Bar Plot
plt.figure(figsize=(10, 6))
colors = ['skyblue' if i < num_components else 'gray' for i in range(len(explained_variance))]
plt.bar(range(1, len(explained_variance) + 1), explained_variance, color=colors)
for i, var in enumerate(explained_variance):
    plt.text(i + 1, var, f"{var:.2f}", ha='center', va='bottom', fontsize=9)
plt.axvline(x=num_components + 0.5, color='red', linestyle='--', label=f'Components Used (n={num_components})')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.title("Explained Variance per Principal Component")
plt.legend()
plt.show()

# FAMD da R
filepath = "/Users/pietropante/Desktop/Files4Python/famd_80_fromR.csv"
df_famd_80R = pd.read_csv(filepath)

# PCAMIX da R
filepath = "/Users/pietropante/Desktop/Files4Python/pcamix_fromR.csv"
df_pcamix_80R = pd.read_csv(filepath)


df_transforms = {
    'PCA': df_pca_selected,
    'FAMD': df_famd_80,
    'FAMD_R': df_famd_80R,
    'PCAMIX_R': df_pcamix_80R
}

for tag, df_scaled in df_transforms.items():

    def find_optimal_clusters(X, tag):
        linkage_methods = ['ward', 'average', 'complete', 'single']
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        for linkage_method in linkage_methods:
            silhouette_avgs = []
            distortions = []
            for k in range(2, 9):
                clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
                clustering.fit(X)
                labels = clustering.labels_

                # Calcolo del Silhouette Score
                silhouette_avg = silhouette_score(X, labels)
                silhouette_avgs.append(silhouette_avg)

                # Calcolo delle distorsioni per l'Elbow Method
                cluster_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
                dist = sum(
                    np.sum(cdist(X[labels == i], cluster_centers[i].reshape(1, -1), 'euclidean')) for i in range(k))
                distortions.append(dist)

            axs[0].plot(range(2, 9), silhouette_avgs, marker='o', label=f'{linkage_method}')
            for x, y in zip(range(2, 9), silhouette_avgs):
                axs[0].text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

            axs[1].plot(range(2, 9), distortions, marker='o', label=f'{linkage_method}')
            for x, y in zip(range(2, 9), distortions):
                axs[1].text(x, y, f'{y:.0f}', ha='center', va='bottom', fontsize=9)

        axs[0].set_xlabel('Number of clusters')
        axs[0].set_ylabel('Silhouette Score')
        axs[0].set_title(f'{tag} - Silhouette Score for Linkage Methods')
        axs[1].set_xlabel('Number of clusters')
        axs[1].set_ylabel('Sum of Intra-cluster Distances')
        axs[1].set_title(f'{tag} - Elbow Method for Linkage Methods')
        axs[0].legend(title="Linkage", loc='best')
        axs[1].legend(title="Linkage", loc='best')
        plt.tight_layout()
        plt.show()

    # Funzione per confrontare Agglomerative e K-Medoids e aggiungere i risultati al DataFrame
    def compare_clustering(X, tag, df):
        silhouette_avgs_ward = []
        silhouette_avgs_kmedoids = []
        distortions_ward = []
        distortions_kmedoids = []
        ward_labels = []
        kmedoids_labels = []
        #!!!!ward perche dall'analisi della figure che plotta silhouette score e inertia permette di raggiungere il miglior trade-off
        for k in range(2, 9):
            # Agglomerative clustering
            model_ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
            model_ward.fit(X)
            labels_ward = model_ward.labels_
            silhouette_avg_ward = silhouette_score(X, labels_ward)
            silhouette_avgs_ward.append(silhouette_avg_ward)
            ward_labels.append(labels_ward)

            # Distortion per Agglomerative
            cluster_centers = np.array([X[labels_ward == i].mean(axis=0) for i in range(k)])
            dist_ward = sum(
                np.sum(cdist(X[labels_ward == i], cluster_centers[i].reshape(1, -1), 'euclidean')) for i in range(k))
            distortions_ward.append(dist_ward)

            # K-Medoids clustering
            kmedoids = KMedoids(n_clusters=k, init='k-medoids++', metric='manhattan', max_iter=500, random_state=0)
            kmedoids.fit(X)
            labels_kmedoids = kmedoids.labels_
            silhouette_avg_kmedoids = silhouette_score(X, labels_kmedoids)
            silhouette_avgs_kmedoids.append(silhouette_avg_kmedoids)
            kmedoids_labels.append(labels_kmedoids)

            # Distortion per K-Medoids
            medoid_centers = np.array([X[labels_kmedoids == i].mean(axis=0) for i in range(k)])
            dist_kmedoids = sum(
                np.sum(cdist(X[labels_kmedoids == i], medoid_centers[i].reshape(1, -1), 'euclidean')) for i in range(k))
            distortions_kmedoids.append(dist_kmedoids)

        if tag == 'PCA':
            best_ward_labels = ward_labels[1]
            best_kmedoids_labels = kmedoids_labels[1]
        elif tag == 'FAMD':
            best_ward_labels = ward_labels[3]
            best_kmedoids_labels = kmedoids_labels[1]
        elif tag == 'FAMD_R':
            best_ward_labels = ward_labels[1]
            best_kmedoids_labels = kmedoids_labels[2]
        elif tag == 'PCAMIX_R':
            best_ward_labels = ward_labels[1]
            best_kmedoids_labels = kmedoids_labels[1]

            # Aggiungi i risultati al dataframe
        df[f'{tag}_Agglomerative_Cluster'] = best_ward_labels
        df[f'{tag}_KMedoids_Cluster'] = best_kmedoids_labels

        # Plot dei risultati
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs[0].plot(range(2, 9), silhouette_avgs_ward, marker='o', label='Agglomerative (Ward)', color='blue')
        for x, y in zip(range(2, 9), silhouette_avgs_ward):
            axs[0].text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

        axs[0].plot(range(2, 9), silhouette_avgs_kmedoids, marker='o', label='K-Medoids', color='red')
        for x, y in zip(range(2, 9), silhouette_avgs_kmedoids):
            axs[0].text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

        axs[1].plot(range(2, 9), distortions_ward, marker='o', label='Agglomerative (Ward)', color='blue')
        for x, y in zip(range(2, 9), distortions_ward):
            axs[1].text(x, y, f'{y:.0f}', ha='center', va='bottom', fontsize=9)

        axs[1].plot(range(2, 9), distortions_kmedoids, marker='o', label='K-Medoids', color='red')
        for x, y in zip(range(2, 9), distortions_kmedoids):
            axs[1].text(x, y, f'{y:.0f}', ha='center', va='bottom', fontsize=9)

        axs[0].set_xlabel('Number of clusters')
        axs[0].set_ylabel('Silhouette Score')
        axs[0].set_title(f'{tag} - Silhouette Score Comparison')
        axs[0].legend()

        axs[1].set_xlabel('Number of clusters')
        axs[1].set_ylabel('Sum of Intra-cluster Distances')
        axs[1].set_title(f'{tag} - Elbow Method Comparison')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
        if tag in ['PCA', 'FAMD_R']:
            return df, silhouette_avgs_ward, distortions_ward
        elif tag in ['FAMD', 'PCAMIX_R']:
            return df, silhouette_avgs_kmedoids, distortions_kmedoids

    if tag == 'PCA':
            dfPCA=df.copy()
            find_optimal_clusters(df_scaled, tag)
            dfPCA, silhPCA, distPCA=compare_clustering(df_scaled, tag, dfPCA)
    elif tag == 'FAMD':
            dfFAMD=df.copy()
            find_optimal_clusters(df_scaled, tag)
            dfFAMD, silhFAMD, distFAMD=compare_clustering(df_scaled, tag, dfFAMD)
    elif tag == 'FAMD_R':
            dfFAMD_R=df.copy()
            find_optimal_clusters(df_scaled, tag)
            dfFAMD_R, silhFAMD_R, distFAMD_R=compare_clustering(df_scaled, tag, dfFAMD_R)
    elif tag == 'PCAMIX_R':
            dfPCAMIX_R=df.copy()
            find_optimal_clusters(df_scaled, tag)
            dfPCAMIX_R, silhPCAMIX_R, distPCAMIX_R=compare_clustering(df_scaled, tag, dfPCAMIX_R)

#PCA
dfPCA.to_csv("/Users/pietropante/Desktop/Files4Python/PCA_perFisher.csv", index=False)  # 'index=False' evita di salvare gli indici

dfPCA['PCA_Agglomerative_Cluster'] = dfPCA['PCA_Agglomerative_Cluster'].astype('object')
dfPCA['PCA_KMedoids_Cluster'] = dfPCA['PCA_KMedoids_Cluster'].astype('object')

num_unique_values_aggl = dfPCA['PCA_Agglomerative_Cluster'].nunique()
num_unique_values_kmed = dfPCA['PCA_KMedoids_Cluster'].nunique()
print(f"The number of clusters in PCA_Agglomerative_Cluster is {num_unique_values_aggl}, while in PCA_KMedoids_Cluster it is {num_unique_values_kmed}.")

numeric_vars = df.select_dtypes(include=['float64', 'int64']).columns

# Funzione per l'analisi della normalità e Kruskal-Wallis
def perform_statistical_analysis(df, target_col):
    print(f"\nAnalisi statistica per il clustering su '{target_col}':")

    # Analisi della normalità
    normality_results = []
    for var in numeric_vars:
        data = df[var]
        if data.nunique() > 1:
            shapiro_stat, shapiro_p = shapiro(data)
            ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
            normality_results.append((var, shapiro_stat, shapiro_p, ks_stat, ks_p))
        else:
            print(f"Insufficient data or zero variance for Shapiro test on variable '{var}'")

    normality_results_df = pd.DataFrame(normality_results,
                                        columns=["Variable", "Shapiro_Statistic", "Shapiro_P", "KS_Statistic", "KS_P"])

    non_normal_vars_df = normality_results_df[
        (normality_results_df['Shapiro_P'] < 0.05) | (normality_results_df['KS_P'] < 0.05)]

    print("\nVariabili che non seguono una distribuzione normale:")
    print(non_normal_vars_df)

    if len(non_normal_vars_df) == len(numeric_vars):
        print("\nTutte le variabili numeriche non sono distribuite normalmente.")

    # Kruskal-Wallis
    kruskal_results = []
    for var in numeric_vars:
        data = df[[var, target_col]]
        groups = [data[data[target_col] == group][var].values for group in data[target_col].unique()]
        kruskal_stat, kruskal_p = kruskal(*groups)
        kruskal_results.append((var, kruskal_stat, kruskal_p))

    kruskal_results_df = pd.DataFrame(
        kruskal_results,
        columns=["Variable", "Kruskal_Statistic", "Kruskal_P"]
    )

    significant_vars_df = kruskal_results_df[kruskal_results_df['Kruskal_P'] < 0.05]

    print("\nVariabili significative dopo Kruskal-Wallis (p-value < 0.05):")
    print(significant_vars_df)

    # Test di Mann-Whitney per le variabili significative
    mannwhitney_results = []

    for var in significant_vars_df['Variable']:
        data = df[[var, target_col]]
        for (group1, group2) in combinations(data[target_col].unique(), 2):
            group1_data = data[data[target_col] == group1][var].values
            group2_data = data[data[target_col] == group2][var].values
            mw_stat, mw_p = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            mannwhitney_results.append((var, group1, group2, mw_stat, mw_p))

    mannwhitney_results_df = pd.DataFrame(
        mannwhitney_results,
        columns=["Variable", "Group1", "Group2", "MW_Statistic", "MW_P"]
    )

    # Correzione di Bonferroni
    mannwhitney_results_df['Bonferroni_P'] = multipletests(mannwhitney_results_df['MW_P'], method='bonferroni')[1]
    significant_mw_results_df = mannwhitney_results_df[mannwhitney_results_df['Bonferroni_P'] < 0.05]

    if len(significant_mw_results_df) > 0:
        print("\nRisultati significativi di Mann-Whitney (p-value Bonferroni corretto < 0.05):")
        print(significant_mw_results_df[['Variable', 'Group1', 'Group2', 'MW_Statistic', 'MW_P', 'Bonferroni_P']])
    else:
        print("\nNessun risultato significativo di Mann-Whitney dopo la correzione di Bonferroni.")

    # Selezioniamo le variabili significative da Kruskal-Wallis e Mann-Whitney
    significant_mw_vars = significant_mw_results_df['Variable'].unique()
    final_significant_vars = significant_vars_df[significant_vars_df['Variable'].isin(significant_mw_vars)]

    print("\nVariabili finali significative da mantenere:")
    print(final_significant_vars.iloc[:]['Variable'])

# Ciclo per eseguire l'analisi per entrambi i metodi di clustering
for target in ['PCA_Agglomerative_Cluster', 'PCA_KMedoids_Cluster']:
    perform_statistical_analysis(dfPCA, target)

# FISHER IN R
'''> df <- read.csv("/Users/pietropante/Desktop/Files4Python/PCA_perFisher.csv", stringsAsFactors = FALSE)
> 
> # Modifica il data frame per adattarsi ai vari metodi di clustering
> df$gender <- as.factor(df$gender)
> df$education <- as.factor(df$education)
> df$marital <- as.factor(df$marital)
> 
> # Funzione per eseguire il test di Fisher per tutte le variabili categoriali
> run_pairwise_fisher <- function(df_categorical) {
+   cluster_var <- "cluster"
+   cluster_levels <- levels(df_categorical[[cluster_var]])
+   variables <- names(df_categorical)[-which(names(df_categorical) == cluster_var)]
+   pairwise_results <- list()
+   
+   # Esegui il test di Fisher per ciascuna variabile categoriale
+   for (var in variables) {
+     full_table <- table(df_categorical[[var]], df_categorical[[cluster_var]])
+     results_var <- combn(cluster_levels, 2, function(pair) {
+       subset_table <- full_table[, pair]
+       fisher_test <- fisher.test(subset_table, simulate.p.value = TRUE)
+       data.frame(
+         Variable = var,
+         Cluster_1 = pair[1],
+         Cluster_2 = pair[2],
+         Fisher_P = fisher_test$p.value
+       )
+     }, simplify = FALSE)
+     pairwise_results[[var]] <- do.call(rbind, results_var)
+   }
+   
+   # Combina i risultati in un unico data frame
+   all_results <- do.call(rbind, pairwise_results)
+   
+   # Applica la correzione di Bonferroni
+   m <- nrow(all_results)  # numero totale di test (combinazioni di cluster)
+   alpha <- 0.05  # livello di significatività globale
+   all_results$Bonferroni_P <- p.adjust(all_results$Fisher_P, method = "bonferroni")
+   
+   # Determina se i risultati sono significativi o meno
+   all_results$Significance <- ifelse(all_results$Bonferroni_P < alpha, "Significativa", "Non significativa")
+   
+   return(all_results)
+ }
> 
> # Esegui il test di Fisher per il primo clustering (Agglomerative)
> df$cluster <- as.factor(df$PCA_Agglomerative_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> pairwise_results_agg <- run_pairwise_fisher(df_categorical)
> 
> # Visualizza i risultati per Agglomerative Clustering
> cat("Agglomerative Clustering - Test di Fisher:\n")
Agglomerative Clustering - Test di Fisher:
> cat("   Variable Cluster_1 Cluster_2     Fisher_P\n")
   Variable Cluster_1 Cluster_2     Fisher_P
> apply(pairwise_results_agg, 1, function(row) {
+   cat(sprintf("%10s %9s %9s %13.10f\n", row['Variable'], row['Cluster_1'], row['Cluster_2'], as.numeric(row['Fisher_P'])))
+ })
    gender         0         1  0.2138930535
    gender         0         2  0.6741629185
    gender         1         2  0.4257871064
 education         0         1  0.0004997501
 education         0         2  0.0004997501
 education         1         2  0.0004997501
   marital         0         1  0.0004997501
   marital         0         2  0.0004997501
   marital         1         2  0.0004997501
NULL
> cat("  Bonferroni_P      Significance\n")
  Bonferroni_P      Significance
> apply(pairwise_results_agg, 1, function(row) {
+   cat(sprintf("%13.9f %s\n", as.numeric(row['Bonferroni_P']), row['Significance']))
+ })
  1.000000000 Non significativa
  1.000000000 Non significativa
  1.000000000 Non significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
NULL
> 
> # Esegui il test di Fisher per il secondo clustering (KMedoids)
> df$cluster <- as.factor(df$PCA_KMedoids_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> pairwise_results_kmedoids <- run_pairwise_fisher(df_categorical)
> 
> # Visualizza i risultati per KMedoids Clustering
> cat("\nKMedoids Clustering - Test di Fisher:\n")

KMedoids Clustering - Test di Fisher:
> cat("   Variable Cluster_1 Cluster_2     Fisher_P\n")
   Variable Cluster_1 Cluster_2     Fisher_P
> apply(pairwise_results_kmedoids, 1, function(row) {
+   cat(sprintf("%10s %9s %9s %13.10f\n", row['Variable'], row['Cluster_1'], row['Cluster_2'], as.numeric(row['Fisher_P'])))
+ })
    gender         0         1  0.2663668166
    gender         0         2  0.5327336332
    gender         1         2  0.5692153923
 education         0         1  0.0004997501
 education         0         2  0.0004997501
 education         1         2  0.0004997501
   marital         0         1  0.0004997501
   marital         0         2  0.0004997501
   marital         1         2  0.0004997501
NULL
> cat("  Bonferroni_P      Significance\n")
  Bonferroni_P      Significance
> apply(pairwise_results_kmedoids, 1, function(row) {
+   cat(sprintf("%13.9f %s\n", as.numeric(row['Bonferroni_P']), row['Significance']))
+ })
  1.000000000 Non significativa
  1.000000000 Non significativa
  1.000000000 Non significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
NULL
> 
> # Funzione per eseguire il test di Fisher globale
> run_fisher_global <- function(df_categorical) {
+   cluster_var <- "cluster"
+   variables <- names(df_categorical)[-which(names(df_categorical) == cluster_var)]
+   results <- list()
+   
+   # Esegui il test di Fisher per ciascuna variabile categoriale
+   for (var in variables) {
+     # Crea la tabella di contingenza completa per la variabile rispetto ai cluster
+     contingency_table <- table(df_categorical[[var]], df_categorical[[cluster_var]])
+     
+     # Esegui il test di Fisher
+     fisher_test <- fisher.test(contingency_table, simulate.p.value = TRUE)
+     
+     # Salva i risultati
+     results[[var]] <- data.frame(
+       Variable = var,
+       P_Value = fisher_test$p.value
+     )
+   }
+   
+   # Combina i risultati in un unico data frame
+   all_results <- do.call(rbind, results)
+   
+   # Determina se i risultati sono significativi o meno
+   all_results$Significance <- ifelse(all_results$P_Value < 0.05, "Significativa", "Non significativa")
+   
+   return(all_results)
+ }
> 
> # Esegui il test di Fisher globale per Agglomerative Clustering
> df$cluster <- as.factor(df$PCA_Agglomerative_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> global_results_agg <- run_fisher_global(df_categorical)
> 
> # Visualizza i risultati per Agglomerative Clustering
> cat("Agglomerative Clustering - Test di Fisher (Globale):\n")
Agglomerative Clustering - Test di Fisher (Globale):
> print(global_results_agg)
           Variable      P_Value      Significance
gender       gender 0.4502748626 Non significativa
education education 0.0004997501     Significativa
marital     marital 0.0004997501     Significativa
> 
> # Esegui il test di Fisher globale per KMedoids Clustering
> df$cluster <- as.factor(df$PCA_KMedoids_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> global_results_kmedoids <- run_fisher_global(df_categorical)
> 
> # Visualizza i risultati per KMedoids Clustering
> cat("\nKMedoids Clustering - Test di Fisher (Globale):\n")

KMedoids Clustering - Test di Fisher (Globale):
> print(global_results_kmedoids)
           Variable      P_Value      Significance
gender       gender 0.5102448776 Non significativa
education education 0.0004997501     Significativa
marital     marital 0.0004997501     Significativa
'''
# Meglio Agglomerative
dfPCA = dfPCA.drop(columns=['PCA_KMedoids_Cluster'])
final_table = pd.DataFrame()

# Calcoliamo il numero di persone per cluster
cluster_counts = dfPCA['PCA_Agglomerative_Cluster'].value_counts()

# Aggiungiamo i conteggi dei membri dei cluster nell'header
final_table.loc['Cluster Size', cluster_counts.index] = cluster_counts.values
for var in dfPCA.columns:
    if var != 'PCA_Agglomerative_Cluster':
        if dfPCA[var].dtype in ['float64', 'int64']:
            stats_per_cluster = dfPCA.groupby('PCA_Agglomerative_Cluster')[var].agg(['median', 'min', 'max'])
            stats_per_cluster = stats_per_cluster.round()

            final_table.loc[var, stats_per_cluster.index] = [
                f"{int(median)} ({int(min_val)} - {int(max_val)})"
                for median, min_val, max_val in
                zip(stats_per_cluster['median'], stats_per_cluster['min'], stats_per_cluster['max'])
            ]
        else:
            mode_per_cluster = dfPCA.groupby('PCA_Agglomerative_Cluster')[var].agg(lambda x: x.mode()[0])
            mode_freq_per_cluster = dfPCA.groupby('PCA_Agglomerative_Cluster')[var].agg(lambda x: (x == x.mode()[0]).sum())

            total_counts = dfPCA.groupby('PCA_Agglomerative_Cluster')[var].count()
            freq_per_cluster = mode_freq_per_cluster / total_counts * 100
            final_table.loc[var, mode_per_cluster.index] = [
                f"{mode} ({freq:.2f}%)" for mode, freq in zip(mode_per_cluster, freq_per_cluster)
            ]

final_table.columns = [f'Cluster {cluster}' for cluster in final_table.columns]
print(
    "\nTabella con media (min-max) per variabili numeriche e moda con frequenza per variabili categoriche per ogni cluster:")
print(final_table)

#final_table.to_excel("Persona_Table.xlsx", sheet_name="PCA Cluster", index=True)

#FAMD
dfFAMD.to_csv("/Users/pietropante/Desktop/Files4Python/FAMD_perFisher.csv", index=False)

dfFAMD['FAMD_Agglomerative_Cluster'] = dfFAMD['FAMD_Agglomerative_Cluster'].astype('object')
dfFAMD['FAMD_KMedoids_Cluster'] = dfFAMD['FAMD_KMedoids_Cluster'].astype('object')
num_unique_values_aggl = dfFAMD['FAMD_Agglomerative_Cluster'].nunique()
num_unique_values_kmed = dfFAMD['FAMD_KMedoids_Cluster'].nunique()
print(f"The number of clusters in FAMD_Agglomerative_Cluster is {num_unique_values_aggl}, while in FAMD_KMedoids_Cluster it is {num_unique_values_kmed}.")
numeric_vars = df.select_dtypes(include=['float64', 'int64']).columns

# Funzione per l'analisi della normalità e Kruskal-Wallis
def perform_statistical_analysis(df, target_col):
    print(f"\nAnalisi statistica per il clustering su '{target_col}':")

    # Analisi della normalità
    normality_results = []
    for var in numeric_vars:
        data = df[var]
        if data.nunique() > 1:
            shapiro_stat, shapiro_p = shapiro(data)
            ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
            normality_results.append((var, shapiro_stat, shapiro_p, ks_stat, ks_p))
        else:
            print(f"Insufficient data or zero variance for Shapiro test on variable '{var}'")

    normality_results_df = pd.DataFrame(normality_results,
                                        columns=["Variable", "Shapiro_Statistic", "Shapiro_P", "KS_Statistic", "KS_P"])

    non_normal_vars_df = normality_results_df[
        (normality_results_df['Shapiro_P'] < 0.05) | (normality_results_df['KS_P'] < 0.05)]

    print("\nVariabili che non seguono una distribuzione normale:")
    print(non_normal_vars_df)

    if len(non_normal_vars_df) == len(numeric_vars):
        print("\nTutte le variabili numeriche non sono distribuite normalmente.")

    # Kruskal-Wallis
    kruskal_results = []
    for var in numeric_vars:
        data = df[[var, target_col]]
        groups = [data[data[target_col] == group][var].values for group in data[target_col].unique()]
        kruskal_stat, kruskal_p = kruskal(*groups)
        kruskal_results.append((var, kruskal_stat, kruskal_p))

    kruskal_results_df = pd.DataFrame(
        kruskal_results,
        columns=["Variable", "Kruskal_Statistic", "Kruskal_P"]
    )

    significant_vars_df = kruskal_results_df[kruskal_results_df['Kruskal_P'] < 0.05]

    print("\nVariabili significative dopo Kruskal-Wallis (p-value < 0.05):")
    print(significant_vars_df)

    # Test di Mann-Whitney per le variabili significative
    mannwhitney_results = []

    for var in significant_vars_df['Variable']:
        data = df[[var, target_col]]
        for (group1, group2) in combinations(data[target_col].unique(), 2):
            group1_data = data[data[target_col] == group1][var].values
            group2_data = data[data[target_col] == group2][var].values
            mw_stat, mw_p = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            mannwhitney_results.append((var, group1, group2, mw_stat, mw_p))

    mannwhitney_results_df = pd.DataFrame(
        mannwhitney_results,
        columns=["Variable", "Group1", "Group2", "MW_Statistic", "MW_P"]
    )

    # Correzione di Bonferroni
    mannwhitney_results_df['Bonferroni_P'] = multipletests(mannwhitney_results_df['MW_P'], method='bonferroni')[1]
    significant_mw_results_df = mannwhitney_results_df[mannwhitney_results_df['Bonferroni_P'] < 0.05]

    if len(significant_mw_results_df) > 0:
        print("\nRisultati significativi di Mann-Whitney (p-value Bonferroni corretto < 0.05):")
        print(significant_mw_results_df[['Variable', 'Group1', 'Group2', 'MW_Statistic', 'MW_P', 'Bonferroni_P']])
    else:
        print("\nNessun risultato significativo di Mann-Whitney dopo la correzione di Bonferroni.")

    # Selezioniamo le variabili significative da Kruskal-Wallis e Mann-Whitney
    significant_mw_vars = significant_mw_results_df['Variable'].unique()
    final_significant_vars = significant_vars_df[significant_vars_df['Variable'].isin(significant_mw_vars)]

    print("\nVariabili finali significative da mantenere:")
    print(final_significant_vars.iloc[:]['Variable'])

# Ciclo per eseguire l'analisi per entrambi i metodi di clustering
for target in ['FAMD_Agglomerative_Cluster', 'FAMD_KMedoids_Cluster']:
    perform_statistical_analysis(dfFAMD, target)

# FISHER IN R
'''> df <- read.csv("/Users/pietropante/Desktop/Files4Python/FAMD_perFisher.csv", stringsAsFactors = FALSE)
> 
> # Modifica il data frame per adattarsi ai vari metodi di clustering
> df$gender <- as.factor(df$gender)
> df$education <- as.factor(df$education)
> df$marital <- as.factor(df$marital)
> 
> # Funzione per eseguire il test di Fisher per tutte le variabili categoriali
> run_pairwise_fisher <- function(df_categorical) {
+   cluster_var <- "cluster"
+   cluster_levels <- levels(df_categorical[[cluster_var]])
+   variables <- names(df_categorical)[-which(names(df_categorical) == cluster_var)]
+   pairwise_results <- list()
+   
+   # Esegui il test di Fisher per ciascuna variabile categoriale
+   for (var in variables) {
+     full_table <- table(df_categorical[[var]], df_categorical[[cluster_var]])
+     results_var <- combn(cluster_levels, 2, function(pair) {
+       subset_table <- full_table[, pair]
+       fisher_test <- fisher.test(subset_table, simulate.p.value = TRUE)
+       data.frame(
+         Variable = var,
+         Cluster_1 = pair[1],
+         Cluster_2 = pair[2],
+         Fisher_P = fisher_test$p.value
+       )
+     }, simplify = FALSE)
+     pairwise_results[[var]] <- do.call(rbind, results_var)
+   }
+   
+   # Combina i risultati in un unico data frame
+   all_results <- do.call(rbind, pairwise_results)
+   
+   # Applica la correzione di Bonferroni
+   m <- nrow(all_results)  # numero totale di test (combinazioni di cluster)
+   alpha <- 0.05  # livello di significatività globale
+   all_results$Bonferroni_P <- p.adjust(all_results$Fisher_P, method = "bonferroni")
+   
+   # Determina se i risultati sono significativi o meno
+   all_results$Significance <- ifelse(all_results$Bonferroni_P < alpha, "Significativa", "Non significativa")
+   
+   return(all_results)
+ }
> 
> # Esegui il test di Fisher per il primo clustering (Agglomerative)
> df$cluster <- as.factor(df$FAMD_Agglomerative_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> pairwise_results_agg <- run_pairwise_fisher(df_categorical)
> 
> # Visualizza i risultati per Agglomerative Clustering
> cat("Agglomerative Clustering - Test di Fisher:\n")
Agglomerative Clustering - Test di Fisher:
> cat("   Variable Cluster_1 Cluster_2     Fisher_P\n")
   Variable Cluster_1 Cluster_2     Fisher_P
> apply(pairwise_results_agg, 1, function(row) {
+   cat(sprintf("%10s %9s %9s %13.10f\n", row['Variable'], row['Cluster_1'], row['Cluster_2'], as.numeric(row['Fisher_P'])))
+ })
    gender         0         1  0.0029985007
    gender         0         2  0.5807096452
    gender         0         3  0.3908045977
    gender         0         4  0.0899550225
    gender         1         2  0.0889555222
    gender         1         3  0.3293353323
    gender         1         4  0.0029985007
    gender         2         3  0.9815092454
    gender         2         4  0.0319840080
    gender         3         4  0.0334832584
 education         0         1  0.0004997501
 education         0         2  0.0004997501
 education         0         3  0.0004997501
 education         0         4  0.0004997501
 education         1         2  0.0004997501
 education         1         3  0.0004997501
 education         1         4  0.0004997501
 education         2         3  0.0004997501
 education         2         4  0.0039980010
 education         3         4  0.0019990005
   marital         0         1  0.0004997501
   marital         0         2  0.0004997501
   marital         0         3  0.0004997501
   marital         0         4  0.0004997501
   marital         1         2  0.0004997501
   marital         1         3  0.0004997501
   marital         1         4  0.0004997501
   marital         2         3  0.0004997501
   marital         2         4  0.0004997501
   marital         3         4  0.0004997501
NULL
> cat("  Bonferroni_P      Significance\n")
  Bonferroni_P      Significance
> apply(pairwise_results_agg, 1, function(row) {
+   cat(sprintf("%13.9f %s\n", as.numeric(row['Bonferroni_P']), row['Significance']))
+ })
  0.089955020 Non significativa
  1.000000000 Non significativa
  1.000000000 Non significativa
  1.000000000 Non significativa
  1.000000000 Non significativa
  1.000000000 Non significativa
  0.089955020 Non significativa
  1.000000000 Non significativa
  0.959520240 Non significativa
  1.000000000 Non significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.119940030 Non significativa
  0.059970010 Non significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
  0.014992500 Significativa
NULL
> 
> # Esegui il test di Fisher per il secondo clustering (KMedoids)
> df$cluster <- as.factor(df$FAMD_KMedoids_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> pairwise_results_kmedoids <- run_pairwise_fisher(df_categorical)
> 
> # Visualizza i risultati per KMedoids Clustering
> cat("\nKMedoids Clustering - Test di Fisher:\n")

KMedoids Clustering - Test di Fisher:
> cat("   Variable Cluster_1 Cluster_2     Fisher_P\n")
   Variable Cluster_1 Cluster_2     Fisher_P
> apply(pairwise_results_kmedoids, 1, function(row) {
+   cat(sprintf("%10s %9s %9s %13.10f\n", row['Variable'], row['Cluster_1'], row['Cluster_2'], as.numeric(row['Fisher_P'])))
+ })
    gender         0         1  0.5447276362
    gender         0         2  0.0059970015
    gender         1         2  0.0669665167
 education         0         1  0.0004997501
 education         0         2  0.0004997501
 education         1         2  0.0004997501
   marital         0         1  0.0009995002
   marital         0         2  0.0004997501
   marital         1         2  0.0004997501
NULL
> cat("  Bonferroni_P      Significance\n")
  Bonferroni_P      Significance
> apply(pairwise_results_kmedoids, 1, function(row) {
+   cat(sprintf("%13.9f %s\n", as.numeric(row['Bonferroni_P']), row['Significance']))
+ })
  1.000000000 Non significativa
  0.053973013 Non significativa
  0.602698651 Non significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.008995502 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
NULL
> 
> # Funzione per eseguire il test di Fisher globale
> run_fisher_global <- function(df_categorical) {
+   cluster_var <- "cluster"
+   variables <- names(df_categorical)[-which(names(df_categorical) == cluster_var)]
+   results <- list()
+   
+   # Esegui il test di Fisher per ciascuna variabile categoriale
+   for (var in variables) {
+     # Crea la tabella di contingenza completa per la variabile rispetto ai cluster
+     contingency_table <- table(df_categorical[[var]], df_categorical[[cluster_var]])
+     
+     # Esegui il test di Fisher
+     fisher_test <- fisher.test(contingency_table, simulate.p.value = TRUE)
+     
+     # Salva i risultati
+     results[[var]] <- data.frame(
+       Variable = var,
+       P_Value = fisher_test$p.value
+     )
+   }
+   
+   # Combina i risultati in un unico data frame
+   all_results <- do.call(rbind, results)
+   
+   # Determina se i risultati sono significativi o meno
+   all_results$Significance <- ifelse(all_results$P_Value < 0.05, "Significativa", "Non significativa")
+   
+   return(all_results)
+ }
> 
> # Esegui il test di Fisher globale per Agglomerative Clustering
> df$cluster <- as.factor(df$FAMD_Agglomerative_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> global_results_agg <- run_fisher_global(df_categorical)
> 
> # Visualizza i risultati per Agglomerative Clustering
> cat("Agglomerative Clustering - Test di Fisher (Globale):\n")
Agglomerative Clustering - Test di Fisher (Globale):
> print(global_results_agg)
           Variable      P_Value  Significance
gender       gender 0.0124937531 Significativa
education education 0.0004997501 Significativa
marital     marital 0.0004997501 Significativa
> 
> # Esegui il test di Fisher globale per KMedoids Clustering
> df$cluster <- as.factor(df$FAMD_KMedoids_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> global_results_kmedoids <- run_fisher_global(df_categorical)
> 
> # Visualizza i risultati per KMedoids Clustering
> cat("\nKMedoids Clustering - Test di Fisher (Globale):\n")

KMedoids Clustering - Test di Fisher (Globale):
> print(global_results_kmedoids)
           Variable      P_Value  Significance
gender       gender 0.0269865067 Significativa
education education 0.0004997501 Significativa
marital     marital 0.0004997501 Significativa
'''
# Meglio k-medoids
dfFAMD = dfFAMD.drop(columns=['FAMD_Agglomerative_Cluster'])
final_table = pd.DataFrame()

# Calcoliamo il numero di persone per cluster
cluster_counts = dfFAMD['FAMD_KMedoids_Cluster'].value_counts()

# Aggiungiamo i conteggi dei membri dei cluster nell'header
final_table.loc['Cluster Size', cluster_counts.index] = cluster_counts.values
for var in dfFAMD.columns:
    if var != 'FAMD_KMedoids_Cluster':
        if dfFAMD[var].dtype in ['float64', 'int64']:
            stats_per_cluster = dfFAMD.groupby('FAMD_KMedoids_Cluster')[var].agg(['median', 'min', 'max'])
            stats_per_cluster = stats_per_cluster.round()

            final_table.loc[var, stats_per_cluster.index] = [
                f"{int(median)} ({int(min_val)} - {int(max_val)})"
                for median, min_val, max_val in
                zip(stats_per_cluster['median'], stats_per_cluster['min'], stats_per_cluster['max'])
            ]
        else:
            mode_per_cluster = dfFAMD.groupby('FAMD_KMedoids_Cluster')[var].agg(lambda x: x.mode()[0])
            mode_freq_per_cluster = dfFAMD.groupby('FAMD_KMedoids_Cluster')[var].agg(lambda x: (x == x.mode()[0]).sum())

            total_counts = dfFAMD.groupby('FAMD_KMedoids_Cluster')[var].count()
            freq_per_cluster = mode_freq_per_cluster / total_counts * 100
            final_table.loc[var, mode_per_cluster.index] = [
                f"{mode} ({freq:.2f}%)" for mode, freq in zip(mode_per_cluster, freq_per_cluster)
            ]

final_table.columns = [f'Cluster {cluster}' for cluster in final_table.columns]
print(
    "\nTabella con media (min-max) per variabili numeriche e moda con frequenza per variabili categoriche per ogni cluster:")
print(final_table)

'''# Aggiunge un nuovo foglio al file esistente
with ExcelWriter("Persona_Table.xlsx", mode="a", engine="openpyxl") as writer:
    final_table.to_excel(writer, sheet_name="FAMD Cluster", index=True)
'''
#FAMD_R
dfFAMD_R.to_csv("/Users/pietropante/Desktop/Files4Python/FAMD_R_perFisher.csv", index=False)

dfFAMD_R['FAMD_R_Agglomerative_Cluster'] = dfFAMD_R['FAMD_R_Agglomerative_Cluster'].astype('object')
dfFAMD_R['FAMD_R_KMedoids_Cluster'] = dfFAMD_R['FAMD_R_KMedoids_Cluster'].astype('object')
num_unique_values_aggl = dfFAMD_R['FAMD_R_Agglomerative_Cluster'].nunique()
num_unique_values_kmed = dfFAMD_R['FAMD_R_KMedoids_Cluster'].nunique()
print(f"The number of clusters in FAMD_R_Agglomerative_Cluster is {num_unique_values_aggl}, while in FAMD_R_KMedoids_Cluster it is {num_unique_values_kmed}.")
numeric_vars = df.select_dtypes(include=['float64', 'int64']).columns

# Funzione per l'analisi della normalità e Kruskal-Wallis
def perform_statistical_analysis(df, target_col):
    print(f"\nAnalisi statistica per il clustering su '{target_col}':")

    # Analisi della normalità
    normality_results = []
    for var in numeric_vars:
        data = df[var]
        if data.nunique() > 1:
            shapiro_stat, shapiro_p = shapiro(data)
            ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
            normality_results.append((var, shapiro_stat, shapiro_p, ks_stat, ks_p))
        else:
            print(f"Insufficient data or zero variance for Shapiro test on variable '{var}'")

    normality_results_df = pd.DataFrame(normality_results,
                                        columns=["Variable", "Shapiro_Statistic", "Shapiro_P", "KS_Statistic", "KS_P"])

    non_normal_vars_df = normality_results_df[
        (normality_results_df['Shapiro_P'] < 0.05) | (normality_results_df['KS_P'] < 0.05)]

    print("\nVariabili che non seguono una distribuzione normale:")
    print(non_normal_vars_df)

    if len(non_normal_vars_df) == len(numeric_vars):
        print("\nTutte le variabili numeriche non sono distribuite normalmente.")

    # Kruskal-Wallis
    kruskal_results = []
    for var in numeric_vars:
        data = df[[var, target_col]]
        groups = [data[data[target_col] == group][var].values for group in data[target_col].unique()]
        kruskal_stat, kruskal_p = kruskal(*groups)
        kruskal_results.append((var, kruskal_stat, kruskal_p))

    kruskal_results_df = pd.DataFrame(
        kruskal_results,
        columns=["Variable", "Kruskal_Statistic", "Kruskal_P"]
    )

    significant_vars_df = kruskal_results_df[kruskal_results_df['Kruskal_P'] < 0.05]

    print("\nVariabili significative dopo Kruskal-Wallis (p-value < 0.05):")
    print(significant_vars_df)

    # Test di Mann-Whitney per le variabili significative
    mannwhitney_results = []

    for var in significant_vars_df['Variable']:
        data = df[[var, target_col]]
        for (group1, group2) in combinations(data[target_col].unique(), 2):
            group1_data = data[data[target_col] == group1][var].values
            group2_data = data[data[target_col] == group2][var].values
            mw_stat, mw_p = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            mannwhitney_results.append((var, group1, group2, mw_stat, mw_p))

    mannwhitney_results_df = pd.DataFrame(
        mannwhitney_results,
        columns=["Variable", "Group1", "Group2", "MW_Statistic", "MW_P"]
    )

    # Correzione di Bonferroni
    mannwhitney_results_df['Bonferroni_P'] = multipletests(mannwhitney_results_df['MW_P'], method='bonferroni')[1]
    significant_mw_results_df = mannwhitney_results_df[mannwhitney_results_df['Bonferroni_P'] < 0.05]

    if len(significant_mw_results_df) > 0:
        print("\nRisultati significativi di Mann-Whitney (p-value Bonferroni corretto < 0.05):")
        pd.set_option('display.max_columns', 500)
        print(significant_mw_results_df[['Variable', 'Group1', 'Group2', 'MW_Statistic', 'MW_P', 'Bonferroni_P']])
    else:
        print("\nNessun risultato significativo di Mann-Whitney dopo la correzione di Bonferroni.")

    # Selezioniamo le variabili significative da Kruskal-Wallis e Mann-Whitney
    significant_mw_vars = significant_mw_results_df['Variable'].unique()
    final_significant_vars = significant_vars_df[significant_vars_df['Variable'].isin(significant_mw_vars)]

    print("\nVariabili finali significative da mantenere:")
    print(final_significant_vars.iloc[:]['Variable'])

# Ciclo per eseguire l'analisi per entrambi i metodi di clustering
for target in ['FAMD_R_Agglomerative_Cluster', 'FAMD_R_KMedoids_Cluster']:
    perform_statistical_analysis(dfFAMD_R, target)

# FISHER IN R
'''
> df <- read.csv("/Users/pietropante/Desktop/Files4Python/FAMD_R_perFisher.csv", stringsAsFactors = FALSE)
> 
> # Modifica il data frame per adattarsi ai vari metodi di clustering
> df$gender <- as.factor(df$gender)
> df$education <- as.factor(df$education)
> df$marital <- as.factor(df$marital)
> 
> # Funzione per eseguire il test di Fisher per tutte le variabili categoriali
> run_pairwise_fisher <- function(df_categorical) {
+   cluster_var <- "cluster"
+   cluster_levels <- levels(df_categorical[[cluster_var]])
+   variables <- names(df_categorical)[-which(names(df_categorical) == cluster_var)]
+   pairwise_results <- list()
+   
+   # Esegui il test di Fisher per ciascuna variabile categoriale
+   for (var in variables) {
+     full_table <- table(df_categorical[[var]], df_categorical[[cluster_var]])
+     results_var <- combn(cluster_levels, 2, function(pair) {
+       subset_table <- full_table[, pair]
+       fisher_test <- fisher.test(subset_table, simulate.p.value = TRUE)
+       data.frame(
+         Variable = var,
+         Cluster_1 = pair[1],
+         Cluster_2 = pair[2],
+         Fisher_P = fisher_test$p.value
+       )
+     }, simplify = FALSE)
+     pairwise_results[[var]] <- do.call(rbind, results_var)
+   }
+   
+   # Combina i risultati in un unico data frame
+   all_results <- do.call(rbind, pairwise_results)
+   
+   # Applica la correzione di Bonferroni
+   m <- nrow(all_results)  # numero totale di test (combinazioni di cluster)
+   alpha <- 0.05  # livello di significatività globale
+   all_results$Bonferroni_P <- p.adjust(all_results$Fisher_P, method = "bonferroni")
+   
+   # Determina se i risultati sono significativi o meno
+   all_results$Significance <- ifelse(all_results$Bonferroni_P < alpha, "Significativa", "Non significativa")
+   
+   return(all_results)
+ }
> 
> # Esegui il test di Fisher per il primo clustering (Agglomerative)
> df$cluster <- as.factor(df$FAMD_R_Agglomerative_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> pairwise_results_agg <- run_pairwise_fisher(df_categorical)
> 
> # Visualizza i risultati per Agglomerative Clustering
> cat("Agglomerative Clustering - Test di Fisher:\n")
Agglomerative Clustering - Test di Fisher:
> cat("   Variable Cluster_1 Cluster_2     Fisher_P\n")
   Variable Cluster_1 Cluster_2     Fisher_P
> apply(pairwise_results_agg, 1, function(row) {
+   cat(sprintf("%10s %9s %9s %13.10f\n", row['Variable'], row['Cluster_1'], row['Cluster_2'], as.numeric(row['Fisher_P'])))
+ })
    gender         0         1  0.5107446277
    gender         0         2  0.1704147926
    gender         1         2  0.5417291354
 education         0         1  0.0004997501
 education         0         2  0.0004997501
 education         1         2  0.0004997501
   marital         0         1  0.0004997501
   marital         0         2  0.0004997501
   marital         1         2  0.0004997501
NULL
> cat("  Bonferroni_P      Significance\n")
  Bonferroni_P      Significance
> apply(pairwise_results_agg, 1, function(row) {
+   cat(sprintf("%13.9f %s\n", as.numeric(row['Bonferroni_P']), row['Significance']))
+ })
  1.000000000 Non significativa
  1.000000000 Non significativa
  1.000000000 Non significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
NULL
> 
> # Esegui il test di Fisher per il secondo clustering (KMedoids)
> df$cluster <- as.factor(df$FAMD_R_KMedoids_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> pairwise_results_kmedoids <- run_pairwise_fisher(df_categorical)
> 
> # Visualizza i risultati per KMedoids Clustering
> cat("\nKMedoids Clustering - Test di Fisher:\n")

KMedoids Clustering - Test di Fisher:
> cat("   Variable Cluster_1 Cluster_2     Fisher_P\n")
   Variable Cluster_1 Cluster_2     Fisher_P
> apply(pairwise_results_kmedoids, 1, function(row) {
+   cat(sprintf("%10s %9s %9s %13.10f\n", row['Variable'], row['Cluster_1'], row['Cluster_2'], as.numeric(row['Fisher_P'])))
+ })
    gender         0         1  0.7281359320
    gender         0         2  0.0004997501
    gender         0         3  0.1329335332
    gender         1         2  0.0004997501
    gender         1         3  0.1734132934
    gender         2         3  0.0024987506
 education         0         1  0.0004997501
 education         0         2  0.0004997501
 education         0         3  0.0004997501
 education         1         2  0.4097951024
 education         1         3  0.0004997501
 education         2         3  0.0004997501
   marital         0         1  0.0004997501
   marital         0         2  0.0004997501
   marital         0         3  0.0004997501
   marital         1         2  0.0004997501
   marital         1         3  0.0004997501
   marital         2         3  0.0004997501
NULL
> cat("  Bonferroni_P      Significance\n")
  Bonferroni_P      Significance
> apply(pairwise_results_kmedoids, 1, function(row) {
+   cat(sprintf("%13.9f %s\n", as.numeric(row['Bonferroni_P']), row['Significance']))
+ })
  1.000000000 Non significativa
  0.008995502 Significativa
  1.000000000 Non significativa
  0.008995502 Significativa
  1.000000000 Non significativa
  0.044977511 Significativa
  0.008995502 Significativa
  0.008995502 Significativa
  0.008995502 Significativa
  1.000000000 Non significativa
  0.008995502 Significativa
  0.008995502 Significativa
  0.008995502 Significativa
  0.008995502 Significativa
  0.008995502 Significativa
  0.008995502 Significativa
  0.008995502 Significativa
  0.008995502 Significativa
NULL
> 
> 
> # Funzione per eseguire il test di Fisher globale
> run_fisher_global <- function(df_categorical) {
+   cluster_var <- "cluster"
+   variables <- names(df_categorical)[-which(names(df_categorical) == cluster_var)]
+   results <- list()
+   
+   # Esegui il test di Fisher per ciascuna variabile categoriale
+   for (var in variables) {
+     # Crea la tabella di contingenza completa per la variabile rispetto ai cluster
+     contingency_table <- table(df_categorical[[var]], df_categorical[[cluster_var]])
+     
+     # Esegui il test di Fisher
+     fisher_test <- fisher.test(contingency_table, simulate.p.value = TRUE)
+     
+     # Salva i risultati
+     results[[var]] <- data.frame(
+       Variable = var,
+       P_Value = fisher_test$p.value
+     )
+   }
+   
+   # Combina i risultati in un unico data frame
+   all_results <- do.call(rbind, results)
+   
+   # Determina se i risultati sono significativi o meno
+   all_results$Significance <- ifelse(all_results$P_Value < 0.05, "Significativa", "Non significativa")
+   
+   return(all_results)
+ }
> 
> # Esegui il test di Fisher globale per Agglomerative Clustering
> df$cluster <- as.factor(df$FAMD_R_Agglomerative_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> global_results_agg <- run_fisher_global(df_categorical)
> 
> # Visualizza i risultati per Agglomerative Clustering
> cat("Agglomerative Clustering - Test di Fisher (Globale):\n")
Agglomerative Clustering - Test di Fisher (Globale):
> print(global_results_agg)
           Variable      P_Value      Significance
gender       gender 0.3843078461 Non significativa
education education 0.0004997501     Significativa
marital     marital 0.0004997501     Significativa
> 
> # Esegui il test di Fisher globale per KMedoids Clustering
> df$cluster <- as.factor(df$FAMD_R_KMedoids_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> global_results_kmedoids <- run_fisher_global(df_categorical)
> 
> # Visualizza i risultati per KMedoids Clustering
> cat("\nKMedoids Clustering - Test di Fisher (Globale):\n")

KMedoids Clustering - Test di Fisher (Globale):
> print(global_results_kmedoids)
           Variable      P_Value  Significance
gender       gender 0.0004997501 Significativa
education education 0.0004997501 Significativa
marital     marital 0.0004997501 Significativa
'''
# Meglio Agglomerative
dfFAMD_R = dfFAMD_R.drop(columns=['FAMD_R_KMedoids_Cluster'])
final_table = pd.DataFrame()

# Calcoliamo il numero di persone per cluster
cluster_counts = dfFAMD_R['FAMD_R_Agglomerative_Cluster'].value_counts()

# Aggiungiamo i conteggi dei membri dei cluster nell'header
final_table.loc['Cluster Size', cluster_counts.index] = cluster_counts.values

for var in dfFAMD_R.columns:
    if var != 'FAMD_R_Agglomerative_Cluster':
        if dfFAMD_R[var].dtype in ['float64', 'int64']:
            stats_per_cluster = dfFAMD_R.groupby('FAMD_R_Agglomerative_Cluster')[var].agg(['median', 'min', 'max'])
            stats_per_cluster = stats_per_cluster.round()

            final_table.loc[var, stats_per_cluster.index] = [
                f"{int(median)} ({int(min_val)} - {int(max_val)})"
                for median, min_val, max_val in
                zip(stats_per_cluster['median'], stats_per_cluster['min'], stats_per_cluster['max'])
            ]
        else:
            mode_per_cluster = dfFAMD_R.groupby('FAMD_R_Agglomerative_Cluster')[var].agg(lambda x: x.mode()[0])
            mode_freq_per_cluster = dfFAMD_R.groupby('FAMD_R_Agglomerative_Cluster')[var].agg(lambda x: (x == x.mode()[0]).sum())

            total_counts = dfFAMD_R.groupby('FAMD_R_Agglomerative_Cluster')[var].count()
            freq_per_cluster = mode_freq_per_cluster / total_counts * 100
            final_table.loc[var, mode_per_cluster.index] = [
                f"{mode} ({freq:.2f}%)" for mode, freq in zip(mode_per_cluster, freq_per_cluster)
            ]

final_table.columns = [f'Cluster {cluster}' for cluster in final_table.columns]
print(
    "\nTabella con media (min-max) per variabili numeriche e moda con frequenza per variabili categoriche per ogni cluster:")
print(final_table)

'''with ExcelWriter("Persona_Table.xlsx", mode="a", engine="openpyxl") as writer:
    final_table.to_excel(writer, sheet_name="FAMD_R Cluster", index=True)
'''
#PCAMIX da R
dfPCAMIX_R.to_csv("/Users/pietropante/Desktop/Files4Python/PCAMIX_R_perFisher.csv", index=False)

dfPCAMIX_R['PCAMIX_R_Agglomerative_Cluster'] = dfPCAMIX_R['PCAMIX_R_Agglomerative_Cluster'].astype('object')
dfPCAMIX_R['PCAMIX_R_KMedoids_Cluster'] = dfPCAMIX_R['PCAMIX_R_KMedoids_Cluster'].astype('object')
num_unique_values_aggl = dfPCAMIX_R['PCAMIX_R_Agglomerative_Cluster'].nunique()
num_unique_values_kmed = dfPCAMIX_R['PCAMIX_R_KMedoids_Cluster'].nunique()
print(f"The number of clusters in PCAMIX_R_Agglomerative_Cluster is {num_unique_values_aggl}, while in PCAMIX_R_KMedoids_Cluster it is {num_unique_values_kmed}.")
numeric_vars = df.select_dtypes(include=['float64', 'int64']).columns

dfPCAMIX_R.to_csv("PCAMIX_R_perFisher.csv", index=False)

dfPCAMIX_R['PCAMIX_R_Agglomerative_Cluster'] = dfPCAMIX_R['PCAMIX_R_Agglomerative_Cluster'].astype('object')
dfPCAMIX_R['PCAMIX_R_KMedoids_Cluster'] = dfPCAMIX_R['PCAMIX_R_KMedoids_Cluster'].astype('object')

numeric_vars = df.select_dtypes(include=['float64', 'int64']).columns

# Funzione per l'analisi della normalità e Kruskal-Wallis
def perform_statistical_analysis(df, target_col):
    print(f"\nAnalisi statistica per il clustering su '{target_col}':")

    # Analisi della normalità
    normality_results = []
    for var in numeric_vars:
        data = df[var]
        if data.nunique() > 1:
            shapiro_stat, shapiro_p = shapiro(data)
            ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
            normality_results.append((var, shapiro_stat, shapiro_p, ks_stat, ks_p))
        else:
            print(f"Insufficient data or zero variance for Shapiro test on variable '{var}'")

    normality_results_df = pd.DataFrame(normality_results,
                                        columns=["Variable", "Shapiro_Statistic", "Shapiro_P", "KS_Statistic", "KS_P"])

    non_normal_vars_df = normality_results_df[
        (normality_results_df['Shapiro_P'] < 0.05) | (normality_results_df['KS_P'] < 0.05)]

    print("\nVariabili che non seguono una distribuzione normale:")
    print(non_normal_vars_df)

    if len(non_normal_vars_df) == len(numeric_vars):
        print("\nTutte le variabili numeriche non sono distribuite normalmente.")

    # Kruskal-Wallis
    kruskal_results = []
    for var in numeric_vars:
        data = df[[var, target_col]]
        groups = [data[data[target_col] == group][var].values for group in data[target_col].unique()]
        kruskal_stat, kruskal_p = kruskal(*groups)
        kruskal_results.append((var, kruskal_stat, kruskal_p))

    kruskal_results_df = pd.DataFrame(
        kruskal_results,
        columns=["Variable", "Kruskal_Statistic", "Kruskal_P"]
    )

    significant_vars_df = kruskal_results_df[kruskal_results_df['Kruskal_P'] < 0.05]

    print("\nVariabili significative dopo Kruskal-Wallis (p-value < 0.05):")
    print(significant_vars_df)

    # Test di Mann-Whitney per le variabili significative
    mannwhitney_results = []

    for var in significant_vars_df['Variable']:
        data = df[[var, target_col]]
        for (group1, group2) in combinations(data[target_col].unique(), 2):
            group1_data = data[data[target_col] == group1][var].values
            group2_data = data[data[target_col] == group2][var].values
            mw_stat, mw_p = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            mannwhitney_results.append((var, group1, group2, mw_stat, mw_p))

    mannwhitney_results_df = pd.DataFrame(
        mannwhitney_results,
        columns=["Variable", "Group1", "Group2", "MW_Statistic", "MW_P"]
    )

    # Correzione di Bonferroni
    mannwhitney_results_df['Bonferroni_P'] = multipletests(mannwhitney_results_df['MW_P'], method='bonferroni')[1]
    significant_mw_results_df = mannwhitney_results_df[mannwhitney_results_df['Bonferroni_P'] < 0.05]

    if len(significant_mw_results_df) > 0:
        print("\nRisultati significativi di Mann-Whitney (p-value Bonferroni corretto < 0.05):")
        print(significant_mw_results_df[['Variable', 'Group1', 'Group2', 'MW_Statistic', 'MW_P', 'Bonferroni_P']])
    else:
        print("\nNessun risultato significativo di Mann-Whitney dopo la correzione di Bonferroni.")

    # Selezioniamo le variabili significative da Kruskal-Wallis e Mann-Whitney
    significant_mw_vars = significant_mw_results_df['Variable'].unique()
    final_significant_vars = significant_vars_df[significant_vars_df['Variable'].isin(significant_mw_vars)]

    print("\nVariabili finali significative da mantenere:")
    print(final_significant_vars.iloc[:]['Variable'])

# Ciclo per eseguire l'analisi per entrambi i metodi di clustering
for target in ['PCAMIX_R_Agglomerative_Cluster', 'PCAMIX_R_KMedoids_Cluster']:
    perform_statistical_analysis(dfPCAMIX_R, target)

# FISHER IN R
'''
> df <- read.csv("/Users/pietropante/Desktop/Files4python/PCAMIX_R_perFisher.csv", stringsAsFactors = FALSE)
> 
> # Modifica il data frame per adattarsi ai vari metodi di clustering
> df$gender <- as.factor(df$gender)
> df$education <- as.factor(df$education)
> df$marital <- as.factor(df$marital)
> 
> # Funzione per eseguire il test di Fisher per tutte le variabili categoriali
> run_pairwise_fisher <- function(df_categorical) {
+   cluster_var <- "cluster"
+   cluster_levels <- levels(df_categorical[[cluster_var]])
+   variables <- names(df_categorical)[-which(names(df_categorical) == cluster_var)]
+   pairwise_results <- list()
+   
+   # Esegui il test di Fisher per ciascuna variabile categoriale
+   for (var in variables) {
+     full_table <- table(df_categorical[[var]], df_categorical[[cluster_var]])
+     results_var <- combn(cluster_levels, 2, function(pair) {
+       subset_table <- full_table[, pair]
+       fisher_test <- fisher.test(subset_table, simulate.p.value = TRUE)
+       data.frame(
+         Variable = var,
+         Cluster_1 = pair[1],
+         Cluster_2 = pair[2],
+         Fisher_P = fisher_test$p.value
+       )
+     }, simplify = FALSE)
+     pairwise_results[[var]] <- do.call(rbind, results_var)
+   }
+   
+   # Combina i risultati in un unico data frame
+   all_results <- do.call(rbind, pairwise_results)
+   
+   # Applica la correzione di Bonferroni
+   m <- nrow(all_results)  # numero totale di test (combinazioni di cluster)
+   alpha <- 0.05  # livello di significatività globale
+   all_results$Bonferroni_P <- p.adjust(all_results$Fisher_P, method = "bonferroni")
+   
+   # Determina se i risultati sono significativi o meno
+   all_results$Significance <- ifelse(all_results$Bonferroni_P < alpha, "Significativa", "Non significativa")
+   
+   return(all_results)
+ }
> 
> # Esegui il test di Fisher per il primo clustering (Agglomerative)
> df$cluster <- as.factor(df$PCAMIX_R_Agglomerative_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> pairwise_results_agg <- run_pairwise_fisher(df_categorical)
> 
> # Visualizza i risultati per Agglomerative Clustering
> cat("Agglomerative Clustering - Test di Fisher:\n")
Agglomerative Clustering - Test di Fisher:
> cat("   Variable Cluster_1 Cluster_2     Fisher_P\n")
   Variable Cluster_1 Cluster_2     Fisher_P
> apply(pairwise_results_agg, 1, function(row) {
+   cat(sprintf("%10s %9s %9s %13.10f\n", row['Variable'], row['Cluster_1'], row['Cluster_2'], as.numeric(row['Fisher_P'])))
+ })
    gender         0         1  0.7881059470
    gender         0         2  0.6491754123
    gender         1         2  0.1789105447
 education         0         1  0.0004997501
 education         0         2  0.0004997501
 education         1         2  0.0004997501
   marital         0         1  0.0004997501
   marital         0         2  0.0004997501
   marital         1         2  0.0004997501
NULL
> cat("  Bonferroni_P      Significance\n")
  Bonferroni_P      Significance
> apply(pairwise_results_agg, 1, function(row) {
+   cat(sprintf("%13.9f %s\n", as.numeric(row['Bonferroni_P']), row['Significance']))
+ })
  1.000000000 Non significativa
  1.000000000 Non significativa
  1.000000000 Non significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
NULL
> 
> # Esegui il test di Fisher per il secondo clustering (KMedoids)
> df$cluster <- as.factor(df$PCAMIX_R_KMedoids_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> pairwise_results_kmedoids <- run_pairwise_fisher(df_categorical)
> 
> # Visualizza i risultati per KMedoids Clustering
> cat("\nKMedoids Clustering - Test di Fisher:\n")

KMedoids Clustering - Test di Fisher:
> cat("   Variable Cluster_1 Cluster_2     Fisher_P\n")
   Variable Cluster_1 Cluster_2     Fisher_P
> apply(pairwise_results_kmedoids, 1, function(row) {
+   cat(sprintf("%10s %9s %9s %13.10f\n", row['Variable'], row['Cluster_1'], row['Cluster_2'], as.numeric(row['Fisher_P'])))
+ })
    gender         0         1  0.0009995002
    gender         0         2  0.6726636682
    gender         1         2  0.0004997501
 education         0         1  0.0004997501
 education         0         2  0.0004997501
 education         1         2  0.0004997501
   marital         0         1  0.0004997501
   marital         0         2  0.0004997501
   marital         1         2  0.0004997501
NULL
> cat("  Bonferroni_P      Significance\n")
  Bonferroni_P      Significance
> apply(pairwise_results_kmedoids, 1, function(row) {
+   cat(sprintf("%13.9f %s\n", as.numeric(row['Bonferroni_P']), row['Significance']))
+ })
  0.008995502 Significativa
  1.000000000 Non significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
  0.004497751 Significativa
NULL
> 
> # Funzione per eseguire il test di Fisher globale
> run_fisher_global <- function(df_categorical) {
+   cluster_var <- "cluster"
+   variables <- names(df_categorical)[-which(names(df_categorical) == cluster_var)]
+   results <- list()
+   
+   # Esegui il test di Fisher per ciascuna variabile categoriale
+   for (var in variables) {
+     # Crea la tabella di contingenza completa per la variabile rispetto ai cluster
+     contingency_table <- table(df_categorical[[var]], df_categorical[[cluster_var]])
+     
+     # Esegui il test di Fisher
+     fisher_test <- fisher.test(contingency_table, simulate.p.value = TRUE)
+     
+     # Salva i risultati
+     results[[var]] <- data.frame(
+       Variable = var,
+       P_Value = fisher_test$p.value
+     )
+   }
+   
+   # Combina i risultati in un unico data frame
+   all_results <- do.call(rbind, results)
+   
+   # Determina se i risultati sono significativi o meno
+   all_results$Significance <- ifelse(all_results$P_Value < 0.05, "Significativa", "Non significativa")
+   
+   return(all_results)
+ }
> 
> # Esegui il test di Fisher globale per Agglomerative Clustering
> df$cluster <- as.factor(df$PCAMIX_R_Agglomerative_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> global_results_agg <- run_fisher_global(df_categorical)
> 
> # Visualizza i risultati per Agglomerative Clustering
> cat("Agglomerative Clustering - Test di Fisher (Globale):\n")
Agglomerative Clustering - Test di Fisher (Globale):
> print(global_results_agg)
           Variable      P_Value      Significance
gender       gender 0.5642178911 Non significativa
education education 0.0004997501     Significativa
marital     marital 0.0004997501     Significativa
> 
> # Esegui il test di Fisher globale per KMedoids Clustering
> df$cluster <- as.factor(df$PCAMIX_R_KMedoids_Cluster)
> df_categorical <- df[, c("gender", "education", "marital", "cluster")]
> global_results_kmedoids <- run_fisher_global(df_categorical)
> 
> # Visualizza i risultati per KMedoids Clustering
> cat("\nKMedoids Clustering - Test di Fisher (Globale):\n")

KMedoids Clustering - Test di Fisher (Globale):
> print(global_results_kmedoids)
           Variable      P_Value  Significance
gender       gender 0.0009995002 Significativa
education education 0.0004997501 Significativa
marital     marital 0.0004997501 Significativa
'''
# Meglio KMedoids
dfPCAMIX_R = dfPCAMIX_R.drop(columns=['PCAMIX_R_Agglomerative_Cluster'])
final_table = pd.DataFrame()

# Calcoliamo il numero di persone per cluster
cluster_counts = dfPCAMIX_R['PCAMIX_R_KMedoids_Cluster'].value_counts()

# Aggiungiamo i conteggi dei membri dei cluster nell'header
final_table.loc['Cluster Size', cluster_counts.index] = cluster_counts.values

for var in dfPCAMIX_R.columns:
    if var != 'PCAMIX_R_KMedoids_Cluster':
        if dfPCAMIX_R[var].dtype in ['float64', 'int64']:
            stats_per_cluster = dfPCAMIX_R.groupby('PCAMIX_R_KMedoids_Cluster')[var].agg(['median', 'min', 'max'])
            stats_per_cluster = stats_per_cluster.round()

            final_table.loc[var, stats_per_cluster.index] = [
                f"{int(median)} ({int(min_val)} - {int(max_val)})"
                for median, min_val, max_val in
                zip(stats_per_cluster['median'], stats_per_cluster['min'], stats_per_cluster['max'])
            ]
        else:
            mode_per_cluster = dfPCAMIX_R.groupby('PCAMIX_R_KMedoids_Cluster')[var].agg(lambda x: x.mode()[0])
            mode_freq_per_cluster = dfPCAMIX_R.groupby('PCAMIX_R_KMedoids_Cluster')[var].agg(lambda x: (x == x.mode()[0]).sum())

            total_counts = dfPCAMIX_R.groupby('PCAMIX_R_KMedoids_Cluster')[var].count()
            freq_per_cluster = mode_freq_per_cluster / total_counts * 100
            final_table.loc[var, mode_per_cluster.index] = [
                f"{mode} ({freq:.2f}%)" for mode, freq in zip(mode_per_cluster, freq_per_cluster)
            ]

final_table.columns = [f'Cluster {cluster}' for cluster in final_table.columns]
print(
    "\nTabella con media (min-max) per variabili numeriche e moda con frequenza per variabili categoriche per ogni cluster:")
print(final_table)

'''with ExcelWriter("Persona_Table.xlsx", mode="a", engine="openpyxl") as writer:
    final_table.to_excel(writer, sheet_name="PCAMIX_R Cluster", index=True)
'''
k_range = range(2, 9)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
pastel_colors = {
    'silhPCA': '#FFB3B3',
    'silhFAMD': '#B3E0FF',
    'silhFAMD_R': '#D1F2A5',
    'silhPCAMIX_R': '#FFB3E6',
    'distPCA': '#FFB3B3',
    'distFAMD': '#B3E0FF',
    'distFAMD_R': '#D1F2A5',
    'distPCAMIX_R': '#FFB3E6'
}
bright_colors = {
    'silhPCA': '#FF0000',
    'silhFAMD': '#0000FF',
    'silhFAMD_R': '#228B22',
    'silhPCAMIX_R': '#FF00FF',
    'distPCA': '#FF0000',
    'distFAMD': '#0000FF',
    'distFAMD_R': '#228B22',
    'distPCAMIX_R': '#FF00FF'
}

ax1.plot(k_range, silhPCA, label='silhPCA', color=pastel_colors['silhPCA'], marker='o')
ax1.plot(k_range, silhFAMD, label='silhFAMD', color=pastel_colors['silhFAMD'], marker='o')
ax1.plot(k_range, silhFAMD_R, label='silhFAMD_R', color=pastel_colors['silhFAMD_R'], marker='o')
ax1.plot(k_range, silhPCAMIX_R, label='silhPCAMIX_R', color=pastel_colors['silhPCAMIX_R'], marker='o')
ax1.scatter(3, silhPCA[1], color=bright_colors['silhPCA'], s=100, zorder=5)
ax1.scatter(3, silhFAMD[1], color=bright_colors['silhFAMD'], s=100, zorder=5)
ax1.scatter(3, silhFAMD_R[1], color=bright_colors['silhFAMD_R'], s=100, zorder=5)
ax1.scatter(3, silhPCAMIX_R[1], color=bright_colors['silhPCAMIX_R'], s=100, zorder=5)
ax1.axvline(x=3, color='black', linestyle='--', label='k=3')
ax1.set_title('Silhouette Scores')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Silhouette Score')
ax1.legend()
ax2.plot(k_range, distPCA, label='distPCA', color=pastel_colors['distPCA'], marker='o')
ax2.plot(k_range, distFAMD, label='distFAMD', color=pastel_colors['distFAMD'], marker='o')
ax2.plot(k_range, distFAMD_R, label='distFAMD_R', color=pastel_colors['distFAMD_R'], marker='o')
ax2.plot(k_range, distPCAMIX_R, label='distPCAMIX_R', color=pastel_colors['distPCAMIX_R'], marker='o')
ax2.scatter(3, distPCA[1], color=bright_colors['distPCA'], s=100, zorder=5)
ax2.scatter(3, distFAMD[1], color=bright_colors['distFAMD'], s=100, zorder=5)
ax2.scatter(3, distFAMD_R[1], color=bright_colors['distFAMD_R'], s=100, zorder=5)
ax2.scatter(3, distPCAMIX_R[1], color=bright_colors['distPCAMIX_R'], s=100, zorder=5)
ax2.axvline(x=3, color='black', linestyle='--', label='k=3')
ax2.set_title('Distortion Scores')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Distortion')
ax2.legend()
fig.suptitle('Comparison of Silhouette and Distortion Scores for Different Cluster Counts', fontsize=16)
plt.tight_layout()
plt.show()

#:)
