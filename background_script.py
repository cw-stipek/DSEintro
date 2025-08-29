### Name: Clinton Stipek
### Fun Fact: Clinton likes to travel alot and used to live in a van picking fruit in New Zealand. 


def main():
    print("Would you like to know about Clinton's undergraduate, graduate school or his professional career?")
    print("Options: undergraduate, graduate, professional, personal")
    clinton_phase = input("Enter your choice: ").strip().lower()

    if clinton_phase == "undergraduate":
        print("\nClinton attended the University of Washington in Seattle where he was a student-athlete in swimming. He absolutely loves Seattle and it will always be home for him. His undergraduate degree was in Oceanography with a focus in geophysics and he thoroughly enjoyed his time there. Some of his favorite memories are biking around the San Juan Islands and camping.")
    elif clinton_phase == "graduate":
        print("\nAfter working for a few years after his undergraduate Clinton moved to Miami to attend the University of Miami for his MSc. He studied how seagrass communities react to freshwater pulses using machine learning and spatial analysis. He had a blast in Miami and really enjoyed the professional development. One of his favorite memories was going scuba diving in the Florida Keys.")
    elif clinton_phase == "professional":
        print("\nIn Clinton's professional career he is a research scientist at Oak Ridge National Laboratory where he works at the intersection of big data, artificial intelligence, and the built environment. His work helps to better inform population modeling, extreme events, the electrical grid infrastructure, among others. Some of his favorite memories thus far have been working in a highly collaborative setting and solving multi-faceted problems with his teammates.")
    elif clinton_phase =="personal":
        print("\nClinton loves to swim still in his personal time and also enjoys hiking, traveling, and spending time with his wonderful partner who keeps him sane but he also drives her crazy sometimes.")
    else:
        print("\nPlease enter the following options: 'undergraduate', 'graduate', 'professional'.")

if __name__ == "__main__":
    main()


###Additional workflows - ML
df = pd.read_sql_query("""
                            SELECT a.height, b.*
                            FROM xp_clustering.chicago a
                            LEFT JOIN phase2_output_gauntlet_v2.illinois b
                            USING (build_id)
                            WHERE a.height IS NOT NULL""",
                          con = engine)
print('columns for training data:',df.columns)
print('data:',df.head())
df.set_index('build_id', inplace=True)
out_index = df.index

num_df = df.select_dtypes(include=[np.number]).copy()
num_df.replace([np.inf, -np.inf], np.nan, inplace=True)
print("NaNs per column (top 20):")
print(num_df.isna().sum().sort_values(ascending=False).head(20))

print("\nZero-variance columns:")
zero_var_cols = [c for c in num_df.columns if num_df[c].nunique(dropna=True) <= 1]
print(zero_var_cols[:20], "... total:", len(zero_var_cols))
num_df = num_df.dropna()

df = num_df.copy()
print(df.info())
print(df.head())
df.drop(columns=['sqft', 'lon', 'lat'], inplace=True)
df.info()

scaled_data = StandardScaler().fit_transform(df)

# ============================================
# 1. KMeans++
# ============================================
range_k = range(2, 15)
kmeans_scores = []

for k in range_k:
    km = KMeans(n_clusters=k, init="k-means++", random_state=13)
    labels = km.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    kmeans_scores.append((k, score))

# Best KMeans
best_k, best_kmeans_sil = max(kmeans_scores, key=lambda x: x[1])
best_kmeans = KMeans(n_clusters=best_k, init="k-means++", random_state=13).fit(scaled_data)
kmeans_labels = best_kmeans.labels_

# ============================================
# 2. Gaussian Mixture Model (GMM)
# ============================================
gmm_scores = []
best_gmm, best_gmm_sil, best_gmm_k, best_gmm_bic = None, -1, None, np.inf

for k in range(2, 15):
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=13, n_init=5)
    gmm.fit(scaled_data)
    labels = gmm.predict(scaled_data)
    if len(set(labels)) > 1:  # need at least 2 clusters
        sil = silhouette_score(scaled_data, labels)
        bic = gmm.bic(scaled_data)
        gmm_scores.append((k, sil, bic))
        # prefer higher silhouette, use BIC to break ties
        if sil > best_gmm_sil or (sil == best_gmm_sil and bic < best_gmm_bic):
            best_gmm, best_gmm_sil, best_gmm_k, best_gmm_bic = gmm, sil, k, bic

gmm_labels = best_gmm.predict(scaled_data)
gmm_probs = best_gmm.predict_proba(scaled_data)   # shape (n_samples, n_components)
gmm_max_p = gmm_probs.max(axis=1)  

# ============================================
# 3. HDBSCAN
# ============================================
hdb = hdbscan.HDBSCAN(min_cluster_size=20, prediction_data=True).fit(scaled_data)
hdb_labels = hdb.labels_

# Exclude noise (-1) for silhouette
mask_nz = hdb_labels != -1
if mask_nz.sum() > 1 and len(set(hdb_labels[mask_nz])) > 1:
    hdb_sil = silhouette_score(scaled_data[mask_nz], hdb_labels[mask_nz])
else:
    hdb_sil = np.nan

hdb_outlier = hdb.outlier_scores_   # shape (n_samples,)

results = pd.DataFrame([
    {"Algorithm": "KMeans++", "Optimal_k": best_k, "Silhouette": best_kmeans_sil},
    {"Algorithm": "GaussianMixture", "Optimal_k": best_gmm_k, "Silhouette": best_gmm_sil, "BIC": best_gmm_bic},
    {"Algorithm": "HDBSCAN", "Optimal_k": len(set(hdb_labels) - {-1}), "Silhouette": hdb_sil}
])
print(results.head())
results.to_csv('/clust/variance/chicago/sil_chicago.csv', index=False)

results_2 = pd.DataFrame({
    'build_id': out_index,                 # ensure this lines up with reduced_data rows
    'cluster_kmeans': kmeans_labels,
    'cluster_gmm': gmm_labels,
    'gmm_max_prob': gmm_max_p,             # now defined
    'cluster_hdbscan': hdb_labels,
    'hdb_outlier_score': hdb_outlier
})
