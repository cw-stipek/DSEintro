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


###Additional workflows - Unsupervised ML
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


###Workflow - Supervised ML

print(df.info())
df.rename(columns={'median':'height'}, inplace=True)


#Removing all buildings less than 2 m due to potential unwanted structures (sheds, bike stops, bus stops, etc...)
df = df[df['height']>2]
df = df[df['height']<200]

print(df.height.describe())

# Calculating median
df['median']=df.height.median()

height = df['height'].to_numpy()
median = df['median'].to_numpy()

# 4. calculating the mean absolute error
print('MAE:',mean_absolute_error(height, median))
# 5. calculating the root mean squared error
print('RMSE:',np.sqrt(mean_squared_error(height, median)))
df.drop(columns=['median'], inplace=True)


y = df.loc[:, df.columns == 'height']
X = df.loc[:, df.columns != 'height']

# split data into train and test sets
X_train, X_test_beta, y_train, y_test_beta = train_test_split(X, y, test_size=0.3, random_state=13)

#Standard scaler
sc = StandardScaler()
sc.fit(X_train)
#Fitting to the training data
X_train_scaled = sc.transform(X_train)

#converting from array to dataframe with original columns
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

#saving scaler
dump(sc, 'my_standard_scaler.pkl')

#loading scaler
same_scaler = load('my_standard_scaler.pkl')

#applying scaler to testing data
X_test_scaled_beta = same_scaler.transform(X_test_beta)

#converting from array to dataframe with original columns
X_test_scaled_beta = pd.DataFrame(X_test_scaled_beta, columns=X_test_beta.columns)

#Alpha training data
X = X_train_scaled
y = y_train
print(len(X))
print(len(y))

# Assuming you have defined your dataset as X and y
print('Starting the bayesian optimization')

start_time = datetime.now()

# Assuming you have defined your dataset as X and y

space = {
    'max_depth': hp.quniform("max_depth", 1, 15, 1),
    'gamma': hp.uniform('gamma', 1, 10),
    'reg_alpha': hp.quniform('reg_alpha', 20, 100, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 20, 1),
    'n_estimators': hp.quniform('n_estimators', 10, 600, 10),
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.5)
}

def hyper_tuning(space):
    model = XGBRegressor(n_estimators=int(space['n_estimators']),
                         max_depth=int(space['max_depth']),
                         gamma=space['gamma'],
                         reg_alpha=space['reg_alpha'],
                         reg_lambda=space['reg_lambda'],
                         colsample_bytree=space['colsample_bytree'],
                         min_child_weight=space['min_child_weight'],
                         learning_rate=space['learning_rate'],
                         random_state=13)
    
    # Perform k-fold cross-validation (e.g., k=5)
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        cv_scores.append(mse)
    
    # Calculate the mean RMSE (neg_mean_squared_error gives negative values)
    mse_mean = np.mean(cv_scores)
    print("Mean Cross-Validation RMSE:", np.sqrt(mse_mean))
    
    return {'loss': mse_mean, 'status': STATUS_OK, 'model': model}

trials = Trials()
best = fmin(fn=hyper_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=100, # Increase max_evals for more iterations
            trials=trials)

end_time = datetime.now()

print('Duration of hyper_tuning: {}'.format(end_time-start_time))

print('\n')

print(best)

best_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'gamma': best['gamma'],
    'reg_alpha': best['reg_alpha'],
    'reg_lambda': best['reg_lambda'],
    'colsample_bytree': best['colsample_bytree'],
    'min_child_weight': int(best['min_child_weight']),
    'learning_rate': best['learning_rate'],
}

# Convert best_params to a DataFrame
best_params_df = pd.DataFrame([best_params])

# Save to CSV
best_params_df.to_csv("/data/hyper_params/microsoft/best_hyperparameters_davao_philippines_ms.csv", index=False)

X_train = X
y_train = y
X_test = X_test_scaled_beta
y_test = y_test_beta

def ml(X_train, y_train, X_test, y_test):
    '''ml is the fifth function after clean_df, engineering_df, median_calc, and train_test_val. It performs:
    1. Linear regression on the train, test, and val data
    2. XGBoost on the train, test, and val data
    3. Transfers predictions back into the dataframe to investigate the error difference'''
    print('Starting ml')
    start_time=datetime.now()
    # 1. Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    
    # Linear Regression evaluation for training set
    y_train_predict = lin_model.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
    mae = mean_absolute_error(y_train, y_train_predict)
    r2 = r2_score(y_train, y_train_predict)

    print("The linear regression performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('MAE: {}'.format(mae))
    print('R2 score is {}'.format(r2))
    print("\n")
    
    # Linear Regression evaluation for testing set
    y_test_predict = lin_model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    mae = mean_absolute_error(y_test, y_test_predict)
    r2 = r2_score(y_test, y_test_predict)

    print("The linear regression performance for testing set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('MAE: {}'.format(mae))
    print('R2 score is {}'.format(r2))
    print("\n")
    
    # 2. XGBoost
    reg = XGBRegressor(random_state=13)#**best_params)
    # XGBoost evaluation for training set

    reg.fit(X_train, y_train)
    train_pred = reg.predict(X_train)
    print("The XGBoost performance for training set")
    print("--------------------------------------")
    print('RMSE:',np.sqrt(mean_squared_error(y_train, train_pred)))
    print('MAE:',mean_absolute_error(y_train, train_pred))
    print('R^2:',r2_score(y_train, train_pred))
    print('\n')
    
    # XGBoost evaluation for testing set
    target_test_pred = reg.predict(X_test)
    print("The XGBoost performance for testing set")
    print("--------------------------------------")
    print('RMSE:',np.sqrt(mean_squared_error(y_test, target_test_pred)))
    print('MAE:',mean_absolute_error(y_test, target_test_pred))
    print('R^2:',r2_score(y_test, target_test_pred))
    print('\n')


    # # 3. Error metrics to list
    # #Adding in predicted values for validation data
    y_test['predicted']=target_test_pred.tolist()
    
    # #Caculating the difference between height and predicted
    y_test['difference']=y_test['height']-y_test['predicted']
    
    # #Calculating the absolute error of the difference column
    y_test['abs_error']=y_test['difference'].abs()
    y_test.to_csv('/data/results/microsoft/all_microsoft.csv')
    print('successfully output data')


    end_time=datetime.now()
    print('Duration of ml: {}'.format(end_time-start_time))
    # return y_test
ml(X_train, y_train, X_test, y_test)

def cross_val(X_train, y_train):
    '''cross_val is the sixth function and its purpose is to cross validate the scores. It peforms:
    1. splitting up the parameters into a dictionary
    2. generates a XGBoost regressor model with the parameters from step 1
    3. fits the model on the training data
    4. runs 5 splits in a KFold function
    5. stores the scores after training'''
    print('Staring cross_val now')
    start_time=datetime.now()
    
    # 2. Instantiating a model with best_params from above
    xgb_cv=XGBRegressor(random_state=13)
    
    # 3. Fitting XGBoost on the training data
    xgb_cv.fit(X_train, y_train)

    # 4. determining number of splits on the data
    kfold = KFold(n_splits=10,
                  shuffle=True,
                  random_state=13)
    
    # 5. Running cross validation on the training data
    scores = cross_val_score(xgb_cv,
                             X_train,
                             y_train,
                             cv=kfold,
                             scoring='neg_mean_squared_error')
    
    end_time=datetime.now()

    mse = -np.mean(scores)
    print('Mean cross validation score: ', np.sqrt(mse))
    print('Duration of cross validation: {}'.format(end_time-start_time))
    return scores

cross_val(X_train, y_train)
