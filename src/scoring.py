from sklearn.cross_validation import KFold

CV_FOLDS = 5
folds = KFold(len(all_ratings), n_folds=CV_FOLDS, shuffle=True)

# TODO(andrei) Implement predictions as a subclass of BaseEstimator
# and/or RegressionEstimator so we can use it with e.g. grid searches.

k = 3

# A 2D array where every row is a data point. The first column specifies
# the row in the data matrix where the rating belongs, the second column,
# its column. The third one is the actual rating.
all_ratings_np = np.array(all_ratings)
all_ratings_np.shape

# We will not use 'svd_predict' and instead we will use a dirty trick
# to significantly speed up our cross-validation scores.

# For every fold, results will contain a list of len(ks) scores.
# This is an unorthodox way of doing CV, but it's much faster since
# we only do one SVD per fold, instead of one per (k * folds).
results = []

ks = range(1, 30, 1)
# imputation_fn = predict_by_avg_item_rating
imputation_fn = predict_by_avg_avg
for train_index, test_index in folds:
    train = all_ratings_np[train_index]
    test = all_ratings_np[test_index]

    train_matrix = ratings_to_matrix(train, USER_COUNT, ITEM_COUNT)
    test_matrix = ratings_to_matrix(test, USER_COUNT, ITEM_COUNT)

    # print("Performing imputation...")
    imputed = imputation_fn(train_matrix)
    if len(imputed[np.isnan(imputed)]):
        raise ValueError("Found NaNs in imputed data. Aborting.")

    # print("Performing SVD...")
    U, d, V = np.linalg.svd(imputed, full_matrices=True)
    # print("SVD done.")
    V = V.T

    D = np.zeros_like(imputed)
    D[:d.shape[0], :d.shape[0]] = np.diag(d)

    rmses = []
    for k in ks:
        U_k = U[:, :k]
        D_k = D[:k, :]
        V_k = V[:, :]

        k_prediction = np.dot(U_k, np.dot(D_k, V_k.T))
        k_prediction = k_prediction[:, :1000]
        rmses.append((k, score_predictions(k_prediction, test_matrix)))

    results.append(rmses)

print("Done.")

cv_rmses = []
for index, k in enumerate(ks):
    score = 0
    for fold in range(CV_FOLDS):
        score += results[fold][index][1]

    avg_score = score / CV_FOLDS
    cv_rmses.append(avg_score)


plt.plot(ks, cv_rmses)
plt.ylabel("RMSE with %d CV folds" % CV_FOLDS)
plt.xlabel("Number of kept singular values $\sigma$")
plt.title("%d-fold CV result" % CV_FOLDS)


plt.plot(ks[6:15], cv_rmses[6:15], label="CV RMSE")
plt.plot([e[0] for e in rmses][6:15], [e[1] for e in rmses][6:15], label="Simple validation RMSE")
plt.legend()