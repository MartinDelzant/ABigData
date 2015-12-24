from sklearn.linear_model import LogisticRegressionCV


model = LogisticRegressionCV(Cs = np.logspace(-1,1,6),penalty ="l2",solver = "lbfgs")
scores_accuracy = cross_val_score(model, data_GoW, y, cv=cv, n_jobs=3)
print("Accuracy :\n", round(np.mean(scores_accuracy), 4), "+/-", round(2*np.std(scores_accuracy),4))

