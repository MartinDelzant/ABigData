from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV

parameters = { "alpha" : np.logspace(-5,-3,5) }

# penalty l1 or l2

n = 25000
model = SGDClassifier(loss ="log", average = 10,penalty = "l2",n_jobs = 3,n_iter = np.ceil(10**6 / n))
clf = GridSearchCV(model,parameters,n_jobs = 3,cv = cv)

scores_accuracy = cross_val_score(clf, data_GoW, y, cv=cv, n_jobs=3)
print("Accuracy :\n", round(np.mean(scores_accuracy), 4), "+/-", round(2*np.std(scores_accuracy),4))

