# a being this triple ( text, real value, proba to be misled )

import matplotlib.pyplot as plt

import seaborn as sns

res = [ s[2] for s in a ]

plt.hist(res, bins = 20)
plt.title("Certainty of the wrong answer \nfor erroneous predictions : " + str(len(res)) )
plt.xlabel("Certainty of the wrong answer")
plt.ylabel("Number of concerned observations")


