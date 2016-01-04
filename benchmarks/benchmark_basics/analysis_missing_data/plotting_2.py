
import matplotlib.pyplot as plt
import seaborn as sns

p = []
n = []

for s in a :
    if s[1] == 0 :
        n.append(s[2])
    if s[1] == 1 : 
        p.append(s[2])       

sns.distplot(p, bins=20, kde=False, rug=False , label = "Positive reviews");
sns.distplot(n,bins = 20, kde = False, rug = False, label = "Negative reviews");
plt.legend()
plt.title("Certainty of the wrong answer : by type of review\nTotal misclassified reviews : " + str(len(a)))

