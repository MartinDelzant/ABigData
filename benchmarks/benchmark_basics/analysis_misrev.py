f = open("data_misclassified_review.txt","r")

a = []
l1 = " "
l2 = 0
l3 = 0.1

while True :
    l1 = f.readline()
    if len(l1) ==  0 :
        break 
    l2 = int(f.readline())
    l3 = float(f.readline())
    a.append((l1,l2,l3))
f.close()
