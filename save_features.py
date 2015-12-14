import os

def save_textual_data(data,complete_path):
    saved_path = os.getcwd()
    os.chdir(complete_path)
    i = 0
    for s in data :
        with open("data"+str(i)+".txt","w") as f :
            f.write(s)
        i+=1
    os.chdir(saved_path)
    return 0
    
def load_textual_data(complete_path):
    # Pour charger, faire un sort sur la longueur et puis sur la string
    data = list()
    a = list(os.walk(complete_path))
    list_files = a[0][2]
    list_files = sorted(list_files, key = lambda item : (len(item),item))
    for filenames in list_files :
        with open(complete_path + filenames,"r") as f :
            data.append(f.read())
    return data
