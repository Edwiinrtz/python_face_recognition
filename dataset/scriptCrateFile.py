import pickle

data = []
f = open("./dataset/known.pkl","wb")
f.write(pickle.dumps(data))
f.close()

