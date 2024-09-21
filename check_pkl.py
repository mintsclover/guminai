import pickle

with open('docstore.pkl', 'rb') as file:
    data = pickle.load(file)
    print(data)