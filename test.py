import pickle

encoder = open('output/encoder.pkl', 'rb')      
le = pickle.load(encoder) 
print(len(le.classes_.tolist()))
encoder.close() 