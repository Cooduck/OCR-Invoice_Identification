import pickle as pkl

alphabet_list = pkl.load(open('./crnn/alphabet.pkl','rb'))
alphabet = [ord(ch) for ch in alphabet_list]
alphabet_v2 = alphabet
print(alphabet_list)
print(alphabet_v2)