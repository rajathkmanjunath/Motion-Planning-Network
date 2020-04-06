import pickle

string = 'Environment '
dbfile = open('datafile.pickle', 'ab+')
environment = 1

dic = [1,2,3,4,5]


# for i in range(100):
#     pickle.dump(i, dbfile)
pickle.dump(dic, dbfile)
dbfile.close()