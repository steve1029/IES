import os

def rename_srtwith(path,srtwith):

    to_change = [x for x in os.listdir(path) if x.startswith(srtwith)]

    #print(to_change)

    for old in to_change:

        a = old[5:]
        new = 'lambz'+a
        os.rename(path+old, path+new)
        #print(old, new)

def rename_len(path, length):

    to_change = [x for x in os.listdir(path) if len(x)==length]

    #print(to_change)

    for old in to_change:

        srt = old[:5]
        wv = old[5:9]
        a = old[9:]
        new = srt+'0'+wv+a
        #print(srt, '0'+wv, a)
        #print(old, new)
        os.rename(path+old, path+new)

if __name__ == '__main__':

    path = './graph/'

    #rename_srtwith(path, 'lambz')
    rename_len(path,22)
