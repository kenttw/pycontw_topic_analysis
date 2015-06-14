from sets import Set
def load_stop_words(fpath):
    r = Set()
    with open(fpath,'rb') as file :
        for line in file :
            s = line.strip().decode('utf-8')
            r.add(s)
    return r