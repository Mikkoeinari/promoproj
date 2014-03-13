from sys  import argv, stderr, maxsize
from math import ceil
from time import time
from collections import Counter
import numpy as np

def probabilities(rows):
    singleprobs=[]
    for i in np.array(rows).transpose(): #go through all columns of the input data
        c=Counter(i)                    #Count occurences of different values
        rowcount=sum(c.values())        #total number of rows
        singleprobs.append([c['1']/float(rowcount), c['2']/float(rowcount), c['3']/float(rowcount)]) #store pobabilitydistribution to a list
        dip=np.array(rows).transpose()-i
        for j in dip:
            joined=i+j
            print(joined)
    print(singleprobs)
    
if __name__ == '__main__':
    desc    = 'Attempt to find optimal bayesian network'
    version = 'version 1.0 ' \
            + '(c) 2014 Mikko Hakila'
    opts    = {'t': ['target', 's'],
               'm': ['min',     1 ],
               'n': ['max',    -1 ],
               's': ['supp',   10 ] }
    fixed   = []                # list of program options

    if len(argv) <= 1:          # if no arguments are given
        opts = dict([(o,opts[o][1]) for o in opts])
        print('usage: PromoProj.py [options] infile [outfile]')
        print(desc); print(version)
        print('-t#      target type                            '
                      +'(default: '+str(opts['t'])+')')
        print('         (s: frequent, c: closed, m: maximal item sets)')
        print('-m#      minimum number of items per item set   '
                      +'(default: '  +str(opts['m'])+')')
        print('-n#      maximum number of items per item set   '
                      +'(default: no limit)')
        print('-s#      minimum support                        '
                      +'(default: '  +str(opts['s'])+'%)')
        print('         (positive: percentage,'
                      +'negative: absolute number)')
        print('infile   file to read transactions from         '
                      +'[required]')
        print('outfile  file to write frequent item sets to    '
                      +'[optional]')
        exit()                  # print a usage message

    stderr.write('PromoProj.py')    # print a startup message
    stderr.write(' - ' +desc +'\n' +version +'\n')

    for a in argv[1:]:          # traverse the program arguments
        if a[0] != '-':         # check for a fixed argument
            if len(fixed) < 2: fixed.append(a); continue
            else: print('too many fixed arguments'); exit()
        if a[1] not in opts:    # check for an option
            print('unknown option: '+a[:2]); exit()
        o = opts[a[1]]          # get the corresponding option
        v = a[2:]               # and get the option argument
        if   isinstance(o[1],(0).__class__): o[1] = int(v)
        elif isinstance(o[1],0.0.__class__): o[1] = float(v)
        else:                                o[1] = v
    opts = dict([opts[o] for o in opts])

    t = time()
    stderr.write('reading ' +fixed[0] +' ... ')
    with open(fixed[0], 'rt') as inp:
        rows = [list(line.split()) for line in inp]
    rows.pop(0)
    n = len(rows)
    stderr.write('[%d transaction(s)] ' % n)
    stderr.write('done [%.2fs].\n' % (time()-t)) 
    probabilities(rows)
    

        

