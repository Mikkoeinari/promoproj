from sys  import argv, stderr, maxsize
from math import ceil
from time import time
from collections import Counter
import numpy as np
import itertools


def probabilities(rows):
    singleprobs=[]
    dualprobs=[]
    triprobs=[]
    for i in np.array(rows).transpose(): #go through all columns of the input data
        c=Counter(i)                    #Count occurences of different values
        rowcount=sum(c.values())        #total number of rows
        singleprobs.append([(c['1']+1)/float(rowcount+3), (c['2']+1)/float(rowcount+3), (c['3']+1)/float(rowcount+3)]) #store pobabilitydistribution to a list
        for j in np.array(rows).transpose()[:]:
            if (j==i).all():
                j=i;
            else:
                k=([m+str(n) for m,n in zip(i,j)])
                d=Counter(m)
                dualprobs.append([(d['11']+1)/float(rowcount+9), (d['12']+1)/float(rowcount+9), (d['13']+1)/float(rowcount+9),
                    (d['21']+1)/float(rowcount+9), (d['22']+1)/float(rowcount+9), (d['23']+1)/float(rowcount+9),
                    (d['31']+1)/float(rowcount+9), (d['32']+1)/float(rowcount+9), (d['33']+1)/float(rowcount+9)])
                for l in np.array(rows).transpose()[:]:
                    if (l==j).all() or (l==i).all():
                        l=l
                    else:
                        l=([m+str(n) for m,n in zip(k,l)])
                        e=Counter(l)
                        triprobs.append([(e['111']+1)/float(rowcount+27), (e['112']+1)/float(rowcount+27), (e['113']+1)/float(rowcount+27),
                                           (e['121']+1)/float(rowcount+27), (e['122']+1)/float(rowcount+27), (e['123']+1)/float(rowcount+27),
                                           (e['131']+1)/float(rowcount+27), (e['132']+1)/float(rowcount+27), (e['133']+1)/float(rowcount+27),
                                           (e['211']+1)/float(rowcount+27), (e['212']+1)/float(rowcount+27), (e['213']+1)/float(rowcount+27),
                                           (e['221']+1)/float(rowcount+27), (e['222']+1)/float(rowcount+27), (e['223']+1)/float(rowcount+27),
                                           (e['231']+1)/float(rowcount+27), (e['232']+1)/float(rowcount+27), (e['233']+1)/float(rowcount+27),
                                           (e['311']+1)/float(rowcount+27), (e['312']+1)/float(rowcount+27), (e['313']+1)/float(rowcount+27),
                                           (e['321']+1)/float(rowcount+27), (e['322']+1)/float(rowcount+27), (e['323']+1)/float(rowcount+27),
                                           (e['331']+1)/float(rowcount+27), (e['332']+1)/float(rowcount+27), (e['333']+1)/float(rowcount+27)])
    #Output to file for validation of output, testing only
    writeOutput('duals.dat', dualprobs)
    writeOutput('singles.dat',singleprobs)
    writeOutput('tris.dat', triprobs)
    return singleprobs, dualprobs, triprobs
def checkIndependence(a,b, singleprobs, dualprobs, triprobs):
##    print(singleprobs[a])
##    print(singleprobs[b])
##    print(dualprobs[(a-1*26+b)])
    print((a)*26+b)
    singlesproduct=[]
    for element in itertools.product(*[singleprobs[a],singleprobs[b]]):
        singlesproduct.append(element[0]*element[1])
    #print singlesproduct
    #print dualprobs[(a)*26+b]
    if (singlesproduct==dualprobs[(a)*26+b]):
        #print(singlesproduct)
        #print(dualprobs[(a)*26+b])
        return 1
    else:
        return 0
def writeOutput(filename, outputTable):
    f = open(filename, 'w')
    for row in outputTable:
        f.write("%s\n" % row)
    f.close()
    
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
    t = time()
    stderr.write('counting probabilities')
    singles, duals, tris=probabilities(rows)
    stderr.write('done [%.2fs].\n' % (time()-t)) 
    for i in range(24):
        for j in range(25):
            for k in range(26):
            checkIndependence(i,j, singles, duals, tris)
        

