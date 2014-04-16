from sys  import argv, stderr, maxsize
from math import ceil, log, fmod, exp
from time import time
from collections import Counter
import numpy as np #for array transformations
import itertools as it
import pickle, copy
import random as rd
import operator
from scipy.special import gammaln #for log gamma in BDeu
import argparse
from multiprocessing import Process, Pool

#Counts the frequency of parent&node tuples from the data
def countOfNode(setti, params, alreadyCountedDict={}):
    if params in alreadyCountedDict.keys():
        return alreadyCountedDict[params]
    else:
        rowsCo=np.array(setti).transpose()[:]
        T2=[''.join(j) for j in zip(*(rowsCo[i].tolist() for i in params))]
        d=Counter(T2)
        a=[''.join(w) for w in it.product('123', repeat=len(params))]
        retval= ([d[s] for s in a])
        alreadyCountedDict[params]=tuple(retval)
        return tuple(retval)
    
#Returns the probability of a node and its parents (params). P(a,b,c,..)
def probOfNode(setti, params):
    rowsCo=np.array(setti).transpose()[:]
    T2=[''.join(j) for j in zip(*(rowsCo[i].tolist() for i in params))]
    d=Counter(T2)
    a=[''.join(w) for w in it.product('123', repeat=len(params))]
    #Probabilities, (+1/+len(a)) for smoothing
    retval= ([(d[s]+1)/float(len(setti)+len(a)) for s in a])
    retval=[i/sum(retval) for i in retval]
    return tuple(retval)

def bdeu(counts, alpha):
    BD=0
    for i in range(len(counts)/3):
        startingpoint=i*3
        cnt=counts[startingpoint:startingpoint+3]
        BD+=(gammaln(alpha/float((len(counts)/3)))
            -gammaln((alpha/float((len(counts)/3)))+sum(cnt)))
        for j in cnt:
            BD+=(gammaln((alpha/float(len(counts)))+j)
            -gammaln(alpha/float(len(counts))))
    return -BD

#Returns the Minimum Description Lenght of  a given 
#node, takes the counts of node and its parent configurations as 
#input. AlreadyCalculatedDict stores the results for faster executiontimes.
def mdl(counts):
    MDL=0
    for i in range(len(counts)/3):
        #process in settis of three(in separate parent configurations)
        #Must rewrite if used for different datasettis
        startingpoint=i*3
        cnt=counts[startingpoint:startingpoint+3]
        #Remove zero counts to avoid div by 0
        cnt=[x for x in cnt if x != 0] 
        for j in cnt:
            MDL+=j*log(j/float(sum(cnt)))
    MDL-=(log(float(sum(counts)))/2)*(len(counts)/3)
    MDL=-MDL
    return MDL

#Write data to disk in pickle.dump. Used for storing
#good results in testing phase
def writeOutput(filename, outputTable):
    pickle.dump(outputTable, open(filename, "wb"))

def readInput(file):
    result=[]
    for a in file:
        result.append(pickle.load(open(a, "rb")))
    return result

# iterateAndImprove is the main logic in greedy hillclimbing
# it counts the move that reduces the total score the most and executes it.product
# If no move is left, the function has found a local minimum and it returns None
# The most improvement to be made in this approach is in random restarts and
# random moves if local minimum found.
def iterateAndImprove(setti, testi, parentG, countDict, arcList):
    originAndTargets={}
    bestEdge=[]
    bestNodeScore=0
    ng=getParentform(parentG)
    #find best addition
    for target in parentG.keys():
        originAndTargets[target]=set(parentG.keys())-set(parentG.get(target))-set([target])
        for origin in originAndTargets[target]:
            #Here we have all possible edges to add
            originalScore=getNodeScore(setti,(parentG.get(target)+[target]), countDict)
            current=tuple([1,target,origin])
            if parentG.get(target)==None:
                parentG[target]=origin
            else:
                parentG[target].append(origin)
            loops=validatePath(target,[],parentG, [])
            moveScore=(getNodeScore(setti,(parentG.get(target)+[target]), countDict)-originalScore)
            if not tuple([origin, target]) in arcList:
                arcList[origin, target]=moveScore
            if arcList[origin, target]>moveScore:
                arcList[origin, target]=moveScore
            if moveScore<bestNodeScore  and len(loops)==0:
                bestEdge=current
                bestNodeScore=moveScore
            parentG[target].remove(origin)
    #find best removal
    for target in parentG.keys():
        originAndTargets[target]=set(parentG.get(target))
        for origin in originAndTargets[target]:
            current=tuple([2,target,origin])
            originalScore=getNodeScore(setti, (parentG.get(target)+[target]), countDict)
            #Here we have all possible edges to remove
            parentG[target].remove(origin)
            #If last edge removed, change none to empty list
            if parentG[target] == None:
                parentG[target]=[]
            loops=validatePath(target,[],parentG, [])
            moveScore=(getNodeScore(setti, (parentG.get(target)+[target]), countDict)-originalScore)
            if moveScore<bestNodeScore  and len(loops)==0:
                bestEdge=current
                bestNodeScore=moveScore
            #put the edge back to restore graph
            parentG[target].append(origin)
    #find best reversal
    for target in parentG.keys():
        originAndTargets[target]=set(parentG.get(target))
        for origin in originAndTargets[target]:
            current=tuple([3,target,origin])
            originalScore=getNodeScore(setti, (parentG.get(target)+[target]), countDict)
            originalScore+=getNodeScore(setti, (parentG.get(origin)+[origin]), countDict)
            #Here we have all possible edges to reverse
            parentG[target].remove(origin)
            if parentG[target] == None:
                parentG[target]=[]
            parentG[origin].append(target)
            loops=validatePath(target,[],parentG, [])
            moveScore=getNodeScore(setti, (parentG.get(target)+[target]), countDict)
            moveScore+=(getNodeScore(setti, (parentG.get(origin)+[origin]), countDict)-originalScore)
            if moveScore<bestNodeScore  and len(loops)==0:
                bestEdge=current
                bestNodeScore=moveScore
            parentG[origin].remove(target)
            parentG[target].append(origin)
            if parentG[origin] == None:
                parentG[origin]=[]
    #If the function found a task that improves score the function executes it
    if bestEdge:
        bestNodeScore=0
        #add
        if bestEdge[0]==1:
            if parentG.get(bestEdge[1])==None:
                parentG[bestEdge[1]]=bestEdge[2]
            else:
                parentG[bestEdge[1]].append(bestEdge[2])
            arcList[bestEdge[2], bestEdge[1]]=moveScore
        #remove
        if bestEdge[0]==2:
            parentG[bestEdge[1]].remove(bestEdge[2])
            if parentG.get(bestEdge[1])==None:
                parentG[bestEdge[1]]=[]
        #reverse
        if bestEdge[0]==3:
            parentG[bestEdge[1]].remove(bestEdge[2])
            if parentG.get(bestEdge[1])==None:
                parentG[bestEdge[1]]=[]
            parentG[bestEdge[2]].append(bestEdge[1])
    #else the function returns None value
    else:
        return [None, arcList]
    #Final check
    loops=validatePath(target,[],parentG, [])
    score=getGraphScore(setti, testi, parentG, countDict)
    if score<getBest() and len(loops)==0:
        setBest(score)
        return [parentG, arcList]
    else:
        return  [None, arcList]

#Changes graph representation from Parent:Child to Child:parent 
#Works both ways.
def getParentform(normGraph):
    parentRep={i:[] for i in normGraph.keys()}
    for node, children in normGraph.items():
        for w in children:
                parentRep[w]+=[node]
    return parentRep

#Gets score for single node. 
def getNodeScore(setti, params, countDict):
    if argv[6]=='-b':
        Score=bdeu(countOfNode(setti, tuple(params), countDict),  10)
    else:
        Score=mdl(countOfNode(setti, tuple(params), countDict))
    return Score

#Gets the score for the graph
def getGraphScore(setti, testi, parentG, countDict):
    TotalScore=0
    for node, parents in parentG.items():
        params=parents+[node]
        if argv[1]=='-b':
            TotalScore+=bdeu(countOfNode(setti, tuple(params), countDict), 10)
        elif argv[1]=='-m':
            TotalScore+=mdl(countOfNode(setti, tuple(params), countDict))
    if argv[1]=='-p':
        return -log(testModel(setti, parentG, testi))
    return TotalScore

#Check if the graph is cycle free after adding on reversing edge from anode
#anode is a node in graph, path are the nodes the fuction has visited in recursive steps
def validatePath(anode, path, graph, loops):
    if anode in path:
        loops.append([1])
        return loops
    path+=[anode]
    for i in graph.get(anode):
        validatePath(i, path, graph, loops)
    return loops

#Stores the best score so far
def setBest(best):
    global bestScore
    bestScore=best
    
#returns the best value so far
def getBest():
    return bestScore

def setBestPred(best):
    global bestPrediction
    bestPrediction=best
    
#returns the best value so far
def getBestPred():
    return bestPrediction

#Creates a random DAG, used in random restarts
#NEEDS TO BE REWRITTEN AT SOME POINT, REMOVE HARDCODED VALUES
def getRandomDag(vars):
    graph={i:[] for i in vars}
    originAndTargets={}
    arcList=[]
    arcList=[x for x in it.permutations('abcdefghijlkmnopqrstuvwxyz', 2)]
    for j in range(10):
        anarc=rd.sample(arcList, 1).pop()
        target=ord(anarc[1])-97
        origin=ord(anarc[0])-97
        if graph.get(origin)==None:
            graph[origin]=target
        else:
            graph[origin].append(target)
        loops=validatePath(origin,[],graph, [])
        if len(loops)!=0:
            graph[origin].remove(target)
    return graph
                
#update and report most frequent arcs in results
def reportGraph(graph, topArcs):
    for a, b in graph.items():
        for x in b:
            arc=chr(a+65)+chr(x+65) 
            topArcs.append(arc)
    return topArcs
def bestGraphOfTopArcs(training, topArcs, validation):
    graph=getParentform({i:[] for i in range(len(training[0]))})
    retGraph={}
    TotalScore=getGraphScore(training,validation, graph, {})
    #TotalPred=testModel(training, graph, validation)
    for arc in topArcs.keys():
        origin=ord(arc[0])-65
        target=ord(arc[1])-65
        if graph.get(target)==None:
                graph[target]=origin
        else:
            graph[target].append(origin)
        #if len(validatePath(target,[],graph, []))==0:
        score=getGraphScore(training, validation, graph, {})
        #pred=testModel(training, graph, validation)
        if score<TotalScore:# and pred>TotalPred:
            TotalScore=score
           # TotalPred=pred
            retGraph=copy.deepcopy(graph)
        #else: graph[target].remove(origin)
    #return [getParentform(retGraph), TotalScore, TotalPred]
    
    return getParentform(retGraph)

def hc(training, validation, loops):
    cols=range(len(training[1]))
    graphAlpha=getParentform({i:[] for i in cols})
    iterations=0
    countDict={}
    localG=copy.deepcopy(graphAlpha)
    localBest=getGraphScore(training, validation, graphAlpha, countDict)
    setBest(localBest)
    setBestPred(0)
    arcList={}
    topArcs=[]
    for l in range(loops):
        t=time()
        #graphAlpha=getParentform({i:[] for i in cols})
        graphAlpha=getParentform(getRandomDag(range(len(training[1]))))
        repGraph=graphAlpha
        setBest(getGraphScore(training, validation, graphAlpha, countDict))
        lastMove=[]
        unchanged={}
        progress=0
        #Range 100 to ensure the hillclimbing continues as long as theres room for improvement
        #Usual breaking point is 20-25 iterations
        for i in range(100):
            stderr.write('.')
            progress+=1
##            print progress
##            print 'countDict length: %s ' % str(len(countDict))
            #if the algorithm gets stuck in a hairy situation
            if time()-t>10 and progress>2: break
##            print 'timelapse: %f' %(time()-t)
            t=time()
            graphAlpha, arcList=iterateAndImprove(training, validation, graphAlpha, countDict, arcList)
            if not graphAlpha or progress==20:
                reportGraph(getParentform(repGraph), topArcs)
##                print 'timelapse: %f' %(time()-t)
##                print 'current best score: %f' % getBest()
##                print 'global best score: %f' % localBest
##                print 'current best predictions: %.20f' % getBestPred()
##                print 'countDict length: %s ' % str(len(countDict))
##                print 'iterations: %i' %iterations
                break
            iterations+=1
            repGraph=graphAlpha
            if getBest()<localBest:
                if testModel(training, graphAlpha, validation)>getBestPred():
                    progress=0
                    setBestPred(testModel(training, graphAlpha, validation))
                    localBest=getBest()
                    localG=copy.deepcopy(graphAlpha)
##                    print getBest()
##                    print getBestPred()
##                    print iterations
    localG=getParentform(localG)  
##    print 'countDict length: %s ' % str(len(countDict))
##    for attribute, value in localG.items():
##        targs=''
##        for x in value:
##            targs+=chr(x+65)
##        print('{}\t: \t{}'.format(chr(attribute+65),targs))
##    print 'Score: %f' % getGraphScore(training, getParentform(localG), countDict)
##    print 'TestProb: %.20f' % testModel(training, localG, validation)
##    print 'iterations: %i' % iterations
    return [localG, arcList, testModel(training, localG, validation), localBest, topArcs]

#Generates a probabilitytable for Node in parameters:(parents, node)
def getPTable(setti, parameters):
    if len(parameters)>1:
        #split into node and parents
        parentsC=probOfNode(setti, tuple(parameters[:-1]))
        totC=probOfNode(setti, tuple(parameters))
        rettable=[]
        for i in range(len(parentsC)):
            case=[]
            for j in range(3):
                #P(a,b,c)/P(b,c)
                case.append(totC[3*i+j]/float(parentsC[i]))
            #Another normalization because of rounding errors
            case=[i/sum(case) for i in case]
            rettable.extend(case)
        #rettable=[totC[3*i+j]/float(parentsC[i]) for i in range(len(parentsC)) for j in range(3)]
    else:
        rettable=probOfNode(setti, tuple(parameters))
    return rettable

#get probabilitytable for each variable
#P(Variable | parentvariables)
#and package these in return value
def formula(setti, BGraph):
    BNForm={}
    pgraph=getParentform(BGraph)
    variables=enumerate
    for node, parents in pgraph.items():
        PTable={}
        params=parents+[node]
        PTable=getPTable(setti, params)
        BNForm[node]=PTable
    #print PTable
    #print BNForm
    return BNForm

#Counts the probability of given row in given graph
def probOfRow(row, BGraph, BNForm):
    prob=0
    pform=getParentform(BGraph)
    #multiply the probability of all items in a row together
    for i in pform.keys():
        lap=0
        pointer=0
        for j in pform[i]:
            #find the pointer to the right probabilitytable.
            pointer+=(int(row[j])-1)*3**(len(pform[i])-lap)
            lap+=1
        pointer+=int(row[i])-1
        prob+=log(BNForm[i][pointer])
    return exp(prob)

def writeResult(filename, outputTable):
    f = open(filename, 'w')
    for row in outputTable:
        f.write("%s\n" % row)
    f.close()

#A test of graph, basically return the sum of probabilities of rows
#If the model represents the data well the probability should be bigger
#so this is used to test if a model is better than the alternative.
def testModel(setti, model, testData):
    form=formula(setti, model)
    rowProb=[]
    for i in testData:
        rowProb.append(probOfRow(i, model, form))
    accuracy=float(sum(rowProb))/len(testData)
    return accuracy

#Basic bootstrapping with replacement
def bootstrap(data, samples):
    retData=[]
    for i in range(samples):
            retData.append(rd.sample(data, 1).pop())
    return retData
    
if __name__ == '__main__':
    desc    = 'Attempt to find optimal bayesian network'
    version = 'version 1.0 ' \
            + '(c) 2014 Mikko Hakila'
    
    if len(argv)==5:
        stderr.write('usage: promo_proj.py ')
        stderr.write('-m/b int(random restarts) -r/n int(K-folds) int(nr of processes)')
        exit()
    file='training_data.txt'
    t = time()
    stderr.write('reading ' +file +' ... ')
    with open(file, 'rt') as inp:
        rows = [tuple(line.split()) for line in inp]
    rows.pop(0)
    n = len(rows)
    stderr.write('[%d transaction(s)] ' % n)
    stderr.write('done [%.2fs].\n' % (time()-t)) 
    t = time()
    stderr.write('Hillclimbing...')
    
    #K-folds arvg[4] is the number of folds
    K=int(argv[4])
    #if argv[3] if -r take a random sample to training set
    if argv[3]=='-r':
        test= [rows.pop(rd.randrange(len(rows))) for _ in xrange(len(rows)/K)]
        train=rows
    #With -n take every K:th row to test set
    else:
        for k in range(K):
            train = [x for i, x in enumerate(rows) if i % K != k]
            test = [x for i, x in enumerate(rows) if i % K == k]
    #Multiprocess for efficiency, argv[5] is the number of cores
    
    pool = Pool(processes=int(argv[5]))
    contenders={}       
    loops=int(argv[2])/int(argv[5])
    if argv[7]=='-bs':
        for i in range(int(argv[5])):
            data=bootstrap(rows, int(argv[8]))
            if argv[3]=='-r':
                test= [data.pop(rd.randrange(len(data))) for _ in xrange(len(data)/K)]
                train=data
    #With -n take every K:th row to test set
            else:
                for k in range(K):
                    train = data
                    test = rows
            stderr.write('\ntrainN: {} , testN: {} '.format(str(len(train)), str(len(test))))
            contenders[i]=pool.apply_async(hc, (train, test, loops))
    else:
        for i in range(int(argv[5])):
            contenders[i]=pool.apply_async(hc, (train, test, loops))
    #pick the winner
    graph={}
    arcs={}
    score=0
    kprob=0
    topArcs=Counter()
    for i in range(4):
        Tgraph, Tarcs, Tprob, Tscore, TtopArcs=contenders[i].get()
        if Tprob>kprob:
            graph=Tgraph
            arcs=Tarcs
            kprob=Tprob
            score=Tscore
        topArcs+=Counter(TtopArcs)
    print topArcs.most_common(50)
    tarclist= [x[0] for x in topArcs.most_common(50)]
    stderr.write('\n\n')
    for attribute, value in graph.items():
        targs=''
        for x in value:
            targs+=chr(x+65)
        print('{}\t: \t{}'.format(chr(attribute+65),targs))
    print 'graph score: %f' % score
    print 'TestProb: %.20f' % kprob
##    bestGraph, bestScore, bestProb=bestGraphOfTopArcs(train, topArcs, test)
##    stderr.write('\nBESTGRAPH\n')
##    for attribute, value in bestGraph.items():
##        targs=''
##        for x in value:
##            targs+=chr(x+65)
##        print('{}\t: \t{}'.format(chr(attribute+65),targs))
##    print 'bestGraph score: %f' % bestScore
##    print 'bestGraph TestProb: %.20f' % bestProb
    #Save the graph
    
    #--------------------------Compute probabilities from different dags and take the average probs.
    #take the dags, compute probabilities, average probabilities.
    writeOutput('current2.txt', graph)
    #get it back!
    graph=pickle.load(open('current2.txt', "rb"))
    #Get formula for prob counting
    form=formula(rows, graph)
    with open('test_data.txt', 'rt') as inp:
            testrows = [tuple(line.split()) for line in inp]
    testrows.pop(0)
    n = len(testrows)
    rowProb=[]
    #count the probabilities
    for j in testrows:
        rowProb.append(probOfRow(j, graph, form))
    normalizedProbs=[]
    for i in rowProb:
        normalizedProbs.append(i/sum(rowProb))
    writeResult('lHakilaMikko_1_probs.txt', normalizedProbs)
    #these functions parse the arc list from 
    def charify(pair):
        return chr(pair[0]+65)+' '+chr(pair[1]+65)
    def injectGraph(graph, arcs):
        loc=[]
        if type(arcs)==dict:
            loc=[charify(x) for x,y in sorted(arcs.iteritems(), key=operator.itemgetter(1))]
        else: loc=arcs
        print 'Popped arcs %i' %len(graph.values())
        for a, b in graph.items():
            for x in b:
                #print [chr(a+65)+' '+chr(x+65)]
                if chr(a+65)+' '+chr(x+65) in loc: loc.remove(chr(a+65)+' '+chr(x+65))
                loc= [chr(a+65)+' '+chr(x+65)]+loc
        return loc
    #arcs=injectGraph(tarclist, arcs)
    writeResult('lHakilaMikko_1_arcs.txt', injectGraph(graph,arcs))
    stderr.write('done [%.2fs].\n' % (time()-t)) 
