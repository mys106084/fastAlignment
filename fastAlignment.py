from __future__ import division
from time import time
import math
import sys
import scipy
import collections
import cPickle
import re, json


class Alignment(object):
    def __init__(self, nullprob = 0.1, dirT = 0.001, lamb = 10.0): # (1) nullAlignment (2) dirT (3) dirQ
        #
        self.url_e = 'data//corpus.en'
        self.url_f = 'data//corpus.es'
        self.url_a = 'alignments.txt'
        self.url_dev_e = 'data//dev.en'
        self.url_dev_f = 'data//dev.es'
        self.url_dev_out = 'data//dev.out'
        self.url_dev_gold = 'data//dev.key'

        
        self.iterations = 30
        self.iter = 0
        self.infinitesimal = 0.0000001
        
        
        # Parameters to be Tuned
        self.nullprob = nullprob
        self.normprob = 1 - self.nullprob
        self.dirT = dirT
        self.lamb = lamb
        
        # Parameters for Gradient Descent
        self.emp_feat = 0
        self.mode_feat = 0

        # initialization for the docs
        self.wordmap_e = {}
        self.wordmap_f = {}
        self.words_e = []
        self.words_f = []
        
        self.sum_s = 0 # size of sentences
        self.sum_e = 0
        self.sum_f = 0
                     
        self.lenval_e = [] # lengths value
        self.lenval_f = []

        # used in inference
        self.sentences_e = []
        self.sentences_f = []
        self.lengths_e = []
        self.lengths_f = []
        self.alignments = []

        # used in dev
        self.sentences_dev_e = []
        self.sentences_dev_f = []
        self.lengths_dev_e = []
        self.lengths_dev_f = []
        self.alignments_dev = []
        
        # counts
        self.count_e = collections.defaultdict(int)
        self.count_fe = collections.defaultdict(int)
        
        # delta
        self.delta = {}
        self.t = collections.defaultdict(lambda:self.infinitesimal) #translation probability
        
        
        print "MODEL:"+ 'NullAligenment:' +str(self.nullprob)+ '  dirT:' +str(self.dirT)+ '  Lambda:' +str(self.lamb)


    def Inputcorpus(self):
        fin_e = open(self.url_e,'r')
        for line in fin_e:
            words = line.strip().split()
            words_idx = []
            for word in words:
                if word not in self.wordmap_e:
                    self.wordmap_e[word] = len(self.wordmap_e) # build wordmap
                    self.words_e.append(word) # save words
                self.count_e[word] += 1
                words_idx.append(self.wordmap_e[word])
            if len(words_idx) not in self.lenval_e: # restore all the length values
                self.lenval_e.append(len(words_idx))
            self.sentences_e.append(words_idx)
            self.lengths_e.append(len(words_idx))
            self.sum_s = len(self.sentences_e)# count sentence
        fin_e.close()
        fin_f = open(self.url_f,'r')
        for line in fin_f:
            words = line.strip().split()
            words_idx = []
            for word in words:
                if word not in self.wordmap_f:
                    self.wordmap_f[word] = len(self.wordmap_f) # build wordmap
                    self.words_f.append(word) # save words
                words_idx.append(self.wordmap_f[word])
            if len(words_idx) not in self.lenval_f: # restore all the length values
                self.lenval_f.append(len(words_idx))
            self.sentences_f.append(words_idx)
            self.lengths_f.append(len(words_idx))
        fin_f.close()
        print "Sentences:"+str(self.sum_s)
        print "E words:"+str(len(self.words_e))
        print "F words:"+str(len(self.words_f))
        
        
    #-----------------------------------------------PakageFunction-------------------------------------------#   
    def GetT(self,idx_f,idx_e):         #t(f|e)
        self.t.setdefault((idx_f,idx_e),1.0/len(self.wordmap_f))
        return self.t[(idx_f,idx_e)]
    
    def GetQ(self,j,i,l,m,z):             #q(j|i,l,m)     move +1 for fast alignment
        return UnnormalizedProb(i+1,j+1,l,m,self.lamb)/z
    
    def GetZ(self,i,l,m,lamb):
        return ComputeZ(i+1,l,m,lamb)
    
    def GetLogZ(self,i, l, m, lamb):
        return ComputeDLogZ(i+1,l,m,lamb)
        
        
    def GetDelta(self,s,i,j):
        return self.delta[(s,i,j)]
    
    def GetCount_fe(self,idx_f,idx_e):
        return self.count_fe[(idx_f,idx_e)]

    def GetCount_e(self,idx_e):
        return self.count_e[idx_e]
    
    def GetCount_jilm(self,j,i,l,m):
        return self.count_jilm[(j,i,l,m)]
    
    def GetCount_ilm(self,i,l,m):
        return self.count_ilm[(i,l,m)]
    

    
     #-----------------------------------------------ComputeFunction-------------------------------------------#

    def ComputeT(self):
        #self.t = {}
        for (idx_f,idx_e),val in self.count_fe.iteritems():
            self.t[(idx_f,idx_e)] = ( self.GetCount_fe(idx_f,idx_e) + self.dirT ) / ( self.GetCount_e(idx_e) + self.dirT*len(self.wordmap_f) )  # dirT
            
    def ComputeLamb(self):
        for ii in xrange(0,20):
            self.mod_feat = 0
            for s in xrange(0,self.sum_s):
                m = self.lengths_f[s]
                l = self.lengths_e[s]
                for i in xrange(0,m):
                    if l==0:
                        continue
                    else:
                        self.mod_feat +=  self.GetLogZ(i, l, m, self.lamb)  # null aligned words
                    
            #self.mod_feat /= len(self.wordmap_f)
            #self.emp_feat /= len(self.wordmap_f)
            delta = (self.emp_feat - self.normprob*self.mod_feat) * 20.0/len(self.wordmap_f)
            print 'Delata: '+ str(delta)
            self.lamb += delta
            if self.lamb <= 0.1:
                self.lamb = 0.1
            if self.lamb > 150:
                self.lamb = 150
                
            print 'Lambda: '+str(self.lamb)
           
    def UpdateCounts(self):
        self.count_e.clear()
        self.count_fe.clear() # define a new counts in every iteration
        #------------------------ComputeDelta-------------------------------------
        self.emp_feat = 0
        for s in xrange(0,self.sum_s):
            #if s%1000 == 0:
            #    print "E-step - ComputeDelta - Sentence:"+str(s)
            m = self.lengths_f[s]
            l = self.lengths_e[s]
            for i in xrange(0,m):
                normalization = 0
                prob = []
                if l==0:            # in case of l==0
                    continue
                else:
                    z = self.GetZ(i,l,m,self.lamb)
                for j in xrange(0,l):
                    prob.append(self.GetT(self.sentences_f[s][i],self.sentences_e[s][j])*self.GetQ(j,i,l,m,z)*self.normprob)
                    normalization += prob[j]
                prob.append(self.GetT(self.sentences_f[s][i],-1)*self.nullprob)
                norm_noNA = normalization
                normalization += prob[l]
                # Print alignments
                for j in xrange(0,l):
                    self.delta[(s,i,j)] = prob[j]/normalization
                    
                    #self.emp_feat += prob[j]/norm_noNA*Feature(i+1,j+1,l,m)
                    
                #nullAlignment
                self.delta[(s,i,-1)] = prob[l]/normalization
         #---------------------------------------UpdateCounts---------------------------------------
                for j in xrange(0,l):
                    # Count C(e,f)
                    self.count_fe[(self.sentences_f[s][i],self.sentences_e[s][j])] += self.GetDelta(s,i,j)
                    # Count C(e)
                    self.count_e[self.sentences_e[s][j]] += self.GetDelta(s,i,j)
                    
                    self.emp_feat += self.GetDelta(s,i,j)*Feature(i+1,j+1,l,m)
                # nullAlignment
                self.count_fe[(self.sentences_f[s][i],-1)] += self.GetDelta(s,i,-1)
                self.count_e[-1] += self.GetDelta(s,i,-1)
                
            
    def UpdateCounts_IBM1(self):
        self.count_e.clear()
        self.count_fe.clear() # define a new counts in every iteration
        for s in xrange(0,self.sum_s):
            #if s%1000 == 0:
            #    print "E-step- ComputeDelta - Sentence:"+str(s)
            m = self.lengths_f[s]
            l = self.lengths_e[s]
            for i in xrange(0,m):
                #------------------------ComputeDelta-------------------------------------
                normalization = 0
                prob = []
                for j in xrange(0,l):
                    prob.append(self.GetT(self.sentences_f[s][i],self.sentences_e[s][j])*self.normprob)
                    normalization += prob[j]
                prob.append(self.GetT(self.sentences_f[s][i],-1)*self.nullprob)
                normalization += prob[l]   
                for j in xrange(0,l):
                    self.delta[(s,i,j)] = prob[j]/normalization
                #nullAlignment
                self.delta[(s,i,-1)] = prob[l]/normalization
                #------------------------UpdateCounts-------------------------------------
                for j in xrange(0,l):
                    # Count C(e,f)
                    self.count_fe[(self.sentences_f[s][i],self.sentences_e[s][j])] += self.GetDelta(s,i,j)
                    # Count C(e)
                    self.count_e[self.sentences_e[s][j]] += self.GetDelta(s,i,j)
                # nullAlignment
                self.count_fe[(self.sentences_f[s][i],-1)] += self.GetDelta(s,i,-1)
                self.count_e[-1] += self.GetDelta(s,i,-1)
    def EM(self):
        for self.iter in xrange(0,self.iterations):
            print "EM processing in iteration:"+str(self.iter)
        #E-step
            print "E-step-UpdateCounts."
            if self.iter >=5:
                self.UpdateCounts()
            else:
                self.UpdateCounts_IBM1()
        #M-step
             # compute t
            print "M-step-ComputeT."
            self.ComputeT()
             # compute q  j i l m
            
            if self.iter >=5:
                print "M-step-ComputeQ"
                self.ComputeLamb()
                
    def DEV(self):
        fin_e = open(self.url_dev_e,'r')
        for line in fin_e:
            words = line.strip().split()
            words_idx = []
            for word in words:
                if word not in self.wordmap_e:
                    words_idx.append(-2)            #-1 is null alignment
                else:
                    words_idx.append(self.wordmap_e[word])
            self.sentences_dev_e.append(words_idx)
            self.lengths_dev_e.append(len(words_idx))
            
        fin_f = open(self.url_dev_f,'r')
        for line in fin_f:
            words = line.strip().split()
            words_idx = []
            for word in words:
                if word not in self.wordmap_f:
                    words_idx.append(-2)
                else:
                    words_idx.append(self.wordmap_f[word])        
            self.sentences_dev_f.append(words_idx)
            self.lengths_dev_f.append(len(words_idx))
            
        # get alignments for dev
        self.alignments_dev = []
        for s in xrange(0,len(self.sentences_dev_e)):
            m = self.lengths_dev_f[s]
            l = self.lengths_dev_e[s]
            self.alignments_dev.append([])
            for i in xrange(0,m):
                self.alignments_dev[s].append(0)
                maximum = 0
                if self.sentences_dev_f[s][i] == -2: #filter the French words which are not in wordmap
                    self.alignments_dev[s][i] = -1
                    continue
                if l==0:            # in case of l==0
                    continue
                else:
                    z = self.GetZ(i,l,m,self.lamb)
                #print 'TMP prob:'
                for j in xrange(0,l):    # starts from 1                    
                    if self.sentences_dev_e[s][j] == -2: #filter the English words which are not in wordmap
                        continue                    
                    #tmp = scipy.log(self.GetT(self.sentences_e[s][j],self.sentences_f[s][i]))+scipy.log(self.GetQ_IBM2(j,i,l,m))
                    tmp = self.GetT(self.sentences_dev_f[s][i],self.sentences_dev_e[s][j])*self.GetQ(j,i,l,m,z)*self.normprob
                    if tmp  >= maximum:
                        self.alignments_dev[s][i] = j
                        maximum = tmp   
                # nullAlignment
                # Compare to "null alignemnt probablity"
                tmp = self.GetT(self.sentences_dev_f[s][i],-1)*self.nullprob # nullAlignment
                #print tmp
                if tmp >= maximum:
                    self.alignments_dev[s][i] = -1
        print "\n DEV- Alignments - Sentence:"+str(s)
        fout = open(self.url_dev_out,'w')
        for s in xrange(0,len(self.sentences_dev_e)):
            for i in xrange(0,len(self.alignments_dev[s])):
                if self.alignments_dev[s][i] ==-1:
                    continue
                fout.write(str(s+1)+' '+str(self.alignments_dev[s][i]+1)+' '+str(i+1))
                fout.write('\n')
        fout.close()
    #-----------------------------------------------FastAlignementComputing-------------------------------------------#
#@staticmethod
def Feature(i,j,l,m):
    return -math.fabs(float(j) / l - float(i) / m)

#@staticmethod
def UnnormalizedProb(i,j,l,m,lamb):         # There're some restrictions of calling this method
    return math.exp(Feature(i, j, l, m) * lamb)    

#@staticmethod
def ComputeZ(i,l,m,lamb):
    split = float(i) * l / m
    floor = int(split)
    ceil = floor + 1
    ratio = math.exp(-lamb / l)
    num_top = l - floor
    ezt = 0
    ezb = 0
    if (num_top):
        ezt = UnnormalizedProb(i, ceil, l, m, lamb) * (1.0 - pow(ratio, num_top)) / (1.0 - ratio)
    if (floor):
        ezb = UnnormalizedProb(i, floor, l, m, lamb) * (1.0 - pow(ratio, floor)) / (1.0 - ratio)
    return ezb + ezt;

#@staticmethod
def ComputeDLogZ(i,l,m,lamb):
    z = ComputeZ(i,l,m,lamb)
    split = float(i) * l / m
    floor = int(split)
    ceil = floor + 1
    ratio = math.exp(-lamb / l)
    d = -1.0 / l
    num_top = l - floor
    pct = 0
    pcb = 0
    if (num_top):
        pct = arithmetico_geometric_series(Feature(i, ceil, l, m), UnnormalizedProb(i, ceil, l, m, lamb), ratio, d, num_top)
    if (floor):
        pcb = arithmetico_geometric_series(Feature(i, floor, l, m), UnnormalizedProb(i, floor, l, m, lamb), ratio, d, floor)
    return (pct + pcb) / z
#@staticmethod        
def arithmetico_geometric_series(a_1, g_1, r, d, n):
    g_np1 = g_1 * math.pow(r, n)
    a_n = d * (n - 1) + a_1
    x_1 = a_1 * g_1
    g_2 = g_1 * r
    rm1 = r - 1
    return (a_n * g_np1 - x_1) / rm1 - d*(g_np1 - g_2) / (rm1 * rm1)
    
def runfastAlignment(arg):
    t = time()
    myAlignment = Alignment(float(arg[0]),float(arg[1]),float(arg[2]))
    myAlignment.Inputcorpus()
    #print min(myAlignment.lengths_e)
    
    
    myAlignment.EM()
    myAlignment.DEV()
    print "total run time:"
    print time()-t
    return 1
    

if __name__ == "__main__":
    #main(sys.argv[1:]) # 0.08 0.001 0.1
    arg = [0.08,0.001,10]
    print runfastAlignment(arg)
               
                
                
                
                
                
                
                
                
                
                