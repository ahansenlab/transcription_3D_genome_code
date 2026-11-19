# MIT License

# Copyright (c) 2019 Massachusetts Institute of Technology 

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The following is adapted from a polychrom loop extrusion example
# see https://github.com/open2c/polychrom/tree/master/examples/loopExtrusion

#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True

import numpy as np

cimport numpy as np

import cython

cimport cython


cdef extern from "<stdlib.h>":
    double drand48()   

cdef cython.double randnum():
    return drand48()


cdef class LEFTranslocatorDirectional(object):
    cdef int N
    cdef int M
    cdef cython.double [:] emission
    cdef cython.double [:] stallLeft
    cdef cython.double [:] stallRight
    cdef cython.double [:] stallFalloff
    cdef cython.double [:] stallFalloff_typeII
    cdef cython.double [:] falloff
    cdef cython.double [:] falloff_typeII
    cdef cython.double [:] pause
    cdef cython.double [:] pause_typeII
    cdef cython.double [:] cumEmission
    cdef cython.double [:] permeability
    cdef cython.long [:] LEFs1
    cdef cython.long [:] LEFs2
    cdef cython.long [:] stalled1
    cdef cython.long [:] stalled2
    cdef cython.long [:] stallSite1 # site at which each leg last made its CTCF stall coin flip (-1 = none)
    cdef cython.long [:] stallSite2
    cdef cython.long [:] occupied
    cdef cython.long [:] LEFtype
    cdef cython.long [:] collision1 # list of whether LEF has collided with an RNAP (xy default only flags collisions where RNAP runs into one of the legs)    
    cdef cython.long [:] collision2
    cdef cython.double [:] unstall

    cdef double kin
    cdef double [:] kter
    cdef double kon
    cdef double koff
    cdef bint bursting
    cdef double dissoc
    cdef cython.double [:] PausePol
    cdef double [:] StallPol # probability of stalling
    cdef double [:] UnstallPol
    cdef double [:] PolPermLeft #rate of a cohesin arm that moves leftward jumping over Pol 
    cdef double [:] PolPermRight #rate of a cohesin arm that moves rightward jumping over Pol 
    cdef cython.long [:] PolLoadingSite
    cdef cython.long [:] PolTerminationSite
    cdef cython.long [:] PolPos # list of 1’s and 0’s indicating polymerase occupancy at sites along polymer 
    cdef cython.long [:] PolStalled # list with positions of stalled Pols
    cdef int ngenes
    cdef int strongStall # True if LEF stalling at CTCF prevents RNAP from pushing LEF past CTCF, False by default
    
    cdef int use_lef_permeability
    cdef int permeable
    
    cdef int maxss
    cdef int curss
    cdef cython.long [:] ssarray  
 
    
    def __init__(self, emissionProb, deathProb, stallProbLeft, stallProbRight, unstallProb, pauseProb,
                stallFalloffProb, permeability, numLEF,kinpol,kterpol, pauseProbPol,
                PolLoad,PolTer,stallProbPol, unstallProbPol, k_on, k_off,
                PolPermL=1,PolPermR=1,deathProbII=0, stallFalloffProbII=0, pauseProbII=0,
                poldissoc=0.,LEFtype=0
                ):
        emissionProb[0] = 0
        emissionProb[len(emissionProb)-1] = 0
        emissionProb[stallProbLeft > 0.9] = 0        
        emissionProb[stallProbRight > 0.9] = 0        
        
        self.N = len(emissionProb)
        self.M = numLEF
        self.emission = emissionProb
        self.stallLeft = stallProbLeft
        self.stallRight = stallProbRight
        self.stallLeft[0] = 1 # stall LEFs at two chain ends
        self.stallRight[self.N-1] = 1 
        self.falloff = deathProb
        self.collision1 = np.zeros(self.M, int) - 1
        self.collision2 = np.zeros(self.M, int) - 1
        self.occupied = np.zeros(self.N, int)
        if type(deathProbII) in [int, float, np.float64, np.double]:
            self.falloff_typeII = deathProbII*np.ones(self.N, dtype=np.double)
        else:
            self.falloff_typeII = deathProbII
        # pauseProb is supplied by the caller as a PAUSE probability (compSim.py builds it as
        # 1-vlef), matching the upstream polychrom convention where the test is `randnum() > pause`.
        # The stepping code below instead partitions a single random draw as
        #   r < pause          -> grow,   pause <= r < pause+pause_typeII -> shrink,
        # so it needs self.pause to be a STEP probability. Convert once here, which restores
        # p(step per attempt) = vlef and leaves the partition arithmetic in step() intact.
        self.pause = 1. - np.asarray(pauseProb, dtype=np.double)
        if type(pauseProbII) in [int, float, np.float64, np.double]:
            self.pause_typeII = pauseProbII*np.ones(self.N, dtype=np.double)
        else:
            self.pause_typeII = pauseProbII
        cumem = np.cumsum(emissionProb)
        cumem = cumem / float(cumem[len(cumem)-1])
        self.cumEmission = np.array(cumem, np.double)
        self.LEFs1 = np.zeros((self.M), int)
        self.LEFs2 = np.zeros((self.M), int)
        self.stalled1 = np.zeros(self.M, int)
        self.stalled2 = np.zeros(self.M, int)
        self.stallSite1 = np.zeros(self.M, int) - 1
        self.stallSite2 = np.zeros(self.M, int) - 1
        self.unstall = unstallProb
        self.permeability = permeability 
        if (numLEF>0) and (np.max(permeability)>0):
            self.permeable = 1
        else:
            self.permeable = 0
        if type(LEFtype) in [int, float, np.float64, np.double]:
            self.LEFtype = np.zeros(self.N, dtype=int) + int(LEFtype)
        else:
            self.LEFtype = LEFtype
        self.use_lef_permeability=1 # wrote this in for flexibility in code later
        self.occupied = np.zeros(self.N, int)
        self.stallFalloff = stallFalloffProb
        if type(stallFalloffProbII) in [int, float, np.float64, np.double]:
            self.stallFalloff_typeII = stallFalloffProbII*np.ones(self.N, np.double)
        else:
            self.stallFalloff_typeII = stallFalloffProbII
        self.occupied[0] = 1
        self.occupied[self.N - 1] = 1

        self.kin=kinpol              # Transcription initiation probability per time step
        if type(kterpol) in [int,float,np.float64,np.double]: # not the preferred method of initialization since it will make kter>0 including in genes
            self.kter=np.ones(self.N, np.double)*kterpol
        else:
            self.kter=kterpol            # Transcription termination probability per time step

        self.PausePol= pauseProbPol  # pause sites for RNAP
        self.StallPol= stallProbPol   # Pol II stall probabilities, only stalled PolII can terminate
        self.UnstallPol= unstallProbPol
        self.dissoc= poldissoc       # Pol II dissociation rate from the TSS         
        self.bursting = True
        self.kon = k_on
        self.koff = k_off
        self.PolLoadingSite = PolLoad    # list with Pol loading sites
        self.PolTerminationSite = PolTer # list with Transcription termination sites
        self.PolPos     = np.zeros((self.N),int)
        self.PolStalled = np.zeros((self.N),int)
        self.ngenes = len(self.PolLoadingSite)

        if type(PolPermL) in [int,float,np.float64,np.double]:
            print('setting left perm from float')
            self.PolPermLeft = np.zeros(self.N, np.double)
            for i in range(self.N):
                self.PolPermLeft[i] = PolPermL
        else:
            #setting permeabilities from input list
            self.PolPermLeft = PolPermL

        if type(PolPermR) in [int,float,np.float64,np.double]:
            print('setting right perm from float')
            self.PolPermRight = np.zeros(self.N, np.double)
            for i in range(self.N):
                self.PolPermRight[i] = PolPermR
        else:
            self.PolPermRight = PolPermR

 
        for i in self.PolTerminationSite:
            self.StallPol[i]=1 # Always unload at termination site, no PolII beyond termination site
        self.maxss = 1000000
        self.curss = 99999999

        for ind in xrange(self.M):
            self.birth(ind)
    cdef PolLoading(self):
        # Consider a bursty promoter? Or not, model is complicated enough as it is
        if not self.bursting and randnum()<self.kon:
            self.bursting = True
        elif self.bursting and randnum()<self.koff:
            self.bursting = False
        if self.bursting:
            for i in self.PolLoadingSite:
                if self.occupied[i]==0:
                    if self.PolPos[i]==0:
                        if randnum()<self.kin:
                            self.PolPos[i]=1        # Pol 2 loads to initiation site
                else: # add dissociation
                    if self.dissoc > 0.:
                        for i in self.PolLoadingSite:
                            if self.PolPos[i]==1:
                                if randnum()<self.dissoc:
                                    self.PolPos[i]=0
        return


    cdef PolTermination(self):
        for i in range(self.N):
            if self.PolStalled[i]==1:
                if randnum()<self.kter[i]:
                    self.PolPos[i]=0       # Stalled Pol 2 unloads from site of termination 
                    self.PolStalled[i]=0
        return
    
    
    cdef PolStallatTermination(self): #actually, this just handles stalls everywhere and unstalls
        
        for i in range(self.N):
            if self.PolPos[i]==1:
                if self.PolStalled[i]==0:
                    if randnum()<self.StallPol[i]:
                        self.PolStalled[i]=1
                else:
                    if randnum()<self.UnstallPol[i]:
                        self.PolStalled[i]=0
        return
    
    cdef PolStepping(self):
        
        cdef int i,j,k,l,m,it,itt, ncoleft,ncoright,direction,genelength
        cdef bint NextSiteOccupied
        cdef cython.long [:] LeftIndices
        cdef cython.long [:] RightIndices
        cdef int stalled_lef # indicator for whether RNAP is trying to push a stalled LEF
        
        # some explanation about what the above indices mean:
        # it - indexes genes
        # m - counter for num RNAP that have moved, used to avoid double stepping of RNAP. (after looking at site i and RNAP moves to i+1, next site to look for RNAP is i+2, not i+1)
        # itt - indexes positions in gene relative to TSS
        # i - indexes relative position within gene, accounting for TSS position and index m
        # j - indexes positions next to RNAP while checking for cohesin occupancy
        # k - indexes cohesin subunits while searching for which cohesin is at a particular polymer site
        # ncoleft, ncoright -- number of left/right cohesin subunits encountered in adjacent chain of cohesins

        for it in range(self.ngenes):
            if self.PolTerminationSite[it]-self.PolLoadingSite[it]<0:
                direction=-1
                genelength=self.PolLoadingSite[it]-self.PolTerminationSite[it]
            else:
                direction=1
                genelength=self.PolTerminationSite[it]-self.PolLoadingSite[it]
            
            if direction==1:
                m=0
                for itt in range(genelength+1):                                 
                    i=itt+self.PolLoadingSite[it]+m
                    # If Pol 2 makes a step, add 1 to position to prevent counting double (see end of for loop)
                    if (self.PolPos[i+m]==1) and (self.PolPos[i+1+m]==0) and ((i+1+m)<=self.PolTerminationSite[it]) and (self.PolStalled[i+m]==0):
                    # above condition checks: RNAP here (at i+m) but not next site (i+m+1), and RNAP is not at term (i+m+1) and is not currently stalled
                    # Note: Make sure the termination site is not at the end of the lattice
                        ncoleft=0
                        ncoright=0
                        stalled_lef=0
                        j=1
                        LeftIndices=np.zeros((self.M), int)
                        RightIndices=np.zeros((self.M),int)
                        NextSiteOccupied=True
                        while NextSiteOccupied==True: # Find number of consecutive cohesins after Pol, nco 
                            if (self.occupied[i+m+j]==1) and ((i+m+j)<(self.N-1)): #check if next site is occupied, starting next to RNAP, and verify site is not at end of polymer
                                if self.strongStall and (self.stalled2[i+m+j] == 1): # only check right subunits, which would be pushed past oriented CTCF (left subunit can be pushed off its CTCF)
                                    stalled_lef=True
                                    break # no point in continuing to check, LEFs won't be moved
                                # Find the cohesin arms corresponding to these positions
                                for k in range(self.M):
                                    if self.LEFs1[k]==(i+m+j): #is cohesin k at the next position (i+m+j)
                                        LeftIndices[ncoleft]=k #add it to list of LeftIndices to move
                                        ncoleft=ncoleft+1 #increment n left cohesins 

                                    elif self.LEFs2[k]==(i+m+j): #check for right cohesin subunits
                                        RightIndices[ncoright]=k                                       
                                        ncoright=ncoright+1
                                j=j+1 #move to next position
                            else:
                                NextSiteOccupied=False
                      
                        if (self.PolPos[i+m+j]==0) and ((i+m+ncoright+ncoleft)<self.N-2):  # Check if there is a Pol II right after the cohesins, if not, then the trailing pol will push all the nco cohesins forward.   
                            if randnum()<self.PausePol[i+m]:
                                self.PolPos[i+m]=0
                                self.PolPos[i+m+1]=1 # add extra number to prevent overcounting Pol                              

                                #here, we actually push the LEFs, as long as none of those LEFs were stalled
                                #note, as of 210611, we are only flipping the collision bool if cohesin moves due to RNAP
                                if not stalled_lef: #stalled_lef=False when strongStall=False
                                    for l in range(ncoleft):
                                        self.occupied[self.LEFs1[LeftIndices[l]]+1]=1 # mark last site in chain as occupied
                                        self.LEFs1[LeftIndices[l]]=self.LEFs1[LeftIndices[l]]+1 # move LEF
                                        self.collision1[LeftIndices[l]] = i+m+1 # change indicator of collision for LEF. indicator shows which RNAP is involved in collision
                                        self.stalled1[LeftIndices[l]] = 0 # if LEF was stalled (e.g., due to CTCF) it should not be stalled any more, compatible with strongStall since left LEFs can move off CTCF by RNAP

                                    for l in range(ncoright): 
                                        self.occupied[self.LEFs2[RightIndices[l]]+1]=1
                                        self.LEFs2[RightIndices[l]]=self.LEFs2[RightIndices[l]]+1   
                                        self.collision2[RightIndices[l]] = i+m+1
                                        self.stalled2[RightIndices[l]] = 0 # compatible with strongStall because this must be 0 to enter this the above if statement

                                    self.occupied[i+m+1]=0 # only the Polymerase now occupies this site and cohesins do not because they were pushed away
                                #if a LEF is stalled, the chain of LEFs will not be pushed, but we will nonetheless permit RNAP to advance (i.e., assume dominance of CTCF site when self.strongStall=True)
                                m=m+1

            elif direction==-1:#Pol moves from right to left
                m=0
                for itt in range(genelength+1):             
                     # If Pol 2 makes a step, add 1 to position to prevent counting double
                    i=self.PolLoadingSite[it]-itt-m
                    
                    if (self.PolPos[i-m]==1) and (self.PolPos[i-1-m]==0) and ((i-1-m)>=self.PolTerminationSite[it]) and (self.PolStalled[i-m]==0):
                    # Make sure the termination set is not at the beginning of the lattice
                        ncoleft=0
                        ncoright=0
                        stalled_lef=0
                        j=1
                        LeftIndices=np.zeros((self.M), int)
                        RightIndices=np.zeros((self.M),int)
                        NextSiteOccupied=True
                        while NextSiteOccupied==True: # Find number of consecutive cohesins after Pol, nco 
                            if (self.occupied[i-m-j]==1) and ((i-m-j)>0):
                                if self.strongStall and (self.stalled1[i-m-j] == 1): # only check left subunits, which would be pushed past oriented CTCF (right subunit can be pushed off its CTCF)
                                    stalled_lef=True
                                    break
                                # Find the cohesin arms corresponding to these positions

                                for k in range(self.M):
                                    if self.LEFs1[k]==(i-m-j):
                                        LeftIndices[ncoleft]=k                                     
                                        ncoleft=ncoleft+1

                                    elif self.LEFs2[k]==(i-m-j):
                                        RightIndices[ncoright]=k                                       
                                        ncoright=ncoright+1
                                j=j+1
                            else:
                                NextSiteOccupied=False                    

                        if (self.PolPos[i-m-j]==0) and ((i-m-ncoright-ncoleft)>1):  # Check if there is a Pol II right after the cohesins, if not, then the trailing pol will push all the nco cohesins forward.   
                            if randnum()<self.PausePol[i-m]:
                                self.PolPos[i-m]=0
                                self.PolPos[i-m-1]=1 # add extra number to prevent overcounting Pol                              

                                if not stalled_lef:
                                    for l in range(ncoleft):
                                        self.occupied[self.LEFs1[LeftIndices[l]]-1]=1
                                        self.LEFs1[LeftIndices[l]]=self.LEFs1[LeftIndices[l]]-1
                                        self.collision1[LeftIndices[l]] = i-m-1
                                        self.stalled1[LeftIndices[l]] = 0

                                    for l in range(ncoright): 
                                        self.occupied[self.LEFs2[RightIndices[l]]-1]=1
                                        self.LEFs2[RightIndices[l]]=self.LEFs2[RightIndices[l]]-1   
                                        self.collision2[RightIndices[l]] = i-m-1
                                        self.stalled2[RightIndices[l]] = 0

                                    self.occupied[i-m-1]=0 # only the Polymerase now occupies this site  
                                m=m+1
        return

    cdef birth(self, cython.int ind):
        cdef int pos,i 
  
        while True:
            pos = self.getss()
            if pos >= self.N - 1:
                print("bad value", pos, self.cumEmission[len(self.cumEmission)-1])
                continue 
            if pos <= 0:
                print("bad value", pos, self.cumEmission[0])
                continue 
 
            
            if self.occupied[pos] == 1:
                continue
            if self.occupied[pos+1] == 1:
                continue

            self.LEFs1[ind] = pos
            self.LEFs2[ind] = pos+1
            self.occupied[pos] = 1
            self.occupied[pos+1] = 1
            self.stallSite1[ind] = -1 # a newly loaded LEF has not been tested anywhere yet
            self.stallSite2[ind] = -1

            return

    cdef death(self):
        cdef int i,jj
        cdef double falloff1, falloff2 
        cdef double falloff 
        cdef int multi_occ1, multi_occ2 # variables to indicate whether site of interest is occupied by multiple LEFs
        multi_occ1=0 # set to 0 so that in case self.permeable==False, we can still flip the site occupied flag off
        multi_occ2=0

        for i in xrange(self.M):
            if self.stalled1[i] == 0:
                if self.LEFtype[i]==0:
                    falloff1 = self.falloff[self.LEFs1[i]]
                else:
                    falloff1 = self.falloff_typeII[self.LEFs1[i]]
            else: 
                if self.LEFtype[i]==0:
                    falloff1 = self.stallFalloff[self.LEFs1[i]]
                else:
                    falloff1 = self.stallFalloff_typeII[self.LEFs1[i]]
            if self.stalled2[i] == 0:
                if self.LEFtype[i]==0:
                    falloff2 = self.falloff[self.LEFs2[i]]
                else:
                    falloff2 = self.falloff_typeII[self.LEFs2[i]]
            else:
                if self.LEFtype[i]==0:
                    falloff2 = self.stallFalloff[self.LEFs2[i]]
                else:
                    falloff2 = self.stallFalloff_typeII[self.LEFs2[i]]
            
            falloff = max(falloff1, falloff2)
            if randnum() < falloff:
                if self.permeable:
                    multi_occ1 = 0
                    multi_occ2 = 0
                    for jj in range(self.M):
                        if (self.LEFs1[jj] == self.LEFs1[i]) and not (jj == i):
                            multi_occ1 = 1
                        elif (self.LEFs2[jj] == self.LEFs1[i]):
                            multi_occ1 = 1
                        if (self.LEFs2[jj] == self.LEFs2[i]) and not (jj==i):
                            multi_occ2 = 1
                        elif (self.LEFs1[jj] == self.LEFs2[i]):
                            multi_occ2 = 1

                if not multi_occ1:
                    self.occupied[self.LEFs1[i]] = 0
                if not multi_occ2:
                    self.occupied[self.LEFs2[i]] = 0
                self.stalled1[i] = 0
                self.stalled2[i] = 0
                self.birth(i)
    
    cdef int getss(self):
    
        if self.curss >= self.maxss - 1:
            foundArray = np.array(np.searchsorted(self.cumEmission, np.random.random(self.maxss)), dtype = np.int_)
            self.ssarray = foundArray
            self.curss = -1
        
        self.curss += 1         
        return self.ssarray[self.curss]
        
        

    cdef step(self): #cohesin stepping
        cdef int i, kk, ll, mm, nn
        cdef double pause1, pause2 #, pause 
        cdef double pause_typeII1, pause_typeII2
        cdef double stall1, stall2 
        cdef int cur1
        cdef int cur2 
        cdef double r



        for i in range(self.M):            
            stall1 = self.stallLeft[self.LEFs1[i]]
            stall2 = self.stallRight[self.LEFs2[i]]
            
            # The stall coin is flipped once per arrival at a site, not once per timestep.
            # stallSite tracks where the last flip happened; a leg that lost (or that unstalled)
            # is not re-tested until it reaches a different site. Without this, a leg parked at a
            # CTCF gets a fresh chance to stall every timestep, so the per-encounter stall
            # probability tends to 1 as vlef falls, instead of staying at stallLeft/stallRight.
            if self.stalled1[i] == 1:
                if randnum() < self.unstall[i]:
                    self.stalled1[i] = 0
            elif self.stallSite1[i] != self.LEFs1[i]:
                self.stallSite1[i] = self.LEFs1[i]
                if randnum() < stall1:
                    self.stalled1[i] = 1

            if self.stalled2[i] == 1:
                if randnum() < self.unstall[i]:
                    self.stalled2[i] = 0
            elif self.stallSite2[i] != self.LEFs2[i]:
                self.stallSite2[i] = self.LEFs2[i]
                if randnum() < stall2:
                    self.stalled2[i] = 1

                         
            cur1 = self.LEFs1[i]
            cur2 = self.LEFs2[i]

            if self.stalled1[i] == 0:
                #allow a bypass event if lattice site that is 2 units away is >0 and not occupied by a LEF or an RNAP
                bypass_left_allowed = (cur1-2>0) and (self.occupied[cur1-2]==0) and (self.PolPos[cur1-2]==0)
                if ((self.occupied[cur1-1]==0) or bypass_left_allowed): # this second line allows bypassing of a single cohesin
                    pause1 = self.pause[self.LEFs1[i]]
                    r= randnum()
                    if r < pause1: # this is probability of making either the bypassing step or the non-bypassing step 
                        if self.occupied[cur1-1]==0:# here, we recheck whether this is a bypassing extrusion step or not. if site is empty, no bypassing necessary
                            if (self.PolPos[cur1-1]==0) or ((self.PolPos[cur1-1]==1) and (randnum()<self.PolPermLeft[cur1-1])):#Pol conditions
                                self.occupied[cur1-1]=1
                                self.occupied[cur1]=0
                                self.LEFs1[i]=cur1-1
                                self.collision1[i]= -1 # if bypassing Pol II or Pol II is not present, collision indicator = False.
                        elif randnum() < self.permeability[i]: #for LEF bypassing steps, already checked that Pol II is not present at cur1-2 so don't need to check again
                            self.occupied[cur1-2]=1 
                            self.occupied[cur1]=0
                            self.LEFs1[i]=cur1-2
                            self.collision1[i]= -1 #even if there is a PolII at cur1-3, they have not yet collided.
                        #else, no step occurs. collision1 variable unchanged.  a check on that variable appears at end of this function
                    elif (self.occupied[cur1+1]==0):#left and right of cohesin are unocc by cohesin, but r>pause1, so test for shrink 
                                                    #(not allowing bypasses on shrinking events)
                        pause_typeII1 = self.pause_typeII[self.LEFs1[i]]
                        if r < pause1+pause_typeII1: #already know r >pause1; checking to see if pause1 < r < pause1+pause_typeII1
                            if (self.PolPos[cur1+1]==0) or ((self.PolPos[cur1+1]==1) and (randnum()<self.PolPermRight[cur1+1])): #Pol conditions
                                self.occupied[cur1+1]=1
                                self.occupied[cur1]=0
                                self.LEFs1[i]=cur1+1
                                self.collision1[i]= -1
                        #when step not attempted, collision1 is unchanged (final check on this variable appears at end of function, below)
                elif (self.occupied[cur1+1]==0):#only right of cohesin unocc by cohesin, so only shrinking allowed by left side
                    pause_typeII1 = self.pause_typeII[self.LEFs1[i]]
                    r=1.
                    if pause_typeII1 > 0:
                        r= randnum()
                    if r < pause_typeII1:
                        if (self.PolPos[cur1+1]==0) or ((self.PolPos[cur1+1]==1) and (randnum()<self.PolPermRight[cur1+1])): #Pol conditions
                            #again, not allowing bypassing on shrinking events
                            self.occupied[cur1+1]=1
                            self.occupied[cur1]=0
                            self.LEFs1[i]=cur1+1
                            self.collision1[i]=-1
                   #again, if stepping attempt is not made, collision1 = 1 can remain. 

            if self.stalled2[i] == 0:
                bypass_right_allowed= (cur2+2<self.N) and (self.occupied[cur2+2]==0) and (self.PolPos[cur2+2]==0)
                if ((self.occupied[cur2+1]==0) or bypass_right_allowed): 
                    pause2 = self.pause[self.LEFs2[i]]
                    r= randnum()
                    if r < pause2: # this is probability of making either the bypassing step or the non-bypassing step
                        if self.occupied[cur2+1]==0:
                            if (self.PolPos[cur2+1]==0) or ((self.PolPos[cur2+1]==1) and (randnum()<self.PolPermRight[cur2+1])):#Pol conditions
                                self.occupied[cur2+1]=1
                                self.occupied[cur2]=0
                                self.LEFs2[i]=cur2+1
                                self.collision2[i]=-1
                        elif randnum() < self.permeability[i]:
                            self.occupied[cur2+2]=1
                            self.occupied[cur2]=0
                            self.LEFs2[i]=cur2+2
                            self.collision2[i]=-1
                    elif (self.occupied[cur2-1]==0):#left and right unocc by cohesin so test for shrink
                        pause_typeII2 = self.pause_typeII[self.LEFs2[i]]
                        if r < pause2+pause_typeII2: #already know r >pause2; checking to see if pause2 < r < pause_typeII2
                            if (self.PolPos[cur2-1]==0) or ((self.PolPos[cur2-1]==1) and (randnum()<self.PolPermLeft[cur2-1])): #Pol conditions
                                self.occupied[cur2-1]=1
                                self.occupied[cur2]=0
                                self.LEFs2[i]=cur2-1
                                self.collision2[i]=-1
                elif (self.occupied[cur2-1]==0):#only site to left unocc by cohesin, so only shrinking allowed by right
                    r=1.
                    pause_typeII2 = self.pause_typeII[self.LEFs2[i]]
                    if pause_typeII2>0:
                        r= randnum()
                    if r < pause_typeII2:
                        if (self.PolPos[cur2-1]==0) or ((self.PolPos[cur2-1]==1) and (randnum()<self.PolPermLeft[cur2-1])): #Pol conditions
                            self.occupied[cur2-1]=1
                            self.occupied[cur2]=0
                            self.LEFs2[i]=cur2-1
                            self.collision2[i]=-1
            
            #now, after both sides of LEF have been moved (or not) check to see if they remain adjacent to RNAP or a series of LEFs that ends in an RNAP 
            #note that if LEFs have been moved, their collision indicator gets reset to 0 regardless, see above.
            #this check is for LEFs that do not undergo motion, which could have lost a connection to RNAP due to other LEFs moving or due to RNAP unbinding
            #also need to check cur2+1, and do this for all lefs, after all lefs have moved...
        for i in range(self.M):
            cur1 = self.LEFs1[i]
            cur2 = self.LEFs2[i]
            if cur1>0:
                kk=cur1-1
            else:
                kk=0
            ll=cur1+1#cur1+1 guaranteed < N
            mm=cur2-1
            if cur2<self.N-1:
                nn=cur2+1
            else:
                nn=self.N-1

            while (self.occupied[kk]==1) and (kk>0):  # first have to get to the beginning of the chain of LEFs, if applicable
                kk -= 1
            while (self.occupied[ll]==1) and (ll<self.N-1):
                ll += 1
            if (not (self.PolPos[kk] or self.PolPos[ll])) or (not ((self.collision1[i]==kk) or (self.collision1[i]==ll))):
                #if adjacent sites don't have pol II, or don't have the pol II that matches the collision indicator
                self.collision1[i]=-1
                #this is necessary because of the scenario where e.g., a right-moving pol from a collision unbinds but a left-moving pol approaches the LEF, without colliding yet. this is why we use a collision indicator that identifies the RNAP that collided 
            while (self.occupied[mm]==1) and (mm>0):
                mm -= 1
            while (self.occupied[nn]==1) and (nn<self.N):
                nn += 1
            if (not (self.PolPos[mm] or self.PolPos[nn])) or (not ((self.collision2[i]==mm) or (self.collision2[i]==nn))):
                self.collision2[i]=-1
 
    def steps(self,N):
        cdef int i 
        for i in xrange(N):
            self.death()
            self.step()
            self.PolLoading()
            self.PolTermination()
            self.PolStallatTermination()#Always has to come before PolStepping!
            self.PolStepping()
            
            
    def getOccupied(self):
        return np.array(self.occupied)
    
    def getLEFs(self):
        return np.array(self.LEFs1), np.array(self.LEFs2)
        
    def getPolOccupied(self):
        return np.array(self.PolPos)
        
    def getBurstState(self):
        return self.bursting

    def updateMap(self, cmap):
        cmap[self.LEFs1, self.LEFs2] += 1
        cmap[self.LEFs2, self.LEFs1] += 1

    def updatePos(self, pos, ind):
        pos[ind, self.LEFs1] = 1
        pos[ind, self.LEFs2] = 1


