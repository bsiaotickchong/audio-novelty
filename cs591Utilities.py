"""
 File: cs591Utilities.py
 Author: Wayne Snyder

 Date: 1/28/17
 Purpose: This collects together the most important algorithms used in
          CS 591, in order to work interactively; for the most part
          signals are manipulated as arrays, not as wave files.
          This file assumes you have scipy and numpy.
          
          The main difference from previous version is that
          we are using numpy arrays exclusively. 
"""

import array
import contextlib
import wave
import numpy as np
import matplotlib.pyplot as plt
#from numpy import pi, sin, cos, exp, abs
#from scipy.io.wavfile import read, write


"""
 Basic parameters
"""


numChannels   = 1                      # mono
sampleWidth   = 2                      # in bytes, a 16-bit short
SR            = 44100                  #  sample rate
MAX_AMP       = (2**(8*sampleWidth - 1) - 1)    #maximum amplitude is 2**15 - 1  = 32767
MIN_AMP       = -(2**(8*sampleWidth - 1))       #min amp is -2**15


"""
 Basic utilities
"""

# round to 4 decimal places

# round to 4 decimal places

def round4(x):
    return round(float(x)+0.00000000001,4)
    
def roundList(S):
    return [round4(s) for s in S]
    
def gcd(n,m):
    if(m == 0):
        return n
    else:
        return gcd( m, n % m)
        
def lcm(n,m):
    return (n*m)//gcd(n,m)
    
def clipZero(x):
    return (x if(x >= 0) else 0)
    
    
def clip(x):
    if(x > MAX_AMP):
        return MAX_AMP
    elif(x < MIN_AMP):
        return MIN_AMP
    else:
        return int(x)
        
def multSignals(X,Y):
    Z = [0]*len(X)
    for k in range(len(X)):
        Z[k] = X[k]*Y[k]
    return Z
    
def sumSignals(X,Y):
    Z = [0]*len(X)
    for k in range(len(X)):
        Z[k] = X[k]+Y[k]
    return Z
    
def dotProduct(X,Y):
    sum = 0.0
    for k in range(len(X)):
        sum += X[k]*Y[k]
    return sum
        
"""
 File I/O        
"""

# Read a wave file and return the entire file as a standard array
# infile = filename of input Wave file
# If you set withParams to True, it will return the parameters of the input file

def readWaveFile(infile,withParams=False,asNumpy=True):
    with contextlib.closing(wave.open(infile)) as f:
        params = f.getparams()
        frames = f.readframes(params[3])
        if(params[0] != 1):
            print("Warning in reading file: must be a mono file!")
        if(params[1] != 2):
            print("Warning in reading file: must be 16-bit sample type!")
        if(params[2] != 44100):
            print("Warning in reading file: must be 44100 sample rate!")
    if asNumpy:
        X = array.array('h', frames)
        X = np.array(X,dtype='int16')
    else:  
        X = array.array('h', frames)
    if withParams:
        return X,params
    else:
        return X


#  Symmetric to last one, write an array of ints out to a file named
#  fname; if values exceed 16 bit int range, will be clipped.
        
def writeWaveFile(fname, X):
    X = [clip(x) for x in X]
    params = [1,2, SR , len(X), "NONE", None]
    data = array.array("h",X)
    with contextlib.closing(wave.open(fname, "w")) as f:
        f.setparams(params)
        f.writeframes(data.tobytes())
    print(fname + " written.")
    


def makeSignal(spectrum, duration):
    X = [0]*int(SR*duration)  
    for (f,A,phi) in spectrum:
        for i in range(len(X)):           
            X[i] += MAX_AMP * A * np.sin( 2 * np.pi * f * i / SR + phi)
    return X


"""    
 Display a signal with various options
   
   X is an array of samples
   xUnits are scale of x axis: "Seconds" (default), "Milliseconds", or "Samples"
   yUnits are "Relative" [-1..1] (default) or "Absolute" [-MAX_AMP-1 .. MAX_AMP])
   left and right delimit range of signal displayed: [left .. right) in xUnits
   width is width of figure (height is 3)


"""

def displaySignal(X, left = 0, right = -1, title='Signal Window for X',xUnits = "Seconds", yUnits = "Relative",width=10):

        
    minAmplitude = -(2**15 + 100)        # just to improve visibility of curve
    maxAmplitude = 2**15 + 300    
    
    if(xUnits == "Samples"):
        if(right == -1):
            right = len(X)
        T = range(left,right)
        Y = X[left:right]
    elif(xUnits == "Seconds"):
        if(right == -1):
            right = len(X)/44100
        T = np.arange(left, right, 1/44100)
        leftSampleNum = int(left*44100)
        Y = X[leftSampleNum:(leftSampleNum + len(T))]
    elif(xUnits == "Milliseconds"):
        if(right == -1):
            right = len(X)/44.1
        T = np.arange(left, right, 1/44.1)
        leftSampleNum = int(left*44.1)
        Y = X[leftSampleNum:(leftSampleNum + len(T))]
    else:
        print("Illegal value for xUnits")
        
    if(yUnits == "Relative"):
        minAmplitude = -1.003            # just to improve visibility of curve
        maxAmplitude = 1.01
        Y = [x/32767 for x in Y]

    fig = plt.figure(figsize=(width,4))   # Set x and y dimensions of window: may need to redo for your display
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = plt.axes()
    ax.set_xlabel(xUnits)
    ax.set_ylabel(yUnits + ' Amplitude')
    ax.set_ylim([minAmplitude,maxAmplitude])
    ax.set_xlim([left, right])
    plt.axhline(0, color='black')      # draw the 0 line in black
    plt.plot(T,Y) 
    if(    (xUnits == "Samples" and (right - left) < 51)
        or (xUnits == "Seconds" and (right - left) < 0.001)
        or (xUnits == "Milliseconds" and (right - left) < 1) ):
            plt.plot(T,Y, 'bo')                     
    plt.grid(True)                     # if you want dotted grid lines
    plt.show()

    

def makeSpectrum(instr,freq=220):
    if(instr=="triangle"):
        return np.array([(freq,1.0,0.0),     # triples will be converted arrays
        (freq*3,-1/(9),0.0), 
        (freq*5,1/(25),0.0), 
        (freq*7,-1/(49),0.0), 
        (freq*9,1/(81),0.0), 
        (freq*11,-1/(121),0.0), 
        (freq*13,1/(13*13),0.0)])
    elif(instr=="square"):
        return np.array([(freq,2/(np.pi),0.0), 
        (freq*3,2/(3*np.pi),0.0), 
        (freq*5,2/(5*np.pi),0.0), 
        (freq*7,2/(7*np.pi),0.0), 
        (freq*9,2/(9*np.pi),0.0), 
        (freq*11,2/(11*np.pi),0.0), 
        (freq*13,2/(13*np.pi),0.0),
        (freq*15,2/(15*np.pi),0.0),
        (freq*17,2/(17*np.pi),0.0),
        (freq*19,2/(19*np.pi),0.0),
        (freq*21,2/(21*np.pi),0.0)])
    elif(instr=="clarinet"):
        return np.array([(freq,0.314,0.0), 
        (freq*3,.236,0.0), 
        (freq*5,0.157,0.0), 
        (freq*7,0.044,0.0), 
        (freq*9,0.157,0.0), 
        (freq*11,0.038,0.0), 
        (freq*13,0.053,0.0)] ) 
    elif(instr=="bell"):
        return np.array([(freq,0.1666,0.0), 
        (freq*2,0.1666,0.0), 
        (freq*3,0.1666,0.0), 
        (freq*4.2,0.1666,0.0), 
        (freq*5.4,0.1666,0.0), 
        (freq*6.8,0.1666,0.0)])  
    elif(instr=="steelstring"):
        return np.array([(freq*0.7272, .00278,0.0),
                (freq, .0598,0.0),
                (freq*2, .2554,0.0),
                (freq*3, .0685,0.0),
                (freq*4, .0029,0.0),
                (freq*5, .0126,0.0),
                (freq*6, .0154,0.0),
                (freq*7, .0066,0.0),
                (freq*8, .0033,0.0),
                (freq*11.0455, .0029,0.0),
                (freq*12.0455, .0094,0.0),
                (freq*13.0455, .0010,0.0),
                (freq*14.0455, .0106,0.0),
                (freq*15.0455, .0038,0.0)])
    else:
        return np.array([])   
    

  
# wrapper around numpy fft to produce real spectrum
# This will produce

def realFFT(X):
    return 2*abs(np.fft.rfft(X))/len(X)
       
# return the phase spectrum

def phaseFFT(X):
    return [np.angle(x) for x in np.fft.rfft(X)]
    
# return fft coefficients in polar form

def polarFFT(X):
    return [(abs(2*x/len(X)), np.angle(2*x/len(X))) for x in np.fft.rfft(X)]

def spectrumFFT(X):
    S = []
    R = np.fft.rfft(X)
    WR = 44100/len(X)
    for i in range(len(R)):
        S.append( ( i*WR, 2.0 * np.absolute(R[i])/len(X),np.angle(R[i]) ))
    return S
 
# This takes a list of frequencies F (for the x axis) and a list of 
#   corresponding amplitudes S
 
def displaySpectrum(F,S=[],relative=False, labels=True, printSpectrum=True, logscaleX = False, logscaleY = False):
    fig = plt.figure(figsize=(10,3))          # Set x and y dimensions of window: may need to redo for your display
    fig.suptitle('Spectrum', fontsize=14, fontweight='bold')
    ax = plt.axes()
    
    # convert from pairs or triples to F and S

    if(type(F[0])==tuple or type(F[0])==list):
        if(len(F[0]) == 3):
            S = [a for (f,a,phi) in F]
            F= [f for (f,a,phi) in F]
        elif(len(F[0]) == 2):
            S = [a for (f,a) in F]
            F= [f for (f,a) in F]

    # cleanup by removing all close to 0 if only a few above 0
    count = 0           
    for i in range(len(F)):
        if(S[i] >= 0.001 or S[i] <= -0.001):
            count +=1
            
    if(count <= 20):
        tempS = S
        tempF = F
        S = []
        F = []
        for k in range(len(tempS)):
            if(tempS[k] >= 0.001 or tempS[k] <= -0.001):
                S.append(tempS[k])
                F.append(tempF[k])
                
    if(relative and max(S) > 100):
        for k in range(len(S)):
            S[k] = S[k]/MAX_AMP
            
    S = roundList(S)            # round to 4 decimal places
                
    if(logscaleX):
        ax.set_xscale('log')
        minX = 10
        maxX = 22050
    else:
        if(max(F) < 0):
            maxX = 0.0
        else:
            maxX = min(SR/2,max(F) * 1.2)
        if(min(F) < 0):          # negative frequencies
            minX = min(F) * 1.2
        else:
            minX = 0

    if(logscaleY):
        ax.set_yscale('log')
        minY = 1
        maxY = 32767
    else:
        if(min(S) < 0):          # negative amplitudes
            minY = min(S) * 1.2
            plt.plot([minX,maxX],[0,0],color='k', linestyle='-', linewidth=1)
        else:
            minY = 0
        if(max(S) < 0):
            maxY = 0.0
        else:
            maxY = max(S) * 1.2
        
    ax.set_xlim([minX,maxX])
    ax.set_ylim([minY,maxY])
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')
    width = maxX - minX
    
    numBins = 50     # upper bound for when print lollipop spectrum
    
    if(len(F) <= numBins):
        for i in range(len(F)):
            if(S[i] > 0.0):
                plt.plot([F[i], F[i]], [0,S[i]], color='k', linestyle='-', linewidth=1)
                plt.plot([F[i]], [S[i]],'ro')
                if(labels):
                    plt.text(F[i]+width/70,S[i], str(S[i]),fontsize=8)
            elif(S[i] < 0.0):
                plt.plot([F[i], F[i]], [S[i],0], color='k', linestyle='-', linewidth=1)
                plt.plot([F[i]], [S[i]],'ro')
                if(labels):
                    plt.text(F[i]+width/70,S[i], str(S[i]),fontsize=8)
    else:
        ax.set_xlim([minX/1.2,maxX/1.2+10])
        plt.plot(F,S)
        
    plt.show()

    if(printSpectrum and len(F) <= numBins):
        print('Freq\tAmp')
        for k in range(len(F)):
            if(S[k] != 0.0):
                print(str(F[k]) + '\t' + str(S[k]))
        print()
    elif(printSpectrum):
        print('Spectrum too large to print -- more than ' + str(numBins) + ' bins.')


  
   
def displayLollipopSpectrum(F,S,logscaleX = False, logscaleY = False, printSpectrum = False):
    fig = plt.figure(figsize=(10,3))          # Set x and y dimensions of window: may need to redo for your display
    fig.suptitle('Spectrum', fontsize=14, fontweight='bold')
    ax = plt.axes()
    if (max(S) > 2):
        S = [s/32767 for s in S]
    if(logscaleX):
        ax.set_xscale('log')
    if(logscaleY):
        ax.set_yscale('log')
    rangeF = max(F) - min(F)
    rangeS = max(S) - min(S)
    if(logscaleX):
        ax.set_xlim([1,max(F)+(rangeF/10.0)])
    else:
        ax.set_xlim([0,max(F)+(rangeF/10.0)])
    ax.set_ylim([0,max(S)+(rangeS/10.0)])
    if(logscaleX):
        ax.set_xlabel('Frequency (Log Scale)')
    else:
        ax.set_xlabel('Frequency')
    if(logscaleY):
        ax.set_ylabel('Amplitude (Log Scale)')
    else:
        ax.set_ylabel('Amplitude')

    for i in range(len(F)):
        if(S[i] > 0.0001):
            plt.plot([F[i], F[i]], [0,S[i]], color='k', linestyle='-', linewidth=1)
            plt.plot([F[i]], [S[i]],'ro')
    plt.show()
    
    if(printSpectrum):
        print("\nFreq\tAmp\n")
        for f in range(len(S)):
            if(abs(S[f]) > 0.01):
                print(str(F[f]) + "\t" + str(round4(S[f])))
        print()
      
    

# display the spectrum of the signal window X
# The frequency bins will be for frequencies 0, w, 2w, ...., up to Nyquist Limit
# where w = 44100/len(X) = frequency of a sine wave whose period = length of X.

def analyzeSpectrum(X,limit=5000,relative = True,logscaleX = False, logscaleY = False):
    S = realFFT(X)
    incr = SR/((len(S)-1)*2)
    F = [i*incr for i in range(len(S))]
    # now cleanup spectrum by removing all values close to 0.0
    lim = int(limit/F[1])
    if(not logscaleX):
        F = F[:lim]
        S = S[:lim]
    displaySpectrum(F,S,True,True,printSpectrum=False)
    
    