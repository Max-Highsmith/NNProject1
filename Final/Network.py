import matplotlib.pyplot as plt
import imageio
import pandas as pd
import pdb
import numpy as np


def saveImagesAndTimeLines(numFilters,filterW, trainAccTimeline, trainLossTimeline,
                            testAccTimeline, testLossTimeline, kwargs):
    SaveString=""
    for k,v in kwargs.items():
      SaveString+=str(k)+"_"+str(v)+"_"
    cmaps = plt.colormaps()
    for l in range(0, numFilters):
        plt.imshow(filterW[l], cmap=cmaps[2*l])#, interpolation='nearest')
        plt.savefig("Filters/part"+str(kwargs['ExNum'])+"/filter"+str(l)+"_"+SaveString+".jpg")

    #Save Timelines
    np.save("Timelines/part"+str(kwargs['ExNum'])+"/trainAcc"+SaveString, trainAccTimeline)
    np.save("Timelines/part"+str(kwargs['ExNum'])+"/trainLoss"+SaveString, trainLossTimeline)
    np.save("Timelines/part"+str(kwargs['ExNum'])+"/testAcc"+SaveString, testAccTimeline)
    np.save("Timelines/part"+str(kwargs['ExNum'])+"/testLoss"+SaveString, testLossTimeline)


def getTrainingData(expNum=2):
    if expNum == 2:
        trainDF1   = pd.read_csv("Data/part2/Part2_1_Train.csv")
        testDF1    = pd.read_csv("Data/part2/Part2_1_Test.csv")
        trainDF3   = pd.read_csv("Data/part2/Part2_3_Train.csv")
        testDF3    = pd.read_csv("Data/part2/Part2_3_Test.csv")

        trainArr1  = trainDF1.values
        trainArr3  = trainDF3.values
        testArr1   = testDF1.values
        testArr3   = testDF3.values

        #[1,0] = 1
        #[0,1] = 3
        train_labels = np.vstack((np.repeat([[1,0]], trainArr1.shape[0],0),
                                 np.repeat([[0,1]], trainArr3.shape[0],0)))
        test_labels  = np.vstack((np.repeat([[1,0]], testArr1.shape[0], 0),
                                 np.repeat([[0,1]], testArr3.shape[0],0)))

        train_data = np.vstack((trainArr1, trainArr3))
        test_data  = np.vstack((testArr1, testArr3))
        return train_data, test_data, train_labels, test_labels

    if expNum ==3:
        trainList    =[]
        testList     =[]
        trainLabList =[]
        testLabList  =[]

        numOfEachClass =10
        for o in range(0,10):
            tempTrain    = pd.read_csv("Data/part3/Part3_"+str(o)+"_Train.csv").values[0:numOfEachClass]
            tempTest     = pd.read_csv("Data/part3/Part3_"+str(o)+"_Test.csv").values[0:numOfEachClass]
            hotEncode    = np.repeat([0],10)
            hotEncode[o] = 1
            tempTrainLab = np.repeat([hotEncode], numOfEachClass,0) #tempTrain.shape[0],0)
            tempTestLab  = np.repeat([hotEncode], numOfEachClass,0) #tempTest.shape[0],0)

            trainList.append(tempTrain)
            testList.append(tempTest)
            trainLabList.append(tempTrainLab)
            testLabList.append(tempTestLab)

        train_data   = np.vstack(trainList)
        test_data    = np.vstack(testList)
        train_labels = np.vstack(trainLabList)
        test_labels  = np.vstack(testLabList)
        return train_data, test_data, train_labels, test_labels


def decisionLayer(inVec, label):
    if inVec.shape != label.shape:
        print("dimension error")
    predict = np.zeros(inVec.shape)
    predict[np.argmax(inVec)] = 1
    if np.equal(predict, label).all():
        return 1
    else:
        return 0

def sigmoidAct(x, derive=False):
  if derive:
    return x*(1-x)
  return 1/(1+np.exp(-1*x))

def tanhAct(x, derive=False):
  if derive:
    return 1-(np.power(x,2))
  return np.tanh(x)

def reluAct(x, derive=False):
  if derive:
    y = np.ones(x.shape)
    fi = (x<=0)
    y[fi] = 0
    return y
  return np.maximum(x,0)

def noAct(x, derive=False):
  if derive:
    return np.ones(x.shape)
  return x

def SquaredError(guess, label, derive=False):
  if derive:
    return (guess-label)
  return (1.0/2.0)*np.sum(np.power((guess-label), 2.0))


#sanity check
'''
T=1
L=2
I=2
J=2
C=3
D=3
M=C-I+1
N=D-J+1
K=M*N*L
R=2
W = np.array([[1,1,1,1,1,1,1,1],[0,1,2,0,1,2,0,1]])
A = np.array([[[1,1],[1,1]],[[1,0],[0,2]]])
train_data =np.array([[[1,2,3],[4,5,6],[7,8,9]]])
train_labels=np.array([[150,125]])
ACT = noAct
'''
def runNetwork(**kwargs):
  Epochs = kwargs['Epochs']
  ExNum  = kwargs['ExNum']
  T      = kwargs['T']     # num training samples
  TT     = kwargs['TT']
  L      = kwargs['L']     # num layers
  I      = kwargs['I']     # index of conv kernel height
  J      = kwargs['J']     # index of conv kernel width
  C      = kwargs['C']     # index of Image height
  D      = kwargs['D']     # index of Image width
  M      = C-I+1           # index of receptive field height #TODO includes no padding or stride
  N      = D-J+1           # index of receptive field width
  K      = M*N*L           # inputs to Perceptron 
  R      = kwargs['R']     # outpus of Perceptron
  lrW    = kwargs['lrW']   # learning rate Perceptron
  lrA    = kwargs['lrA']   # learning rate Conv Layer 
  ACT    = kwargs['ACT']
  LOSS   = kwargs['LOSS']
  ORDER  = kwargs['ORDER']
  vis    = kwargs['vis']
  INIT   = kwargs['INIT']
  
  #SEt LOSS AND ACT
  if ACT == 'tanhAct':
    ACT = tanhAct
  if ACT =='sigmoidAct':
    ACT = sigmoidAct
  if ACT =='noAct':
    ACT = noAct
  if ACT == 'reluAct':
    ACT = reluAct

  if LOSS=='SquaredError':
    LOSS = SquaredError

  #Initialize Layers
  np.random.seed(0)
  # W      = np.random.random((R,K))
  if INIT == "Freeze":
    W      = np.array([[1,0],[0,1]])
    A      = np.random.random((L,I,J))/(L*I*J)
  if INIT == "Uniform":
    W      = np.random.random((R,K))/R
    A      = np.random.random((L,I,J))/(I*J)
  trainLossTimeline = []
  trainAccTimeline  = []
  testLossTimeline  = []
  testAccTimeline   = []
  train_data, test_data, train_labels, test_labels = getTrainingData(ExNum)



  for epoch in range(0, Epochs):
    EpochTrainLoss  = 0
    EpochTrainAcc   = 0
    EpochTestLoss   = 0
    EpochTestAcc    = 0
    trainIndexList = list(range(0,T))
    if ORDER == "Random":
      np.random.shuffle(trainIndexList)
    if ORDER == "Alternate":
      trainIndexList=[]
      for t in range(0,int(T/2)-1):
        trainIndexList.append(t)
        trainIndexList.append(T-t)
    if ORDER == "Fixed":
      pass
    for t in trainIndexList:
      #print(str(epoch)+":"+str(t)) 

      P   = train_labels[t]
      X_0 = train_data[t]
      X_0 = X_0.reshape((C,D))

 
      assert X_0.shape==(C,D)


      VISUALIZE_DATA=vis
      if VISUALIZE_DATA:
        imageio.imwrite("SanityCheckSample/part"+str(ExNum)+"/train_"+str(t)+"label_"+str(P)+".png", X_0)




    #ConvLayer
      Z_0 = np.zeros((L,M,N))
      for n in range(0,N):
        for m in range(0,M):
          for l in range(0,L):
            for i in range(0,I):
              for j in range(0,J):
                assert m+i<C
                Z_0[l,m,n] += A[l,i,j]*X_0[m+i,n+j]
    
      #Activation 1
      X_1 = ACT(Z_0)
      assert X_1.shape == Z_0.shape
      #Reshape
      V   = X_1.reshape(K)
      #Forward Pass Perceptron
      Z_1 = np.zeros((R))
      for k in range(0,K):
        for r in range(0,R):
          Z_1[r] += V[k]*W[r,k]
    
      #Activation 2
      X_2 = ACT(Z_1)

      E        = LOSS(X_2, P)
      decision = decisionLayer(X_2, P)
  
      EpochTrainLoss+=E
      EpochTrainAcc+=decision
      #Back Prop
      dE_dXr   =  LOSS(X_2, P, derive=True)
      assert dE_dXr.shape == (R,)
      dXr_dZr  =  ACT(X_2, derive=True)
      assert dXr_dZr.shape ==(R,)
      dZr_Wrk  = np.zeros((R,K))

      for r in range(0, R):
        for k in range(0, K):
          dZr_Wrk[r,k] = V[k]
   
      assert dZr_Wrk.shape ==(R,K)
      dE_dW    = np.ones((R,K))

      dZr_dVk = W      
      dE_V    = np.zeros(V.shape)
      #BackProp Perceptron
      for r in range(0,R):
        for k in range(0,K):
          dE_dW[r,k] = dE_dXr[r]*dXr_dZr[r]*dZr_Wrk[r,k]
          dE_V[k]    = dE_dXr[r]*dXr_dZr[r]*dZr_dVk[r,k]


      dE_dA   = np.zeros((L,I,J))

      
      #part 2
      dXlmn_dZlmn = X_1.reshape(L)
      for l in range(0,L):
        for i in range(0,J):
          for i in range(0,I):
             dE_dA[l,i,j] = X_0[i,j]*ACT(dXlmn_dZlmn[l])*dE_dA[l,i,j]           

      #BackProp Conv
      '''
      dZr_dXlmn   = np.zeros((L,I,J,M,N))
      dXlmn_dZlmn = np.zeros((L,I,J,M,N))
      dZlmn_dAlij = np.zeros((L,I,J,M,N))    
    
       
      dE_dX_col    = np.sum(dE_dXr)
      dXr_dZr_col  = np.sum(dXr_dZr)
      W_col        = np.sum(W, 0)
      for l in range(0, L):
        for i in range(0, I):
          for j in range(0, J):
            for m in range(0, M):
              for n in range(0, N):
                k = (n*M*L)+(m*N)+l
                dZr_dXlmn[l,i,j,m,n]   = W_col[k]
                dXlmn_dZlmn[l,i,j,m,n] = ACT(X_1[l,m,n], derive=True) 
                dZlmn_dAlij[l,i,j,m,n] = X_0[m+i,n+j]
                dE_dA[l,i,j] +=  (dXlmn_dZlmn[l,i,j,m,n] * dZlmn_dAlij[l,i,j,m,n] *dZr_dXlmn[l,i,j,m,n] * dE_dX_col*dXr_dZr_col)
      '''
      #apply deltas
      A -= lrA * dE_dA
      if not INIT=="Freeze":
      	W -= lrW * dE_dW

    for t in range(0, TT):
      #print(str(epoch)+":"+str(t)) 

      P   = test_labels[t]
      X_0 = test_data[t]
      X_0 = X_0.reshape((C,D))

      assert X_0.shape==(C,D)

      VISUALIZE_DATA=vis
      if VISUALIZE_DATA:
        imageio.imwrite("SanityCheckSample/part"+str(ExNum)+"/test_"+str(t)+"label_"+str(P)+".jpg", X_0)

    #ConvLayer
      Z_0 = np.zeros((L,M,N))
      for n in range(0,N):
        for m in range(0,M):
          for l in range(0,L):
            for i in range(0,I):
              for j in range(0,J):
                assert m+i<C
                Z_0[l,m,n] += A[l,i,j]*X_0[m+i,n+j]

      #Activation 1
      X_1 = ACT(Z_0)
      #Reshape
      V   = X_1.reshape(K)
      #Forward Pass Perceptron
      Z_1 = np.zeros((R))
      for k in range(0,K):
        for r in range(0,R):
          Z_1[r] += V[k]*W[r,k]

      #Activation 2
      X_2 = ACT(Z_1)

      E        = LOSS(X_2, P)
      decision = decisionLayer(X_2, P)

      EpochTestLoss+=E
      EpochTestAcc+=decision

    print("\nEpoch:"+str(epoch)+"\tTrainLoss:"+str(EpochTrainLoss)+"\tTrainAcc:"+
	str(EpochTrainAcc/T)+"\tTestLoss"+str(EpochTestLoss)+"\tTestAcc:"+
	str(EpochTestAcc/TT))

    trainLossTimeline.append(EpochTrainLoss)
    trainAccTimeline.append(EpochTrainAcc/T)
    testLossTimeline.append(EpochTestLoss)
    testAccTimeline.append(EpochTestAcc/TT)
    #break early
    if (len(trainLossTimeline)>40 and (trainLossTimeline[-30:-1] == trainLossTimeline[-1]).all()):
      break

 
  saveImagesAndTimeLines(L,A, trainAccTimeline, trainLossTimeline,
                            testAccTimeline, testLossTimeline, kwargs)


#Initial Dimension
param ={
'Epochs'  : 20000,
'ExNum'   : 2,
'T'       : 31,    # num training samples
'TT'      : 98,    # num test samples
'L'       : 2,     # num Conv Layers 
'I'       : 28,    # index of conv kernel height
'J'       : 28,    # index of conv kernel width
'C'       : 28,    # index of Image height
'D'       : 28,    # index of Image width
'R'       : 2,    # outpus of Perceptron
'lrW'     : 10, # learning rate Perceptron
'lrA'     : 10,  #learning rate Conv Layer
'ACT'     : "reluAct",
'LOSS'    : "SquaredError",
'vis'     : False,
'ORDER'   : 'Alternate',
'INIT'    : 'Uniform',

}
runNetwork(**param)
'''
for lr in [0.0001,0.001,0.01,0.1]:
  param['lrW']=lr
  param['lrA']=lr
  runNetwork(**param)

for ACT in ['tanhAct','sigmoidAct','noAct', 'reluAct']:
  param['ACT']=ACT
  runNetwork(**param)
'''
print("done")
