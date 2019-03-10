import pandas as pd
import pdb
import numpy as np

#Activation functions
def sigmoid(x, derive=False):
    if derive:
        return x*(1-x)
    return 1/(1+np.exp(-x))

def tanhActivation(x, derive=False):
    if derive:
        return 1-(np.power(x,2))
    return np.tanh(x)

#Error functions
def SquaredError(guess, label, derive=False):
    if derive:
        return (guess - label)
    return (1.0/2.0)*np.sum(np.power((guess-label), 2.0))


def crossEntropy(guess, label, derive=False):
        return -99

    #data_presentation   = 'serial', 'batch', 'mini-batch'
    #order               = 'fixed', 'shuffle'
    #activation_function = 'tanh', 'sigmoid', 'relu'
    #initialization      = 'normal', 'uniform'
    #learning_rate       = 0.01, 0.001, 0.0001
def getTrainingData(expNum=2):
    #/TODO These files are mislabeled. the test.csv are the 3s and the train.csv are the 1's
    #/TODO TODO TODO TODO
    trainDF1   = pd.read_csv("Part2_3_Train.csv")
    testDF1    = pd.read_csv("Part2_1_Train.csv")
    trainDF3   = pd.read_csv("Part2_3_Test.csv")
    testDF3    = pd.read_csv("Part2_1_Test.csv")
    #/TODO TODO TODO TODO

    trainArr1  = trainDF1.values
    trainArr3  = trainDF3.values
    testArr1   = testDF1.values
    testArr3   = testDF3.values

    train_labels = np.vstack((np.repeat([[0,1]], trainArr3.shape[0],0), np.repeat([[1,0]], trainArr1.shape[0],0)))
    test_labels  = np.vstack((np.repeat([[0,1]], testArr3.shape[0], 0), np.repeat([[1,0]], testArr1.shape[0],0)))

    train_data = np.vstack((trainArr1, trainArr3))
    test_data  = np.vstack((testArr1, testArr3))
    return train_data, test_data, train_labels, test_labels


class combo:
    def __init__(self, inShape, dimR):
        #order should be channels, i, j
        self.inShape   = inShape
        self.outSize   = 1
        self.dimR      = dimR
        self.lastInvec = 'err';
        for i in inShape:
            self.outSize *= i

    def forwardPass(self, inVec, Eval=False):
        if Eval==False:
            self.lastInvec = inVec
        return inVec.reshape(self.outSize, order='C')
    
    def backProp(self, chainRulePrefix): #TODO
        return chainRulePrefix.reshape((self.dimR,)+self.inShape, order='C')

class perceptron:
    def __init__(self, inSize, outSize, activ='tanh'):
        self.inSize  = inSize
        self.outSize = outSize
        self.bias    = 0#TODO incorporate Bias
        self.weights = np.random.random((outSize, inSize)).astype(np.float64)/(inSize*outSize) #TODO varry initStrat
        self.lastInvec = 'err';
        if activ == 'sigmoid':
            self.activation = sigmoid
        if activ =='tanh':
            self.activation = tanhActivation

    def forwardPass(self, inVec, Eval=False):
        if Eval==False:
            self.lastInvec = inVec
        z =  np.matmul(self.weights, inVec) + self.bias
        x =  self.activation(z)
        return x

    def backProp(self, chainRulePrefix, lr):
        del_act        = self.activation(chainRulePrefix, derive=True)
        chainRule      = del_act * chainRulePrefix
        #evaluates dW/dError
        del_W  = np.zeros(self.weights.shape)
        repInVec       = np.repeat([self.lastInvec], self.outSize, axis=0)
        repChainRule   = np.transpose(np.repeat([chainRule], self.inSize, axis=0))
        del_W          = repInVec*repChainRule
        #updates weight for w
        self.weights -= lr* del_W; 
        # returns dE/dz
        del_V         =  repChainRule*self.weights
        return  del_V
        
class conv_filter:
    def __init__(self, size, initStrat, numOfFilters = 1,  activation='tanh'):
        self.size      = size;
        self.kernel    = np.random.random((size, size)).astype(np.float64)/(size*size)  #TODO make dependent on initStrat 
        self.bias      = np.random.random(1)
        self.act       = activation
        self.lastInVec = 'err'
        if activation=='sigmoid':
            self.act = sigmoid;
        if activation=='tanh':
            self.act = tanhActivation
    def forwardPass(self, inVec, Eval=False):
        outputVal = np.ones((inVec.shape[0]-self.kernel.shape[0]+1, inVec.shape[1]-self.kernel.shape[1]+1))
        if Eval==False:
            self.lastInVec = inVec
        for i in range(0, inVec.shape[0]-self.kernel.shape[0]+1):
            for j in range(0, inVec.shape[1]-self.kernel.shape[1]+1):
                matProduct = inVec[i:i+self.kernel.shape[0], j:j+self.kernel.shape[0]]* self.kernel #element wise product
                outputVal[i][j] = np.sum(matProduct)
            return self.act(outputVal)

    def backProp(self, chainRulePrefix, lr):
        del_act   = self.act(chainRulePrefix, derive=True)
        chainRule = del_act * chainRulePrefix
        del_A     = np.zeros(self.kernel.shape)
        for m in range(0, chainRulePrefix.shape[0]):
            for n in range(0, chainRulePrefix.shape[1]):
                for i in range(0, self.kernel.shape[0]):
                    for j in range(0, self.kernel.shape[1]):
                       del_A[i,j] +=  self.lastInVec[m+i,n+j]*chainRulePrefix[m,n]
        #update kernel weights
        self.kernel -= lr*del_A
    
    def getKernel(self):
        return self.kernel

def decisionLayer(inVec, label):
    if inVec.shape != label.shape:
        pdb.set_trace()
        print("dimension error")
    predict = np.zeros(inVec.shape)
    predict[np.argmax(inVec)] = 1
    if np.equal(predict, label).all():
        return 1
    else:
        return 0

    

def getBatchSize(data_presentation, trainingArray):
    if data_presentation=='serial':
        return trainingArray.shape[0]
    if data_presentation=='minimatch':
        return 7
    if data_presentation=='batch':
        return 1

#data_presentation   = 'serial', 'batch', 'mini-batch'
#order               = 'fixed', 'shuffle'
#activation_function = 'tanh', 'sigmoid', 'relu'
#initialization      = 'normal', 'uniform'
#learning_rate       = 0.01, 0.001, 0.0001
def evaluateNetwork(epochs=1000,
                    data_presentation='serial',
                    order='fixed', 
                    activation_function='tanh',
                    initialization='uniform',
                    learning_rate=0.001,
                    projectPart='2',
                    loss_fn="mse"):
    train_data, test_data, train_labels, test_labels = getTrainingData()
    numTrainSamples = train_data.shape[0]
    numTestSamples  = test_data.shape[0]
    #greenW, orangeW, outputW, greenV, orangeV = initializeWeights(initialization)
    inputDim     = 28
    filterDim    = 28
    greenW       = conv_filter(filterDim, initialization)
    orangeW      = conv_filter(filterDim, initialization)
    numFilters   = 2
    rfM          = inputDim - filterDim+1  #receptive field Height
    rfN          = inputDim - filterDim+1  #receptive field Width
    dimVecOutput = 2
    combineLayer = combo((numFilters, rfM, rfN), dimVecOutput)# num
    dimVecInput  = numFilters*rfM*rfN
    outP       = perceptron(dimVecInput, dimVecOutput)#numfilters, numOutputs
    #Training
    trainLossTimeline = []
    trainAccTimeline  = []
    testLossTimeline  = []
    testAccTimeline   = []
    for epoch in range(0, epochs):
       EpochTrainLoss  = 0
       EpochTrainAcc = 0
       EpochTestLoss   = 0
       EpochTestAcc  = 0
       trainIndexList = list(range(0, train_data.shape[0]))
       if order=='shuffle':
           np.random.shuffle(trainIndexList)
       for sIndx in trainIndexList:
            # get data
            sample    = train_data[sIndx].reshape(inputDim,inputDim)
            label     = train_labels[sIndx]

            #Forward Pass Train Data
            orangeV   = orangeW.forwardPass(sample)
            greenV    = greenW.forwardPass(sample)
            innerV    = np.stack((greenV, orangeV))
            V         = combineLayer.forwardPass(innerV)
            output    = outP.forwardPass(V)

    
            # currentPred = np.zeros(label.shape)
            #currentPred[np.argmax(output)]=1
            #print("label:"+str(label)+"\toutput:"+str(output)+"\tpred:"+str(currentPred))
            
            #Train Error Evaluation
            loss  = SquaredError(output, label)
            EpochTrainLoss += loss
            trainAcc = decisionLayer(output, label)
            EpochTrainAcc += trainAcc

            
            BACKPROPAGATE = True 
            if BACKPROPAGATE:
            #backward Pass
                del_Er    = SquaredError(output, label, derive=True)
                del_V     = outP.backProp(del_Er, learning_rate)
                del_X1    = combineLayer.backProp(del_V)
                for r in range(0, del_X1.shape[0]):
                    orangeW.backProp(del_X1[r][0],learning_rate)
                    greenW.backProp(del_X1[r][1], learning_rate)
     
       for tIndx in range(0, test_data.shape[0]):
           #get data
           sample = test_data[tIndx].reshape(inputDim, inputDim)
           label  = test_labels[tIndx]

           #Forward Pass Test Data
           orangeV = orangeW.forwardPass(sample, Eval=True)
           greenV  = greenW.forwardPass(sample, Eval=True)
           innerV  = np.stack((greenV, orangeV))
           V       = combineLayer.forwardPass(innerV, Eval=True)
           output  = outP.forwardPass(V, Eval=True)

           #Test Error Evaluation
           testLoss = SquaredError(output, label)
           EpochTestLoss += testLoss
           testAcc = decisionLayer(output, label)
           EpochTestAcc += testAcc

       print("Ep:"+str(epoch)+"\ttrL:"+
               str(EpochTrainLoss)+"\ttrA:"+
               str(EpochTrainAcc)+"\tteL:"+
               str(EpochTestLoss)+"\tteA:"+
               str(EpochTestAcc))

       trainLossTimeline.append(EpochTrainLoss)
       trainAccTimeline.append(EpochTrainAcc/numTrainSamples)
       testLossTimeline.append(EpochTestLoss)
       testAccTimeline.append(EpochTestAcc/numTestSamples)



    #Create String
    SaveString = ("_order_"+str(order)+""
                 "_activation_"+ str(activation_function)+""
                "_initialization_"+str(initialization)+""
                "_learning_rate_"+str(learning_rate)+""
                "_dataPres_"+str(data_presentation)+""
                "_loss_"+str(loss_fn)+""
                "_epoch_"+str(epochs))

    #Save images
    import matplotlib.pyplot as plt
    plt.imshow(orangeW.getKernel(), cmap='autumn')#, interpolation='nearest')
    plt.savefig("Filters/orange_"+SaveString+".jpg")
    plt.imshow(greenW.getKernel(), cmap='summer')#, interpolation='nearest')
    plt.savefig("Filters/green_"+SaveString+".jpg")

    #Save Timelines
    np.save("Timelines/trainAcc"+SaveString, trainAccTimeline)
    np.save("Timelines/trainLoss"+SaveString, trainLossTimeline)
    np.save("Timelines/testAcc"+SaveString, testAccTimeline)
    np.save("Timelines/testLoss"+SaveString, testLossTimeline)

if __name__=="__main__":
    #data_presentation   = 'serial', 'batch', 'mini-batch'
    #order               = 'fixed', 'shuffle'
    #activation_function = 'tanh', 'sigmoid', 'relu'
    #initialization      = 'normal', 'uniform'
    #learning_rate       = 0.01, 0.001, 0.0001
    print("ok")
    evaluateNetwork(order='shuffle')
