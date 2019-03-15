import pandas as pd
import matplotlib.pyplot as plt
import pdb
from PIL import Image
import imageio
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

def getTrainingData(expNum=2):
    if expNum == 2:
        trainDF1   = pd.read_csv("Data/Part2_1_Train.csv")
        testDF1    = pd.read_csv("Data/Part2_1_Test.csv")
        trainDF3   = pd.read_csv("Data/Part2_3_Train.csv")
        testDF3    = pd.read_csv("Data/Part2_3_Test.csv")

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

        numOfEachClass =20
        for o in range(0,10):
            tempTrain    = pd.read_csv("Data/Part3_"+str(o)+"_Train.csv").values[0:numOfEachClass]
            tempTest     = pd.read_csv("Data/Part3_"+str(o)+"_Test.csv").values[0:numOfEachClass]
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
    def __init__(self, inSize, outSize, activation, initialization, freezeP):
        self.inSize  = inSize
        self.outSize = outSize
        self.freezeP = freezeP
        self.bias    = 0#TODO incorporate Bias

        #handle initialization
        if initialization == 'uniform':
            self.weights = np.random.random((outSize, inSize)).astype(np.float64)/(inSize*outSize) 
        if initialization == 'normal':
            self.weights = np.random.normal(size=(outSize, inSize)).astype(np.float64)/(inSize*outSize)
        if initialization == 'He':
            self.weights = np.random.randn(outSize, inSize)*np.sqrt(2/inSize)
        if initialization == 'Xav':
            self.weights = np.random.randn(outSize, inSize)*np.sqrt(2/(inSize+outSize))
        if freezeP:
            self.weights = np.identity(outSize)
        #used in back prop
        self.lastInvec = 'err';

        #handle activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
        if activation =='tanh':
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
        del_W           = np.zeros(self.weights.shape)
        repInVec       = np.repeat([self.lastInvec], self.outSize, axis=0)
        repChainRule   = np.transpose(np.repeat([chainRule], self.inSize, axis=0))
        del_W          = repInVec*repChainRule
        #updates weight for w
        if self.freezeP==False:
            self.weights -= lr* del_W; 
        # returns dE/dz
        del_V         =  repChainRule*self.weights
        return  del_V


 
class conv_filter_alt:
    def __init__(self, size, initialization, activation, numFilters):
        self.size      = size;

        #handling initialization strategy
        if initialization =='He':
            self.kernel  = np.random.randn(numFilters, size,size)*np.sqrt(2/size)
        if initialization  =='uniform':
            self.kernel    = np.random.random((numFilters, size, size)).astype(np.float64)/(size*size)  #TODO make dependent on initStrat 
            self.bias      = np.random.random(1)
        if initialization  =='normal':
            self.kernel    = np.random.normal(size=(numFilters, size, size)).astype(np.float64)/(size*size)
            self.bias      = np.random.normal(1)
        if initialization  =='Xav':
            self.kernel = np.random.randn(numFilters, size, size)*np.sqrt(2/(size+size))
            self.bias      = np.random.normal(1)
        #used in backprop
        self.numFilters = numFilters
        self.lastInVec = 'err'

        #handling activation function
        if activation=='sigmoid':
            self.act = sigmoid;
        if activation=='tanh':
            self.act = tanhActivation

    def forwardPass(self, inVec, Eval=False):
        outputVal = np.ones((self.numFilters, inVec.shape[0]-self.kernel.shape[1]+1, inVec.shape[1]-self.kernel.shape[2]+1))
        if Eval==False:
            self.lastInVec = inVec
        for l in range(0, self.numFilters):
            for i in range(0, inVec.shape[0]-self.kernel.shape[1]+1):
                for j in range(0, inVec.shape[1]-self.kernel.shape[2]+1):
                    matProduct = inVec[i:i+self.kernel.shape[1], j:j+self.kernel.shape[2]]* self.kernel[l] #element wise product
                    outputVal[l][i][j] = np.sum(matProduct)
        return self.act(outputVal) #CHECK THIS OUT

    def backProp(self, chainRulePrefix, lr):
        del_act   = self.act(chainRulePrefix, derive=True)
        chainRule = del_act * chainRulePrefix
        del_A     = np.zeros(self.kernel.shape)
        for l in range(0, self.numFilters):
            for m in range(0, chainRulePrefix.shape[1]):
                for n in range(0, chainRulePrefix.shape[2]):
                    for i in range(0, self.kernel.shape[1]):
                        for j in range(0, self.kernel.shape[2]):
                           del_A[l][i,j] +=  self.lastInVec[m+i,n+j]*chainRulePrefix[l][m,n]
        #update kernel weights
        self.kernel -= lr*del_A
    
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

def saveImagesAndTimeLines(numFilters,filterW, trainAccTimeline, trainLossTimeline,
                            testAccTimeline, testLossTimeline,SaveString):
    cmaps = plt.colormaps()
    for l in range(0, numFilters):
        plt.imshow(filterW.kernel[l], cmap=cmaps[2*l])#, interpolation='nearest')
        plt.savefig("Filters/part3/filter"+str(l)+"_"+SaveString+".jpg")

    #Save Timelines
    np.save("Timelines/part3/trainAcc"+SaveString, trainAccTimeline)
    np.save("Timelines/part3/trainLoss"+SaveString, trainLossTimeline)
    np.save("Timelines/part3/testAcc"+SaveString, testAccTimeline)
    np.save("Timelines/part3/testLoss"+SaveString, testLossTimeline)


def getReceptiveField(inputDim, filterDim):
	return inputDim-filterDim+1

def setUpNetwork(netParams):
	conv_filter =  conv_filter_alt(filterDim = netParams["filterDim"],
			initialization=netParams["initialization"], 
			activation=netParams["activation_function"],
			numFilters=netParams["numFilters"])

def evaluateNetwork(epochs,
                    data_presentation,
                    order, 
                    activation_function,
                    initialization,
                    learning_rate,
                    project_part,
                    loss_fn,
                    freezeP,
                    filterDim,
                    numFilters,
                    trial_num):
    train_data, test_data, train_labels, test_labels = getTrainingData(project_part)
    numTrainSamples = train_data.shape[0]
    numTestSamples  = test_data.shape[0]

    inputDim     = 28
    filterW      = conv_filter_alt(filterDim, initialization=initialization, activation=activation_function, numFilters=numFilters)
    rfM          = inputDim - filterDim+1  #receptive field Height
    rfN          = inputDim - filterDim+1  #receptive field Width
    if project_part==2:
        dimVecOutput = 2
    if project_part==3:
        dimVecOutput = 10
    combineLayer = combo((numFilters, rfM, rfN), dimVecOutput)# num
    dimVecInput  = numFilters*rfM*rfN
    outP       = perceptron(dimVecInput, dimVecOutput, initialization=initialization, activation=activation_function, freezeP=freezeP)#numfilters, numOutputs


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

            VISUALIZE_DATA=False
            if VISUALIZE_DATA:
                sample = np.flip(sample)
                sample = np.rot90(sample)
                sample = np.rot90(sample)
                imageio.imwrite("SanityCheckSample/train_"+str(sIndx)+"label_"+str(label)+".jpg", sample)

            #Forward Pass Train Data
            filterV   = filterW.forwardPass(sample)
            V         = combineLayer.forwardPass(filterV)#innerV
            output    = outP.forwardPass(V)

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
                    filterW.backProp(del_X1[r], learning_rate)
     
       for tIndx in range(0, test_data.shape[0]):
           #get data
           sample = test_data[tIndx].reshape(inputDim, inputDim)
           label  = test_labels[tIndx]

           if VISUALIZE_DATA:
                sample = np.flip(sample)
                sample = np.rot90(sample)
                sample = np.rot90(sample)
                imageio.imwrite("SanityCheckSample/test_"+str(tIndx)+"label_"+str(label)+".jpg", sample)
           #Forward Pass Test Data
           filterV  = filterW.forwardPass(sample, Eval=True)
           V       = combineLayer.forwardPass(filterV, Eval=True)#innerV
           output  = outP.forwardPass(V, Eval=True)

           #Test Error Evaluation
           testLoss = SquaredError(output, label)
           EpochTestLoss += testLoss
           testAcc = decisionLayer(output, label)
           EpochTestAcc += testAcc

       print("Ep:"+str(epoch)+"\ttrL:"+
               str(EpochTrainLoss)+"\ttrA:"+
               str(EpochTrainAcc/numTrainSamples)+"\ttrAR:"+
               str(EpochTrainAcc)+"\tteL:"+
               str(EpochTestLoss)+"\tteA:"+
               str(EpochTestAcc/numTestSamples)+"\tteAR:"+
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
                "_epoch_"+str(epochs)+""
                "_freezeP_"+str(freezeP)+""
                "_filterDim_"+str(filterDim)+""
                "_numFilter_"+str(numFilters)+""
                "_trial_"+str(trial_num))

    
    saveImagesAndTimeLines(numFilters,filterW,
                            trainAccTimeline, trainLossTimeline,
                            testAccTimeline, testLossTimeline,
                            SaveString)

    '''
    #Save images
    import matplotlib.pyplot as plt
    #cmaps=['autumn','summer','winter', 'spring']
    cmaps = plt.colormaps()
    for l in range(0, numFilters):
        plt.imshow(filterW.kernel[l], cmap=cmaps[2*l])#, interpolation='nearest')
        plt.savefig("Filters/part3/filter"+str(l)+"_"+SaveString+".jpg")

    #Save Timelines
    np.save("Timelines/part3/trainAcc"+SaveString, trainAccTimeline)
    np.save("Timelines/part3/trainLoss"+SaveString, trainLossTimeline)
    np.save("Timelines/part3/testAcc"+SaveString, testAccTimeline)
    np.save("Timelines/part3/testLoss"+SaveString, testLossTimeline)
    '''
if __name__=="__main__":
    #TODO data_presentation   = 'serial', 'batch', 'mini-batch'
    #order               = 'fixed', 'shuffle'
    #activation_function = 'tanh', 'sigmoid'
    #TODO initialization      = 'normal', 'uniform'
    #learning_rate       = 0.01, 0.001, 0.0001

    learning_rates       = [0.01, 0.001, 0.0001]
    initializations      = ['uniform']
    activation_functions = ['tanh']
    orders               = ['fixed','shuffle']

    epochs              = 1000
    order               = 'shuffle'
    activation_function = 'tanh'
    learning_rate       = 0.0001
    initialization      = 'uniform'
    data_presentation   = 'serial'
    loss_fn             = 'SquaredError'
    project_part        = 2
    trial_num           = 2
    freezeP             = False #can only be done if num filters = num outputs to error function
    filterDim           = 28
    numFilters          = 2


    evaluateNetwork(epochs = epochs,order=order,
                    activation_function=activation_function,
                    learning_rate=learning_rate,
                    initialization=initialization,
                    data_presentation=data_presentation,
                    loss_fn =loss_fn,
                    project_part=project_part,
                    trial_num=trial_num,
                    freezeP=freezeP,
                    filterDim=filterDim,
                   numFilters=numFilters)

  #  pdb.set_trace()
    for learning_rate in learning_rates:
        for initialization in initializations:
            for activation_function in activation_functions:
                for order in orders:
                    evaluateNetwork(epochs = epochs,order=order,
                                    activation_function=activation_function,
                                    learning_rate=learning_rate,
                                    initialization=initialization,
                                    data_presentation=data_presentation,
                                    loss_fn =loss_fn,
                                    project_part=project_part,
                                    trial_num=trial_num,
                                    freezeP=freezeP,
                                    filterDim=filterDim,
                                    numFilters=numFilters)
