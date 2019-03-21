import pdb
import matplotlib.pyplot as plt
import numpy as np

def buildTimeline(**kwargs):
  SaveString = ""
  for k,v in kwargs.items():
    SaveString+=str(k)+"_"+str(v)+"_"
  ExNum     = kwargs['ExNum'] 
  ParamString = SaveString+".npy"
  #ParamString = "epochs_1000_dataPres_serial_initi_uniform_lr_0.001_projNum_2_lossFn_SquaredError_freezeP_False_filterDim_28_numFilters_2_trialNum_0_inputDim_28_dimVecOut_2_seed_0_optimized_1_order_fixed_activ_tanh_.npy"
  trainLoss = np.load("Timelines/part"+str(ExNum)+"/trainLoss"+ParamString)
  trainAcc  = np.load("Timelines/part"+str(ExNum)+"/trainAcc"+ParamString)
  testLoss  = np.load("Timelines/part"+str(ExNum)+"/testLoss"+ParamString)
  testAcc   = np.load("Timelines/part"+str(ExNum)+"/testAcc" +ParamString)
  
  plt.clf()
  plt.title("Loss vs Epoch")
  plt.xlabel("Epoch")
  plt.ylabel("MSE")
  plt.plot(trainLoss, label="trainLoss")
  plt.plot(testLoss,  label="testLoss")
  plt.gca().legend()
  plt.savefig("Plots/part"+str(ExNum)+"/"+SaveString+"MSE.png")
  plt.clf()
  plt.title("Accuracy vs Epoch")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.plot(trainAcc, label="trainAcc")
  plt.plot(testAcc, label="testAcc")
  plt.gca().legend()
  plt.savefig("Plots/"+SaveString+"Acc.png")

def getTimeLine(**kwargs):
  SaveString = ""
  for k,v in sorted(kwargs.items()):
    SaveString+=str(k)+"_"+str(v)+"_"
  ExNum     = kwargs['ExNum']
  ParamString = SaveString+".npy"
  #ParamString = "epochs_1000_dataPres_serial_initi_uniform_lr_0.001_projNum_2_lossFn_SquaredError_freezeP_False_filterDim_28_numFilters_2_trialNum_0_inputDim_28_dimVecOut_2_seed_0_optimized_1_order_fixed_activ_tanh_.npy"
  trainLoss = np.load("Timelines/part"+str(ExNum)+"/trainLoss"+ParamString)
  trainAcc  = np.load("Timelines/part"+str(ExNum)+"/trainAcc"+ParamString)
  testLoss  = np.load("Timelines/part"+str(ExNum)+"/testLoss"+ParamString)
  testAcc   = np.load("Timelines/part"+str(ExNum)+"/testAcc" +ParamString)
  return trainLoss, trainAcc, testLoss, testAcc



def GenerateGraphs(GraphName, Param_Variation_Keys, Param_Variation_Values, Varying_Param_String, **baseParam):
  param = baseParam
  ##LEARNING RATE GRAPHS
  TrainLoss = []
  TestLoss  = []
  TrainAcc  = []
  TestAcc   = []
  TimeLines = [[],[],[],[]]
  trl, tra, tel,tea = getTimeLine(**param)
  #Training Loss
  for pv in Param_Variation_Values:
    for it, pk in enumerate(Param_Variation_Keys):
      param[pk] = pv[it]
    
    trl, tra, tel, tea = getTimeLine(**param)
    TimeLines[0].append(trl)
    TimeLines[1].append(tel)
    TimeLines[2].append(tra)
    TimeLines[3].append(tea)
  Graph_Types = ["Train Loss", "Test Loss", "Train Accuracy", "Test Accuracy"]
  for i, gtype in enumerate(Graph_Types): 
    plt.clf()
    plt.title(gtype+" vs Epoch by "+Varying_Param_String)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    for j, cuParam in enumerate(Param_Variation_Values):
      labelStr = ''
      for k, ke in enumerate(Param_Variation_Keys):
        labelStr += str(ke)+"="+str(cuParam[k])+" "
      plt.plot(TimeLines[i][j], label=str(labelStr))
    plt.gca().legend()
    plt.savefig("Plots/part"+str(param['ExNum'])+"/"+GraphName+gtype+".png")
    plt.show()

#Varying Learning Rate Graph
param = {
'Epochs': 200,
'ExNum' : 2,
'T'     : 31,
'TT'    : 98,
'L'     : 2,
'I'     : 28,
'J'     : 28,
'C'     : 28,
'D'     : 28,
'R'     : 2,
'lrA'   : 0.001,
'lrW'   : 0.001,
'ACT'   : 'tanhAct',
'LOSS'  : 'SquaredError',
'vis'   : False,
'ORDER' : 'Random',
'INIT'  : 'Uniform',
'TrialNum':1
}
#GenerateGraphs("Order", ['ORDER'], [['Alternate'], ['Fixed'], ['Random']], "Order", **param)
#GenerateGraphs("Activation", ['ACT'], [['tanhAct'], ['reluAct'], ['sigmoidAct'], ['noAct']], "Activation", **param)
#GenerateGraphs("LearningRate", ['lrA'], [[0.1], [0.01], [0.001], [0.0001]], "Learning Rate", **param)
#GenerateGraphs("Lw=0.1 ", ['lrA','lrW'], [[0.1,0.1], [0.01,0.1], [0.001,0.1], [0.0001,0.1]], "Learning Rate W=0.1", **param)
#GenerateGraphs("Lw=0.01", ['lrA','lrW'], [[0.1,0.01], [0.01,0.01], [0.001,0.01], [0.0001,0.01]], "Learning Rate W=0.01", **param)
#GenerateGraphs("Lw=0.001", ['lrA','lrW'], [[0.1,0.001], [0.01,0.001], [0.001,0.001], [0.0001,0.001]], "Learning Rate W=0.001", **param)
#GenerateGraphs("Lw=0.0001", ['lrA','lrW'], [[0.1,0.0001], [0.01,0.0001], [0.001,0.0001], [0.0001,0.0001]], "Learning Rate W=0.0001", **param)
GenerateGraphs("Lw=0", ['lrA','lrW'], [[0.1,0], [0.01,0], [0.001,0], [0.0001,0]], "Learning Rate W=0", **param)
GenerateGraphs("Lw=0", ['lrA','lrW'], [[0.1,0], [0.01,0], [0.001,0], [0.0001,0]], "Learning Rate W=0", **param)
