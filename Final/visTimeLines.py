import pdb
import matplotlib.pyplot as plt
import numpy as np

def buildTimeline(**kwargs):
  SaveString = ""
  for k,v in kwargs.items():
    SaveString+=str(k)+"_"+str(v)+"_"
  projNum     = kwargs['projNum'] 
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


