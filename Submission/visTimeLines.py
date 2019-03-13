import matplotlib.pyplot as plt
import numpy as np

ParamString = "_order_shuffle_activation_tanh_initialization_uniform_learning_rate_0.0001_dataPres_serial_loss_SquaredError_epoch_100_trial_1.npy"
trainLoss = np.load("Timelines/trainLoss"+ParamString)
trainAcc  = np.load("Timelines/trainAcc"+ParamString)
testLoss  = np.load("Timelines/testLoss"+ParamString)
testAcc   = np.load("Timelines/testAcc" +ParamString)
 
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.plot(trainLoss, label="trainLoss")
plt.plot(testLoss,  label="testLoss")
plt.gca().legend()
plt.show()
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(trainAcc, label="trainAcc")
plt.plot(testAcc, label="testAcc")
plt.gca().legend()
plt.show()
