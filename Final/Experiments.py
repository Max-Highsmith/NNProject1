import Network as nn
'''
param =
{
'Epochs'  : 10,
'ExNum'   : 2,
'T'       : 32,    # num training samples
'TT'      : 98,    # num test samples
'L'       : 2,     # num layers
'I'       : 28,    # index of conv kernel height
'J'       : 28,    # index of conv kernel width
'C'       : 28,    # index of Image height
'D'       : 28,    # index of Image width
'R'       : 2,     # outpus of Perceptron
'lr'      : 0.1, # learning rate
'ACT'     : tanhAct,
'LOSS'    : SquaredError,
'vis'     : False
}

Activations = [nn.tanhAct, nn.sigAct, reluAct]
Loss        = [nn.SquaredError, nn.CrossEntropy, nn.SoftMax]
'''

#exp 2
param = {
'Epochs': 1000,
'ExNum' : 2,
'T'     : 32,
'TT'    : 98,
'L'     : 2,
'I'     : 28,
'J'     : 28,
'C'     : 28,
'D'     : 28,
'R'     : 2,
'lrA'   : 0.001,
'lrW'   : 0.001,
'ACT'   : 'reluAct',
'LOSS'  : 'SquaredError',
'vis'   : False,
'ORDER' : 'Random',
'INIT'  : 'Uniform',
'TrialNum':0
}

for ACT in ["tanhAct","sigmoidAct","reluAct","noAct"]:
  param['ACT'] = ACT
  for ORDER in ["Random","Fixed","Alternate"]:
    param['ORDER'] = ORDER
    for lrW in [0, 0.0001, 0.001, 0.01, 0.1]:
       param['lrW'] = lrW
       for lrA in [0, 0.0001, 0.001, 0.01,0.1]:
         param['lrA'] = lrA     
         nn.runNetwork(**param)  
pdb.set_trace()
print("starting ex[3")
#exp 3
param ={
'Epochs': 1,
'ExNum' : 3,
'T'     : 19,
'TT'    : 19,
'L'     : 16,
'I'     : 26,
'J'     : 26,
'C'     : 28,
'D'     : 28,
'R'     : 10,
'lrA'   : 0.001,
'lrW'   : 0.001,
'ACT'   : 'reluAct',
'LOSS'  : 'SquaredError',
'vis'   :  False,
'ORDER' : 'Random',
'INIT'  : 'Uniform',
'TrialNum':0
}
for ACT in ["tanhAct","sigmoidAct","reluAct","noAct"]:
  param['ACT'] = ACT 
  nn.runNetowk(**param)
for ORDER in ["Random","Fixed","Alternate"]:
  param['ORDER'] = ORDER
  nn.runNetwork(**param) 
for lrW in [0.0001, 0.001, 0.01, 0.1]:
  param['lrW'] = lrW
  for lrA in [0, 0.0001, 0.001, 0.01,0.1]:
     param['lrA'] = lrA
     nn.runNetwork(**param) 
