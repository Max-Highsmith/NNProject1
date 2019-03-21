import Network as nn

#exp 2


##Vary learning rate
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

#LR exp
'''
for lr in [0, 0.0001, 0.001, 0.01, 0.1, 1]:
  param['lrA'] = lr
  param['lrW'] = lr
  nn.runNetwork(**param)
'''
'RUNing'
'''
for lrA in [0, 0.0001, 0.001, 0.01, 0.1, 1]:
  param['lrA'] = lrA
  for lrW in [0, 0.0001, 0.001, 0.01, 0.1, 1]:
    param['lrW'] = lrW
    nn.runNetwork(**param)
'''
'''
for ACT in ['tanhAct', 'sigmoidAct', 'reluAct','noAct']:
  param['ACT'] = ACT
  nn.runNetwork(**param)
'''
'''
for trialNum in range(0,4):
  param['TrialNum']= trialNum
  for ORDER in ['Random', 'Fixed', 'Alternate']:
    param['ORDER'] = ORDER
    nn.runNetwork(**param)
'''
