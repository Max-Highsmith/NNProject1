#exp 3
import Network as nn 
param ={
'Epochs': 100,
'ExNum' : 2,
'T'     : 31,
'TT'    : 98,
'L'     : 2,
'I'     : 28,
'J'     : 28,
'C'     : 28,
'D'     : 28,
'R'     : 2,
'lrA'   : 0.01,
'lrW'   : 0.001,
'ACT'   : 'tanhAct',
'LOSS'  : 'SquaredError',
'vis'   :  False,
'ORDER' : 'Random',
'INIT'  : 'Freeze',  #FreezeTop converges, FreezeBot does not
'TrialNum':1
}
for trial in range(0,5):
  param['TrialNum'] = trial
  nn.runNetwork(**param)

for trial in range(0,5):
  param['TrialNum'] = trial
  param['INIT'] = "Uniform"
  nn.runNetwork(**param)
