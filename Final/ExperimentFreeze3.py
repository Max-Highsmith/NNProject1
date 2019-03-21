#exp 3
import Network as nn 
param ={
'Epochs': 500,
'ExNum' : 3,
'T'     : 20,
'TT'    : 20,
'L'     : 10,
'I'     : 28,
'J'     : 28,
'C'     : 28,
'D'     : 28,
'R'     : 10,
'lrA'   : 0.0001,
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
