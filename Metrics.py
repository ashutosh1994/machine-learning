__author__ = 'Ashutosh Kshirsagar'
import numpy as np
import math

def RRSE(act,pred):
    mean = act.mean()
    return np.sum(np.power(pred - act,2))/np.sum(np.power(mean-act,2))

def RAE(act,pred):
    mean = act.mean()
    return np.sum(np.abs(pred - act))/np.sum(np.abs(mean-act))

def CORR(act,pred):
    n = act.shape[0]
    p_ = pred.mean()
    a_ = act.mean()
    SPA = ((pred - p_) * (act - a_)).sum() / (n - 1)
    SP = (np.power(pred - p_, 2)).sum() / (n - 1)
    SA = (np.power(act - a_, 2)).sum() / (n - 1)
    return SPA / math.sqrt(SP * SA)