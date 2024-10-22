

import numpy as np


def fun_Otsu(Image):

    # Automatic image thresholding

    Thresholds = np.linspace(np.min(Image),np.max(Image),num=100)
    Interclass_Var = np.full((len(Thresholds),1),np.inf)
    for i_Threshold in range(len(Thresholds)):
        idx = Image>Thresholds[i_Threshold]
        if np.array(Image[idx]).size!=0:
            Interclass_Var[i_Threshold] = len(np.nonzero(Image[idx]))*np.var(Image[idx])+len(np.nonzero(Image[~idx]))*np.var(Image[~idx])
    Threshold = Thresholds[np.argmin(Interclass_Var)]
    
    return Threshold