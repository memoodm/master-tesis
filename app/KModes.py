import math
from datetime import datetime

from kmodes.kmodes import KModes

class Kmodes:
    
    def __init__(self,k_max_try=10):
        # configurations
        self.__k_max_try = k_max_try
    
    def segmentate_for_K(self,data,K): 
        print(f"[Kmodes {datetime.now()}] START segmentate_for_K")
        print(f"[Kmodes {datetime.now()}] clustering for value k: {K}")
        kmode = KModes(n_clusters=K, init = "random", n_init = self.__k_max_try )
        kmode.fit_predict(data)
        print(f"[Kmodes {datetime.now()}] END segmentate_for_K")
        return kmode.labels_
