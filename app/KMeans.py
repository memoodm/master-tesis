import math
import numpy as np
from datetime import datetime

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class Kmeans:

	def get_K_from_points(self,data):
		print(f"[Kmeans {datetime.now()}] START get_K_from_points")
		X = data
		scores = []
		for n in range( 2,self.__proposedAmount(len(X)) ):
			print(f"[Kmeans {datetime.now()}] clustering for k: {n}")
			kmeans = KMeans(n_clusters=n,n_init=5).fit(X)
			cluster_labels = kmeans.fit_predict(X)
			scores.append( kmeans.inertia_ )
		print(f"[Kmeans {datetime.now()}] Finding optimal K value from intertias: {scores}")
		K = self.__find_elbow_solution(scores)
		print(f"[Kmeans {datetime.now()}] Optimal K value: {K}")
		print(f"[Kmeans {datetime.now()}] END get_K_from_points")
		return K,scores

	def __proposedAmount(self,data_size):
		segmentsAmount = 1 + 3.3 * math.log10(data_size)
		return math.ceil(segmentsAmount)

	def __find_elbow_solution(self,values):
		x1, y1 = 1, values[0]
		x2, y2 = len(values), values[-1]
		m = (y2 - y1) / (x2 - x1)
		b = y1 - m * x1
		distances = [np.abs(m * i - j + b) / np.sqrt(m ** 2 + 1) for i, j in enumerate(values, 1)] 
		# Find optimal number of clusters using elbow method
		max_distance_index = np.argmax(distances)
		return max_distance_index + 1