import numpy as np
from datetime import datetime

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


class Categoricer:
    
    def __init__(self,ward_linkeage="single",ward_affinity="euclidean",ward_children_limit=20,show_graph_solution=False):
        self.__ward_linkeage = ward_linkeage
        self.__ward_affinity = ward_affinity
        self.__ward_children_limit = ward_children_limit
        self.__show_graph_solution = show_graph_solution
        self.__scaler = StandardScaler()
        
    def generate_cagegories(self,data):
        print(f"[Categoricer {datetime.now()}] START generate_cagegories")
        data_copy = data.copy(deep=True)
        for column in data.columns:
            print(f"[Categoricer {datetime.now()}] PROCESSING column [{column}]")
            if data[column].dtype in ["int64","float64","complex","timedelta64","datetime64"]:
                print(f"[Categoricer {datetime.now()}] Column: {column}, is Numeric")
                self.__transform_numeric_column(data_copy,column)
            else:
                print(f"[Categoricer {datetime.now()}] Column: {column}, is Categorical")
                self.__transform_categorical_column(data_copy,column)
        print(f"[Categoricer {datetime.now()}] END generate_cagegories")
        return data_copy
            
    
    def __transform_numeric_column(self,data,column):
        X = self.__scaler.fit_transform( [[x] for x in data[column].values] )
        K = self.__find_optimal_clusters_number(X)
        print(f"[Categoricer {datetime.now()}] Column: {column}, optimal K: {K}")
        model = AgglomerativeClustering(
            n_clusters=K,
            metric=self.__ward_affinity,
            linkage=self.__ward_linkeage,
            compute_full_tree=True)
        model.fit(X)
        if(self.__show_graph_solution):
            self.__draw_solution(column,X,model.labels_)
        print(f"[Categoricer {datetime.now()}] Column: {column}, creating categories labels")
        for i in range(len(data)):
            category = chr(model.labels_[i]+65)
            data.loc[i,column] = category 
            
            
    def __transform_categorical_column(self,data,column):
        print(f"[Categoricer {datetime.now()}] Column: {column}, creating categories labels")
        for i,value in enumerate(data[column].unique()):
            category = chr(i+65)
            data.loc[data[column]==value, column] = category
            
        
    def __find_optimal_clusters_number(self,X):
        # generate clusters childs for full tree
        model = AgglomerativeClustering(
            n_clusters=1,
            metric=self.__ward_affinity,
            linkage=self.__ward_linkeage,
            compute_full_tree=True)
        model.fit(X)
        # calculate the disstortions with intravariance
        clusters = {}
        intra_variances = []
        # init individual clusters
        for i in range(len(X)):
            clusters[i] = Node([X[i][0]])       
        # Merge clusters for full tree
        count = len(X)
        for merge_step in model.children_:
            nodeA, nodeB = clusters.pop(merge_step[0]) , clusters.pop(merge_step[1])
            clusters[count] = Node( nodeA.points + nodeB.points )
            intra_variances.append( np.sum( [ clusters[key].var*clusters[key].size for key in clusters ] ) / len(X) )
            count += 1
        intra_variances.reverse()
        K = self.__find_elbow_solution(intra_variances[:self.__ward_children_limit])
        return K
            
    def __find_elbow_solution(self,values):
        # Compute distances from points to the line connecting the first and last points
        x1, y1 = 1, values[0]
        x2, y2 = len(values), values[-1]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        distances = [np.abs(m * i - j + b) / np.sqrt(m ** 2 + 1) for i, j in enumerate(values, 1)] 
        # Find optimal number of clusters using elbow method
        max_distance_index = np.argmax(distances)
        return max_distance_index + 1
    
    def __draw_solution(self,column,data_list,labels):
        # Plot input data
        plt.title(column)
        plt.scatter(data_list, np.zeros(len(data_list)))
        plt.show()
        # Plot solution
        for label in np.unique(labels):
            data_label_values = data_list[labels == label]
            plt.scatter(data_label_values, np.zeros(len(data_label_values)))
        plt.show()

class Node:
    
    def __init__(self,points):
        self.points = points
        self.mean = np.mean(points)
        self.var = np.var(points)
        self.size = len(points)
        
    def __repr__(self):
        return str(self.mean)