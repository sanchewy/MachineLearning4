import sys
import random
from csv import reader
import numpy as np

data_set_location = "data/tao-all2.dat"

def kmeans_cluster(data, numClusters):
    clusters = list()
    #Pick initial centroids randomly
    for i in range(numClusters):
        clusters.append(data[random.randint(0,len(data))], [])
    #Put all the data into the first lcentroid temporarily
    clusters[0][1].append(data)
    change_flag = True
    while(change_flag):
        change_flag = False
        centroids = [x[0] for x in cluster]
        for cluster in clusters:
            for point in cluster[1]:
                min_dist = distance(point, cluster[0])
                min_dist_centroid = None
                for centroid in centroids:
                    if(distance(point, centroid) < min_dist):
                        min_dist = distance(point, centroid)
                        min_dist_centroid = centroid
                if min_dist_centroid is not None:
                    change_flag = True
                    cluster[1].remove(point)
                    clusters[min_dist_centroid][1].append(point)
        #Recalculate centroids
        for cluster in clusters:
            cluster[0] = sum(cluster[1])/float(len(cluster))
    pass

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file, delimiter=' ')
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def distance(point1, point2):
    dim = len(point1)
    if(dim != len(point2)):
        raise ValueError('Tried to calculate distance between two points of differenent dimensions')
    summ = sum(*[(x-y) ** dim for x,y in zip(list1,list2)])
    return summ ** (1/dim)

#change data string to number
def str_to_float(dataset):
    for row in range(len(dataset)):
        for x in range(len(dataset[row])):
            dataset[row][x] = float(dataset[row][x].strip())

if __name__ == '__main__':
    dataset = load_csv(data_set_location)
    dataset = [i[1:7] for i in dataset]
    print(type(dataset[1][0]))
    str_to_float(dataset)
    print(dataset)
    holder = dataset
    length = int(len(holder)/5)
    kmeans_cluster(dataset, 5)
