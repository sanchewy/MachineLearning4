import sys
import random
from csv import reader
import numpy as np
import functools

data_set_location = "data/tao-all2.txt"

def kmeans_cluster(data, numClusters):
    clusters = list()
    #Pick initial centroids randomly
    for i in range(numClusters):
        clusters.append([data[random.randint(0,len(data))], []])
    #Put all the data into the first lcentroid temporarily
    for d in data:
        clusters[0][1].append(d)
    change_flag = True
    while(change_flag):
        change_flag = False
        for cluster in clusters:
            for point in cluster[1]:
                centroids = [x[0] for x in clusters]
                min_dist = distance(point, cluster[0])
                min_dist_centroid = None
                for centroid in range(len(centroids)):
                    if(distance(point, clusters[centroid][0]) < min_dist):
                        print("Changed cluster for point %s" % (str(point)))
                        min_dist = distance(point, clusters[centroid][0])
                        min_dist_centroid = centroid
                if min_dist_centroid is not None:
                    change_flag = True
                    cluster[1].remove(point)
                    clusters[min_dist_centroid][1].append(point)
        #Recalculate centroids
        for cluster in clusters:
            #print(cluster[1])
            centroid_attr = list()
            if len(cluster[1]) > 0:
                for attr in range(len(cluster[1][0])):
                    centroid_attr.append(sum([x[attr] for x in cluster[1]])/float(len(cluster[1])))
    return clusters

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
    #print("Compare: %s %s" % (str(point1),str(point2)))
    dim = len(point1)
    if(dim != len(point2)):
        raise ValueError('Tried to calculate distance between two points of differenent dimensions %s %s')
    summ = sum([(x-y) ** dim for x,y in zip(point1,point2)])
    return summ ** (1/dim)

#change data string to number
def str_to_float(dataset):
    for row in range(len(dataset)):
        for x in range(len(dataset[row])):
            dataset[row][x] = float(dataset[row][x].strip())

if __name__ == '__main__':
    dataset = load_csv(data_set_location)
    dataset = [i[1:7] for i in dataset]
    #print(type(dataset[1][0]))
    str_to_float(dataset)
    #print(dataset)
    holder = dataset
    length = int(len(holder)/5)
    clusters = kmeans_cluster(dataset, 5)
    print(len(clusters))
    for c in clusters:
        print(c[1])
