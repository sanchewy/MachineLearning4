import sys
import random
from csv import reader
import numpy as np
import functools

data_set_location = "data/tao-all2.dat"

def kmeans_cluster(data, numClusters):
    clusters = list()
    #Pick initial centroids randomly
    for i in range(numClusters):
        clusters.append([data[random.randint(0,len(data)-1)], []])
    #Put all the data into the first centroid temporarily
    for d in data:
        clusters[0][1].append(d)
    change_flag = True
    while(change_flag):
        change_flag = False
        for cluster in clusters:
            if len(cluster[1]) == 0:
                fill_cluster = clusters[random.randint(0,numClusters-1)]
                while len(fill_cluster[1]) == 0:
                    fill_cluster = clusters[random.randint(0,numClusters-1)]
                try:
                    cluster[1].append(fill_cluster[1].pop(random.randint(0,len(fill_cluster[1])-1)))
                except IndexError:
                    print("Error tried to fill empty cluster with data from empty cluster.")
            for point in cluster[1]:
                centroids = [x[0] for x in clusters]
                min_dist = distance(point, cluster[0])
                min_dist_centroid = None
                for centroid in range(len(centroids)):
                    if(distance(point, clusters[centroid][0]) < min_dist):
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
    
def dbscan_cluster(data, minpts, theta):
    #label all points core = 2, noise = 0, or border = 1
    core = list()
    for point in range(len(data)):
        thresh_points = 0
        for neighbor in range(len(data)):
            if distance(data[point], data[neighbor]) < theta:
                thresh_points += 1
        if thresh_points > minpts:
            core.append(point)
            
    #Assign clusters
    cluster_tuples = [[0,x] for x in data]
    currCluster = 0
    for c in core:
        if cluster_tuples[c][0] == 0:
            currCluster += 1
            cluster_tuples[c][0] = currCluster
        for point in range(len(cluster_tuples)):
            if cluster_tuples[point][0] == 0:
                if distance(cluster_tuples[point][1], data[c]) < theta:
                    cluster_tuples[point][0] = currCluster
        
    clusters = [list()] * currCluster
    for p in cluster_tuples:
        if p[0] != 0:
            clusters[p[0]-1].append(p[1])
    return clusters
    
def get_centroids(cluster):
    cent = list()
    for x in range(len(cluster[0])):
        cent.append(sum([point[x] for point in cluster]))
    cent = [x/len(cluster) for x in cent]
    print(cent)
    return cent
                
def cluster_scatter(cluster, centroid):
    dim = len(centroid)
    sum_centroid_dist = 0
    for point in cluster:
        if dim != len(point):
            raise ValueError('Tried to calculate distance between two points of differenent dimensions %s %s')
        sum_centroid_dist += sum([abs(x-y) ** dim for x,y in zip(point,centroid)])
    if len(cluster) > 0:
        return (sum_centroid_dist/len(cluster)) ** (1/dim)
    else:
        print("Error, there was an empty cluster.")
        return (sum_centroid_dist/.001) ** (1/dim)
    
def cluster_seperation(centroid1, centroid2):
    #Minimum square distance betwen cluster centers
    return distance(centroid1,centroid2)
    
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
    summ = sum([abs(x-y) ** dim for x,y in zip(point1,point2)])
    return summ ** (1/dim)

#change data string to number
def str_to_float(dataset):
    for row in range(len(dataset)):
        for x in range(len(dataset[row])):
            dataset[row][x] = float(dataset[row][x].strip())

if __name__ == '__main__':
    dataset = load_csv(data_set_location)
    dataset = [i[1:7] for i in dataset]
    str_to_float(dataset)
    holder = dataset
    length = int(len(holder)/5)
    #clusters = kmeans_cluster(dataset, 20)
    #print("Number of clusters: %s" % (len(clusters)))
    #r = list()
    #for c in clusters:
    #    other_clusters = list(clusters)
    #    other_clusters.remove(c)
    #    try:
    #        r.append(max([(cluster_scatter(c[1],c[0])+cluster_scatter(x[1],x[0]))/cluster_seperation(c[0],x[0]) for x in other_clusters]))
    #    except ZeroDivisionError:
    #        print("Error, there was an empty cluster that caused dividion by zero.")
    #        sys.exit()
    #print("Davies-Bouldin index: %s" % (str(sum(r)/len(r))))
    
    clusters = dbscan_cluster(dataset, 10, 10)
    for c in range(len(clusters)):
        clusters[c] = [get_centroids(clusters[c]), clusters[c]]
    #print(clusters[0])
    #print("Number of clusters: %s" % (len(clusters)))
    r = list()
    for c in clusters:
        other_clusters = list(clusters)
        other_clusters.remove(c)
        try:
            r.append(max([(cluster_scatter(c[1],c[0])+cluster_scatter(x[1],x[0]))/cluster_seperation(c[0],x[0]) for x in other_clusters]))
        except ZeroDivisionError:
           print("Error, there was an empty cluster that caused dividion by zero.")
           sys.exit()
    print("Davies-Bouldin index: %s" % (str(sum(r)/len(r))))