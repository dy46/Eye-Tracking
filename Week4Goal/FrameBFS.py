import cv2
# import multiprocessing as mp
import math
# import numpy as np
# import datamani
# import drMatches
import sys
# from drMatches import Position, getXY
import time
# import polygons_overlapping
# import Queue

# class Vertex:
# 	def __init__(self,key, coordinates):
# 		self.id = key
# 		self.connectedTo = {}
# 		self.previouslyVisited = False
# 		self.coordinates = coordinates

# 	def addNeighbor(self,nbr,weight=0):
# 		self.connectedTo[nbr] = weight

# 	def __str__(self):
# 		return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

# 	def getConnections(self):
# 		return self.connectedTo.keys()

# 	def getId(self):
# 		return self.id

# 	def setVisited(self):
# 		self.previouslyVisited = True

# 	def checkVisited(self):
# 		return self.previouslyVisited

# 	def getCoordinates(self):
# 		return self.coordinates
# class Graph:
# 	def __init__(self):
# 		self.vertList = {}
# 		self.numVertices = 0

# 	def addVertex(self,key, coordinates):
# 		self.numVertices = self.numVertices + 1
# 		newVertex = Vertex(key, coordinates)
# 		self.vertList[key] = newVertex
# 		return newVertex

# 	def getVertex(self,n):
# 		if n in self.vertList:
# 			return self.vertList[n]
# 		else:
# 			return None

# 	def __contains__(self,n):
# 		return n in self.vertList

# 	def addEdge(self,f,t,cost=0):
# 		if f not in self.vertList:
# 			nv = self.addVertex(f)
# 		if t not in self.vertList:
# 			nv = self.addVertex(t)
# 		self.vertList[f].addNeighbor(self.vertList[t], cost)

# 	def getVertices(self):
# 		return self.vertList.keys()

# 	def __iter__(self):
# 		return iter(self.vertList.values())


# frame_graph = Graph()
# graph_dictionary = {}
# box_vertices = []
# previous_vertices = []
# height, width = 0, 0

def determineOrder(poly_arr):
	# height, width = frame.shape[:2]
	# populateGraph()
	box_coordinates = []
	for p in poly_arr:
		temp_point = p[0]
		point = [int(temp_point[0]), int(temp_point[1])]
		box_coordinates.append((p[0][0], p[0][1]))
	box_order = []
	while len(box_coordinates) > 0:
		min_distance = sys.maxint
		min_point = ()
		for point in box_coordinates:
			distance = math.sqrt(pow(point[0], 2) + pow(point[1], 2))
			if distance < min_distance:
				min_distance, min_point = distance, point
		box_order.append(min_point)
		box_coordinates.remove(min_point)
	# queue = Queue.Queue()
	# queue.put(frame_graph.getVertex(0))
	# box_order = []
	# while queue.qsize() > 0 and len(box_order) < len(box_coordinates):
	# 	current_vertex = queue.get()
	# 	for v in current_vertex.getConnections():
	# 		if not v.checkVisited():
	# 			queue.put(v)
	# 			v.setVisited()
	# 			if v.getId() in box_coordinates:
	# 				box_order.append(v.getCoordinates())
	return box_order

# def inFrame(x, y):
# 	return x in range(width) and y in range(height)

# def populateGraph():
# 	global graph_dictionary, frame_graph
# 	graph_counter = 0
# 	for j in range(height):
# 		for i in range(width):
# 			graph_dictionary[(i, j)] = graph_counter
# 			frame_graph.addVertex(graph_counter, (i,j))
# 			graph_counter += 1
# 	for j in range(height):
# 		for i in range(width):
# 			checkPointsAroundPoint(i, j)

			
# def checkPointsAroundPoint(x, y):
# 	global frame_graph, graph_dictionary
# 	for i in range(-1, 2):
# 		for j in range(-1, 2):
# 			if inFrame(x + i, y + j) and i != 0 and j != 0:
# 				frame_graph.addEdge(graph_dictionary[(x, y)], graph_dictionary[(x+i,y+j)])

