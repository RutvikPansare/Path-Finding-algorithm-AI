from PIL import Image, ImageDraw, ImageFont
import math
from PIL import Image, ImageFilter
import heapq
import time
totalTime = 0
import sys
""" Foundation of Artificial Intelligence, Lab 1
    Author: Rutvik Pansare     rp2832@g.rit.edu
    Rochester Institute Of Technology
"""
def readElevationFile(fileName):
    """
    function to read elevation file
    :param fileName: name of the elevation file
    :return: list containing elevation values
    """
    f = open(fileName, "r")
    elevationList = [line.strip(",").split() for line in f.readlines()]
    return elevationList

class Node:
    """
    Class to represent the nodes in the graph
    """
    __slots__ = "id", "rgb", "elevation", "connectedTo", "heuristic", "g", "f", "parent","cost"

    def __init__(self, id, heuristic=100000, rgb=0, elevation: object = 0, g=0, f=1000000):
        """

        :param id: the id of the node or (x,y) coordinates
        :param heuristic: the heuristic value of the node
        :param rgb: the rgb value at the node
        :param elevation: the elevation at the node
        :param g: the g cost to reach that node
        :param f: the f cost to reach that node
        """
        self.id = id
        self.rgb = rgb
        self.elevation = elevation
        self.connectedTo = {}
        self.heuristic = heuristic
        self.g = g
        self.f = f
        self.cost = 0

    def addNeighbor(self, nbr, weight=0):
        """
        function to add the neighbor of the node
        :param nbr:
        :param weight:
        :return: None
        """
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + "heuristic" + str(self.f) + ' connectedTo: ' + str(
            [str(x.id) for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def getHeuristic(self):
        return self.heuristic

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(str(self.id))


class Graph:
    """
    class to create a grpah consisting of nodes
    """
    __slots__ = "nodeList", "numNodes"

    def __init__(self):
        self.nodeList = {}
        self.numNodes = 0

    def addNode(self, key, heuristic, rgb=0, elevation=0):
        """
        function to add the node into the graph
        :param key: the Id of the node
        :param heuristic: the heuristic value of the node
        :param rgb: the rgb value of the node
        :param elevation: the elevation value of the node
        :return: node if new node was added, else None
        """
        if self.getNode(key) is None:
            self.numNodes += 1
            node = Node(key, heuristic, rgb, elevation)
            self.nodeList[key] = node
            return node
        else:
            node = self.getNode(key)
            node.heuristic = heuristic
            node.elevation = elevation
            node.rgb = rgb
            return None

    def getNode(self, key):
        """
        function to get the node from the graph
        :param key: the id of the node
        :return: the node
        """
        if key in self.nodeList:
            return self.nodeList[key]
        else:
            return None

    def addEdge(self, src, dest, heuristic=0, src_rgb=0, dest_rgb=0, src_elevation=0, dest_elevation=0, cost=0):
        """
        function to add the edge in the graph
        :param src: the source node
        :param dest: the destination node
        :param heuristic: the heuristic value of the node
        :param src_rgb: the rgb value at source node
        :param dest_rgb: the rgb value at destination
        :param src_elevation: the elevation at source
        :param dest_elevation: the elevation at destination
        :param cost:the cost to reach the destination node
        :return: None
        """
        if src not in self.nodeList:
            self.addNode(src, src_rgb)
        if dest not in self.nodeList:
            self.addNode(dest, heuristic)
        self.nodeList[src].addNeighbor(self.nodeList[dest], cost)

    def __iter__(self):
        return iter(self.nodeList.values())


def makeRGBDictionary():
    """
    function to make the rgb dictionary
    :return: dictionary
    """
    dictionary = {}
    dictionary[(248, 148, 18)] = 1
    dictionary[(255, 192, 0)] = 2
    dictionary[(255, 255, 255)] = 2
    dictionary[(2, 208, 60)] = 16
    dictionary[(2, 136, 40)] = 8
    dictionary[(5, 73, 24)] = 9
    dictionary[(0, 0, 255)] = 1000
    dictionary[(71, 51, 3)] = 5
    dictionary[(0, 0, 0)] = 7
    dictionary[(205, 0, 101)] = 1000
    dictionary[(205, 255, 0)] = 1000
    return dictionary


def MakeGraphDictionary(image, elevationFile=None):
    """
    function to make the graph.
    :param image: image object
    :param elevationFile: elevation file
    :return: nodeDictionary and graph
    """
    if elevationFile is None:
        elevationFile = [[]]
    graph = Graph()
    startTime = time.time()
    x, y = image.size
    outOfBounds = (205, 0, 101)
    rgb_dictionary = makeRGBDictionary()
    nodeDictionary = {}
    neighbors = []
    #checking if the pixel has neighbors
    for i in range(x): 
        for j in range(y):
            rgb = image.getpixel((i, j))
            #checking if the pixel is out of bounds or not
            if outOfBounds == rgb[0:3]:
                heuristic = 0
                elevation = float(elevationFile[j][i])
                nodeName = (i, j)
                node = graph.addNode(nodeName, heuristic, rgb_dictionary[rgb[0:3]], elevation)
                continue
            heuristic = 0
            nodeName = (i, j)
            elevation = float(elevationFile[j][i])
            node = graph.addNode(nodeName, heuristic, rgb_dictionary[rgb[0:3]], elevation)
            # checking if the pixel neighbor is valid
            if i - 1 in range(x):
                if j - 1 in range(y):
                    rgb = image.getpixel((i - 1, j - 1))
                    if outOfBounds == rgb[0:3]:
                        pass
                    else:
                        neighbors.append((i - 1, j - 1))
                        graph.addEdge(nodeName, (i - 1, j - 1))
                ###
                rgb = image.getpixel((i - 1, j))
                if outOfBounds == rgb[0:3]:
                    pass
                else:
                    neighbors.append((i - 1, j))
                    graph.addEdge(nodeName, (i - 1, j))

                ####
                if j + 1 in range(y):
                    rgb = image.getpixel((i - 1, j + 1))
                    if outOfBounds == rgb[0:3]:
                        pass
                    else:
                        neighbors.append((i - 1, j + 1))
                        graph.addEdge(nodeName, (i - 1, j + 1))
            if j - 1 in range(y):
                rgb = image.getpixel((i, j - 1))
                if outOfBounds == rgb[0:3]:
                    pass
                else:
                    neighbors.append((i, j - 1))
                    graph.addEdge(nodeName, (i, j - 1))
                if i + 1 in range(x):
                    rgb = image.getpixel((i + 1, j - 1))
                    if outOfBounds == rgb[0:3]:
                        pass
                    else:
                        neighbors.append((i + 1, j - 1))
                        graph.addEdge(nodeName, (i + 1, j - 1))
            if j + 1 in range(y):
                rgb = image.getpixel((i, j + 1))
                if outOfBounds == rgb[0:3]:
                    pass
                else:
                    neighbors.append((i, j + 1))
                    graph.addEdge(nodeName, (i, j + 1))
                if i + 1 in range(x):
                    rgb = image.getpixel((i + 1, j + 1))
                    if outOfBounds == rgb[0:3]:
                        pass
                    else:

                        neighbors.append((i + 1, j + 1))
                        graph.addEdge(nodeName, (i + 1, j + 1))
            if i + 1 in range(x):
                rgb = image.getpixel((i + 1, j))
                if outOfBounds == rgb[0:3]:
                    pass
                else:
                    neighbors.append((i + 1, j))
                    graph.addEdge(nodeName, (i + 1, j))
            nodeDictionary[nodeName] = neighbors
    end = time.time()
    return nodeDictionary, graph


def FindOptimalPath(graph, start, destination):
    """
    function to find the optimal path between the start node and destination node
    :param graph: the graph used to find the path
    :param start: the start node
    :param destination: the destination node
    :return: the path and total distance travelled
    """
    totalDistance = 0
    startTime = time.time()
    closedList = set()
    openList = []
    openList.append(graph.getNode(start))
    startNode = graph.getNode(start)
    destNode = graph.getNode(destination)
    while (len(openList) > 0):
        currentNode = (heapq.heappop(openList))
        (x, y) = currentNode.id
        if currentNode == destNode:
            path = []
            while currentNode != startNode:
                #print(currentNode)
                path.append(currentNode.id)
                totalDistance += currentNode.cost
                currentNode = currentNode.parent
            path.append(start)
            # Return reversed path
            end = time.time()
            #print(end - startTime)
            return path[::-1],totalDistance
        for neighbor in currentNode.getConnections():
            (x1, y1) = neighbor.id
            (x2, y2) = destNode.id
            if (neighbor in closedList):
                continue
            neighbor.heuristic = getHeuristic(x1, y1, currentNode.elevation, neighbor.elevation, neighbor.rgb, x2, y2)
            neighbor.cost = float(math.sqrt((((x - x1) * 10.29) ** 2) + (((y - y1) * 7.55) ** 2)) +
                             (neighbor.elevation - currentNode.elevation) ** 2)
            neighbor.g = currentNode.g + neighbor.cost
            f = neighbor.g + neighbor.heuristic
            if neighbor in openList and neighbor.f <= f:
                continue
            neighbor.f = f
            neighbor.parent = currentNode
            heapq.heappush(openList, neighbor)
        closedList.add(currentNode)
    return [],0


def getHeuristic(x, y, src_elevation, dest_elevation, landType, x1, y1):
    """
    function to calculate the heuristic value of the given node
    :param x: the x coordinate of source pixel
    :param y: the y coordinate of source pixel
    :param src_elevation: elevation at source
    :param dest_elevation: elevation at destination
    :param landType: the land type value at the given location
    :param x1: the x coordinate of destination node
    :param y1: the y coordinate of destination node
    :return: heuristic value
    """
    heuristic = src_elevation + landType + int(
        math.sqrt((((x1 - x) * 10.29) ** 2) + (((y1 - y) * 7.55) ** 2)
                  + ((src_elevation - dest_elevation) ** 2)))
    return heuristic

def makePath(pathFile):
    """
    function to create the path file
    :param pathFile: name of the path file
    :return: path list
    """
    course = []
    f = open(pathFile, "r")
    for line in f:
        line = [int(i) for i in line.split()]
        course.append(line)
    return course
def runTheCourse(imageFile,elevationFile,pathFile,outputImageFile):
    """
    function to run the course
    :param imageFile: the image file to run
    :param elevationFile: the elevation file
    :param pathFile: the path file
    :param outputImageFile: the name of the output image file
    :return:image file and total distance of the optimum path
    """
    totalDistance = 0
    totalPath = []
    im = Image.open(imageFile)
    elevation = readElevationFile(elevationFile)
    course = makePath(pathFile)
    Dictionary, graph = MakeGraphDictionary(im, elevation)
    for i in range(0, len(course) - 1):
        path,distance = FindOptimalPath(graph, (course[i][0], course[i][1]), (course[i + 1][0], course[i + 1][1]))
        totalPath.extend(path)
        totalDistance = distance + totalDistance
    if totalPath:
        draw = ImageDraw.Draw(im)
        draw.line(totalPath, fill=(49, 227, 185)),
        im.show()
        im = im.save(outputImageFile)
        #print(totalPath)
        print("Total Distance = " + str(totalDistance))
        return im
    else:
        print("No path available")


if __name__ == "__main__":
    im = runTheCourse(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
