from node import Node
import pickle

import os

modes = ["test", "train", "val"]

graphs = []

def generateDGL(mode):
    path = ""+mode # path to members folder

    for filename in os.listdir(path):
        id = filename.split(".")[0]
        c = Node(id=id, type =mode)
        if len(c[0].ndata)==8:
            graphs.append(c[0])


for mode in modes:
    generateDGL(mode)


with open(".pkl", 'wb') as f:# output pkl path 
    pickle.dump(graphs,f)
