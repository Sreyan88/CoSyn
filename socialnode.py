import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np

import pickle


def generateEdges(type, df):
    path = ""+type+".pickle" # path to ineraction matrix folder

    try:

        with open(path, 'rb') as f:
            sub = pickle.load(f)
        
        for user in sub:
            for follower in sub[user]:
                df.loc[len(df)] = [user,follower]

    except:
        print("Not found: ", path)


    

class Node(DGLDataset):
    def __init__(self):
        super().__init__(name='socialnode')

    def process(self):


        test = pd.read_csv("")
        train = pd.read_csv("")
        val = pd.read_csv("")
        found = 0
        notfound = 0
        def mapFunc(a):
            try:
                return tweet_to_ids[a]
            except:
                returnInd = len(tweet_to_ids)
                tweet_to_ids.update({a:returnInd})
                #nodes_data.loc[len(nodes_data)] = [a]
                return tweet_to_ids[a]


        df = pd.concat([train,test,val])

        unique_users = pd.unique(df['author'])

        



        nodes_data= pd.read_csv("./username2id.csv")# mapping of username to id 
        print(nodes_data.shape[0])
        edges_data = pd.DataFrame( columns=['src','dest'])

        generateEdges("TRAIN",edges_data)

        generateEdges("VAL",edges_data)
        
        generateEdges("TEST",edges_data)

        tweet_to_ids = {}
        
        #print(edges_data)
        edges_data['src'] = edges_data['src'].map(lambda a: mapFunc(a))
        edges_data['dest'] = edges_data['dest'].map(lambda a: mapFunc(a))

        
        print(edges_data)
        print(len(nodes_data))
        g = torch.empty(size = (len(nodes_data),1,768))
        print(nodes_data)
        

        for ind in range(len(nodes_data)):
                
            try:
                temp = torch.load("./user_embed/"+nodes_data['user_name'][ind]+".pt", map_location = "cpu")
                g[ind] = temp[0]
                

                
                found = found+1
                
            except:
                #print(ind)
                print("Username not found",nodes_data['user_name'][ind] )
                g[ind] =  torch.empty(1,768)
                notfound = notfound+1
            #print(g[ind].size())
        print("found",found)
        print("notfound",notfound)
                
        #print(nodes_data)
        id = torch.from_numpy(nodes_data['id'].to_numpy())
        #print(edges_data)
        


        id = torch.from_numpy(nodes_data['id'].to_numpy())

        edges_dst = torch.from_numpy(edges_data['dest'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())


        self.graph = dgl.graph((edges_src, edges_dst), num_nodes = nodes_data.shape[0])
        self.graph.ndata['g'] = g.to(dtype=torch.float32)
        self.graph.ndata['id'] = id
        self.graph.ndata['tweet_id'] = id.to(dtype=torch.int64)
        print(self.graph)
 


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

