import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np

class Node(DGLDataset):
    def __init__(self,id,type):
        self.id = id
        self.type = type
        super().__init__(name='node')

    def process(self):
        nodes_data = pd.read_csv('./members/'+self.type+"/"+self.id+".csv")
        edges_data = pd.read_csv('./interactions/'+self.type+"/"+self.id+".csv")


        tweet_to_ids = {}

        for ind in nodes_data.index:
            tweet_to_ids.update({nodes_data["index"][ind]:ind})

        length = len(nodes_data)
        
        userID = pd.read_csv('./username2id.csv')
        tweetID = pd.read_csv('./tweetid2id.csv')
        if edges_data["src"].equals(edges_data["dest"]):
            length-=1
            #print("hi")
        
        nodes_data['x'] = nodes_data['x'].map(str)
        def mapU2ID(a):
            #if a=="NAN":
            #    return(len(userID))
            if len(userID.loc[userID["user_name"]==a]["id"].values)==0:
                return len(userID)
            return  userID.loc[userID["user_name"]==a]["id"].values[0]
        def maptweet2ID(a):
            return (tweetID.loc[tweetID["tweet_id"]==a]["id"].values[0])

        def transform_edge(p):
            if len(edges_data)==0:
                p = pd.DataFrame({'src': pd.Series(dtype='int'), 'dest': pd.Series(dtype='int'),'wt': pd.Series(dtype='int')})
            return p
        edges_data['src'] = edges_data['src'].map(lambda a: tweet_to_ids[a])
        edges_data['dest'] = edges_data['dest'].map(lambda a: tweet_to_ids[a])
        
        x = torch.empty(size = (len(nodes_data),768))
        print(len(nodes_data))
        for ind in range(len(nodes_data)):
            try:
                yy = torch.load("./tweet_embed/"+nodes_data['x'][ind]+".pt",map_location=torch.device('cpu'))
                x[ind]= torch.mean(yy, dim=0)
            except:
                x[ind] = torch.tensor(np.zeros(shape=(768))) 
 


        y = torch.from_numpy(nodes_data['y'].to_numpy())
        del_t = torch.from_numpy(nodes_data['del_t'].to_numpy())
        train_mask = torch.from_numpy(nodes_data['train_mask'].to_numpy())
        val_mask = torch.from_numpy(nodes_data['val_mask'].to_numpy())
        test_mask = torch.from_numpy(nodes_data['test_mask'].to_numpy())
        #print(nodes_data )
        id = torch.from_numpy((nodes_data['u_name'].map(lambda a: mapU2ID(a))).to_numpy())
        #print(id)
        tweet_id = torch.from_numpy(nodes_data['x'].map(lambda a: maptweet2ID(a)).astype("int64").to_numpy())

        edges_data =transform_edge(edges_data)
        edges_dst = torch.from_numpy(edges_data['dest'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy()) 

        self.graph = dgl.graph((edges_src, edges_dst))

        if length:
            self.graph.ndata['x'] = x.to(dtype=torch.float32) 
        
            self.graph.ndata['y'] = y
            self.graph.ndata['del_t'] = del_t.to(dtype=torch.float32)
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['test_mask'] = test_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['id'] = id
            self.graph.ndata['tweet_id'] = tweet_id.to(dtype=torch.int64)
            


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
