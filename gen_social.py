import json
import pandas as pd
from datetime import datetime
import numpy as np

import pickle




def generategraph(type):
    parent_comment = json.load(open(""+type+".json"))# path to parent_comment nodes
    comment_reply = json.load(open(""+type+".json"))# path to comment_reply nodes
    reply_reply = json.load(open(""+type+".json"))# path to reply_reply nodes

    for k in parent_comment.keys():
        createCSVs(k, parent_comment,comment_reply, reply_reply,type)


def generateGraph():
    test = pd.read_csv("") # path to test csv file
    train = pd.read_csv("") # path to train csv file
    val = pd.read_csv("") # path to val csv file

    df = pd.concat([train,test,val], axis =1)

    interactions_df = pd.DataFrame( columns=['src','dest'])
    members_df = pd.unique(df['user_name'])






def createInteractions(parent, parent_comment,comment_reply, reply_reply,type):
    interactions_df = pd.DataFrame( columns=['src','dest', 'wt'])
    for comment in parent_comment[parent]:
        interactions_df.loc[len(interactions_df)] = [parent,comment,1]
        if comment in comment_reply :
            for reply in comment_reply[comment]:
                interactions_df.loc[len(interactions_df)] = [comment, reply,1]
                if reply in reply_reply :
                    for reply2 in reply_reply[reply]:
                        interactions_df.loc[len(interactions_df)] = [reply,reply2,1]
    interactions_df.to_csv(""+type+"/"+parent+".csv", index=False) # path to interactions(b/w the nodes) folder


def getdelt(t2,t1):

    if len(t1)==0 or len(t2)==0 or t1[0]!=t1[0] or t2[0]!=t2[0]:
        return 0
    else:
        t1 = t1[0]
        t2 = t2[0]
  
    t1_dt =   datetime.strptime(t1,'%Y-%m-%d %H:%M:%S')
    t2_dt =   datetime.strptime(t2,'%Y-%m-%d %H:%M:%S')

    duration = (t1_dt-t2_dt).total_seconds()

    return duration

def createCSVs(parent, parent_comment,comment_reply, reply_reply,t):
    members_df = pd.DataFrame(columns=['index', 'x', 'y','del_t','train_mask', 'test_mask', 'val_mask', 'u_name'])
    interactions_df = pd.DataFrame( columns=['src','dest', 'wt'])

    

    test = pd.read_csv("")
    train = pd.read_csv("")
    val = pd.read_csv("")

    train["tweet_id"] = train["tweet_id"].map(str)

    train_mask = 0
    val_mask = 0
    test_mask = 0

    mode = "."

    if t=="test":
        mode ="test"
        df = test
        test_mask = 1
    else:
        if parent in train["tweet_id"].values:
            mode = "train"
            df = train
            train_mask =1
        else:
            mode = "val"
            df = val
            val_mask = 1

    print(parent)
    print(mode)



    df["tweet_id"] = df["tweet_id"].map(str)

    
    t0 = df.loc[df["tweet_id"]==parent]["timestamp"].values
    label = 0 if df.loc[df["tweet_id"]==parent]["label"].values[0]=="NONE" else 0
    u_name = df.loc[df["tweet_id"]==parent]["user_name"].values[0]
    members_df.loc[len(members_df)] = [parent, parent, label, 0,train_mask,test_mask,val_mask, u_name]


    for comment in parent_comment[parent]:
        t1 = df.loc[df["tweet_id"]==comment]["timestamp"].values
        label = 0 if df.loc[df["tweet_id"]==comment]["label"].values[0]=="NONE" else 1
        u_name = df.loc[df["tweet_id"]==comment]["user_name"].values[0]
        del_t = getdelt(t0,t1)
        members_df.loc[len(members_df)] = [comment, comment, label, del_t,train_mask,test_mask,val_mask, u_name]
        interactions_df.loc[len(interactions_df)] = [parent,comment,1]
        if comment in comment_reply :
            for reply in comment_reply[comment]:
                t1 = df.loc[df["tweet_id"]==reply]["timestamp"].values
                label = 0 if df.loc[df["tweet_id"]==reply]["label"].values[0]=="NONE" else 1
                u_name = df.loc[df["tweet_id"]==reply]["user_name"].values[0]
                del_t = getdelt(t0,t1)
                members_df.loc[len(members_df)] = [reply, reply, label, del_t,train_mask,test_mask,val_mask, u_name]
                interactions_df.loc[len(interactions_df)] = [comment, reply,1]
                if reply in reply_reply :
                    for reply2 in reply_reply[reply]:
                        t1 = df.loc[df["tweet_id"]==reply2]["timestamp"].values
                        label = 0 if df.loc[df["tweet_id"]==reply2]["label"].values[0]=="NONE" else 1
                        u_name = df.loc[df["tweet_id"]==reply2]["user_name"].values[0]
                        del_t = getdelt(t0,t1)
                        members_df.loc[len(members_df)] = [reply2, reply2, label, del_t,train_mask,test_mask,val_mask, u_name]
                        interactions_df.loc[len(interactions_df)] = [reply,reply2,1]
    

    members_df.to_csv(""+mode+"/"+parent+".csv", index=False)
    interactions_df.to_csv(""+mode+"/"+parent+".csv", index=False)



for t in ["test","train"]:
    generategraph(t)




    

    



