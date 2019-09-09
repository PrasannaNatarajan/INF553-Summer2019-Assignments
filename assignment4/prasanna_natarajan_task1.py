import sys
from pyspark import SparkContext
from operator import add,itemgetter
import time
from itertools import combinations


def read_csv(x):
    return x.split(',')

def combine_lists(nodes):
    acc = nodes[0]
    n = nodes[1]
    if acc!=None and n!=None:
        return acc+n
    elif acc!=None:
        return acc
    elif n!=None:
        return n
    else:
        print("some wierd stuff is going on...")
        print(acc,n)
        exit(1)
def make_edges(user_pair,user_businesses_set):
	set_business_id1 = user_businesses_set[user_pair[0]]
	set_business_id2 = user_businesses_set[user_pair[1]]
	l_inter = len(set_business_id1.intersection(set_business_id2))
	return (user_pair,l_inter)

def calc_bet(root,graph):
    temp_queue = []
    temp_queue.append(root)
    visited = [root]
    level_dict = {root:0}
    no_shortest_path = {root:1}
    parentchute = {root:None}
    while(len(temp_queue)!=0):
        curr_node = temp_queue.pop(0)
        for neighbor in graph[curr_node]:
            if neighbor in visited:
                if level_dict[curr_node] == level_dict[neighbor]-1:
                    no_shortest_path[neighbor]+=1
                    parentchute[neighbor].append(curr_node)
                continue
            visited.append(neighbor)
            temp_queue.append(neighbor)
            level_dict[neighbor]=level_dict[curr_node]+1
            no_shortest_path[neighbor]=1
            parentchute[neighbor]=[curr_node]
    nodelabels = {x:1 for x in visited}
    nodelabels[root]=0
    ret = []
    for node in visited[::-1]:
        if parentchute[node]==None:
            break
        for parent in parentchute[node]:
            nodelabels[parent]+=nodelabels[node]/no_shortest_path[node]
            ret.append(((min((parent,node)),max((parent,node))),nodelabels[node]/no_shortest_path[node]))
    
    return ret
'''
G - Graph
S - Samples
m - no of edges
'''
def calcQ(G,S,m,degrees):
    ret = 0
    for s in S:
        for i in s:
            for j in s:
                if i!=j:
                    Aij = 0
                    if j in G[i]:
                        Aij=1
                    ret+=(Aij-(((degrees[i]*degrees[j]))/(2*m)))
    return (1/(2*m))*ret 

'''
G - Graph
edge - (node1,node2)
'''
def checkSplit(G,edge):
    root = edge[0]
    other_node = edge[1]
    temp_queue = []
    temp_queue.append(root)
    visited = [root]
    while(temp_queue):
        curr_node = temp_queue.pop(0)
        for neighbor in G[curr_node]:
            if neighbor==other_node:
                return False
            if neighbor in visited:
                continue
            
            visited.append(neighbor)
            temp_queue.append(neighbor)

    return True

def bfs(root,graph):
    visited = [root]
    temp_queue = [root]
    while(temp_queue):
        curr_node = temp_queue.pop(0)
        for neighbor in graph[curr_node]:
            if neighbor in visited:
                continue
            visited.append(neighbor)
            temp_queue.append(neighbor)
    return {x:graph[x] for x in visited}

def dfs(temp, v, visited,V):
    visited[v] = True
    temp.append(v)
    for i in V[v]: 
        if visited[i] == False: 
            temp = dfs(temp, i, visited,V) 
    return temp 
 
def connectedComponents(V): 
    visited = {k:False for k in V}
    cc = [] 

    for v in V: 
        if visited[v] == False: 
            temp = [] 
            cc.append(dfs(temp, v, visited,V)) 
    return cc

def remove_node(x,biggest):

    if x[0]==biggest[0]:
        x[1].remove(biggest[1])
    elif x[0]==biggest[1]:
        x[1].remove(biggest[0])
    else:
        pass
    return x

if __name__ == "__main__":
    if len(sys.argv)!=5:
        print("This function needs 4 input arguments <filter> <input_file_name> <betweenness_output_file_path> <community_output_file_path>")
        sys.exit(1)
    start = time.time()
    
    filter_threshold = float(sys.argv[1])
    input_file_path = sys.argv[2]
    betweenness_output_file_path = sys.argv[3]
    community_output_file_path = sys.argv[4]

    sc = SparkContext("local[*]")
    
    data = sc.textFile(input_file_path).filter(lambda x: x!="user_id, business_id").map(lambda x: read_csv(x)).persist()
    
    uni_userid = data.map(lambda x: (x[0],1)).reduceByKey(lambda acc,n: n).map(lambda x: x[0]).collect()
    uni_userid.sort()
    m = len(uni_userid)

    d_users_id = {}
    d_id_users = {}
    
    i=0
    for x in uni_userid:
        d_users_id[x] = i
        d_id_users[i] = x
        i+=1
    
    uni_businessid = data.map(lambda x: (x[1],1)).reduceByKey(lambda acc,n: n).map(lambda x:x[0]).collect()
    uni_businessid.sort()

    d_business_id = {}
    d_id_business = {}
    
    i=0
    for x in uni_businessid: 
        d_business_id[x]=i
        d_id_business[i]=x
        i+=1
    
    id_data = data.map(lambda x: (d_users_id[x[0]],d_business_id[x[1]]))
    user_businesses_set = id_data.map(lambda x: (x[0],{x[1]})).reduceByKey(lambda acc,n: acc.union(n)).collectAsMap()

    comb_users = sc.parallelize([y for y in combinations(d_id_users.keys(),2)])
    edges = comb_users.map(lambda x:make_edges(x,user_businesses_set)).filter(lambda x: x[1]>=filter_threshold).map(lambda x: (min(x[0]),max(x[0]))).persist()
    ##nodes = list(edges.map(lambda x: set(x)).reduce(lambda acc,n: acc.union(n)))

    edges_count = edges.count()
    # print("edges_count = ", edges_count)
    edges_one_side = edges.map(lambda x: (x[0],[x[1]])).reduceByKey(lambda acc,n: acc+n)
    edges = edges.map(lambda x: (x[1],[x[0]])).reduceByKey(lambda acc,n: acc+n).fullOuterJoin(edges_one_side)

    graph_rdd = edges.map(lambda x: (x[0],combine_lists(x[1])))
    graph = graph_rdd.collectAsMap()
    
    cc = connectedComponents(graph)

    list_graph = []
    for c in cc:
        graph_comm = dict()
        for x in c:
            graph_comm[x]=graph[x]
        list_graph.append(graph_comm)

    degrees = graph_rdd.map(lambda x: (x[0],len(x[1]))).collectAsMap()

    betweenness = graph_rdd.flatMap(lambda x: calc_bet(x[0],graph)).reduceByKey(lambda acc,n: acc+n).map(lambda x: (x[0][0],x[0][1],x[1]/2)).sortBy(lambda x: (-x[2],x[0],x[1]))
    
    betweenness_out = betweenness.collect()
    out=""

    for v in betweenness_out:
        out+="('"+d_id_users[v[0]]+"', '"+d_id_users[v[1]]+"'), "+str(v[2])+"\n"
    
    with open(betweenness_output_file_path,"w") as f:
        f.write(out)
        f.close()

    biggest = betweenness_out[0]


    Q = []
    S = list(list_graph)

    i=0
    graph_rdd_new = graph_rdd.persist()
    new_graph = graph
    zaq = 100 if edges_count>=300 else 0
    while(i<=edges_count-zaq and biggest!=None):
        i+=1
        if i%15==0:
            graph_rdd_new = sc.parallelize(list(new_graph.items()))

        graph_rdd_new = graph_rdd_new.map(lambda x: remove_node(x,biggest)).persist()
        new_graph = graph_rdd_new.collectAsMap()

        if checkSplit(new_graph,(biggest[0],biggest[1])):
            s1 = bfs(biggest[0],new_graph)
            s2 = bfs(biggest[1],new_graph)

            for s in S:
                if biggest[0] in s and biggest[1] in s:
                    S.append(s1)
                    S.append(s2)
                    S.remove(s)
                    break
            q = calcQ(graph,S,edges_count,degrees)
            
            Q.append((q,list(S)))
            # if q<Q[len(Q)-2][0]:
            #     # print(Q[len(Q)-2])
            #     # print(q)
            #     i+=20
            #     # break
        else:
            for s in S:
                if biggest[0] in s and biggest[1] in s:
                    if biggest[1] in s[biggest[0]]:
                        s[biggest[0]].remove(biggest[1])
                    if biggest[0] in s[biggest[1]]:
                        s[biggest[1]].remove(biggest[0])
            
        betweenness_new = graph_rdd_new.flatMap(lambda x: calc_bet(x[0],new_graph)).reduceByKey(lambda acc,n: acc+n).map(lambda x: (x[0],x[1]/2))
        if betweenness_new.isEmpty():
            break
        biggest = betweenness_new.max(lambda x: x[1])
        if biggest!=[]:
            biggest = biggest[0]
        else:
            biggest = None
        

    r  = max(Q,key=lambda x: x[0])
    # print(r)
    # print(len(r[1]))
    out=""
    for i in sorted(r[1],key=lambda x: (len(x),sorted(list(x.keys()))[0])):
        for k in sorted(i):
            out+= "'"+d_id_users[k]+"', "
        out = out[0:len(out)-2]
        out+="\n"
    out=out[0:len(out)-1]
    with open(community_output_file_path,"w") as f:
        f.write(out)
        f.close()
    end = time.time()
    print("time: ", str(end-start))




