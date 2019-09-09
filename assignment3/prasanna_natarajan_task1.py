from pyspark import SparkContext
import sys
from itertools import combinations
import time
import random

HASH_NUM = 65

def read_csv(x):
    return x.split(',')

def append_list(acc,n):
    acc.append(n)
    return acc

def extend_list(acc,n):
    acc.extend(n)
    return acc

def make_sig(business_id,row_list,total_hashes,m):
    # hashes = [lambda x: ((x*random.randint(0,m)+ random.randint(0,m))%m) for _ in range(0,total_hashes)] ## takes too much time
    hashes = lambda x,y: ((x*y)+(HASH_NUM*x))%m
    ret = [float('Inf') for i in range(0,total_hashes)]
    for row in row_list:
        for h in range(0,total_hashes):
            hash_func = hashes(h,row)
            if hash_func < ret[h]:
                ret[h] = hash_func
    return (business_id,ret)

def make_lsh(row,b,r):
    
    ret = []
    for i in range(0,b):
        x = i*r
        y = (i+1)*r
        # band = (10001*i)+min([h(t) for t in row[1][x:y]])%10000           ## takes too much time
        # band = (10001*i)+int(''.join(str(t) for t in row[1][x:y]))%10000  ## gives 0 total results
        band = row[1][x:y] #.insert(0,i)
        band.insert(0,i) # just to seperate each bucket
        # print(band)
        ret.append((tuple(band),row[0]))
    return ret

def jacc_sim(pair,ori_char_mat):
    
    temp = []
    busi_1 = ori_char_mat[pair[0]]
    busi_2 = ori_char_mat[pair[1]]
    inter = len(busi_1.intersection(busi_2))
    uni = len(busi_1.union(busi_2))
    sim = float(inter)/float(uni)
    if sim>=0.5:
        temp = (frozenset((pair[0],pair[1])),sim)
    return temp

if __name__ == "__main__":
    if len(sys.argv)!=3:
        print("This function needs 2 input arguments <input_file_name> <output_file_name>")
        sys.exit(1)
    start = time.time()
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    sc = SparkContext("local[*]")
    
    data = sc.textFile(input_file_path).filter(lambda x: x!="user_id, business_id, stars").map(lambda x: read_csv(x)).persist()

    uni_userid = data.map(lambda x: (x[0],1)).reduceByKey(lambda acc,n: n).map(lambda x: x[0]).collect()
    uni_userid.sort()
    # print("len(uni_userid)",len(uni_userid))
    m = len(uni_userid)
    d_users = {}
    i=0
    for x in uni_userid:
        d_users[x]=i
        i+=1
    uni_businessid = data.map(lambda x: (x[1],1)).reduceByKey(lambda acc,n: n).map(lambda x:x[0]).collect()
    # print("len(uni_businessid)",len(uni_businessid))
    uni_businessid.sort()
    d_business_id = {}
    d_id_business = {}
    i=0
    for x in uni_businessid:
        
        d_business_id[x]=i
        d_id_business[i]=x
        i+=1
    
    m = len(uni_userid)
    total_hashes = 100
    b = 50
    r = total_hashes//b
    
    ori_matrix = data.map(lambda x: (d_business_id[x[1]],d_users[x[0]])).combineByKey(lambda x: [x],lambda acc,n: append_list(acc,n), lambda acc,n: extend_list(acc,n)).persist()
    ori_char_mat = ori_matrix.map(lambda x: (x[0],set(x[1]))).collectAsMap()

    cand = ori_matrix.map(lambda x: make_sig(x[0],x[1],total_hashes,m))
    cand = cand.flatMap(lambda x: make_lsh(x,b,r)).groupByKey().filter(lambda x: len(x[1])>1)
    cand = cand.flatMap(lambda x: [y for y in combinations(x[1],2)]).distinct().persist()

    ret = cand.map(lambda x: jacc_sim(x,ori_char_mat)).filter(lambda x: x!=[]).collectAsMap()
    # print(len(ret))

    out = "business_id_1, business_id_2, similarity\n"
    x = []
    for key in ret.keys():
        k = [sorted(tuple(key))]
        k.append(ret[key])
        x.append(list(k))
    
    for k in sorted(x):
        out+= d_id_business[k[0][0]]+","+d_id_business[k[0][1]]+","+str(k[1])+"\n"

    with open(output_file_path,"w") as f:
        f.write(out)
        f.close()
    end = time.time()
    print("time: ", str(end-start))