import sys
from pyspark import SparkContext
from operator import add
import time
import itertools
from collections import OrderedDict, Counter

def read_csv(x,case):
    ret = x.split(',')
    if case==2:
        return (ret[1],ret[0])
    else:
        return (ret[0],ret[1])

def append_list(acc,n):
    acc.add(n)
    return acc

def extend_list(acc,n):
    acc.update(n)
    return acc

def construct(all_items,i):
    ret = set()
    if i==1:
        for x in all_items:
            ret = x[1]|ret
        ret = list(ret)
    else:
        if isinstance(all_items[0],str):
            all_items = {frozenset([k]) for k in all_items}
        ret = list({a.union(b) for a in all_items for b in all_items if len(a.union(b))==i})
    return ret

def filter_items(transactions,candidates,support):
    d = {}
    fl = 0

    for x in candidates:
        d[x] = 0

    if not isinstance(candidates[0],frozenset):
        for transaction in transactions:

            for k in d:
                if k in transaction[1]:
                    d[k]+=1
    else:
        for transaction in transactions:
            itemset = set(transaction[1])
            for k in d:
                if k.issubset(itemset):
                    d[k]+=1
    ret = []
    for k in d:
        if d[k]>=support:
            ret.append(k)
    return ret

def apriori_algo(transactions,support,num_rdds):
    i=1
    trans = []
    ret = []
    for t in transactions:
        trans.append(t)
    support_par = support / (num_rdds/len(trans))
    
    candidates = construct(trans,i)
    while len(candidates)!=0:
        i+=1
        freq_items = filter_items(trans,candidates,support_par)
        for x in freq_items:
            ret.append((x,i-1))
        if len(freq_items)!=0:
            candidates = construct(list(freq_items),i)
        else:
            break
    return ret

def freq_count(transactions,freq_items):
    d = {}
    fl=0
    for item in freq_items:
        d[item[0]]=0
    trans = []
    for t in transactions:
        trans.append(t)
    
    for transaction in trans:
        for item in freq_items:
            if isinstance(item[0],frozenset):
                for x in item[0]:
                    if x in transaction[1]:
                        fl=1
                    else:
                        fl=0
                        break
                if fl==1:
                    d[item[0]]+=1
                    fl=0
            
            else:
                if item[0] in transaction[1]:
                    d[item[0]]+=1
    ret = []
    for k in d:
        ret.append((k,d[k]))
    
    return ret

if __name__ == "__main__":
    if len(sys.argv)!=5:
        print("This function needs 4 input arguments <input_file_name> <output_file_name>")
        sys.exit(1)
    start = time.time()
    case_no = int(sys.argv[1])
    support = float(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]
    sc = SparkContext("local[*]")
    
    users = sc.textFile(input_file_path).filter(lambda x: x!="user_id,business_id").map(lambda x: read_csv(x,case_no)).combineByKey(lambda x: {x},lambda acc,n: append_list(acc,n),lambda acc,n: extend_list(acc,n)).persist()
    num_users = users.count()

    can = users.mapPartitions(lambda x:apriori_algo(x,float(support),num_users),True).distinct().collect()
    
    ## phase 2
    ph2 = users.mapPartitions(lambda x:freq_count(x,can)).reduceByKey(add).filter(lambda x: x[1]>=support).collect()

    with open(output_file_path,'w') as f:
        out = "Candidates: \n"        

        ret = []
        k=0
        d = dict()
        
        while(k<len(can)):
            if can[k][1] in d:
                if can[k][1] == 1:
                    d[can[k][1]].append(list([can[k][0]]))
                    k+=1
                    continue
                d[can[k][1]].append(list([x for x in can[k][0]]))
            else:
                if can[k][1] ==1:
                    d[can[k][1]] = list([[can[k][0]]])
                    k+=1
                    continue
                d[can[k][1]] = list([[x for x in can[k][0]]])
            k+=1
        
        for k in sorted(d.keys()):
            d[k] = [sorted(list(x)) for x in d[k]]
            d[k] = sorted(d[k])
            for x in d[k]:
                if len(tuple(x))==1:
                    out+=str(tuple(x))[0:len(str(tuple(x)))-2]+"),"
                else:
                    out+=str(tuple(x))+","
            out=out[0:len(out)-1]
            out+="\n\n"

        out = out[0:len(out)-1]
        out += "\nFrequent itemsets:\n"

        ret = []
        k=0
        d = dict()
        # print("temp")
        while k<len(ph2):
            if isinstance(ph2[k][0],frozenset):
                if len(ph2[k][0]) in d:
                    d[len(ph2[k][0])].append(list([x for x in ph2[k][0]]))
                else:
                    d[len(ph2[k][0])] = list([[x for x in ph2[k][0]]])
            else:
                if not 1 in d:
                    d[1] = list([[ph2[k][0]]])
                else:
                    d[1].append(list([ph2[k][0]]))
            k+=1

        for k in sorted(d.keys()):
            d[k] = [sorted(list(x)) for x in d[k]]
            d[k] = sorted(d[k])
            for x in d[k]:
                if len(tuple(x))==1:
                    out+=str(tuple(x))[0:len(str(tuple(x)))-2]+"),"
                else:
                    out+=str(tuple(x))+","
            out = out[0:len(out)-1]
            out+="\n\n"
        out = out[0:len(out)-2]

        f.write(out)

    end = time.time()
    print("Duration:",end-start)
    
