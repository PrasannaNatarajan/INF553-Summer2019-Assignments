import sys
import json
from pyspark import SparkContext
from operator import add
import time

def task1(reviews,businesses,output_file_path1):
    start = time.time()
    
    reviews_list = reviews.map(lambda x: json.loads(x)).map(lambda x: (x["business_id"],(x["stars"],1)))
    ret = reviews_list.reduceByKey(lambda acc,n: (acc[0]+n[0],acc[1]+n[1]))
    
    businesses_list = businesses.map(lambda x: json.loads(x)).map(lambda x:(x["business_id"],x["state"]))
    
    res = ret.join(businesses_list)
    
    res_map = res.map(lambda x: (x[1][1],x[1][0]))
    res_task2 = res_map.reduceByKey(lambda acc,n: (acc[0]+n[0],acc[1]+n[1]))
    res_final = res_task2.collect()
    res_final = [(x[0],x[1][0]/x[1][1]) for x in res_final]
    res_final.sort(key=lambda x: (-x[1],x[0]))
    
    with open(output_file_path1,"w") as f:
        f.write("state, stars \n")
        for x in res_final:
            f.write(""+str(x[0])+","+str(x[1])+"\n")
    
    end = time.time()
    with open("logsB.txt","w") as f:
        f.write(str(end-start))
    return res_task2

def task2(res_task2,output_file_path2):    
    
    # s_b = time.time()
    # res_task2_b = res_task2.map(lambda x:(x[0],x[1][0]/x[1][1])).sortBy(lambda x: (-x[1],x[0]),ascending=True).take(5)
    # print(res_task2_b)
    # e_b = time.time()

    s_b = time.time()
    res_task2_b = res_task2.map(lambda x:(x[0],x[1][0]/x[1][1])).takeOrdered(5,key =lambda x: (-x[1],x[0]) )
    print(res_task2_b)
    e_b = time.time()

    s_a = time.time()
    res_task2_a = res_task2.map(lambda x:(x[0],x[1][0]/x[1][1])).sortBy(lambda x: (-x[1],x[0]),ascending=True).collect()[0:5]
    print(res_task2_a)
    e_a = time.time()



    out = {}
    out["m1"] = e_a - s_a
    out["m2"] = e_b - s_b
    out["explaination"] = "When we use collect on an RDD, all the data from all the worker nodes will be copied to the driver (master node) and it returns all the values. Unlike collect, take/takeOrdered returns just the argument number of elements. In this particular implementation, I have used sort with collect (in method 1), which helps in sorting based on a value. I have also used takeOrdered, which in itself gives the sorted top 5 elements. I think that also saves some time."

    json_out = json.dumps(out)
    with open(output_file_path2,"w") as f:
        f.write(json_out)



    

if __name__ == "__main__":
    if len(sys.argv)!=5:
        print("This function needs 4 input arguments <input_file_name1> <input_file_name2> <output_file_name1> <output_file_name2>")
        sys.exit(1)
    
    input_file_path1 = sys.argv[1]
    input_file_path2 = sys.argv[2]
    output_file_path1 = sys.argv[3]
    output_file_path2 = sys.argv[4]

    sc = SparkContext("local[*]")
    
    reviews = sc.textFile(input_file_path1).persist()
    businesses = sc.textFile(input_file_path2).persist()
    
    rt = task1(reviews,businesses,output_file_path1)
    task2(rt,output_file_path2)
