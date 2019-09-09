import sys
import json
from pyspark import SparkContext
from operator import add
import time

def task1(reviews):
    reviews_list = reviews.map(lambda x:json.loads(x)).filter(lambda x: x["useful"]>0).map(lambda x: ("n_review_useful",1))
    ret = reviews_list.reduceByKey(add).collect()
    return ret

def task2(reviews):
    reviews_list = reviews.map(lambda x:json.loads(x)).filter(lambda x: x["stars"]==5).map(lambda x: ("n_review_5_star",1))
    ret = reviews_list.reduceByKey(add).collect()
    return ret

def task3(reviews):
    reviews_list = reviews.map(lambda x:json.loads(x)).map(lambda x: ("n_characters",len(x["text"])))
    ret = reviews_list.reduceByKey(max).collect()
    return ret

def task4(reviews):
    reviews_list = reviews.map(lambda x:json.loads(x)).map(lambda x:(x["user_id"],1))
    ret = reviews_list.groupByKey().mapValues(lambda x: 1).count()
    return [("n_user",ret)]

def task5(reviews):
    reviews_list = reviews.map(lambda x:json.loads(x)).map(lambda x:(x["user_id"],1))
    ret = reviews_list.reduceByKey(add).sortBy(lambda x: (-x[1],x[0]),ascending=True).take(20)
    return ret

def task6(reviews):
    reviews_list = reviews.map(lambda x:json.loads(x)).map(lambda x:(x["business_id"],1))
    ret = reviews_list.groupByKey().mapValues(lambda x: 1).count()
    return [("n_business",ret)]

def task7(reviews):
    reviews_list = reviews.map(lambda x:json.loads(x)).map(lambda x:(x["business_id"],1))
    ret = reviews_list.reduceByKey(add).sortBy(lambda x: (-x[1],x[0]),ascending=True).take(20)
    return ret

if __name__ == "__main__":
    if len(sys.argv)!=3:
        print("This function needs 2 input arguments <input_file_name> <output_file_name>")
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    sc = SparkContext("local[*]")
    reviews = sc.textFile(input_file_path).persist()
    out = {}
    t1 = task1(reviews)
    t2 = task2(reviews)
    t3 = task3(reviews)
    t4 = task4(reviews)
    t5 = task5(reviews)
    t6 = task6(reviews)
    t7 = task7(reviews)

    out[t1[0][0]] = t1[0][1]
    out[t2[0][0]] = t2[0][1]
    out[t3[0][0]] = t3[0][1]
    out[t4[0][0]] = t4[0][1]
    out["top20_user"] = t5
    out[t6[0][0]] = t6[0][1]
    out["top20_business"] = t7

    json_out = json.dumps(out)
    with open(output_file_path,"w") as f:
        f.write(json_out)
        
