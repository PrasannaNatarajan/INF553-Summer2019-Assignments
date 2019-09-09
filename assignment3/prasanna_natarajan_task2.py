import sys
from pyspark import SparkContext
from operator import add
import time
from collections import defaultdict
from itertools import combinations
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

def read_csv(x):
    return x.split(',')

def dict_update(acc,n):
    if not isinstance(acc,dict):
        return n
    else:
        acc.update(n)
        return acc
## case 2
'''
user_id_1: int user id
user_id_2: int user id 
set_items_1: set of items rated by user 1
set_items_2: set of items rated by user 2
ratings_1: dict {item_id:rating} of user 1
ratings_2: dict {item_id:rating} of user 2
'''

def calc_weights_user_user(user_id_1,user_id_2,set_items_1,set_items_2, ratings_1,ratings_2):

    corated_items = list(set_items_1.intersection(set_items_2))
    if len(corated_items)==0:
        return 0.2
    avg_rating_1 = [ratings_1[x] for x in corated_items]
    avg_rating_1 = sum(avg_rating_1)/len(avg_rating_1)
    avg_rating_2 = [ratings_2[x] for x in corated_items]
    avg_rating_2 = sum(avg_rating_2)/len(avg_rating_2)

    sum_u_v = 0
    sq_sum_u = 0
    sq_sum_v = 0

    for item in corated_items:
        sum_u_v += (ratings_1[item]-avg_rating_1)*(ratings_2[item]-avg_rating_2)
        sq_sum_u += (ratings_1[item]-avg_rating_1)**2
        sq_sum_v += (ratings_2[item]-avg_rating_2)**2
    if sum_u_v==0:
        return 0.2
    return sum_u_v/((sq_sum_u**0.5)*(sq_sum_v**0.5))

'''
new_user_id: int user id
new_item_id: int item id
ratings_user: rating of all users dict {user_id:{item_id:rating}}
user_items: dict of {user_id: {item_ids}}
items_avg: dict with {item_id:avg}
users_avg: dict with {user_id: (sum(ratings),len(ratings))}
'''
def predict_all_users(new_user_id, new_item_id,ratings_user, user_items, items_avg, users_avg,item_users):
    if new_user_id not in ratings_user and new_item_id not in items_avg:
        return ((new_user_id,new_item_id),3)
    elif new_user_id not in ratings_user:
        value = items_avg[new_item_id]
        if value<1:
            value = 1
        return ((new_user_id,new_item_id),value)
    elif new_item_id not in items_avg:
        value = float(users_avg[new_user_id][0]/users_avg[new_user_id][1])
        if value<1:
            value=1
        return ((new_user_id,new_item_id),value)
    prod = 0
    weight = 0
    abs_sum = 0

    users_rated_item = item_users[new_item_id]

    weights = dict()
    for user in users_rated_item:

        weight = calc_weights_user_user(new_user_id,user,user_items[new_user_id], user_items[user], ratings_user[new_user_id], ratings_user[user])
        weights[user]=weight
        avg_rating_user = (users_avg[user][0]-ratings_user[user][new_item_id])/(users_avg[user][1]-1)
        prod+=(weight*(ratings_user[user][new_item_id]-avg_rating_user))
        abs_sum+=abs(weight)
    ra = float(users_avg[new_user_id][0]/users_avg[new_user_id][1])
    
    if prod==0:
        return ((new_user_id,new_item_id),1)
    value = (ra+(prod/abs_sum))
    if value<1:
        value = 1
    if value>5:
        value = 5
    return ((new_user_id,new_item_id),value)

## case 3
'''
item_id1: int item id 
item_id2: int item id
id1_users: set of users who have rated item id 1
id2_users: set of users who have rated item id 2
id1_ratings: dictionary of ratings for item id 1 {user_id: rating}
id2_ratings: dictionary of ratings for item id 2 {user_id: rating}
'''
def calc_weight_item_item(busi_id1, busi_id2, id1_users, id2_users, id1_ratings, id2_ratings):
    corated_users = id1_users.intersection(id2_users)

    sum_id1 = 0
    n = 0
    sum_id2 = 0
    if len(corated_users)==0:
        return 0.2
    for user in corated_users:
        sum_id1+=id1_ratings[user]
        sum_id2+=id2_ratings[user]
        n+=1
    avg_id1 = sum_id1/n
    avg_id2 = sum_id2/n

    sum_w_i_j = 0
    sq_sum_w_i = 0
    sq_sum_w_j = 0
    for user in corated_users:
        sum_w_i_j += ((id1_ratings[user]- avg_id1)*(id2_ratings[user] - avg_id2))
        sq_sum_w_i += (id1_ratings[user]- avg_id1)**2
        sq_sum_w_j += (id2_ratings[user] - avg_id2)**2

    if sum_w_i_j==0:
        return 0.2
    
    return sum_w_i_j/(pow(sq_sum_w_i,(1/2))*pow(sq_sum_w_j,(1/2)))

def predict_item_item(new_user_id, new_busi_id, user_ratings_dict, train_dict, business_user_dict,mean_businesses,mean_users):
    if new_user_id not in user_ratings_dict and new_busi_id not in train_dict:
        return ((new_user_id,new_busi_id),3)
    elif new_user_id not in user_ratings_dict:
        return ((new_user_id,new_busi_id),mean_businesses[new_busi_id])
    elif new_busi_id not in train_dict:
        return ((new_user_id,new_busi_id),mean_users[new_user_id])
    
    prod = 0
    mod_sum_w = 0
    ret = []
    user_ratings = user_ratings_dict[new_user_id]
    for i in range(0,len(user_ratings)):
        w_i_n = calc_weight_item_item(new_busi_id,user_ratings[i][0],business_user_dict[new_busi_id],business_user_dict[user_ratings[i][0]],train_dict[new_busi_id],train_dict[user_ratings[i][0]])
        if w_i_n ==0:
            continue
        ret.append((w_i_n,i))

    for i in ret:
        prod += ((user_ratings[i[1]][1]-mean_businesses[user_ratings[i[1]][0]])*i[0])
        mod_sum_w += abs(i[0])
    if prod ==0:
        return ((new_user_id,new_busi_id),1) 

    return ((new_user_id,new_busi_id),float(prod/mod_sum_w+mean_businesses[new_busi_id]))

if __name__ == "__main__":
    if len(sys.argv)!=5:
        print("This function needs 4 input arguments <train_file_name> <test_file_name> <case_id> <output_file_name>")
        sys.exit(1)
    start = time.time()
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    case_id = int(sys.argv[3])
    output_file_name = sys.argv[4]

    sc = SparkContext("local[*]")

    train_data = sc.textFile(train_file_path).filter(lambda x: x!="user_id, business_id, stars").map(lambda x: read_csv(x)).map(lambda x: (x[0],x[1],float(x[2]))).persist()
    val_data = sc.textFile(test_file_path).filter(lambda x: x!="user_id, business_id, stars").map(lambda x: read_csv(x)).map(lambda x: (x[0],x[1],float(x[2]))).persist()   
    # print("val data count = ",val_data.count())
    data = sc.union([train_data,val_data])
    # print("data count = ",data.count())
    uni_userid = data.map(lambda x: (x[0],1)).reduceByKey(lambda acc,n: n).map(lambda x: x[0]).collect()
    uni_userid.sort()
    # print("len(uni_users) = ",len(uni_userid))
    d_user_id = defaultdict()
    d_id_user = defaultdict()
    i=0
    for x in uni_userid:
        # print(x)
        d_user_id[x]=i
        d_id_user[i]=x
        i+=1

    uni_businessid = data.map(lambda x: (x[1],1)).reduceByKey(lambda acc,n: n).map(lambda x:x[0]).collect()
    # print("len(uni_businessid)",len(uni_businessid))
    uni_businessid.sort()
    d_business_id = defaultdict()
    d_id_business = defaultdict()
    i=0
    for x in uni_businessid:
        
        d_business_id[x]=i
        d_id_business[i]=x
        i+=1
    if case_id ==1:
        train_matrix = train_data.map(lambda x: ((d_user_id[x[0]],d_business_id[x[1]]),float(x[2]))).persist()
        ratings = train_data.map(lambda x: Rating(d_user_id[x[0]],d_business_id[x[1]],x[2]))
        val_int_data = val_data.map(lambda x: Rating(d_user_id[x[0]],d_business_id[x[1]],float(x[2])))
        val_ratings = val_int_data.map(lambda x: (x[0],x[1]))
        mean_businesses = train_matrix.map(lambda x: (x[0][1],[x[1]])).reduceByKey(lambda acc,n: acc+n).map(lambda x: (x[0],sum(x[1])/len(x[1]))).collectAsMap()
        mean_users = train_matrix.map(lambda x: (x[0][0],[x[1]])).reduceByKey(lambda acc,n: acc+n).map(lambda x: (x[0],sum(x[1])/len(x[1]))).collectAsMap()

        r = 1
        i = 10
        model = ALS.train(ratings, r, i,nonnegative=True,seed=10)

        all_pred = model.predictAll(val_ratings).map(lambda x: ((x[0],x[1]),x[2]))
        
        cold_start = val_data.map(lambda r: (d_user_id[r[0]],d_business_id[r[1]])).subtract(all_pred.map(lambda r: (r[0][0],r[0][1])))

        def handle_cold_start(pair,avg_users,avg_businesses):
            if pair[0] not in avg_users and pair[1] not in avg_businesses:
                return (pair,3)
            elif pair[0] not in avg_users:
                return (pair,avg_businesses[pair[1]])
            else:
                return (pair,avg_users[pair[0]])
        if cold_start.count()!=0:
            all_pred = sc.union([all_pred,cold_start.map(lambda x: handle_cold_start(x,mean_users,mean_businesses))])
        # validation = val_int_data.map(lambda r: ((r[0], r[1]), r[2])).join(all_pred)
        # MSE = validation.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        # RMSE = pow(MSE,1/2)
        # print("RMSE = ",str(RMSE))
        # print("all_pred count = ",all_pred.count(),"val", val_ratings.count(),"cold_start_count = ", cold_start.count())
        out = "user_id, business_id, prediction\n"
        
        for x in all_pred.map(lambda x: (x[0][0],x[0][1],x[1])).collect():
            out+=d_id_user[x[0]]+","+d_id_business[x[1]]+","+str(x[2])+"\n"
        
        with open(output_file_name,"w") as f:
            f.write(out)
            f.close()

    
    elif case_id==2:
        conv_train_data = train_data.map(lambda x: ((d_user_id[x[0]],d_business_id[x[1]]),x[2])).cache()
        conv_val_data = val_data.map(lambda x: ((d_user_id[x[0]],d_business_id[x[1]]),x[2])).cache()
        ratings_user = conv_train_data.map(lambda x: (x[0][0],{x[0][1]:x[1]})).reduceByKey(lambda acc,n: dict_update(acc,n)).collectAsMap()
        user_items = conv_train_data.map(lambda x: (x[0][0],{x[0][1]})).reduceByKey(lambda acc,n: acc.union(n)).collectAsMap()
        items_avg = conv_train_data.map(lambda x: (x[0][1],[x[1]])).reduceByKey(lambda acc,n: acc+n).map(lambda x: (x[0],sum(x[1])/len(x[1]))).collectAsMap()
        users_avg = conv_train_data.map(lambda x: (x[0][0],[x[1]])).reduceByKey(lambda acc,n: acc+n).map(lambda x: (x[0],(sum(x[1]),len(x[1])))).collectAsMap()
        item_users = conv_train_data.map(lambda x: (x[0][1],{x[0][0]})).reduceByKey(lambda acc,n: acc.union(n)).collectAsMap()
        
        all_preds = conv_val_data.map(lambda x: predict_all_users(x[0][0],x[0][1],ratings_user,user_items,items_avg,users_avg,item_users))
        # validate = all_preds.join(conv_val_data)
        # MSE = validate.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        # print("Mean Squared Error = " + str(MSE))
        # RMSE = pow(MSE,1/2)
        # print("RMSE = ",str(RMSE))
        out = "user_id, business_id, prediction\n"
        for x in all_preds.map(lambda x: (x[0][0],x[0][1],x[1])).collect():
            out+=d_id_user[x[0]]+","+d_id_business[x[1]]+","+str(x[2])+"\n"
        with open(output_file_name,"w") as f:
            f.write(out)
            f.close()

    
    elif case_id==3:
        train_matrix = train_data.map(lambda x: ((d_user_id[x[0]],d_business_id[x[1]]),float(x[2]))).persist()
        val_matrix = val_data.map(lambda x: ((d_user_id[x[0]],d_business_id[x[1]]),float(x[2]))).persist()

        mean_businesses = train_matrix.map(lambda x: (x[0][1],[x[1]])).reduceByKey(lambda acc,n: acc+n).map(lambda x: (x[0],sum(x[1])/len(x[1]))).collectAsMap()
        mean_users = train_matrix.map(lambda x: (x[0][0],[x[1]])).reduceByKey(lambda acc,n: acc+n).map(lambda x: (x[0],sum(x[1])/len(x[1]))).collectAsMap()

        train_dict = train_matrix.map(lambda x: (x[0][1],{x[0][0]:x[1]})).reduceByKey(lambda acc,n: dict_update(acc,n)).collectAsMap()

        business_user_dict = train_matrix.map(lambda x: (x[0][1],{x[0][0]})).reduceByKey(lambda acc,n: acc.union(n)).collectAsMap()

        user_ratings_dict = train_matrix.map(lambda x: (x[0][0],[(x[0][1],x[1])])).reduceByKey(lambda acc,n: acc+n).collectAsMap()

        all_preds = val_matrix.map(lambda x: predict_item_item(x[0][0],x[0][1],user_ratings_dict,train_dict,business_user_dict,mean_businesses,mean_users))

        # validate = all_preds.join(val_matrix)
        # MSE = validate.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        # print("Mean Squared Error = " + str(MSE))
        # RMSE = pow(MSE,1/2)
        # print("RMSE = ",str(RMSE))
        out = "user_id, business_id, prediction\n"

        for x in all_preds.map(lambda x: (x[0][0],x[0][1],x[1])).collect():
            out+=d_id_user[x[0]]+","+d_id_business[x[1]]+","+str(x[2])+"\n"
        
        with open(output_file_name,"w") as f:
            f.write(out)
            f.close()
    else:
        print("There are only 3 cases... so give either 1,2or3 please")
	
    end = time.time()
    print("time: ",str(end-start))



