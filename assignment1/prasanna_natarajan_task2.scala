import org.apache.spark._
import java.io._
import org.json4s._
import org.json4s.native.JsonMethods._
import org.apache.spark.rdd.RDD
object prasanna_natarajan_task2 {
    def map_task1_rev(x: JValue):(String,(Double,Double))={
        implicit val formats = DefaultFormats
        ((x\"business_id").extract[String],((x\"stars").extract[Double],1))
    }

    def map_task1_busi(x: JValue):(String,String)={
        implicit val formats = DefaultFormats
        ((x\"business_id").extract[String],(x\"state").extract[String])
    }
    def task1(reviews: RDD[String],businesses:RDD[String],output_file_path1:String):RDD[(String,Double)]={
        var reviews_list = reviews.map(x=> parse(x)).map(x=> map_task1_rev(x))
        var ret = reviews_list.reduceByKey((acc,n)=>(acc._1+n._1,acc._2+n._2))
        
        var businesses_list = businesses.map(x => parse(x)).map(x=>map_task1_busi(x))
        
        var res = ret.join(businesses_list)
        
        var res_map = res.map(x=> (x._2._2,x._2._1))
        var res_task2 = res_map.reduceByKey((acc,n)=> (acc._1+n._1,acc._2+n._2)).map(x=>(x._1,x._2._1/x._2._2))
        var res_final = res_task2.sortBy(x=> (-x._2,x._1)).collect()
        var out: String = "state, stars \n"
        res_final.foreach(x=>out+=x._1.toString+","+x._2.toString+"\n")
        val pw = new PrintWriter(new File(output_file_path1))
        // print(out)
        pw.write(out)
        pw.close
        return res_task2
    }

    def task2(res_task2:RDD[(String,Double)],output_file_path2:String){

        val s_b = System.nanoTime()
        var res_task2_b = res_task2.takeOrdered(5)(Ordering[(Double,String)].on(x=>(-x._2,x._1)) )
        print(res_task2_b)
        val e_b = System.nanoTime()

        val s_a = System.nanoTime()
        var res_task2_a = res_task2.sortBy(x=> (-x._2,x._1)).collect().slice(0,5)
        print(res_task2_a)
        val e_a = System.nanoTime()



        var out:String = "{\n \"m1\":"+ (e_a - s_a).toString+",\n \"m2\":"+(e_b - s_b).toString+", \n"
        out+="\"explaination\": \"When we use collect on an RDD, all the data from all the worker nodes will be copied to the driver (master node) and it returns all the values. Unlike collect, take/takeOrdered returns just the argument number of elements. In this particular implementation, I have used sort with collect (in method 1), which helps in sorting based on a value. I have also used takeOrdered, which in itself gives the sorted top 5 elements. I think that also saves some time.\"\n"
        out+="}"
        val pw = new PrintWriter(new File(output_file_path2))
        print(out)
        pw.write(out)
        pw.close
    }    
    

    
    


    def main(args: Array[String]) {
        if (args.length != 4){
        println("This function needs 2 input arguments <input_file_name> <output_file_name>")
        return
        }
        val input1 = args(0)
        val input2 = args(1)
        val output1 = args(2)
        val output2 = args(3)

        val conf = new SparkConf()
        conf.set("spark.master","local")
        conf.set("spark.app.name","app2")
        val sc  = new SparkContext(conf)
        
        val reviews  = sc.textFile(input1).persist()
        val businesses = sc.textFile(input2).persist()

        val rt = task1(reviews,businesses,output1)
        task2(rt,output2)
        sc.stop()
    }
}