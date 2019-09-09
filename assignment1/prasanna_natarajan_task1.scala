/* prasanna_natarajan_task1.scala */
import org.apache.spark._
import java.io._
import org.json4s._
import org.json4s.native.JsonMethods._
import org.apache.spark.rdd.RDD
object prasanna_natarajan_task1 {
  def map_task1(x: JValue, col: String, name: String, thres: Int):(String, Int)={
    implicit val formats = DefaultFormats
    if ((x\col).extract[Int] > thres) return (name,1)
    else return (name,0)
  }
  def task1(rdd1: RDD[String]):(String,Int)={
    var reviews_list = rdd1.map(x=> parse(x)).map(x=> map_task1(x,"useful","n_review_useful",0)).filter(x=> x._2 != 0)
    // reviews_list.foreach(x=>println(x))
    var ret = reviews_list.reduceByKey(_+_).collect
    // println(ret(0)._1)
    // println(ret(0)._2)
    return(ret(0)._1,ret(0)._2)
  }
  def task2(rdd1: RDD[String]):(String,Int)={
    var reviews_list = rdd1.map(x=> parse(x)).map(x=> map_task1(x,"stars","n_review_5_star",4)).filter(x=> x._2 != 0)
    // reviews_list.foreach(x=>println(x))
    var ret = reviews_list.reduceByKey(_+_).collect
    // println(ret(0)._1)
    // println(ret(0)._2)
    return(ret(0)._1,ret(0)._2)
  }
  def map_task3(x: JValue):(String,Int)={
    implicit val formats = DefaultFormats
    ("n_characters",(x\"text").extract[String].length)
  }
  def task3(rdd1: RDD[String]):(String,Int)={
    // implicit val formats = DefaultFormats
    var reviews_list = rdd1.map(x=> parse(x)).map(x => map_task3(x))
    var ret = reviews_list.reduceByKey(math.max(_, _)).collect()
    return (ret(0)._1,ret(0)._2)
  }
  def map_task4(x: JValue):(String,Int)={
    implicit val formats = DefaultFormats
    ((x\"user_id").extract[String],1)
  }

  def task4(reviews: RDD[String]):(String,Int)={
    var reviews_list = reviews.map(x => parse(x)).map(x => map_task4(x))
    var ret = reviews_list.groupByKey().mapValues(x=>1).collect()
    return ("n_user",ret.length)
  }
  def map_task5(x: JValue):(String,Int)={
    implicit val formats = DefaultFormats
    ((x\"user_id").extract[String],1)
  }
  def task5( reviews: RDD[String]):Array[(String,Int)]={
    var reviews_list = reviews.map(x => parse(x)).map(x => map_task5(x))
    var ret = reviews_list.reduceByKey(_+_).takeOrdered(20)(Ordering[(Double,String)].on(x=>(-x._2,x._1)))
    return ret
  }
  def map_task6(x:JValue):(String,Int)={
    implicit val formats = DefaultFormats
    ((x\"business_id").extract[String],1)
  }
  def task6( reviews: RDD[String]):(String,Int)={
    var reviews_list = reviews.map(x=> parse(x)).map(x=>map_task6(x))
    var ret = reviews_list.groupByKey().mapValues(x=>1).collect()
    return ("n_business",ret.length)
  }
  def map_task7(x: JValue):(String,Int)={
      implicit val formats = DefaultFormats
      ((x\"business_id").extract[String],1)
  }
  def task7( reviews: RDD[String]):Array[(String,Int)]={
      var reviews_list = reviews.map(x=>parse(x)).map(x=>map_task7(x))
      var ret = reviews_list.reduceByKey(_+_).takeOrdered(20)(Ordering[(Double,String)].on(x=>(-x._2,x._1)))
      return ret
  }

  def main(args: Array[String]) {
    if (args.length != 2){
      println("This function needs 2 input arguments <input_file_name> <output_file_name>")
      return
    }
    val input = args(0)
    val output = args(1)
    println(input)
    println(output)
    val conf = new SparkConf()
    conf.set("spark.master","local")
    conf.set("spark.app.name","app1")
    val sc  = new SparkContext(conf)
    val reviews  = sc.textFile(input).persist()
    // println(sc.getClass)
    // println(rdd1.getClass)
    // rdd1.collect().foreach(println)
    val t1 = task1(reviews)
    val t2 = task2(reviews)
    val t3 = task3(reviews)
    val t4 = task4(reviews)
    val t5 = task5(reviews)
    val t6 = task6(reviews)
    val t7 = task7(reviews)
    //println(t)
    var out: String="{\n"
    out+="\""+t1._1+"\": "+t1._2.toString+",\n"
    out+="\""+t2._1+"\": "+t2._2.toString+",\n"
    out+="\""+t3._1+"\": "+t3._2.toString+",\n"
    out+="\""+t4._1+"\": "+t4._2.toString+",\n"
    out+="\"top20_user\": ["
    t5.foreach(x=>out+="[\""+x._1+"\", "+x._2.toString+"], ")
    out=out.slice(0,out.length-2)
    out+="], \n"
    out+="\""+t6._1+"\":"+t6._2.toString+",\n"
    out+="\"top20_business\":["
    t7.foreach(x=>out+="[\""+x._1+"\", "+x._2.toString+"], ")
    out=out.slice(0,out.length-2)
    out+="] \n"
    out+="}";
    val pw = new PrintWriter(new File(output))
    pw.write(out)
    pw.close
    sc.stop()
  }
}


