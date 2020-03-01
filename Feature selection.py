# You need to import everything below
import pyspark
from pyspark import SparkContext

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F

from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

import lime 
from lime import lime_text
from lime.lime_text import LimeTextExplainer

import numpy as np

sc = SparkContext()

spark = SparkSession \
        .builder \
        .appName("hw3") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

########################################################################################################
# Load data
categories = ["alt.atheism", "soc.religion.christian"]
LabeledDocument = pyspark.sql.Row("category", "text")

def categoryFromPath(path):
    return path.split("/")[-2]
    
    







    
    
def prepareDF(typ):
    rdds = [sc.wholeTextFiles("/user/tbl245/20news-bydate-" + typ + "/" + category)\
              .map(lambda x: LabeledDocument(categoryFromPath(x[0]), x[1]))\
            for category in categories]
    return sc.union(rdds).toDF()









train_df = prepareDF("train").cache()
test_df  = prepareDF("test").cache()

#####################################################################################################
""" Task 1.1
a.	Compute the numbers of documents in training and test datasets. Make sure to write your code here and report
    the numbers in your txt file. - append new id 
b.	Index each document in each dataset by creating an index column, "id", for each data set, with index starting at 0. 

""" 
# Your code starts here
#a 
test_df.count() #No.of the test_df=717
train_df.count() #No.of the train_df=1079





#b 
newtest_df=test_df   #copy for new df as a backup
newtest_df2=newtest_df 
newtest_df3=newtest_df2.rdd.zipWithIndex() #indexing
newtest_df4=newtest_df3.toDF() #changing in df
newtest_df4 # will give the structure  DataFrame[_1: struct<category:string,text:string>, _2: bigint]
df_final = newtest_df4.withColumn('category', newtest_df4['_1'].getItem("category"))  #first term is name of output column, which column we are going to split, last name is the name in structure
df_final = df_final.withColumn('text', df_final['_1'].getItem("text")) #import text
df_final = df_final.withColumn('id', df_final['_2']) #bigints doesnt require splitting inside
test_df=df_final.select("id","category","text") #rearrange the columns


test_df.show(n=5, truncate=True)
"""
+---+-----------+--------------------+
| id|   category|                text|
+---+-----------+--------------------+
|  0|alt.atheism|From: mathew <mat...|
|  1|alt.atheism|From: halat@panth...|
|  2|alt.atheism|From: Nanci Ann M...|
|  3|alt.atheism|From: ch981@cleve...|
|  4|alt.atheism|From: bobbe@vice....|
+---+-----------+--------------------+
only showing top 5 rows
"""


########################################################################################################
# Build pipeline and run
indexer   = StringIndexer(inputCol="category", outputCol="label")
tokenizer = RegexTokenizer(pattern=u'\W+', inputCol="text", outputCol="words", toLowercase=False)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf       = IDF(inputCol="rawFeatures", outputCol="features")
lr        = LogisticRegression(maxIter=20, regParam=0.001)

# Builing model pipeline
pipeline = Pipeline(stages=[indexer, tokenizer, hashingTF, idf, lr])

# Train model on training set
model = pipeline.fit(train_df)   #if you give new names to your indexed datasets, make sure to make adjustments here

# Model prediction on test set
pred = model.transform(test_df)  # ...and here

# Model prediction accuracy (F1-score)
pl = pred.select("label", "prediction").rdd.cache()
metrics = MulticlassMetrics(pl)
metrics.fMeasure()



########################################################################################################
# Build pipeline and run
indexer   = StringIndexer(inputCol="category", outputCol="label") 
tokenizer = RegexTokenizer(pattern=u'\W+', inputCol="text", outputCol="words", toLowercase=False)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf       = IDF(inputCol="rawFeatures", outputCol="features")
lr        = LogisticRegression(maxIter=20, regParam=0.001)

# Builing model pipeline
pipeline = Pipeline(stages=[indexer, tokenizer, hashingTF, idf, lr])

# Train model on training set
model = pipeline.fit(train_df)   #if you give new names to your indexed datasets, make sure to make adjustments here

# Model prediction on test set
pred = model.transform(test_df)  # ...and here

# Model prediction accuracy (F1-score)
pl = pred.select("label", "prediction").rdd.cache()
metrics = MulticlassMetrics(pl)
metrics.fMeasure()  #F-1 score

#####################################################################################################
""" Task 1.2
a.	Run the model provided above. 
    Take your time to carefully understanding what is happening in this model pipeline.
    You are NOT allowed to make changes to this model's configurations.
    Compute and report the F1-score on the test dataset.
b.	Get and report the schema (column names and data types) of the model's prediction output.

""" 
# Your code for this part, IF ANY, starts here

"""
0.9483960948396095
pred.printSchema() 
root
 |-- category: string (nullable = true)
 |-- text: string (nullable = true)
 |-- label: double (nullable = false)
 |-- words: array (nullable = true)
 |    |-- element: string (containsNull = true)
 |-- rawFeatures: vector (nullable = true)
 |-- features: vector (nullable = true)
 |-- rawPrediction: vector (nullable = true)
 |-- probability: vector (nullable = true)
 |-- prediction: double (nullable = false)
 
 """
 
#######################################################################################################
#Use LIME to explain example
class_names = ['Atheism', 'Christian']
explainer = LimeTextExplainer(class_names=class_names)

# Choose a random text in test set, change seed for randomness 
test_point = test_df.sample(False, 0.1, seed = 10).limit(1)
test_point_label = test_point.select("category").collect()[0][0]
test_point_text = test_point.select("text").collect()[0][0]

def classifier_fn(data):
    spark_object = spark.createDataFrame(data, "string").toDF("text")
    pred = model.transform(spark_object)   #if you build the model with a different name, make appropriate changes here
    output = np.array((pred.select("probability").collect())).reshape(len(data),2)
    return output







exp = explainer.explain_instance(test_point_text, classifier_fn, num_features=6)
print('Probability(Christian) =', classifier_fn([test_point_text])[0][0])
print('True class: %s' % class_names[categories.index(test_point_label)])
exp.as_list()

#####################################################################################################
""" 
Task 1.3 : Output and report required details on test documents with ID’s 0, 275, and 664. ###udf
Task 1.4 : Generate explanations for all misclassified documents in the test set, sorted by conf in descending order, 
           and save this output (index, confidence, and LIME's explanation) to netID_misclassified_ordered.csv for submission.
"""
# Your code starts here

#################### 1.3 #########################################################


def LIME_EXP(test_point):
    test_point_label = test_point.select("category").collect()[0][0]
    test_point_text = test_point.select("text").collect()[0][0]
    exp = explainer.explain_instance(test_point_text, classifier_fn, num_features=6)
    print('Probability(Christian) =', classifier_fn([test_point_text])[0][0])
    print('True class: %s' % class_names[categories.index(test_point_label)])
    test=exp.as_list()
    print(test)
  
  
  
  
  
test_point=pred.filter("id like 0 or id like 275 or id like 664")
test_point.show() #df containing all the value like category etc.,

  


 
test_point=pred.filter("id like 0")  #we need Limexplainer for id 0
LIME_EXP(test_point)



test_point=pred.filter("id like 275")     #we need Limexplainer for id 275
LIME_EXP(test_point)




test_point=pred.filter("id like 664")    #we need LIME Explainer for 664 
LIME_EXP(test_point)




################################################        1.4  
pred_test=pred
pred_test.createOrReplaceTempView("pred_test")
query=spark.sql("select * from pred_test where label!=prediction")
query.show()


from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col


firstelement=udf(lambda v:float(v[0]),FloatType())
query2=query.select("*",firstelement('probability'))
query2.show()

query2 = query2.withColumnRenamed("<lambda>(probability)", "conf")

query2 = query2.withColumn('conf', 2*query2.conf-1)
from  pyspark.sql.functions import abs
query2 = query2.withColumn('conf', abs(query2.conf))
query2 = query2.sort(col("conf").desc())
query2.show()       


###############################################################################################################################


class_names = ['Atheism', 'Christian']
explainer = LimeTextExplainer(class_names=class_names)
def classifier_fn(data):
    spark_object = spark.createDataFrame(data, "string").toDF("text")
    pred = model.transform(spark_object)   #if you build the model with a different name, make appropriate changes here
    output = np.array((pred.select("probability").collect())).reshape(len(data),2)
    return output




	
x = sc.emptyRDD()
for val_idf in query2.select("id","text","conf").collect():
    op_lime = explainer.explain_instance(val_idf['text'], classifier_fn, num_features=6).as_list()
    x = x.union(sc.parallelize([(val_idf['id'], val_idf['conf'], op_lime)]))
    
    


    
x.collect()

def toCSVLine(data):
    return ','.join(str(d) for d in data)
    
    

    
lines = x.map(toCSVLine)

lines.saveAsTextFile('LIME_exp.csv')



########################################################################################################
""" Task 1.5
Get the word and summation weight and frequency
"""
# Your code starts here - dictionary of keys and values 
################################
list=x.collect()

words=[i[2] for i in list]


words2=[]
for i in range(len(words)):
   for k in range(len(words[i])):
        words2.append(words[i][k])
    
    
  

  
words3=[i[0] for i in words2] 

word_df = sqlContext.createDataFrame([(str(tup[0]), float(tup[1])) for tup in words2],["word", "weight"])

from pyspark.sql import functions as F

word_df = word_df.withColumn('weight',F.abs(word_df.weight))
word_df.createOrReplaceTempView("word_df")

word_query = "select word as word_col, count(word) as count_col, sum(weight) as weight_col \
                   from word_df group by word"
result = spark.sql(word_query)
result = result.orderBy(result.count_col.desc(),result.word_col) 
result.write.csv('Words_grouped_by_weight1.csv')


########################################################################################################
""" Task 2
Identify a feature-selection strategy to improve the model's F1-score.
Codes for your strategy is required
Retrain pipeline with your new train set (name it, new_train_df)
You are NOT allowed make changes to the test set
Give the new F1-score.
"""

#Your code starts here

################################# Task2 #####################################################

query3=query2.filter(query2.conf>=0.1)


class_names = ['Atheism', 'Christian']
explainer = LimeTextExplainer(class_names=class_names)
def classifier_fn(data):
    spark_object = spark.createDataFrame(data, "string").toDF("text")
    pred = model.transform(spark_object)   #if you build the model with a different name, make appropriate changes here
    output = np.array((pred.select("probability").collect())).reshape(len(data),2)
    return output




	
rdd_task2 = sc.emptyRDD()

for val_idf in query3.select("id","text","conf").collect():
    op_lime = explainer.explain_instance(val_idf['text'], classifier_fn, num_features=6).as_list()
    rdd_task2 = rdd_task2.union(sc.parallelize([(val_idf['id'], val_idf['conf'], op_lime)]))
    
    


    
rdd_task2.collect()



list2=rdd_task2.collect()

task2_words=[i[2] for i in list]


task2_words2=[]
for i in range(len(words)):
   for k in range(len(words[i])):
        task2_words2.append(words[i][k])
    
    
  

  
task2_words3=[i[0] for i in words2] 

task2_word_df = sqlContext.createDataFrame([(str(tup[0]), float(tup[1])) for tup in task2_words2],["word", "weight"])

from pyspark.sql import functions as F

task2_word_df = task2_word_df.withColumn('weight',F.abs(task2_word_df.weight))
task2_word_df.createOrReplaceTempView("task2_word_df")

task2_word_query = "select word as word_col, count(word) as count_col, sum(weight) as weight_col \
                   from word_df group by word"
result_task2 = spark.sql(task2_word_query)
result_task2 = result_task2.orderBy(result_task2.count_col.desc(),result_task2.word_col) 



from pyspark.sql.functions import desc
result_task2=result_task2.sort(desc("weight_col"))
task2_result_3=[str(row['word_col']) for row in result_task2.collect()]

task2_remove_list2=task2_result_3[:50]

cur_f1=0
max_f1=0
prev_f1=0.9483960948396095
candidates=0
removed_words_list=[]
unused_words_list=[]
prev_train_cleaned=train_df
curr_train_cleaned=train_df

for task2_word in task2_remove_list2:
    wordReplace= F.udf(lambda x: x.replace(task2_word, ' '))
    curr_train_cleaned=curr_train_cleaned.withColumn('text', wordReplace('text'))
    model=pipeline.fit(curr_train_cleaned)
    pred=model.transform(test_df)
    pl=pred.select("label","prediction").rdd.cache()
    metrics=MulticlassMetrics(pl)
    print(task2_word)
    cur_f1=metrics.fMeasure()
    if cur_f1>max_f1:
        max_f1=cur_f1
        print("after removing word %s, cur f1 score is %f, prev f1 is %f, and max f1 is %f"%(task2_word,cur_f1,prev_f1, max_f1))
        removed_words_list.append(task2_word)
        prev_f1=cur_f1
        prev_train_cleaned=curr_train_cleaned
        if cur_f1 > 0.97:
            break


    else:
        curr_train_cleaned=prev_train_cleaned
        unused_words_list.append(task2_word)







        
print("The words that contributed for improving F1-Score are" %removed_words_list)
print(" F1 score before feature selection : ", str(metrics.fMeasure()))
print(" F1 score after feature selection : ", str(max_f1))



def calculate_conf(pro):
    return str(abs(pro[0] - pro[1]))
    
    
 



 
calculate_conf_mapping = F.udf(lambda z: calculate_conf(z), StringType())

new_model = pipeline.fit(curr_train_cleaned) 
new_pred = new_model.transform(test_df) 
new_pl = new_pred.select("label", "prediction").rdd.cache()
new_metrics = MulticlassMetrics(new_pl)
final_F1score = new_metrics.fMeasure()
    
new_df_misclass = new_pred.filter(new_pred.label != new_pred.prediction)
new_df_misclass = (
    new_df_misclass.select('id',
              calculate_conf_mapping('probability').alias('conf'), 'category', 'text', 'label', 'words',
              'rawFeatures', 'features', 'rawPrediction','probability', 'prediction') )








new_df_misclass = new_df_misclass.withColumn("conf", new_df_misclass["conf"].cast(DoubleType()))
new_df_misclass = new_df_misclass.orderBy(new_df_misclass.conf.desc()) 

id_misclass = sorted(query3.filter(query3.conf >= 0.1).select('id').rdd.flatMap(lambda x: x).collect())
newid_misclass =sorted( new_df_misclass.filter(new_df_misclass.conf >= 0.1).select('id').rdd.flatMap(lambda x: x).collect())



print("New misclassified id are as follows :\n ", str(newid_misclass))
print("Old misclassified id are as follows :\n ", str(id_misclass))

id_correctly_classified = []
for var in id_misclass:
    if var not in newid_misclass:
        id_correctly_classified.append(var)






        
output_ids_now_correctly_classified = new_pred.where(F.col("id").isin(set(id_correctly_classified)))        
print(output_ids_now_correctly_classified.show())


for i in id_correctly_classified:
    ans_df = new_pred.filter(new_pred.id == i)
    test_point_text = ans_df.select("text").collect()[0][0]
    exp = explainer.explain_instance(test_point_text, classifier_fn, num_features=6)
    print('For id ',str(i),' :')  
    ground_truth = ans_df.select("label").collect()[0][0]
    print('(i) categories (ground truth) : ', predicted_categories[int(ground_truth)]) 
    prob_pred = ans_df.select("probability").collect()[0][0]
    print('(ii) probabilities over categories computed by the classifier :\n')
    print('Probability(Atheism) =', prob_pred[1])
    print('Probability(Christian) =',prob_pred[0])  
    prediction_label = ans_df.select("prediction").collect()[0][0]    
    print('(iii) predicted category for the document : ',predicted_categories[int(prediction_label)])
    print('(iv) LIMEs generated textual explanation, in terms of 6 features : ', str(exp.as_list()),'\n')

        
    
        
