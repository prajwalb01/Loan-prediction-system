#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()
from pyspark.sql import SparkSession


# In[2]:


spark=SparkSession.builder.appName("sproject").config("spark.some.config.option","some-value").getOrCreate()


# In[3]:


df=spark.read.csv("Loan.csv",header=True,inferSchema=True)


# In[4]:


df.toPandas().head()


# In[5]:


pf=df.toPandas()


# In[6]:


pf.head(5)


# In[7]:


gr_count=df[df.Education=="Graduate"].count()
ngr_count=df[df.Education=="Not Graduate"].count()
import matplotlib.pyplot as plt
y=[gr_count,ngr_count]
lb=["Graduate","Not Graduate"]
plt.pie(y,labels=lb)
plt.show()


# In[ ]:





# In[8]:


from pyspark.sql.functions import mean, md5
mean_val= df.select(mean(df['LoanAmount'])).collect()
mean_la=mean_val[0][0]
df=df.na.fill(mean_la,subset=['LoanAmount'])


# In[9]:


df.toPandas().head()


# In[10]:



#removing null in laon amount term
from pyspark.sql.functions import isnan, when, count, col
LA_counts = df.groupBy(['Loan_Amount_Term']).count().alias('counts')
LA_counts.sort(col("count").desc()).show()
LA_mode=LA_counts.agg({"count": "max"}).collect()[0][0]
print(LA_mode)
temp_LA = LA_counts.filter(LA_counts['count']==LA_mode)
temp_LA.printSchema()
LA_mode = temp_LA.select(['Loan_Amount_Term']).collect()[0][0]
df=df.na.fill(LA_mode,subset=['Loan_Amount_Term'])


# In[11]:


#remove null in gender
Gender_counts = df.groupBy(['Gender']).count().alias('counts')
Gender_counts.sort(col("count").desc()).show()
Gender_mode=Gender_counts.agg({"count": "max"}).collect()[0][0]
temp_Gender = Gender_counts.filter(Gender_counts['count']==Gender_mode)
Gender_mode = temp_Gender.select(['Gender']).collect()[0][0]
df=df.na.fill(Gender_mode,subset=['Gender'])


# In[12]:


#removing null in married by mode
Mar_counts = df.groupBy(['Married']).count().alias('counts')
Mar_counts.sort(col("count").desc()).show()
Mar_mode=Mar_counts.agg({"count": "max"}).collect()[0][0]
temp_mar = Mar_counts.filter(Mar_counts['count']==Mar_mode)
Mar_mode = temp_mar.select(['Married']).collect()[0][0]
df=df.na.fill(Mar_mode,subset=['Married'])


# In[13]:


#removing null in dependent
Dep_counts = df.groupBy(['Dependents']).count().alias('counts')
Dep_counts.sort(col("count").desc()).show()
Dep_mode=Dep_counts.agg({"count": "max"}).collect()[0][0]
temp_dep = Dep_counts.filter(Dep_counts['count']==Dep_mode)
Dep_mode = temp_dep.select(['Dependents']).collect()[0][0]
df=df.na.fill(Dep_mode,subset=['Dependents'])


# In[14]:


#Removing null in self employed
emp_counts = df.groupBy(['Self_Employed']).count().alias('counts')
emp_counts.sort(col("count").desc()).show()
emp_mode=emp_counts.agg({"count": "max"}).collect()[0][0]
temp_emp = emp_counts.filter(emp_counts['count']==emp_mode)
emp_mode = temp_emp.select(['Self_Employed']).collect()[0][0]
df=df.na.fill(emp_mode,subset=['Self_Employed'])


# In[15]:


#removing null in credit history
ch_counts = df.groupBy(['Credit_History']).count().alias('counts')
ch_counts.sort(col("count").desc()).show()
ch_mode=ch_counts.agg({"count": "max"}).collect()[0][0]
temp_ch = ch_counts.filter(ch_counts['count']==ch_mode)
ch_mode = temp_ch.select(['Credit_History']).collect()[0][0]
df=df.na.fill(ch_mode,subset=['Credit_History'])


# In[16]:


df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas().head()


# In[17]:


df.toPandas().head()


# In[18]:


#adding new feature total income
df_with_totalincome = df.withColumn('total_income', df['ApplicantIncome']+df['CoapplicantIncome'])
df_with_totalincome.toPandas().head()


# In[19]:


#Adding new feature ratio of total income to loan amount
df_with_ratio = df_with_totalincome.withColumn('ratio', df_with_totalincome['total_income']/df_with_totalincome['LoanAmount'])
df_with_ratio.toPandas().head()


# In[20]:


selected_data = df_with_ratio.select('Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'ApplicantIncome',
 'CoapplicantIncome',
 'LoanAmount',
 'Loan_Amount_Term',
 'Credit_History',
 'Property_Area',
 'Loan_Status','total_income','ratio')
selected_data.toPandas().head()


# In[21]:


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder, StringIndexer
final = selected_data.where(selected_data.Gender.isNotNull())
final = final.where(final.Married.isNotNull())
final = final.where(final.Dependents.isNotNull())
final = final.where(final.Education.isNotNull())
final = final.where(final.Self_Employed.isNotNull())
final.toPandas().head()


# In[22]:


#dummy coding gender Male =0 ; Female =1
gen_indexer = StringIndexer(inputCol="Gender", outputCol="_Gender_index" )
gen_model = gen_indexer.fit(final)
gen_indexed = gen_model.transform(final)
gen_encoder = OneHotEncoder( inputCol="_Gender_index", outputCol="_Gender_vec")
gen_encoder.setDropLast(False)
ohn=gen_encoder.fit(gen_indexed)
final1 = ohn.transform(gen_indexed)
final1.toPandas().head()


# In[23]:


#Dummy coding Married Yes =0, No =1
mar_indexer = StringIndexer(inputCol="Married", outputCol="_Married_index" )
mar_model = mar_indexer.fit(final1)
mar_indexed = mar_model.transform(final1)
mar_encoder = OneHotEncoder( inputCol="_Married_index", outputCol="_Married_vec")
mar_encoder.setDropLast(False)
ohn2=mar_encoder.fit(mar_indexed)
final2 = ohn2.transform(mar_indexed)
final2.toPandas().head()


# In[24]:


#Dummy coding Dependent
dep_indexer = StringIndexer(inputCol="Dependents", outputCol="_Dependents_index" )
dep_model = dep_indexer.fit(final2)
dep_indexed = dep_model.transform(final2)
dep_encoder = OneHotEncoder( inputCol="_Dependents_index", outputCol="_Dependents_vec")
dep_encoder.setDropLast(False)
ohn3=dep_encoder.fit(dep_indexed)
final3 = ohn3.transform(dep_indexed)
final3.toPandas().head()


# In[25]:


#Dummy coding Education graduate =0 ; not graduate =1
edu_indexer = StringIndexer(inputCol="Education", outputCol="_Education_index" )
edu_model = edu_indexer.fit(final3)
edu_indexed = edu_model.transform(final3)
edu_encoder = OneHotEncoder( inputCol="_Education_index", outputCol="_Education_vec")
edu_encoder.setDropLast(False)
ohn4=edu_encoder.fit(edu_indexed)
final4 = ohn4.transform(edu_indexed)
final4.toPandas().head()


# In[26]:


#Dummy coding Self employed No =0 ; Yes= 1
emp_indexer = StringIndexer(inputCol="Self_Employed", outputCol="_Self_Employed_index" )
emp_model = emp_indexer.fit(final4)
emp_indexed = emp_model.transform(final4)
emp_encoder = OneHotEncoder( inputCol="_Self_Employed_index", outputCol="_Self_Employed_vec")
emp_encoder.setDropLast(False)
ohn5=emp_encoder.fit(emp_indexed)
final5 = ohn5.transform(emp_indexed)
final5.toPandas().head()


# In[27]:


#Dummy coding Property Area Urban =1; Rural =0;semiurban =2
area_indexer = StringIndexer(inputCol="Property_Area", outputCol="_Property_Area_index" )
area_model = area_indexer.fit(final5)
area_indexed = area_model.transform(final5)
area_encoder = OneHotEncoder( inputCol="_Property_Area_index", outputCol="_Property_Area_vec")
area_encoder.setDropLast(False)
ohn6=area_encoder.fit(area_indexed)
final6 = ohn6.transform(area_indexed)
final6.toPandas().head()


# In[28]:


loan_indexer = StringIndexer(inputCol="Loan_Status", outputCol="_Loan_Status_index" )
loan_model = loan_indexer.fit(final6)
loan_indexed = loan_model.transform(final6)
loan_encoder = OneHotEncoder( inputCol="_Loan_Status_index", outputCol="_Loan_Status_vec")
loan_encoder.setDropLast(False)
ohn7=loan_encoder.fit(loan_indexed)
final7 = ohn7.transform(loan_indexed)
final7.toPandas().head()


# In[29]:


final7.columns


# In[30]:


assembler = VectorAssembler(inputCols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','total_income','ratio',
 '_Gender_vec', 
 '_Married_vec', 
 '_Dependents_vec','_Education_vec','_Self_Employed_vec','_Property_Area_vec'], outputCol = 'features')


# In[31]:


final8 = assembler.transform(final7)


# In[32]:


final8.columns


# In[33]:


new_final= final8.select("features","_Loan_Status_index")
new_final.toPandas().head()


# In[34]:


train_data, test_data = new_final.randomSplit([0.7, 0.3])


# In[35]:


test_data.toPandas().head(30)


# In[36]:


lr = LogisticRegression(featuresCol = 'features', labelCol = '_Loan_Status_index', maxIter = 10)
train_data.printSchema()
lr_Model = lr.fit(train_data)
pred=lr_Model.transform(test_data)
pred.select("features","_Loan_Status_index").toPandas().head(30)


# In[37]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
my_eval=BinaryClassificationEvaluator(labelCol="_Loan_Status_index")
         


# In[38]:


logi=my_eval.evaluate(pred)*100
logi


# In[39]:


#Random forest
from pyspark.ml.classification import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = RandomForestClassifier(labelCol = "_Loan_Status_index", featuresCol = "features")
rfc_model = rfc.fit(train_data)
rfc_=rfc_model.transform(test_data)
rfc_.select("features","_Loan_Status_index").toPandas().head()


# In[40]:


my_eval3 = BinaryClassificationEvaluator(labelCol = '_Loan_Status_index')
ran=my_eval3.evaluate(rfc_)*100
ran


# In[41]:


#decision tree
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier
dtc = DecisionTreeClassifier()
dtc = DecisionTreeClassifier(labelCol = "_Loan_Status_index", featuresCol = "features")
dtc_model = dtc.fit(train_data)
dtc_pred = dtc_model.transform(test_data)
dtc_pred.select("features","_Loan_Status_index").toPandas().head()


# In[42]:


my_eval2 = BinaryClassificationEvaluator(labelCol = '_Loan_Status_index')
dec=my_eval2.evaluate(dtc_pred)*100
dec


# In[43]:


#Gradient Boosting
gbc = GBTClassifier
gbc = GBTClassifier(labelCol = "_Loan_Status_index", featuresCol = "features")
gbc_model = gbc.fit(train_data)
gbc_pred = gbc_model.transform(test_data)
gbc_pred.select("features","_Loan_Status_index").toPandas().head()


# In[44]:


my_eval4 = BinaryClassificationEvaluator(labelCol = '_Loan_Status_index')
gra=my_eval4.evaluate(gbc_pred)*100
gra


# In[45]:


li=[logi,ran,dec]
la=["logisticRegression","RandomForest"," decisionTree"]
plt.bar(la,li)
plt.show()
li


# In[46]:


pf.head(8)


# In[47]:


pf["Dependents"].fillna(0,inplace=True)


# In[48]:


pf["Education"].fillna("Not Graduate",inplace=True)
pf["Self_Employed"].fillna("No",inplace=True)
pf["ApplicantIncome"].fillna(pf["ApplicantIncome"].mean(),inplace=True)
pf["LoanAmount"].fillna(pf["LoanAmount"].mean(),inplace=True)
pf["Credit_History"].fillna(pf["Credit_History"].mean(),inplace=True)
pf["Property_Area"].fillna("Rural",inplace=True)
pf["Loan_Status"].fillna("Y",inplace=True)


# In[49]:


pf1=pf.drop(["Loan_ID","Gender","Married","CoapplicantIncome","Loan_Amount_Term"],axis=1)


# In[50]:


pf1.columns


# In[51]:


pf1.head(10)


# In[52]:


#testing the na values
pf1["Dependents"].isna().sum()
pf1["Education"].isna().sum()
pf1["Self_Employed"].isna().sum()
pf1["Self_Employed"].isna().sum()
pf1["ApplicantIncome"].isna().sum()
pf1["LoanAmount"].isna().sum()
pf1["Credit_History"].isna().sum()
pf1["Property_Area"].isna().sum()
pf1["Loan_Status"].isna().sum()


# In[53]:


from sklearn.preprocessing import LabelEncoder


# In[54]:


edu=LabelEncoder()
sel_f=LabelEncoder()
pr_area=LabelEncoder()
loan_s=LabelEncoder()
dep=LabelEncoder()


# In[55]:


pf1["Education_"]=edu.fit_transform(pf1["Education"])
pf1["Self_Employed_"]=sel_f.fit_transform(pf1["Self_Employed"])
pf1["Property_Area_"]=pr_area.fit_transform(pf1["Property_Area"])
pf1["Loan_Status_"]=loan_s.fit_transform(pf1["Loan_Status"])
pf1["Dependents_"]=dep.fit_transform(pf1["Dependents"])



# In[56]:


pf2=pf1.drop(["Education","Self_Employed","Property_Area","Loan_Status","Dependents"],axis=1)


# In[57]:


pf2.columns
pf2.head()


# In[58]:


from sklearn.ensemble import RandomForestClassifier


# In[59]:


classifier=RandomForestClassifier(n_estimators=20)
classifier.fit(pf2[["Dependents_", "ApplicantIncome", "LoanAmount", "Credit_History", "Education_", "Self_Employed_", "Property_Area_"]],pf2["Loan_Status_"])


# In[60]:


data=classifier.predict([[1,4583,128,1,0,0,0]])
data


# In[ ]:




