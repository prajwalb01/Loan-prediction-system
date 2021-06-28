from flask import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
app=Flask(__name__)
@app.route("/home")
def home():
	return render_template("home.html")
@app.route("/predict_", methods=["POST","GET"])
def predict():
	if request.method=="POST":
		dep=request.form["dep"]
		edu=request.form["edu"]
		self_em=request.form["sel"]
		inc=request.form["inc"]
		loan_a=request.form["loanam"]
		cre_hist=request.form["crehist"]
		area=request.form["area"]
		pf=pd.read_csv("loan.csv")
		pf["Dependents"].fillna(0,inplace=True)
		pf["Education"].fillna("Not Graduate",inplace=True)
		pf["ApplicantIncome"].fillna(pf["ApplicantIncome"].mean(),inplace=True)
		pf["LoanAmount"].fillna(pf["LoanAmount"].mean(),inplace=True)
		pf["Credit_History"].fillna(pf["Credit_History"].mean(),inplace=True)
		pf["Property_Area"].fillna("Rural",inplace=True)
		pf["Loan_Status"].fillna("Y",inplace=True)
		#select the feature column
		pf1=pf.drop(["Loan_ID","Gender","Married","CoapplicantIncome","Loan_Amount_Term"],axis=1)
		#label encoding
		educ=LabelEncoder()
		sel_f=LabelEncoder()
		pr_area=LabelEncoder()
		loan_s=LabelEncoder()
		depe=LabelEncoder()
		pf1["Education_"]=educ.fit_transform(pf1["Education"])
		pf1["Self_Employed_"]=sel_f.fit_transform(pf1["Self_Employed"])
		pf1["Property_Area_"]=pr_area.fit_transform(pf1["Property_Area"])
		pf1["Loan_Status_"]=loan_s.fit_transform(pf1["Loan_Status"])
		pf1["Dependents_"]=depe.fit_transform(pf1["Dependents"])
		#select the encoded features
		pf2=pf1.drop(["Education","Self_Employed","Property_Area","Loan_Status","Dependents"],axis=1)
		#training the model
		classifier=RandomForestClassifier(n_estimators=20)
		classifier.fit(pf2[["Dependents_", "ApplicantIncome", "LoanAmount", "Credit_History", "Education_", "Self_Employed_", "Property_Area_"]],pf2["Loan_Status_"])
		data=classifier.predict([[dep,inc,loan_a,cre_hist,edu,self_em,area]])
		if data==1:
			return render_template("can_lend.html",depart=dep,educt=edu,self_e=self_em,income=inc,loan_am=loan_a,credit=cre_hist,prop=area)
		if data==0:
			return render_template("no_loan_A.html",depart=dep,educt=edu,self_e=self_em,income=inc,loan_am=loan_a,credit=cre_hist,prop=area)
			



if __name__ == '__main__':
	app.run(debug=True)	
	 #http://127.0.0.1:5000/