

sep = "sep"
cols = "cols"
target = "target"
label = ""
params = "parameters"
datasetpath = "datasets"
lambd = "lambda"
alpha = "alpha"
network = "network"
cancer = "hw3_cancer.csv"
house_votes ="hw3_house_votes_84.csv"
wine = "hw3_wine.csv"
n_features="n_features"
# loan = "loan.csv"
# digits="digits"
# titanic = "titanic.csv"
# parkinsons = "parkinsons.csv"
datasets_dict = {
cancer:{
      sep:"\t",
      cols:"",
      target:"class",
   },
   house_votes:{
      sep:",",
      cols:"",
      target:"class",
   },
   wine:{
      sep:"\\t",
      cols:"",
      target:"class",
   },

#    "digits":{
#       sep:"",
#       cols:"",
#       target:"",
#    },
   # "loan.csv":{
   #    sep:",",
   #    cols:"",
   #    target:"Loan_Status",
   # },
#    "titanic.csv":{
#       sep:",",
#       cols:"",
#       target:"Survived",

#    },
#    "parkinsons.csv":{
#       sep:",",
#       cols:"",
#       target:"Diagnosis",
#    }
}
