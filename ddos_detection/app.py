from flask import Flask, render_template, request
import pandas as pd
import ipaddress
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
enc=LabelEncoder()
os=RandomOverSampler()
global df,path,data


app = Flask(__name__)

def preproceesing(file):
    file['proto']=enc.fit_transform(file['proto'])
    file['flgs'] = enc.fit_transform(file['flgs'])
    file['saddr'] = enc.fit_transform(file['saddr'])
    file['daddr'] = enc.fit_transform(file['daddr'])
    file['sport'] = enc.fit_transform(file['sport'])
    file['state'] = enc.fit_transform(file['state'])
    file['category'] = enc.fit_transform(file['category'])
    file['subcategory'] = enc.fit_transform(file['subcategory'])
    return file

def drop(file):
    file.drop(['Unnamed: 0'],axis=1,inplace=True)
    # Dropping Unnecessary column from the dataset
    return file

def splitting(file):
    X=file.drop(['attack','sport','dport'],axis=1)
    y=file['attack']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=52)
    return X_train,X_test,y_train,y_test


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/help')
def help():
    return render_template('help.html')
@app.route('/prediction')
def predict():
    return render_template('predict.html')


@app.route('/forminput', methods=["POST"])
def forminput():
    f1 = request.form.get('f1')
    print(f1)
    f2 = request.form.get('f2')
    print(f2)
    f3 = request.form.get('f3')
    print(f3)
    f4 = request.form.get('f4')
    print(f4)
    f5 = request.form.get('f5')
    print(f5)
    f6 = request.form.get('f6')
    f7 = request.form.get('f7')
    f8 = request.form.get('f8')
    f9 = request.form.get('f9')
    f10 = request.form.get('f10')
    f11 = request.form.get('f11')
    f12 = request.form.get('f12')
    f13 = request.form.get('f13')
    f14 = request.form.get('f14')
    f15 = request.form.get('f15')
    f16 = request.form.get('f16')
    f17 = request.form.get('f17')
    f18 = request.form.get('f18')
    f19 = request.form.get('f19')
    f20 = request.form.get('f20')
    f21 = request.form.get('f21')
    f22 = request.form.get('f22')
    f23 = request.form.get('f23')
    f24 = request.form.get('f24')
    f25 = request.form.get('f25')
    f26 = request.form.get('f26')
    f27 = request.form.get('f27')
    f28 = request.form.get('f28')
    f29 = request.form.get('f29')
    f30 = request.form.get('f30')
    f31 = request.form.get('f31')
    f32 = request.form.get('f32')
    f33 = request.form.get('f33')
    f34 = request.form.get('f34')
    f35 = request.form.get('f35')
    f36 = request.form.get('f36')
    f37 = request.form.get('f37')
    f38 = request.form.get('f38')
    f39 = request.form.get('f39')
    f40 = request.form.get('f40')
    f41 = request.form.get('f41')
    f42 = request.form.get('f42')
    f43 = request.form.get('f43')

    l = [float(f1), float(f2), float(f3), float(f4), float(f5), float(f6), float(f7), float(int(ipaddress.ip_address(f8))), float(f9),
             float(f10), float(f11), float(f12), float(f13), float(f14), float(f15), float(f16), float(f17), float(f18),
             float(f19), float(f20), float(f21), float(f22), float(f23),
             float(f24), float(f25), float(f26), float(f27), float(f28), float(f29), float(f30), float(f31), float(f32),
             float(f33), float(f34), float(f35), float(f36), float(f37), float(f38), float(f39), float(f40), float(f41),
             float(f42), float(f43)]

    df=pd.read_csv('ddos.csv')
    df.head()
    df.info()
    df.describe()
    pdf = preproceesing(df)
    print(pdf)
    ddf = drop(df)
    print(ddf)
    # X=splitting(file)
    # y=splitting(file)
    #X_train, X_test, y_train, y_test = splitting(df)
    #print(X_train)
    #print(y_test)
    X = df.drop(['attack', 'sport', 'dport'], axis=1)
    y = df['attack']
    X_train_res, y_train_res = os.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size=0.3, random_state=52)
    print(X_train)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dtpred = dt.predict(X_test)
    acs = accuracy_score(y_test, dtpred)
    print("Accuracy is ",acs * 100)
    dtpred = dt.predict([l])
    print(dtpred)
    if dtpred == [0]:
        return render_template('success.html')        
    return render_template('failure.html')
# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
	app.run(debug = True)

