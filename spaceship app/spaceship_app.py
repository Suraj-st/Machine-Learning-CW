import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
import mlflow.sklearn

# MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Titanic_Spaceship")

# Load the production model
production_model_name = "HIST Gradient boost Model Data"
prod_model_uri = f"models:/{production_model_name}@prod"
loaded_model = mlflow.sklearn.load_model(prod_model_uri)

st.write("""
# Titanic Spaceship Survival Prediction App

This app predicts the **Titanic Spaceship Survival** in the Space!
""")

st.sidebar.header('User Input Features')


uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        PassengerId = st.sidebar.text_input('Passenger ID', '0001_01')
        HomePlanet = st.sidebar.selectbox('Home Planet', ('Earth', 'Europa', 'Mars'))
        CryoSleep = st.sidebar.selectbox('Cryo Sleep', [False, True])
        Cabin = st.sidebar.text_input('Cabin', 'Deck/Num/Side')
        Destination = st.sidebar.selectbox('Destination', ('TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22'))
        VIP = st.sidebar.selectbox('VIP', [False, True])
        Name = st.sidebar.text_input('Name', 'FirstName FamilyName')
        Age = st.sidebar.slider('Age', 0, 100, 0)
        RoomService = st.sidebar.slider('Room Service', 0.00, 100000.00, 0.00)
        FoodCourt = st.sidebar.slider('Food Court', 0.00, 100000.00, 0.00)
        ShoppingMall = st.sidebar.slider('Shopping mall', 0.00, 100000.00, 0.00)
        Spa = st.sidebar.slider('Spa', 0.00, 100000.00, 0.00)
        VRDeck = st.sidebar.slider('VR Deck', 0.00, 100000.00, 0.00)
        data = {'PassengerId': PassengerId,
                'HomePlanet': HomePlanet,
                'CryoSleep': CryoSleep,
                'Cabin': Cabin,
                'Destination': Destination,
                'Age': Age,
                'VIP': VIP,
                'RoomService': RoomService,
                'FoodCourt': FoodCourt,
                'ShoppingMall': ShoppingMall,
                'Spa': Spa,
                'VRDeck': VRDeck,
                'Name': Name,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

train_df = pd.read_csv(r"train.csv")
test_df = pd.read_csv(r"test.csv")

train = train_df.copy()
test = test_df.copy()
train_test = pd.concat([train, test], axis=0, ignore_index=True)

spaceship_raw = train_df.copy()
spaceship = spaceship_raw.drop(columns=['Transported'])
spaceship = pd.concat([input_df, spaceship], axis=0)

# Passenger ID
spaceship['PassengerGroup'] = spaceship.PassengerId.str.split('_',expand=True)[0].astype('int')

# HomePlanet
spaceship['HomePlanet'] = spaceship['HomePlanet'].fillna(value='Unknown')

# CryoSleep
spaceship['CryoSleep'] = spaceship['CryoSleep'].fillna(value=False)

# Cabin
spaceship['CabinDeck'] = spaceship['Cabin'].str.split('/',expand=True)[0]
spaceship['CabinDeck'] = spaceship['CabinDeck'].fillna(value='U')

spaceship['CabinSide'] = spaceship['Cabin'].str.split('/',expand=True)[2]
spaceship['CabinSide'] = spaceship['CabinSide'].fillna(value='U')

# Destination
dest_dic = {'TRAPPIST-1e':'A','55 Cancri e':'B','PSO J318.5-22':'C'}
spaceship['Destination'] = spaceship['Destination'].map(dest_dic)
spaceship['Destination'] = spaceship['Destination'].fillna(value='U')

# Age
spaceship['Age'] = spaceship['Age'].fillna(spaceship.groupby('PassengerGroup')['Age'].transform('median'))
spaceship['Age'] = spaceship['Age'].fillna(spaceship.groupby('HomePlanet')['Age'].transform('median'))

spaceship['Adult'] = 1
spaceship.loc[train['Age']<18, 'Adult'] = 0

# VIP
spaceship['VIP'] = spaceship['VIP'].fillna(value=False)

# RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
spaceship[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = spaceship[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(value=0)

# Total Spend
spaceship['TotalSpend'] = spaceship['RoomService']+spaceship['FoodCourt']+spaceship['ShoppingMall']+spaceship['Spa']+spaceship['VRDeck']

# Name
spaceship['FamilyName'] = spaceship['Name'].str.split(' ',expand=True)[1]
spaceship['FamilyName'] = spaceship['FamilyName'].fillna('Unknown')

train_test['FamilyName'] = train_test['Name'].str.split(' ',expand=True)[1]
train_test['FamilyName'] = train_test['FamilyName'].fillna('Unknown')

family_name_dict = train_test['FamilyName'].value_counts().to_dict()
family_name_dict['Unknown'] = 0

spaceship['FamilyMember'] = spaceship['FamilyName']
spaceship['FamilyMember'] = spaceship['FamilyMember'].map(family_name_dict)

spaceship = spaceship.drop(['PassengerId','PassengerGroup','Cabin','Name','FamilyName'],axis=1)

# Convert Bool to Int
spaceship[['CryoSleep','VIP']] = spaceship[['CryoSleep','VIP']].astype(int)

# Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

spaceship['Age'] = sc.fit_transform(spaceship['Age'].values.reshape(-1, 1))
spaceship['RoomService'] = sc.fit_transform(spaceship['RoomService'].values.reshape(-1, 1))
spaceship['FoodCourt'] = sc.fit_transform(spaceship['FoodCourt'].values.reshape(-1, 1))
spaceship['ShoppingMall'] = sc.fit_transform(spaceship['ShoppingMall'].values.reshape(-1, 1))
spaceship['Spa'] = sc.fit_transform(spaceship['Spa'].values.reshape(-1, 1))
spaceship['VRDeck'] = sc.fit_transform(spaceship['VRDeck'].values.reshape(-1, 1))
spaceship['TotalSpend'] = sc.fit_transform(spaceship['TotalSpend'].values.reshape(-1, 1))
spaceship['FamilyMember'] = sc.fit_transform(spaceship['FamilyMember'].values.reshape(-1, 1))

# Encoding
spaceship['HomePlanet'] = spaceship['HomePlanet'].map({'Earth': 0, 'Europa': 1, 'Mars':2, 'Unknown':3})
spaceship['Destination'] = spaceship['Destination'].map({'A': 0, 'B': 1, 'C':2, 'U':3})
spaceship['CabinDeck'] = spaceship['CabinDeck'].map({'A': 0, 'B': 1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7, 'U':8})
spaceship['CabinSide'] = spaceship['CabinSide'].map({'P': 0, 'S': 1, 'U':2})


spaceship = spaceship[:1]

st.subheader('User Input features')

if uploaded_file is not None:
    st.write(spaceship)
else:
    st.write('Awaiting CSV file to be uploaded')
    st.write(spaceship)
    st.write('---')

# load_clf = pickle.load(open('spaceship.pkl', 'rb'))

prediction = loaded_model.predict(spaceship)
prediction_proba = loaded_model.predict_proba(spaceship)


targets = ['False', 'True']
st.subheader('Transported')
columns = ['Transported']
targets_df = pd.DataFrame(targets, columns = columns)
st.write(targets_df)
st.write('---')


st.subheader('Prediction')
student_result = np.array( ['False', 'True'])
col = ['Result']
results = pd.DataFrame(student_result[prediction], columns = col)
st.write(results)
st.write('---')

st.subheader('Prediction probability')
cols = ['False', 'True']
result = pd.DataFrame(prediction_proba, columns = cols)
st.write(result)

st.write('---')


