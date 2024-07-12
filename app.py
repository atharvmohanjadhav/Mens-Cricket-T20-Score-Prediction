import streamlit as st
import pickle
import pandas as pd
import numpy as np
df = pd.read_csv(r"T20 International Dataset.csv")
df.drop(columns=["Unnamed: 0","powerPlay","AverageScore","innings"],inplace=True)

x = df.iloc[:,:-1]
y = df["Final_Score"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=30)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeRegressor
trf = ColumnTransformer([("trf",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),["battingTeam","bowlingTeam","city"])],remainder="passthrough")
from sklearn.pipeline import make_pipeline
dt = DecisionTreeRegressor()

pipe = make_pipeline(trf,dt)
pipe.fit(x_train,y_train)
#pipe = pickle.load(open("score_pre.pkl","rb"))

teams = ['Sri Lanka', 'South Africa', 'New Zealand', 'Pakistan','Australia', 'India', 'Bangladesh', 'England',
        'West Indies','Afghanistan','Netherlands','Zimbabwe','Ireland']
cities = ['Colombo', 'Durban', 'Chattogram', 'King City', 'Manchester','Harare', 'Bristol', 'Dhaka', 'Dubai', 'Auckland', 'Barbados',
       'Pallekele', 'Abu Dhabi', 'Christchurch', 'Sydney', 'Hambantota','Bengaluru', 'Lahore', 'Johannesburg', 'Centurion', 'Delhi',
       'Hamilton', 'Dominica', 'Birmingham', 'Brisbane', 'St Lucia','Mumbai', 'Mirpur', 'Ahmedabad', 'Lauderhill', 'Mount Maunganui',
       'Melbourne', 'Wellington', 'Perth', 'Hyderabad', 'Kandy','St Kitts', 'Nelson', 'Lucknow', 'Southampton', 'London', 'Indore',
       'Port Elizabeth', 'Taunton', 'Cardiff', 'Coolidge', 'Chandigarh','Cape Town', 'Kolkata', 'Nottingham', 'Sharjah', 'Dehradun',
       'Dharamsala', 'Gros Islet', 'Chennai', 'Dunedin', 'Karachi','Leeds', 'Trinidad', 'Visakhapatnam', 'Ranchi', 'Nagpur', 'Guyana',
       'Adelaide', 'Antigua', 'Rajkot', 'Bloemfontein','Chester-le-Street', 'Hobart', 'Napier', "St George's", 'Guwahati',
       'Kanpur', 'Potchefstroom', 'Canberra', 'Basseterre', 'Nairobi','Providence', 'Sylhet', 'St Vincent', 'Thiruvananthapuram',
       'Victoria', 'Paarl', 'Pune', 'Jamaica', 'Cuttack', 'Bridgetown', 'Carrara']

st.title("Mens T20 Score Predictor")

col1,col2 = st.columns(2)

with col1 :
    batting_team = st.selectbox("Batting Team",sorted(teams))
with col1 :
    Bowling_team = st.selectbox("Bowling Team",sorted(teams,reverse=True))

city = st.selectbox("Venue",sorted(cities))
col3,col4 ,col5 = st.columns(3)
with col3:
    current_score = st.number_input("Current Score")
with col4:
    balls_done = st.number_input("Balls Left")
with col5:
    CurrentRunRate	= st.number_input("Current RunRate")

col6,col7,col8 = st.columns(3)
with col6:
    wicketsLeft = st.number_input("wickets Left")
with col7:
    Run_In_Last5 = st.number_input("Runs In Last 5 Overs")
with col8:
    Wickets_In_Last5 = st.number_input("Wickets In Last 5 Overs")

if st.button("Predict Score"):
    input_data = pd.DataFrame({"battingTeam": [batting_team], "bowlingTeam": [Bowling_team], "city": [city],
                               "delivery_left": [balls_done], "score": [current_score],
                               "CurrentRunRate": [CurrentRunRate],
                               "wicketsLeft": [wicketsLeft], "Run_In_Last5": [Run_In_Last5],
                               "Wickets_In_Last5": [Wickets_In_Last5]})
    result = pipe.predict(input_data)
    st.header("Predicted Score : " + str(int(result)))

