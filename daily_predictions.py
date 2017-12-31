import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.models import model_from_json
from keras.models import load_model
import joblib
import warnings
warnings.filterwarnings('ignore')

def get_games():
    sched = pd.read_csv("Data/nba_schedule.csv")
    sched = sched[['Date', 'Home/Neutral', 'Visitor/Neutral']]
    sched.rename(columns={'Visitor/Neutral': 'Visitor', 
                          'Home/Neutral': 'Home'}, inplace=True)
    sched['Date'] = pd.to_datetime(sched['Date'])

    today = dt.datetime.today()
    today = str(today.year) + "-" + str(today.month) + "-" + str(today.day)
    today_df = sched[sched['Date'] == today]

    todays_games = []
    for i in range(len(today_df)):
        todays_games.append((today_df.iloc[i]['Home'], today_df.iloc[i]['Visitor']))

    return todays_games

# build a list of today's games (note: Home team is listed first in the tuple)
todays_games = get_games()

# load the pre-trained models
clf_lr = joblib.load('logreg_model.pkl')
clf_rf = joblib.load('random_forest_model.pkl')
clf_gb = joblib.load('gradient_boost_model.pkl')

# load the CNN model
cnn_model = load_model("cnnModel.h5")


# read in and process the current season data
team_data_df = pd.read_csv("Data/team_stats_2018.csv")

min_max_scaler = MinMaxScaler()
for col in team_data_df.columns.tolist()[1:-1]:
    team_data_df[col] = min_max_scaler.fit_transform(team_data_df[col].reshape(-1, 1))
    

# function to make predictions
def make_prediction(team1, team2):
    team_one = team_data_df[team_data_df['Team'] == team1]
    team_one.reset_index(inplace=True, drop=True)
    team_one.drop('Team', axis=1, inplace=True)
    team_two = team_data_df[team_data_df['Team'] == team2]
    team_two.reset_index(inplace=True, drop=True)
    team_two.drop('Team', axis=1, inplace=True)
    game_data = pd.merge(team_one, team_two, left_index=True, right_index=True)

    logreg_probs = clf_lr.predict_proba(game_data)
    forest_probs = clf_rf.predict_proba(game_data)
    gradboost_probs = clf_gb.predict_proba(game_data)
    cnn_probs = cnn_model.predict_proba(
        game_data.as_matrix().reshape(game_data.shape[0], game_data.shape[1], 1), 
        batch_size=1)
                
    team_one_wins = [logreg_probs[0][0],
    				 forest_probs[0][0],
                     gradboost_probs[0][0],
                     cnn_probs[0][0]]   

    team_two_wins = [logreg_probs[0][1],
    				 forest_probs[0][1],
                     gradboost_probs[0][1],
                     cnn_probs[0][1]]

    return (team_one_wins, team_two_wins)


for game in todays_games:
    team_1 = game[0]
    team_2 = game[1]

    predictions = make_prediction(team_1.lower().replace(" ", "-"), 
                                  team_2.lower().replace(" ", "-"))

    lr_team1 = round(predictions[0][0], 3)
    lr_team2 = round(predictions[1][0], 3)

    rf_team1 = round(predictions[0][1], 3)
    rf_team2 = round(predictions[1][1], 3)

    gb_team1 = round(predictions[0][2], 3)
    gb_team2 = round(predictions[1][2], 3)

    cnn_team1 = round(predictions[0][3], 3)
    cnn_team2 = round(predictions[1][3], 3)

    avg_team1 = round((lr_team1 + rf_team1 + gb_team1 + cnn_team1) / 4., 3)
    avg_team2 = round((lr_team2 + rf_team2 + gb_team2 + cnn_team2) / 4., 3)

    print()
    print("\t\t\t " + str(team_1) + "\t\t\t" + str(team_2))
    print("Log Reg:" "\t\t  " + str(lr_team1) + "\t\t\t\t  " + str(lr_team2))
    print("Random Forest:" "\t\t  " + str(rf_team1) + "\t\t\t\t  " + str(rf_team2))
    print("GradBoost:" "\t\t  " + str(gb_team1) + "\t\t\t\t  " + str(gb_team2))
    print("CNN:" "\t\t\t  " + str(cnn_team1) + "\t\t\t\t  " + str(cnn_team2))
    print("-" * 70)
    print("Average:" "\t\t  " + str(avg_team1) + "\t\t\t\t  " + str(avg_team2))
