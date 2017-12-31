import numpy as np
import pandas as pd
import datetime as dt
import re
import requests
from bs4 import BeautifulSoup
from bs4 import Comment
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# data ranges from 2002 to 2018
#years = range(2002, 2019)
years = [2018]

misc_head_names = ['Team', 'Age', 'PW', 'PL', 'MOV', 'SOS', 'SRS', 'ORtg', 'DRtg', 
				   'Pace', 'FTr', '3PAr', 'TS%', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 
				   'eFG%', 'TOV%', 'DRB%', 'FT/FGA']

team_shoot_head_names = ['Team','FG%','Dist.','2P','0-3','3-10','10-16','16 <3','3P',
               '2P','0-3','3-10','10-16','16 <3','3P',"%Ast'd",'dunks_%FGA','Md.',
               'layups_%FGA','Md.',"%Ast'd",'%3PA','3P%']

opp_shoot_head_names = ['Team', 'opp_FG%','opp_Dist.','opp_2P','opp_0-3','opp_3-10',
						'opp_10-16','opp_16 <3','opp_3P','opp_2P','opp_0-3','opp_3-10',
						'opp_10-16','opp_16 <3','opp_3P',"opp_%Ast'd",'opp_dunks_%FGA',
						'opp_Md.','opp_layups_%FGA','opp_Md.',"opp_%Ast'd",'opp_%3PA','opp_3P%']

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

for year in years:
    year = str(year)
    print("Collecting data for " + str(year))
    url = "http://www.basketball-reference.com/leagues/NBA_"+year+".html"
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data)
    
    # stats are hidden in the html comments
    comments = soup.find_all(string = lambda text:isinstance(text, Comment))
    
    # ===================
    # MISCELLANEOUS STATS
    # ===================
    
    for comment in comments:
        if "Miscellaneous Stats" in comment:
            misc_stats = BeautifulSoup(comment)
            break
                
    # team abbreviations
    team_abbrevs = []
    for a in misc_stats.find_all('a', href=True):
        team_abbrevs.append(a['href'][7:10])

    # team names
    team_names = misc_stats.findAll("td", {"data-stat": "team_name"})
    team_names = [node.getText().lower().replace("*", "").replace(" ", "-") for node in team_names]

    # team average age
    team_age = misc_stats.findAll("td", {"data-stat": "age"})
    team_age = [num(node.getText()) for node in team_age]

    # team Pythagorean Wins
    team_PW = misc_stats.findAll("td", {"data-stat": "wins_pyth"})
    team_PW = [num(node.getText()) for node in team_PW]

    # team Pythagorean Losses
    team_PL = misc_stats.findAll("td", {"data-stat": "losses_pyth"})
    team_PL = [num(node.getText()) for node in team_PL]

    # team Margin-of-Victory
    team_MOV = misc_stats.findAll("td", {"data-stat": "mov"})
    team_MOV = [num(node.getText()) for node in team_MOV]

    # team Strength-of-Schedule
    team_SOS = misc_stats.findAll("td", {"data-stat": "sos"})
    team_SOS = [num(node.getText()) for node in team_SOS]

    # team Simple Rating System
    team_SRS = misc_stats.findAll("td", {"data-stat": "srs"})
    team_SRS = [num(node.getText()) for node in team_SRS]

    # team Offensive Rating
    team_ORtg = misc_stats.findAll("td", {"data-stat": "off_rtg"})
    team_ORtg = [num(node.getText()) for node in team_ORtg]

    # team Defensive Rating
    team_DRtg = misc_stats.findAll("td", {"data-stat": "def_rtg"})
    team_DRtg = [num(node.getText()) for node in team_DRtg]

    # team Pace of play
    team_pace = misc_stats.findAll("td", {"data-stat": "pace"})
    team_pace = [num(node.getText()) for node in team_pace]

    # team Free Throw Attempt Rate  (FT attempted per FG attempted)
    team_FTr = misc_stats.findAll("td", {"data-stat": "fta_per_fga_pct"})
    team_FTr = [num(node.getText()) for node in team_FTr]

    # team Three Point Attempt Rate  (3P attempted per FG attempted)
    team_3PAr = misc_stats.findAll("td", {"data-stat": "fg3a_per_fga_pct"})
    team_3PAr = [num(node.getText()) for node in team_3PAr]

    # team True Shooting Percentage  (takes into account 2PA, 3PA and FTA)
    team_TS = misc_stats.findAll("td", {"data-stat": "ts_pct"})
    team_TS = [num(node.getText()) for node in team_TS]

    # Team Effective Field Goal Percentage
    team_eFG = misc_stats.findAll("td", {"data-stat": "efg_pct"})
    team_eFG = [num(node.getText()) for node in team_eFG]

    # Team Turnover Percentage
    team_TOV = misc_stats.findAll("td", {"data-stat": "tov_pct"})
    team_TOV = [num(node.getText()) for node in team_TOV]

    # team Offensive Rebound Percentage
    team_ORB = misc_stats.findAll("td", {"data-stat": "orb_pct"})
    team_ORB = [num(node.getText()) for node in team_ORB]

    # team Free Throw per Field Goal Attempt
    team_FTFGA = misc_stats.findAll("td", {"data-stat": "ft_rate"})
    team_FTFGA = [num(node.getText()) for node in team_FTFGA]

    # Opponent Effective Field Goal Percentage
    opp_eFG = misc_stats.findAll("td", {"data-stat": "opp_efg_pct"})
    opp_eFG = [num(node.getText()) for node in opp_eFG]

    # Opponent Turnover Percentage
    opp_TOV = misc_stats.findAll("td", {"data-stat": "opp_tov_pct"})
    opp_TOV = [num(node.getText()) for node in opp_TOV]

    # team Defensive Rebound Percentage
    team_DRB = misc_stats.findAll("td", {"data-stat": "drb_pct"})
    team_DRB = [num(node.getText()) for node in team_DRB]

    # Opponent Free Throw per Field Goal Attempt
    opp_FTFGA = misc_stats.findAll("td", {"data-stat": "opp_ft_rate"})
    opp_FTFGA = [num(node.getText()) for node in opp_FTFGA]
    
    misc_stats_list = [team_names, team_age, team_PW, team_PL, team_MOV,
                  team_SOS, team_SRS, team_ORtg, team_DRtg, team_pace,
                  team_FTr, team_3PAr, team_TS, team_eFG, team_TOV,
                  team_ORB, team_FTFGA, opp_eFG, opp_TOV, team_DRB,
                  opp_FTFGA]

    misc_dict = {}
    for i, head in enumerate(misc_head_names):
        misc_dict[head] = misc_stats_list[i]

    misc_data = pd.DataFrame(misc_dict)

    misc_data = misc_data[misc_head_names]
    
    
    # ===================
    # Team Shooting STATS
    # ===================
    
    for comment in comments:
        if "Team Shooting" in comment:
            team_shoot_stats = BeautifulSoup(comment)
            break
    
    # team names
    team_names = team_shoot_stats.findAll("td", {"data-stat": "team_name"})
    team_names = [node.getText().lower().replace("*", "").replace(" ", "-") for node in team_names]

    # team FGpct
    team_FGpct = team_shoot_stats.findAll("td", {"data-stat": "fg_pct"})
    team_FGpct = [num(node.getText()) for node in team_FGpct]

    # team avg shot distance
    team_AvgDist = team_shoot_stats.findAll("td", {"data-stat": "avg_dist"})
    team_AvgDist = [num(node.getText()) for node in team_AvgDist]

    # team % of FGA that are 2P
    team_2PFGA = team_shoot_stats.findAll("td", {"data-stat": "fg2a_pct_fga"})
    team_2PFGA = [num(node.getText()) for node in team_2PFGA]

    # team % of FGA that are between 0-3 feet
    team_03FGA = team_shoot_stats.findAll("td", {"data-stat": "pct_fga_00_03"})
    team_03FGA = [num(node.getText()) for node in team_03FGA]

    # team % of FGA that are between 3-10 feet
    team_310FGA = team_shoot_stats.findAll("td", {"data-stat": "pct_fga_03_10"})
    team_310FGA = [num(node.getText()) for node in team_310FGA]

    # team % of FGA that are between 10-16 feet
    team_1016FGA = team_shoot_stats.findAll("td", {"data-stat": "pct_fga_10_16"})
    team_1016FGA = [num(node.getText()) for node in team_1016FGA]

    # team % of FGA that are beyond 16 feet (but not a 3p)
    team_16xxFGA = team_shoot_stats.findAll("td", {"data-stat": "pct_fga_16_xx"})
    team_16xxFGA = [num(node.getText()) for node in team_16xxFGA]

    # team % of FGA that are that are 3PA
    team_3PFGA = team_shoot_stats.findAll("td", {"data-stat": "fg3a_pct_fga"})
    team_3PFGA = [num(node.getText()) for node in team_3PFGA]

    # team FG % on 2P shots
    team_2PFGPct = team_shoot_stats.findAll("td", {"data-stat": "fg2_pct"})
    team_2PFGPct = [num(node.getText()) for node in team_2PFGPct]

    # team FG % of shots that are between 0-3 feet
    team_03FGPct = team_shoot_stats.findAll("td", {"data-stat": "fg_pct_00_03"})
    team_03FGPct = [num(node.getText()) for node in team_03FGPct]

    # team FG % of shots that are between 3-10 feet
    team_310FGPct = team_shoot_stats.findAll("td", {"data-stat": "fg_pct_03_10"})
    team_310FGPct = [num(node.getText()) for node in team_310FGPct]

    # team FG % of shots that are between 10-16 feet
    team_1016FGPct = team_shoot_stats.findAll("td", {"data-stat": "fg_pct_10_16"})
    team_1016FGPct = [num(node.getText()) for node in team_1016FGPct]

    # team FG % of shots that are beyond 16 feet (but not a 3p)
    team_16xxFGPct = team_shoot_stats.findAll("td", {"data-stat": "fg_pct_16_xx"})
    team_16xxFGPct = [num(node.getText()) for node in team_16xxFGPct]

    # team FG % of shots that are that are 3P
    team_3PFGPct = team_shoot_stats.findAll("td", {"data-stat": "fg3_pct"})
    team_3PFGPct = [num(node.getText()) for node in team_3PFGPct]

    # team % of 2P shots that are assisted
    team_2PAstPct = team_shoot_stats.findAll("td", {"data-stat": "fg2_pct_ast"})
    team_2PAstPct = [num(node.getText()) for node in team_2PAstPct]

    # team % of FGA that are dunks
    team_DunkPct = team_shoot_stats.findAll("td", {"data-stat": "pct_fg2_dunk"})
    team_DunkPct = [num(node.getText()) for node in team_DunkPct]

    # team dunks made
    team_DunkMade = team_shoot_stats.findAll("td", {"data-stat": "fg2_dunk"})
    team_DunkMade = [num(node.getText()) for node in team_DunkMade]

    # team % of FGA that are layups
    team_LayPct = team_shoot_stats.findAll("td", {"data-stat": "pct_fg2_layup"})
    team_LayPct = [num(node.getText()) for node in team_LayPct]

    # team layups made
    team_LayMade = team_shoot_stats.findAll("td", {"data-stat": "fg2_layup"})
    team_LayMade = [num(node.getText()) for node in team_LayMade]

    # team % of 3P shots that are assisted
    team_3PAstPct = team_shoot_stats.findAll("td", {"data-stat": "fg3_pct_ast"})
    team_3PAstPct = [num(node.getText()) for node in team_3PAstPct]

    # team % of 3PA that are from corners
    team_3PCornerPct = team_shoot_stats.findAll("td", {"data-stat": "pct_fg3a_corner"})
    team_3PCornerPct = [num(node.getText()) for node in team_3PCornerPct]

    # team 3PA that are made from corners
    team_3PCornerMade = team_shoot_stats.findAll("td", {"data-stat": "fg3_pct_corner"})
    team_3PCornerMade = [num(node.getText()) for node in team_3PCornerMade]

    team_shoot_stats_list = [team_names, team_FGpct, team_AvgDist, team_2PFGA,
                             team_03FGA, team_310FGA, team_1016FGA, team_16xxFGA,
                             team_3PFGA, team_2PFGPct, team_03FGPct, team_310FGPct,
                             team_1016FGPct, team_16xxFGPct, team_3PFGPct, team_2PAstPct,
                             team_DunkPct, team_DunkMade, team_LayPct, team_LayMade,
                             team_3PAstPct, team_3PCornerPct, team_3PCornerMade]

    team_shoot_dict = {}
    for i, head in enumerate(team_shoot_head_names):
        team_shoot_dict[head] = team_shoot_stats_list[i]

    team_shoot_data = pd.DataFrame(team_shoot_dict)

    team_shoot_data = team_shoot_data[team_shoot_head_names]
    
    # ===================
    # Opp Shooting STATS
    # ===================
    
    for comment in comments:
        if "Opponent Shooting" in comment:
            opp_shoot_stats = BeautifulSoup(comment)
            break
    
    # team names
    team_names = opp_shoot_stats.findAll("td", {"data-stat": "team_name"})
    team_names = [node.getText().lower().replace("*", "").replace(" ", "-") for node in team_names]
    
    # opp FGpct
    opp_FGpct = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg_pct"})
    opp_FGpct = [num(node.getText()) for node in opp_FGpct]
    
    # opp avg shot distance
    opp_AvgDist = opp_shoot_stats.findAll("td", {"data-stat": "opp_avg_dist"})
    opp_AvgDist = [num(node.getText()) for node in opp_AvgDist]
    
    # opp % of FGA that are 2P
    opp_2PFGA = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg2a_pct_fga"})
    opp_2PFGA = [num(node.getText()) for node in opp_2PFGA]
    
    # opp % of FGA that are between 0-3 feet
    opp_03FGA = opp_shoot_stats.findAll("td", {"data-stat": "opp_pct_fga_00_03"})
    opp_03FGA = [num(node.getText()) for node in opp_03FGA]

    # opp % of FGA that are between 3-10 feet
    opp_310FGA = opp_shoot_stats.findAll("td", {"data-stat": "opp_pct_fga_03_10"})
    opp_310FGA = [num(node.getText()) for node in opp_310FGA]

    # opp % of FGA that are between 10-16 feet
    opp_1016FGA = opp_shoot_stats.findAll("td", {"data-stat": "opp_pct_fga_10_16"})
    opp_1016FGA = [num(node.getText()) for node in opp_1016FGA]

    # opp % of FGA that are beyond 16 feet (but not a 3p)
    opp_16xxFGA = opp_shoot_stats.findAll("td", {"data-stat": "opp_pct_fga_16_xx"})
    opp_16xxFGA = [num(node.getText()) for node in opp_16xxFGA]

    # opp % of FGA that are 3PA
    opp_3PFGA = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg3a_pct_fga"})
    opp_3PFGA = [num(node.getText()) for node in opp_3PFGA]

    # opp FG % on 2P shots
    opp_2PFGPct = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg2_pct"})
    opp_2PFGPct = [num(node.getText()) for node in opp_2PFGPct]

    # opp FG % of shots that are between 0-3 feet
    opp_03FGPct = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg_pct_00_03"})
    opp_03FGPct = [num(node.getText()) for node in opp_03FGPct]

    # opp FG % of shots that are between 3-10 feet
    opp_310FGPct = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg_pct_03_10"})
    opp_310FGPct = [num(node.getText()) for node in opp_310FGPct]

    # opp FG % of shots that are between 10-16 feet
    opp_1016FGPct = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg_pct_10_16"})
    opp_1016FGPct = [num(node.getText()) for node in opp_1016FGPct]

    # opp FG % of shots that are beyond 16 feet (but not a 3p)
    opp_16xxFGPct = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg_pct_16_xx"})
    opp_16xxFGPct = [num(node.getText()) for node in opp_16xxFGPct]

    # opp FG % of shots that are 3P
    opp_3PFGPct = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg3_pct"})
    opp_3PFGPct = [num(node.getText()) for node in opp_3PFGPct]

    # opp % of 2P shots that are assisted
    opp_2PAstPct = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg2_pct_ast"})
    opp_2PAstPct = [num(node.getText()) for node in opp_2PAstPct]

    # opp % of FGA that are dunks
    opp_DunkPct = opp_shoot_stats.findAll("td", {"data-stat": "opp_pct_fg2_dunk"})
    opp_DunkPct = [num(node.getText()) for node in opp_DunkPct]

    # opp dunks made
    opp_DunkMade = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg2_dunk"})
    opp_DunkMade = [num(node.getText()) for node in opp_DunkMade]

    # opp % of FGA that are layups
    opp_LayPct = opp_shoot_stats.findAll("td", {"data-stat": "opp_pct_fg2_layup"})
    opp_LayPct = [num(node.getText()) for node in opp_LayPct]

    # opp layups made
    opp_LayMade = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg2_layup"})
    opp_LayMade = [num(node.getText()) for node in opp_LayMade]

    # opp % of 3P shots that are assisted
    opp_3PAstPct = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg3_pct_ast"})
    opp_3PAstPct = [num(node.getText()) for node in opp_3PAstPct]

    # opp % of 3PA that are from corners
    opp_3PCornerPct = opp_shoot_stats.findAll("td", {"data-stat": "opp_pct_fg3a_corner"})
    opp_3PCornerPct = [num(node.getText()) for node in opp_3PCornerPct]

    # opp 3PA that are made from corners
    opp_3PCornerMade = opp_shoot_stats.findAll("td", {"data-stat": "opp_fg3_pct_corner"})
    opp_3PCornerMade = [num(node.getText()) for node in opp_3PCornerMade]

    opp_shoot_stats_list = [team_names, opp_FGpct, opp_AvgDist, opp_2PFGA,
                            opp_03FGA, opp_310FGA, opp_1016FGA, opp_16xxFGA,
                            opp_3PFGA, opp_2PFGPct, opp_03FGPct, opp_310FGPct,
                            opp_1016FGPct, opp_16xxFGPct, opp_3PFGPct, opp_2PAstPct,
                            opp_DunkPct, opp_DunkMade, opp_LayPct, opp_LayMade,
                            opp_3PAstPct, opp_3PCornerPct, opp_3PCornerMade]

    opp_shoot_dict = {}
    for i, head in enumerate(opp_shoot_head_names):
        opp_shoot_dict[head] = opp_shoot_stats_list[i]

    opp_shoot_data = pd.DataFrame(opp_shoot_dict)

    opp_shoot_data = opp_shoot_data[opp_shoot_head_names]

    
    # ===================
    #  Merge the tables
    # ===================

    team_data = pd.merge(misc_data, team_shoot_data, on="Team")
    team_data = pd.merge(team_data, opp_shoot_data, on="Team")

    team_data.to_csv("team_stats_"+year+".csv", index=False)

    # get results of all games
    all_games = []
    for team in team_abbrevs:

        url = "http://www.basketball-reference.com/teams/"+team+"/"+year+"_games.html"
        r = requests.get(url)
        data = r.text
        soup = BeautifulSoup(data)

        # opponent names
        opp_names = soup.findAll("td", {"data-stat": "opp_name"})
        opp_names = [node.getText().lower().replace(" ", "-") for node in opp_names]

        # date of game (make sure it's not in the future)
        game_date = soup.findAll("td", {"data-stat": "date_game"})
        date_game = []
        for date in game_date:
            day = dt.datetime.strptime(date.getText(), "%a, %b %d, %Y")
            day = dt.datetime.strftime(day, '%Y-%m-%d')
            date_game.append(day)


        # game location
        game_location = soup.findAll("td", {"data-stat": "game_location"})
        game_location = [node.getText() for node in game_location]

        # result of game
        game_result = soup.findAll("td", {"data-stat": "game_result"})
        game_result = [node.getText() for node in game_result]
        game_result = [0 if node == 'W' else 1 for node in game_result]

        # record only the "home" (or "neutral site") games
        for opp, date, location, result in zip(opp_names, date_game, game_location, game_result):
            if opp in team_names and location != "@" and date < str(dt.date.today()):
                all_games.append((team, opp, result))


    team_name_dict = {'GSW': 'golden-state-warriors','SAS': 'san-antonio-spurs',
                      'HOU': 'houston-rockets','CLE': 'cleveland-cavaliers',
                      'UTA': 'utah-jazz','LAC': 'los-angeles-clippers','TOR': 'toronto-raptors', 
                      'BOS': 'boston-celtics','MEM': 'memphis-grizzlies','WAS': 'washington-wizards',
                      'OKC': 'oklahoma-city-thunder','MIA': 'miami-heat','DEN': 'denver-nuggets',
                      'CHO': 'charlotte-hornets','MIN': 'minnesota-timberwolves','MIL': 'milwaukee-bucks',
                      'DET': 'detroit-pistons','IND': 'indiana-pacers','ATL': 'atlanta-hawks',
                      'CHI': 'chicago-bulls','POR': 'portland-trail-blazers','DAL': 'dallas-mavericks',
                      'NOP': 'new-orleans-pelicans','SAC': 'sacramento-kings','NYK': 'new-york-knicks',
                      'PHO': 'phoenix-suns','PHI': 'philadelphia-76ers','ORL': 'orlando-magic',
                      'LAL': 'los-angeles-lakers','BRK': 'brooklyn-nets',
                      'NJN': 'new-jersey-nets', 'NOH': 'new-orleans-hornets',
                      'CHA': 'charlotte-bobcats','SEA': 'Seattle Supersonics',
                      'NOK': 'new-orleans/oklahoma-city-hornets', 
                      'CHH': 'charlotte-hornets'}

    game_result_df = pd.DataFrame()
    for game in all_games:
        team_one = team_data[team_data['Team'] == team_name_dict[game[0]]]
        team_one.reset_index(inplace=True, drop=True)
        team_one.drop('Team', axis=1, inplace=True)
        team_two = team_data[team_data['Team'] == game[1]]
        team_two.reset_index(inplace=True, drop=True)
        team_two.drop('Team', axis=1, inplace=True)
        game_data = pd.merge(team_one, team_two, left_index=True, right_index=True)
        game_data['winner'] = game[2]
        game_result_df = game_result_df.append(game_data)


    # normalize the columns
    #min_max_scaler = MinMaxScaler()
    #for col in game_result_df.columns.tolist():
    #    game_result_df[col] = min_max_scaler.fit_transform(game_result_df[col].as_matrix().reshape(-1, 1))

    print("Number of games in "+year+": ", game_result_df.shape[0])
    print()
    game_result_df.to_csv("all_games_" + year + ".csv", index=False)
