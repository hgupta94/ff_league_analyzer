# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:05:30 2021

@author: hirsh
"""


import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import requests
import os
import datetime
import random
from functools import reduce

# %% Data Setup
def load_data(league_id, season, swid="", espn=""):
    '''
    Pull ESPN API data for a particular season (current API version only goes back to 2018).
    '''
    
    url = 'http://fantasy.espn.com/apis/v3/games/ffl/seasons/' + str(int(season)) + '/segments/0/leagues/' + str(int(league_id)) + '?view=mMatchupScore&view=mTeam&view=mSettings'
    r = requests.get(url,
                     cookies = {'SWID': swid,
                                'espn_s2': espn},
                                params={'view':'mMatchup'})
    d = r.json()
    
    return d

def get_params(d):
    '''
    Returns general league information
    '''
    
    # general league setup
    league_size = d['settings']['size']
    roster_size = sum(d['settings']['rosterSettings']['lineupSlotCounts'].values())
    regular_season_end = d['settings']['scheduleSettings']['matchupPeriodCount']
    current_week = d['scoringPeriodId'] - 1
    matchup_week = d['scoringPeriodId']
    weeks_left = regular_season_end - current_week
    playoff_teams = d['settings']['scheduleSettings']['playoffTeamCount']
    
    # roster construction
    slotcodes = {
        0 : 'QB', 1 : 'QB',
        2 : 'RB', 3 : 'RB',
        4 : 'WR', 5 : 'WR',
        6 : 'TE', 7 : 'TE',
        16: 'D/ST',
        17: 'K',
        20: 'Bench',
        21: 'IR',
        23: 'Flex'}
    lineup_slots = d['settings']['rosterSettings']['lineupSlotCounts']
    lineup_slots_df = pd.DataFrame.from_dict(lineup_slots, orient='index').rename(columns={0:'limit'})
    lineup_slots_df['posID'] = lineup_slots_df.index.astype('int')
    lineup_slots_df = lineup_slots_df[lineup_slots_df.limit > 0]
    lineup_slots_df['pos'] = lineup_slots_df.replace({'posID': slotcodes}).posID
    
    # Mapping team ID to team name    
    teams =  [[
            game['abbrev'], game['id']
        ] for game in d['teams']]
    teams = pd.DataFrame(teams, columns=['abbrev', 'id'])
    team_map = dict(zip(teams.id, teams.abbrev))
    
    # Get weekly matchups
# =============================================================================
#     df = [[
#             game['matchupPeriodId'],
#             game['home']['teamId'], game['home']['totalPoints'],
#             game['away']['teamId'], game['away']['totalPoints']
#         ] for game in d['schedule']]
# =============================================================================
    df = pd.DataFrame()
    for game in d['schedule']:
        if game['matchupPeriodId'] > d['settings']['scheduleSettings']['matchupPeriodCount']:
            continue
        if game['matchupPeriodId'] <= d['settings']['scheduleSettings']['matchupPeriodCount']:
            week = game['matchupPeriodId']
            team1 = game['home']['teamId']
            score1 = game['home']['totalPoints']
            team2 = game['away']['teamId']
            score2 = game['away']['totalPoints']
            matchups = pd.DataFrame([[week, team1, score1, team2, score2]], 
                                    columns=["week", "team1", "score1", "team2", "score2"])
            df = df.append(matchups)
    
    df = df.replace({'team1':team_map, 'team2':team_map})
    
    #position = lineup_slots_df.pos.str.lower().drop(labels=['20','21']).tolist()
    position = ['qb', 'rb', 'wr', 'te', 'flex', 'dst', 'k']
    teams = list(team_map.values())
    teams = [x.lower() for x in teams]
    
    return league_size, roster_size, regular_season_end, current_week, matchup_week, weeks_left, playoff_teams, team_map, df, position, teams, slotcodes, lineup_slots_df

def get_ros_projections(d):
    '''
    Returns rest of season projections for all rostered players
    '''
    
    # Map players to teams
    players = []
    teams = []
    league_size = get_params(d)[0]
    roster_size = get_params(d)[1]
    current_week = get_params(d)[3]
    
    for i in range(league_size):
        for player in range(roster_size):
            try:
                players.append(d['teams'][i]['roster']['entries'][player]['playerPoolEntry']['player']['fullName'])
                teams.append(d['teams'][i]['abbrev'])
            except IndexError:
                pass
    
    # Fix player names
    rosters = pd.DataFrame({'team': teams, 'player': players}).apply(lambda x: x.astype(str).str.lower())
    rosters = rosters.replace({'player':[',', '\(', '\)', '\'', ' jr.', '\.']}, {'player':''}, regex=True)
    rosters = rosters.replace(['michael '], 'mike ', regex=True)
    
    # =============================================================================
    # rosters = rosters.replace('allen robinson ii', 'allen robinson')
    # rosters = rosters.replace('melvin gordon iii', 'melvin gordon')
    # rosters = rosters.replace('todd gurley ii', 'todd gurley')
    # rosters = rosters.replace('ronald jones ii', 'ronald jones')
    # rosters = rosters.replace('henry ruggs iii', 'henry ruggs')
    # rosters = rosters.replace('mark ingram ii', 'mark ingram')
    # rosters = rosters.replace('will fuller v', 'will fuller')
    # =============================================================================
    
    rosters['player'] = rosters['player'].str.replace(' iii', '')
    rosters['player'] = rosters['player'].str.replace(' ii', '')
    rosters['player'] = rosters['player'].str.replace(' iv', '')
    rosters['player'] = rosters['player'].str.replace(' v', '')
    
    
    # Add rest of season projections to players
    url = 'https://www.numberfire.com/nfl/fantasy/remaining-projections'
    
    player_data = pd.read_html(url, header=None)[0]
    proj = pd.read_html(url)[1]
    
    # Fix player names
    player_data = player_data.columns.to_frame().T.append(player_data, ignore_index=True)
    player_data = player_data.replace([',', '\(', '\)', '\'', ' Jr.', '\.'], '', regex=True)
    player_data = player_data.replace(['Michael '],'Mike ', regex=True)
    player_data = player_data.iloc[:,0].str.split(expand=True)
    player_data['name'] = player_data[0].str.cat(player_data[1],sep=" ")
    
    # Combine datasets
    projections = pd.concat([player_data.iloc[:,[6,4,5]], proj], axis=1).iloc[:,[0,1,2,3]]
    projections.columns = ['player', 'position', 'nfl_team', 'ros_proj']
    projections = projections.apply(lambda x: x.astype(str).str.lower())
    
    # Calculate points per game
    projections['ppg'] = projections.ros_proj.astype(float) / (17 - current_week)
    #projections['ppg'] = np.where(projections['position'] != 'qb', np.sqrt(projections['ppg']), projections['ppg'])
    
    # use average standard deviation by position from 2018-2019
    # from 'position analysis' R script
    projections['sd'] = np.where(projections['position'] == 'qb', 5.83, 0)
    projections['sd'] = np.where(projections['position'] == 'rb', 6.5, projections['sd'])
    projections['sd'] = np.where(projections['position'] == 'wr', 4.76, projections['sd'])
    projections['sd'] = np.where(projections['position'] == 'te', 5.83, projections['sd'])
    projections_final = pd.merge(rosters, projections, how='right', on='player').dropna(subset=['team'])
    
    return projections_final


# %% Power Ranking
def power_rank(d, league_id, season, week=None):
    # return parameters
    regular_season_end = get_params(d)[2]
    current_week = get_params(d)[3]
    matchup_week = get_params(d)[4]
    
    week = None
    current_year = datetime.datetime.now().year
    if week is None:
        # return last regular season week
        week = np.asscalar(np.where(matchup_week > regular_season_end, 
                                   regular_season_end, 
                                   current_week))
    elif week is None and season < current_year:
        # return last regular season week if prior year is chosen with no week
        week = regular_season_end
    elif week > regular_season_end:
        # return last regular season week if chosen week is in the playoffs
        week = regular_season_end
    elif week > matchup_week:
        # return most recent week if chosen week has not occured
        print('Week has not occured, showing data for Week', current_week)
        week = current_week
    else:
        week = week
        
    # get weekly matchups and scores
    df = get_params(d)[8]
    
    # Calculate W/L    
    df['team1_result'] = np.where(df['score1'] > df['score2'], 1, 0)
    df['team2_result'] = np.where(df['score2'] > df['score1'], 1, 0)
    
    # Account for ties
    mask = (df.score1 == df.score2)
    df.loc[mask, ['team1_result', 'team2_result']] = 0.5
    
    # convert dataframe to long format so each row is a team week, not matchup
    home = df.iloc[:,[0,1,2,5]].rename(columns={'team1':'team', 'score1':'score', 'team1_result':'result'})
    home['id'] = home['team'].astype(str) + home['week'].astype(str)
    away = df.iloc[:,[0,3,4,6]].rename(columns={'team2':'team', 'score2':'score', 'team2_result':'result'})
    away['id'] = away['team'].astype(str) + away['week'].astype(str)
    
    df_current = pd.concat([home, away])
    season_wins = (df_current.groupby(['team', 'week'])
                        .sum()
                        .groupby(level=0)
                        .cumsum()
                        .rename(columns={'result':'wins'})
                        .reset_index())
    season_wins['id'] = season_wins['team'].astype(str) + season_wins['week'].astype(str)
    season_wins = season_wins.drop(['team', 'week', 'score'], axis=1)
    df_current = pd.merge(df_current, season_wins, on='id')   
    
    
    # Calculate total season points by team
    cumul_score = (df_current.groupby(['team', 'week'])
                             .sum()
                             .groupby(level=0)
                             .cumsum()
                             .reset_index())
    cumul_score['id'] = cumul_score['team'].astype(str) + cumul_score['week'].astype(str)
    
    cumul_score = (cumul_score.drop(['team', 'week', 'wins', 'result'], axis=1)
                             .rename(columns={'score':'total_pf'})) 
    
    df_current = pd.merge(df_current, cumul_score, on='id')
    df_current['ppg'] = df_current['total_pf'] / df_current['week']
    
    
    # Calculate median scores
    median_score = (df_current.groupby(['week'])
                            .agg(np.median)
                            .rename(columns={'score':'score_med', 'total_pf':'total_pf_med', 'ppg':'ppg_med'})
                            .drop(['wins', 'result'], axis=1))
    df_current = pd.merge(df_current, median_score, how='left', on='week')
    

    # Calculate rolling standard deviation by team
    st_dev = df_current[['week', 'team', 'score', 'result', 'id']]
    st_dev = (st_dev.set_index('week')
            .sort_index()
            .groupby(['team'])['score']
            .expanding()
            .std()
            .to_frame()
            .rename(columns={'score':'sd'})
            .reset_index())
    st_dev['id'] = st_dev['team'].astype(str) + st_dev['week'].astype(str)
    df_current = st_dev.drop(['team','week'], axis=1).fillna(0).merge(df_current, on='id')
    
    # Calculate luck factor
    df_current['week_luck'] = df_current['result'] - (df_current.groupby('week')['score'].rank() - 1) / 9
    luck_sum = (df_current.groupby(['team', 'week'])
                        .sum()
                        .groupby(level=0)
                        .cumsum()
                        .reset_index()
                        .loc[:,['team','week','week_luck']])
    luck_sum['luck_index'] = luck_sum['week_luck'] / luck_sum['week']
    luck_sum['id'] = luck_sum['team'].astype(str) + luck_sum['week'].astype(str)
    df_current = luck_sum.drop(['team', 'week', 'week_luck'], axis=1).merge(df_current, on='id')
    
    # Calculate ranking metrics
    exp_win = df_current[['week', 'team', 'result', 'week_luck']]
    exp_win['xwins'] = exp_win['result'] - exp_win['week_luck']   
    exp_win = (exp_win.groupby(['team', 'week'])
              .sum()
              .groupby(level=0)
              .cumsum()
              .reset_index()
              .drop(['result', 'week_luck'], axis=1))
    exp_win['id'] = exp_win['team'].astype(str) + exp_win['week'].astype(str)
    df_current = exp_win.drop(['team', 'week'], axis=1).merge(df_current, on='id')    
    
    three_wk_avg = (df_current.groupby('team')['score']
                    .transform(lambda x: x.rolling(3, 1)
                    .mean())
                    .rename('three_wk_avg'))
    df_current = pd.merge(df_current, three_wk_avg, left_index = True, right_index=True)
    
    three_wk_med = (df_current[['week', 'three_wk_avg']]
                    .groupby('week')
                    .agg(np.median)
                    .reset_index()
                    .rename(columns={'three_wk_avg':'three_wk_med'}))
    df_current = pd.merge(df_current, three_wk_med, how='left', on='week')

    df_current['win_index'] = (df_current['wins'] / (df_current['week']))
    df_current['score_index'] = (df_current['three_wk_avg'] / df_current['three_wk_med'])
    df_current['season_index'] = (df_current['total_pf'] / df_current['total_pf_med'])
    df_current['consistency'] = (1 - (df_current['sd'] / df_current['ppg'])) * df_current['season_index']
    
    cons_med = df_current[['week', 'consistency']]
    cons_med = cons_med.groupby('week').agg(np.median).rename(columns={'consistency':'consistency_med'})
    df_current = pd.merge(df_current, cons_med, on='week')
    df_current['consistency_index'] = df_current['consistency'] / df_current['consistency_med']
    
    df_current['power_rank_score'] = (df_current['win_index'] 
                                   #- (df_current['luck_index']/10) 
                                   + (df_current['consistency_index']) 
                                   + df_current['score_index']
                                   + df_current['season_index'])
    
    # Standardize so average=100
    pr_avg = df_current.loc[:,['week', 'power_rank_score']]
    pr_avg = pr_avg.groupby('week').agg(np.mean).rename(columns={'power_rank_score':'power_rank_avg'})
    
    df_current = pd.merge(df_current, pr_avg, on='week')
    
    df_current['power_score'] = df_current.power_rank_score / df_current.power_rank_avg
    
    df_current['team'] = df_current['team'].str.lower()
    df_current['id'] = df_current['id'].str.lower()
    
    return df_current

# %% Simulate Scores
def sim_scores(d):
    '''
    Simulates a weekly score for each team using current roster
    Assumed roster construction:
        1 QB
        2 RB
        2 WR
        1 TE
        1 Flex (RB/WR/TE)
        1 D/ST
        1 K
    Future update: ability to use different settings (2 QB, etc)
    '''
    
    current_week = get_params(d)[3]
    projections_final = get_ros_projections(d)
    score_proj_df = pd.DataFrame(columns=['team','score'])
    position = get_params(d)[9]
    team_list = get_params(d)[10]
    
    for team in team_list:
        # simulate using actual season data
        df_current = power_rank(league_id, season)
        team_stats = df_current[(df_current['week'] == (current_week)) & (df_current['team'] == team)].reset_index(drop=True)
        team_stats = team_stats[['team', 'ppg', 'sd']]
        scores = random.gauss(team_stats['ppg'], team_stats['sd'])
            
        # weight scores
        scores = scores * 0.25
        
        ### Need to clean this up to automatically go through each position
        # simulate using rest of season projections
        team_proj_df = projections_final[projections_final['team'] == team]
        for pos in position:
            pos_df = team_proj_df[team_proj_df['position'] == pos]
            if pos == 'qb':
                if pos not in pos_df:
                    # if there is no qb on the roster (injury, bye, trade), use average ROS projection
                    qb_pts = projections_final[projections_final.position == 'qb']
                    qb_pts = np.mean(qb_pts.ppg)
                else:
                    qb_pts = pos_df.nlargest(1, 'ppg').ppg.values
                    qb_pts = random.gauss(qb_pts, max(pos_df['sd']))
                
            if pos == 'rb':
                rb1_pts = pos_df.sort_values('ppg', ascending=False).groupby(['position']).nth(0).ppg.values
                rb1_pts = random.gauss(rb1_pts, max(pos_df['sd']))
                rb2_pts = pos_df.sort_values('ppg', ascending=False).groupby(['position']).nth(1).ppg.values
                rb2_pts = random.gauss(rb2_pts, max(pos_df['sd']))
                rb_pts = rb1_pts + rb2_pts
                
            if pos == 'wr':
                wr1_pts = pos_df.sort_values('ppg', ascending=False).groupby(['position']).nth(0).ppg.values
                wr1_pts = random.gauss(wr1_pts, max(pos_df['sd']))
                wr2_pts = pos_df.sort_values('ppg', ascending=False).groupby(['position']).nth(1).ppg.values
                wr2_pts = random.gauss(wr2_pts, max(pos_df['sd']))
                wr_pts = wr1_pts + wr2_pts
                
            if pos == 'te':
                te_pts = pos_df.nlargest(1, 'ppg').ppg.values
                te_pts = random.gauss(te_pts, max(pos_df['sd']))
            
            if pos == 'flex':
                # assume only RB or WR in flex
                flex_df = (team_proj_df[(team_proj_df['position'] == 'rb')
                                      | (team_proj_df['position'] == 'wr')])
                # filter for highest of 3rd rb or 3rd wr
                flex_players = flex_df.sort_values('ppg', ascending=False).groupby(['position']).nth(2)
                flex_pts = max(random.gauss(flex_players['ppg'], flex_players['sd']))
                
            if pos == 'dst':
                dst_pts = 6
                dst_pts = random.gauss(flex_pts, 3)
                
            if pos == 'k':
                k_pts = 8
                k_pts = random.gauss(k_pts, 3)
    
        # weight scores
        pts = (qb_pts + rb_pts + wr_pts + te_pts + flex_pts + dst_pts + k_pts) * 0.75
        
        # combine projections
        total_proj_pts = scores + pts
        row = {'team':team,'score': total_proj_pts[0]}
        rowdf = pd.DataFrame(data=[row])
    
        score_proj_df = pd.concat([score_proj_df,rowdf])
    score_proj_df = score_proj_df.reset_index(drop=True)
    
    return score_proj_df

# %% Simulate Matchups
def sim_matchups(d):
    '''
    Simulates rest of season matchups (head to head) using scores from simulate_scores
    and returns final standings
    '''
    
    regular_season_end = get_params(d)[2]
    matchup_week = get_params(d)[4]
    df = get_params(d)[8]
    
    if matchup_week < regular_season_end:
        # if current week is not in playoffs, simulate through matchups
        
        # get weekly matchups
        sim_week = d['scoringPeriodId']
        matchups = df[['week', 'team1', 'score1', 'team2', 'score2']]
        matchups['team1'] = matchups.team1.str.lower()
        matchups['team2'] = matchups.team2.str.lower()
        
        # create separate df for past weeks and append future weeks later
        matchups2 = matchups[matchups['week'] < d['scoringPeriodId']]
        score_sim = sim_scores()
        score_dict = dict(zip(score_sim.team, score_sim.score))
        
        # get scores for future weeks
        for week in matchups.week:
            if week < sim_week:
                continue
            else:
                matchups_new = matchups[matchups['week'] == week]
                a = matchups_new.filter(like='team').columns
                matchups_new['score' + a.str.lstrip('team')] = matchups_new[a].stack().map(score_dict).unstack()
                matchups2 = matchups2.append(matchups_new)
                sim_week += 1
        
        # calculate W/L    
        matchups2['team1_result'] = np.where(matchups2['score1'] > matchups2['score2'], 1, 0)
        matchups2['team2_result'] = np.where(matchups2['score2'] > matchups2['score1'], 1, 0)
        
        # account for ties
        mask = (matchups2.score1 == matchups2.score2)
        matchups2.loc[mask, ['team1_result', 'team2_result']] = 0.5
        
        # convert dataframe to long format so each row is a team week, not matchup
        home = matchups2.iloc[:,[0,1,2,5]].rename(columns={'team1':'team', 'score1':'score', 'team1_result':'wins'})
        away = matchups2.iloc[:,[0,3,4,6]].rename(columns={'team2':'team', 'score2':'score', 'team2_result':'wins'})
        df_sim = pd.concat([home, away]).iloc[:,[1,2,3]]
        
        final_results = df_sim.groupby('team').agg({'wins':'sum', 'score':'sum'}).reset_index()
        
        return final_results
    else:
        # if current week is in playoffs, return final standings
        data = []
        for tm in d['teams']:
            tmid = tmid = tm['abbrev'].lower()
            wins = tm['record']['overall']['wins']
            losses = tm['record']['overall']['losses']
            record = str(wins) + '-' + str(losses)
            score = tm['record']['overall']['pointsFor']
            pa = tm['record']['overall']['pointsAgainst']
            data.append([tmid, wins, losses, record, score, pa])
        
        standings = pd.DataFrame(data,
                                 columns = ["team", "wins", "losses", "record", "score", "pa"])
        standings = standings.sort_values(by=['wins', 'score'], ascending=False)
        return standings

#%% Aggregate Projections
def get_projections(d):
    # get aggregate weekly projections from fantasypros
    positions = get_params(d)[9]
    projections = pd.DataFrame()
    
    # Return PPR scoring type (standard, half, or full) for fantasypros url
    statid = []
    for scoring in d['settings']['scoringSettings']['scoringItems']:
        statid.append(scoring['statId'])
        if 53 in statid:
            if scoring['statId'] == 53:
                if scoring['points'] == 0.5:
                    ppr_type = 'HALF'
                elif scoring['points'] == 1:
                    ppr_type = 'PPR'
        else:
            ppr_type = 'STD'
    
    # Return current week's projections for all positions
    for pos in positions:
        url = 'https://www.fantasypros.com/nfl/projections/' + pos + '.php?scoring=' + ppr_type
        df = pd.read_html(url)[0]
        
        # drop multi index column
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel()
        
        df['POSITION'] = pos
        df = df.loc[:,['Player', 'FPTS', 'POSITION']]
        
        # remove team from player name
        if pos != 'dst':
            df['Player'] = df['Player'].str[:-3]
            df['Player'] = df['Player'].str.rstrip()
        
        if pos == 'dst':
            df['Player'] = df['Player'].str.split().str[-1] + ' D/ST'
        
        df['Player'] = df.Player.str.lower()
        df.columns = df.columns.str.lower()
        df['player'] = df['player'].str.replace('.', '')
        df['player'] = df['player'].str.replace('\'', '')
        df['player'] = df['player'].str.replace(' iii', '')
        df['player'] = df['player'].str.replace(' ii', '')
        df['player'] = df['player'].str.replace(' jr', '')
        
        projections = projections.append(df)
        
        # fix player names
        projections['player'] = projections['player'].replace(['team d/st'],'washington d/st')
    
    return projections

#%% Simulate Week
def sim_week(d, n_sim=100):
    '''
    Simulates current week matchups - used to create betting table
    '''
    
    # get actual and projected scores from espn
    # code thanks to Steven Morse: https://stmorse.github.io/journal/espn-fantasy-projections.html
    league_size = get_params(d)[0]
    current_week = get_params(d)[3]
    matchup_week = get_params(d)[4]
    team_map =  get_params(d)[7]
    df = get_params(d)[8]
    teams = get_params(d)[10]
    slotcodes = get_params(d)[11]
    projections = get_projections()

    data = []
    week = matchup_week
    
    for tm in d['teams']:
        tmid = tm['abbrev']
        for p in tm['roster']['entries']:
            name = p['playerPoolEntry']['player']['fullName']
            slot_id = p['lineupSlotId']
            slot  = slotcodes[slot_id]

            # injured status (try/except bc of D/ST)
            inj = 'NA'
            try:
                inj = p['playerPoolEntry']['player']['injuryStatus']
            except:
                pass

            # projected/actual points
            proj, act = None, None
            for stat in p['playerPoolEntry']['player']['stats']:
                if stat['scoringPeriodId'] != week:
                    continue
                if stat['statSourceId'] == 0:
                    act = stat['appliedTotal']
                elif stat['statSourceId'] == 1:
                    proj = stat['appliedTotal']

            data.append([
                week, tmid, name, slot_id, slot, inj, proj, act
            ])
    
    proj = pd.DataFrame(data, 
                        columns=['week', 'team', 'player', 'slot_id', 
                                 'slot', 'status', 'proj', 'actual'])
    proj = proj.apply(lambda x: x.astype(str).str.lower() if x.dtype=='object' else x)
    proj = proj[(proj['slot'] != 'bench') & (proj['slot'] != 'ir')]
    proj = proj.drop(['proj'], axis=1)
    proj['actual'] = np.where(proj['actual'] == 'none', np.nan, proj['actual'])
    
    # fix player names
    proj['player'] = proj['player'].str.replace('.', '')
    proj['player'] = proj['player'].str.replace('\'', '')
    proj['player'] = proj['player'].str.replace(' iii', '')
    proj['player'] = proj['player'].str.replace(' ii', '')
    proj['player'] = proj['player'].str.replace(' jr', '')
    
    # replace espn projections with aggregate projections
    proj = pd.merge(proj, projections, how='left', on='player').rename(columns={'fpts':'projected'})
    proj['projected'] = np.where(proj['projected'].isnull(), 0, proj['projected'])
    
    # add standard deviations (same as data_setup)
    proj['sd'] = np.where(proj['position'] == 'qb', 5.83, 0)
    proj['sd'] = np.where(proj['position'] == 'rb', 6.5, proj['sd'])
    proj['sd'] = np.where(proj['position'] == 'wr', 4.76, proj['sd'])
    proj['sd'] = np.where(proj['position'] == 'te', 5.83, proj['sd'])
    proj['sd'] = np.where(proj['position'] == 'dst', 3, proj['sd'])
    proj['sd'] = np.where(proj['position'] == 'k', 3, proj['sd']) 
    
    # replace starter if injured or on bye
    #foo = pd.merge(proj, projections_final[['player', 'position']], on='player', how='left')
    
    # get current week matchups
    matchups = df[['week', 'team1', 'score1', 'team2', 'score2']]
    matchups = matchups[matchups['week'] == matchup_week]
    matchups['team1'] = matchups.team1.str.lower()
    matchups['team2'] = matchups.team2.str.lower()
    matchups['game_id'] = range(1, int(league_size/2)+1)
    
    # initialize dicionaries for counts
    teams = list(team_map.values())
    teams = [x.lower() for x in teams]
    teams_dict = {key: 0 for key in teams}
    n_wins = {key: 0 for key in teams}
    n_highest = {key: 0 for key in teams}
    n_lowest = {key: 0 for key in teams}
    
    # simulate current week scores
    score_df = pd.DataFrame(columns={'team', 'score'})
    for sim in range(n_sim):
        proj['score'] = np.nan
        for index, row in proj['actual'].iteritems():
            if pd.isnull(row):
                proj.at[index,'score'] = random.gauss(proj['projected'][index], proj['sd'][index])
            else:
                proj.at[index,'score'] = proj['actual'][index]
        
        for team in teams:
            teams_dict[team] = proj[proj['team'] == team].score.sum()
            score_df = score_df.append(proj[proj['team'] == team].groupby('team').score.sum().reset_index())
        
        a = matchups.filter(like='team').columns
        matchups['score' + a.str.lstrip('team')] = matchups[a].stack().map(teams_dict).unstack()
        
        # calculate wins and losses   
        matchups['team1_result'] = np.where(matchups['score1'] > matchups['score2'], 1, 0)
        matchups['team2_result'] = np.where(matchups['score2'] > matchups['score1'], 1, 0)
        
        # account for ties
        mask = (matchups.score1 == matchups.score2)
        matchups.loc[mask, ['team1_result', 'team2_result']] = 0.5
        
        # convert dataframe to long format so each row is a team week, not matchup
        home = matchups.iloc[:,[0,1,2,5,6]].rename(columns={'team1':'team', 'score1':'score', 'team1_result':'wins'})
        away = matchups.iloc[:,[0,3,4,5,7]].rename(columns={'team2':'team', 'score2':'score', 'team2_result':'wins'})
        df_sim = pd.concat([home, away]).iloc[:,[1,2,3,4]]

        for team in teams:
            n_wins[team] += df_sim[df_sim['team'] == team].wins.values[0].astype(int)
        
        # get highest and lowest scorer
        high = df_sim.sort_values(by='score', ascending=False).iloc[0,0]
        low = df_sim.sort_values(by='score').iloc[0,0]
        
        n_highest[high] += 1
        n_lowest[low] += 1
    
    # convert dicts to df and combine
    game_id = df_sim.loc[:,['team', 'game_id']]
    wins = pd.DataFrame(n_wins.items(), columns=['team', 'n_wins'])
    highest = pd.DataFrame(n_highest.items(), columns=['team', 'n_highest'])
    lowest = pd.DataFrame(n_lowest.items(), columns=['team', 'n_lowest'])
    
    dfs = [game_id, wins, highest, lowest]
    
    week_sim = reduce(lambda left, right: pd.merge(left, right, on='team'), dfs)
    
    return week_sim, score_df

#%% Simulate Season
def sim_season(d, n_sim=10):
    '''
    Simulates rest of season matchups and returns number of times team won n games
    '''
    
    regular_season_end = get_params(d)[2]
    matchup_week = get_params(d)[4]
    teams = get_params(d)[10]
    
    if matchup_week < regular_season_end:
        # if current week is not in playoffs, simulate season
        # initialize empty table to count wins
        table = (pd.DataFrame(index=teams,
                               columns=range(regular_season_end + 1))
                #.reset_index()
                .fillna(0)
                .rename(columns={'index':'team'}))
        
        df = pd.DataFrame()
        for sim in range(n_sim):
            results = sim_matchups()
            results = results.set_index('team')
            df = df.append(results)
            for index, row in results.iterrows():
                wins = row['wins']
                table[wins][index] += 1
        
        # calculate averages and standard deviations
        avg = df.groupby('team').mean().reset_index().rename(columns={'wins':'avg_w', 'score':'avg_pts'})
        sd = df.groupby('team').std().reset_index().rename(columns={'wins':'sd_w', 'score':'sd_pts'})
        df = pd.merge(avg, sd, how='left', on='team').set_index('team')
        df = df.sort_values(by=['avg_w', 'avg_pts'], ascending=False)
        
        table['avg_w'] = 0
        for col in table.columns:
            if (col != 'team') & (col != 'avg_w'):
                table['avg_w'] += col*table[col]
                
        table['avg_w'] = table['avg_w'] / n_sim
        
        table = round(table.sort_values(by='avg_w', ascending=False),1)   
        
        return table, df
    
    else:
        standings = sim_matchups()
        return standings

#%% Simulate Playoffs
def sim_playoffs(d, n_sim=10):
    '''
    Simulates playoffs and returns:
        Probability of making playoffs
        Probability of winning championship
        Probability of winning runner up
        Probability of winning third place
    
    Assumes 4 and 6 team playoffs
    '''
    
    n_teams = get_params(d)[6]
    teams = get_params(d)[10]
    
    # initialize dictionaries to count number of occurances for each team
    n_playoffs = {key: 0 for key in teams}
    n_finals = {key: 0 for key in teams}
    n_champ = {key: 0 for key in teams}
    n_second = {key: 0 for key in teams}
    n_third = {key: 0 for key in teams}
    
    for sim in range(n_sim):
        if n_teams == 4:
            
            # get playoff teams
            playoffs = (sim_matchups()
                       .sort_values(by=['wins', 'score'], ascending=False)
                       .head(4)
                       .reset_index(drop=True)
                       .iloc[:,[0]])
            
            # count playoff appearances
            for team in playoffs.team:
                n_playoffs[team] += 1
            
            # simulate 2 weeks of semifinals matchups
            semi_scores = (sim_scores(playoffs.team.tolist())
                          .set_index('team')
                          .merge(sim_scores(playoffs.team.tolist())
                          .set_index('team'), on='team')
                          .sum(axis=1)
                          .reset_index()
                          .rename(columns={0:'score'}))

            semi_1 = playoffs.iloc[[0,3], :].merge(semi_scores, on='team')
            semi_2 = playoffs.iloc[[1,2], :].merge(semi_scores, on='team')
            
            # get finals matchup
            final_1 = semi_1.sort_values(by='score', ascending=False).head(1)
            final_2 = semi_2.sort_values(by='score', ascending=False).head(1)
            
            finals = pd.concat([final_1, final_2])
            
            # simulate 2 weeks of finals matchup
            final_scores = (sim_scores(finals.team.tolist())
                           .set_index('team')
                           .merge(sim_scores(finals.team.tolist())
                           .set_index('team'), on='team')
                           .sum(axis=1)
                           .reset_index()
                           .rename(columns={0:'score'}))
            
            # count finals appearances
            for team in final_scores.team:
                n_finals[team] +=1
            
            champ = final_scores.sort_values(by='score', ascending=False).iloc[0,0]
            
            # count championships
            n_champ[champ] += 1
            
            # count runner ups
            runner = final_scores.sort_values(by='score').iloc[0,0]
            n_second[runner] += 1
            
            # get third place matchup
            third_1 = semi_1.sort_values(by='score').head(1)
            third_2 = semi_2.sort_values(by='score').head(1)
            
            third = pd.concat([third_1, third_2])
            
            # simulate 2 weeks of third place matchup
            third_scores = (sim_scores(third.team.tolist())
                           .set_index('team')
                           .merge(sim_scores(third.team.tolist())
                           .set_index('team'), on='team')
                           .sum(axis=1)
                           .reset_index()
                           .rename(columns={0:'score'}))
            third = pd.concat([third_1, third_2]).iloc[:,[0]].merge(third_scores)
    
            # count third place
            third_team = third.sort_values(by='score', ascending=False).iloc[0,0]
            n_third[third_team] += 1
           
        if n_teams == 6:
            
            # get playoff teams
            playoffs = (sim_matchups()
                       .sort_values(by=['wins','score'], ascending=False)
                       .head(6)
                       .reset_index(drop=True)
                       .iloc[:,[0]])
            
            # count playoff appearances
            for team in playoffs.team:
                n_playoffs[team] += 1
            
            # simulate 1 week of semifinals
            # top 2 teams get bye
            byes = playoffs.head(2).team.values
            quarter_teams = playoffs.iloc[2:,:]
            quarter_scores = (sim_scores(quarter_teams.team.tolist())
                              .set_index('team')
                              .reset_index()
                              .rename(columns={0:'score'}))
            
            # get quarterfinals matchups and winners
            quarter_1 = quarter_scores.iloc[[0,3],:]
            quarter_1 = quarter_1[quarter_1.score == quarter_1.score.max()].team.values
            quarter_2 = quarter_scores.iloc[[1,2],:]
            quarter_2 = quarter_2[quarter_2.score == quarter_2.score.max()].team.values
            
            # get semifinals matchups and winners
            semi_teams = (playoffs[(playoffs.team.isin(quarter_1))
                         | (playoffs.team.isin(quarter_2))
                         | (playoffs.team.isin(byes))])
            
            semi_scores = (sim_scores(semi_teams.team.tolist())
                           .set_index('team')
                           .reset_index()
                           .rename(columns={0:'score'}))
            # get semi matchups
            semi_1 = semi_teams.iloc[[0,3],:].merge(semi_scores, on='team')
            semi_2 = semi_teams.iloc[[1,2],:].merge(semi_scores, on='team')
            
            # get finals matchup
            final_1 = semi_1.sort_values(by='score', ascending=False).head(1)
            final_2 = semi_2.sort_values(by='score', ascending=False).head(1)
            
            finals_teams = pd.concat([final_1, final_2])
            
            # gimulate 1 week of finals
            final_scores = (sim_scores(finals_teams.team.tolist())
                            .set_index('team')
                            .reset_index()
                            .rename(columns={0:'score'}))
            
            # count finals appearances
            for team in final_scores.team:
                n_finals[team] +=1
            
            champ = final_scores.sort_values(by='score', ascending=False).iloc[0,0]
            
            # count championships
            n_champ[champ] += 1
            
            # count runner ups
            runner = final_scores.sort_values(by='score').iloc[0,0]            
            n_second[runner] += 1
            
            # get third place matchup
            third_1 = semi_1.sort_values(by='score').head(1)
            third_2 = semi_2.sort_values(by='score').head(1)
            
            third_teams = pd.concat([third_1, third_2])
            
            # simulate third place matchup
            third_scores = (sim_scores(third_teams.team.tolist())
                            .set_index('team')
                            .reset_index()
                            .rename(columns={0:'score'}))
    
            # count third place
            third = third_scores.sort_values(by='score', ascending=False).iloc[0,0]
            n_third[third] += 1
    
    # convert dictionary counts to dataframes and combine
    playoffs = pd.DataFrame(n_playoffs.items(), columns=['team', 'n_playoffs'])
    finals = pd.DataFrame(n_finals.items(), columns=['team', 'n_finals'])
    champs = pd.DataFrame(n_champ.items(), columns=['team', 'n_champ'])
    runners = pd.DataFrame(n_second.items(), columns=['team', 'n_second'])
    thirds = pd.DataFrame(n_third.items(), columns=['team', 'n_third'])
    
    dfs = [playoffs, finals, champs, runners, thirds]
    
    playoff_sim = reduce(lambda left, right: pd.merge(left, right, on='team'), dfs)
    playoff_sim = playoff_sim.set_index('team')
    
    return playoff_sim

#%% What If
def scenarios(d, pr):
    '''
    Analyzes how a team would have performed in 3 scenarios:
        1: record vs every team every week
        2: record if team had another team's schedule
        3: record vs each team (1 matchup per week, loop for every team)
    '''
    
    regular_season_end = get_params(d)[2]
    df = get_params(d)[8]
    teams = get_params(d)[10]    
    
    # set up data
    df = df.apply(lambda x: x.astype(str).str.lower())
    df['week'] = df['week'].astype(str).astype(int)
    df['score1'] = df['score1'].astype(str).astype(float)
    df['score2'] = df['score2'].astype(str).astype(float)
    df = df[(df.week <= regular_season_end) & (df.score1 > 0)]
    
    ### 1. Calculate record vs every team
    wins_vs_league = pr[['team', 'week', 'score']]
    wins_vs_league['teams_beat'] = wins_vs_league.groupby('week')['score'].rank() - 1
    
    ### 2. Calculate record given another team's schedule
    # set up matrix
    switched_sched = (pd.DataFrame(index=teams,
                               columns=teams)
                               .fillna(0))
    
    # set up dataframe to return team scores
    hm = df[['week', 'team1', 'score1']].rename(columns={'team1':'team', 'score1':'score'})
    aw = df[['week', 'team2', 'score2']].rename(columns={'team2':'team', 'score2':'score'})
    return_score = hm.append(aw)
    
    # for each team in list (team 1), go through every other team (team 2) 
    # and replace team 1 score with team 2
    # if opponent in a week is the same as team 2, keep original schedule
    for tm1 in teams:
        #tm1 = 'bron'
        # return team 1's schedule
        hm = df[df.team1 == tm1]
        aw = df[df.team2 == tm1]
        aw = aw.rename(columns={"team1":"team2", 
                          "score1":"score2",
                          "team2":"team1",
                          "score2":"score1"})
        sched = hm.append(aw).sort_values("week")
    
        # get team 2's score for the current week to replace with team 1
        for tm2 in teams:
        #tm2 = 'gupt'
            for wk in sched.week:
                #wk = 8
                # return replacement team score
                sched_slice = sched[(sched.week==wk)]
                score = return_score[(return_score.week==wk) & (return_score.team==tm2)]
                
                # replace scores
                # if current week's schedule remains the same, return record; otherwise repace scores
                if sched_slice.team2.values == tm2:
                    result = np.where(sched_slice.score2 > sched_slice.score1, 1, 0)
                else:
                    sched_slice['team1'] = tm2
                    sched_slice['score1'] = score.score.values
                    
                # get result of hypothetica matchup if team 1 does not equal team 2
                if sched_slice.team2.values != tm2:
                    result = np.where(sched_slice.score1 > sched_slice.score2, 1, 0)
                
                switched_sched.loc[tm1, tm2] += result
    
    
    ### 3: record vs each opponent
    wins_vs_opp = (pd.DataFrame(index=teams,
                               columns=teams)
                               .fillna(0))
        
    hm = df[['week', 'team1', 'score1']].rename(columns={'team1':'team', 'score1':'score'})
    aw = df[['week', 'team2', 'score2']].rename(columns={'team2':'team', 'score2':'score'})
    return_score = hm.append(aw)
    
    # for each team (team 1), find how many times they would have beaten team 2
    for tm1 in teams:
        #tm1 = 'bron'
        # return team 1 scores
        hm = df[df.team1 == tm1].iloc[:,0:3]
        aw = df[df.team2 == tm1].iloc[:,[0,3,4]]
        aw = aw.rename(columns={"team1":"team2", 
                          "score1":"score2",
                          "team2":"team1",
                          "score2":"score1"})
        sched = hm.append(aw).sort_values("week")
        
        teams2 = [x for x in teams if x != tm1]
        for tm2 in teams2:
            #tm2 = "gupt"
            # return team 2 scores
            hm2 = df[df.team1 == tm2].iloc[:,0:3]
            hm2 = hm2.rename(columns={"team1":"team2", 
                      "score1":"score2"})
            aw2 = df[df.team2 == tm2].iloc[:,[0,3,4]]
            sched2 = hm2.append(aw2).sort_values("week")
        
            # combine schedules and find result for team 1
            matchups = pd.merge(sched, sched2, how="inner", on="week")
            wins = sum(np.where(matchups.score1 > matchups.score2, 1, 0))
            
            wins_vs_opp.loc[tm1, tm2] += wins
    
    return wins_vs_league, switched_sched, wins_vs_opp

# %% Lineup Efficiency
slotcodes = {
0 : 'QB', 1 : 'QB',
2 : 'RB', 3 : 'RB',
4 : 'WR', 5 : 'WR',
6 : 'TE', 7 : 'TE',
16: 'D/ST',
17: 'K',
20: 'Bench',
21: 'IR',
23: 'Flex'}

def get_matchups(league_id, season, week, swid='', espn=''):
    ''' 
    Pull full JSON of matchup data from ESPN API for a particular week.
    '''
    
    url = 'https://fantasy.espn.com/apis/v3/games/ffl/seasons/' + str(season) + '/segments/0/leagues/' + str(league_id)
    
    r = requests.get(url + '?view=mMatchup&view=mMatchupScore',
                     params={'scoringPeriodId': week, 'matchupPeriodId': week},
                     cookies={"SWID": swid, "espn_s2": espn})
    
    return r.json()

def get_slates(json):
    '''
    Constructs week team slates with slotted position, 
    position, and points (actual and ESPN projected),
    given full matchup info (`get_matchups`)
    '''
    
    slates = {}

    for team in json['teams']:
        slate = []
        for p in team['roster']['entries']:
            # get name
            name  = p['playerPoolEntry']['player']['fullName']

            # get actual lineup slot
            slotid = p['lineupSlotId']
            slot = slotcodes[slotid]

            # get projected and actual scores
            act, proj = 0, 0
            for stat in p['playerPoolEntry']['player']['stats']:
                if stat['scoringPeriodId'] != week:
                    continue
                if stat['statSourceId'] == 0:
                    act = stat['appliedTotal']
                elif stat['statSourceId'] == 1:
                    proj = stat['appliedTotal']
                else:
                    print('Error')

            # get type of player
            pos = 'Unk'
            ess = p['playerPoolEntry']['player']['eligibleSlots']
            if 0 in ess: pos = 'QB'
            elif 2 in ess: pos = 'RB'
            elif 4 in ess: pos = 'WR'
            elif 6 in ess: pos = 'TE'
            elif 16 in ess: pos = 'D/ST'
            elif 17 in ess: pos = 'K'

            slate.append([name, slotid, slot, pos, act, proj])

        slate = pd.DataFrame(slate, columns=['Name', 'SlotID', 'Slot', 'Pos', 'Actual', 'Proj'])
        slates[team['id']] = slate

    return slates

def compute_pts(slates, posns, struc):
    '''
    Given slates (`get_slates`), compute total roster pts:
    actual, optimal, and using ESPN projections
    
    Parameters
    --------------
    slates : `dict` of `DataFrames`
        (from `get_slates`)
    posns : `list`
        roster positions, e.g. ['QB','RB', 'WR', 'TE']
    struc : `list`
        slots per position, e.g. [1,2,2,1]
        
    * This is not flexible enough to handle "weird" leagues
    like 6 Flex slots with constraints on # total RB/WR
    
    Returns
    --------------
    `dict` of `dict`s with actual, ESPN, optimal points
    '''
    
    data = {}
    #positions = ['QB', 'RB', 'WR', 'TE', 'Flex', 'D/ST', 'K']

    for tmid, slate in slates.items():
        #tmid = 1
        #slate = slates[tmid]
        # go through each team roster
        pts = {'opts':0, 'apts':0,
               'QB.opts':0, 'QB.apts':0,
               'RB.opts':0, 'RB.apts':0,
               'WR.opts':0, 'WR.apts':0,
               'TE.opts':0, 'TE.apts':0,
               'Flex.opts':0, 'Flex.apts':0,
               'D/ST.opts':0, 'D/ST.apts':0,
               'K.opts':0, 'K.apts':0}


        # Total actual points - starters
        pts['apts'] = slate.query('Slot not in ["Bench", "IR"]').filter(['Actual']).sum().values[0]

        # Total actual points - by position
        for pos in posns:
            #if pos == 'Flex':
                #new_pos = slate.query('Slot == @pos').Pos.values[0]
                #pts[new_pos+'.apts'] += slate.query('Slot == @pos').filter(['Actual']).sum().values[0]
            pts[pos+'.apts'] += slate.query('Slot == @pos').filter(['Actual']).sum().values[0]        
        
        # Optimal points
        actflex = -100  # actual pts scored by flex
        for pos, num in zip(posns, struc):
            #pos = 'WR'
            #num = 2
            
            # actual points, sorted by actual outcome
            t = slate.query('Pos == @pos').sort_values(by='Actual', ascending=False).filter(['Actual']).values[:,0]

            # sum up points
            pts['opts'] += t[:num].sum()  # total optimal points
            pts[pos+'.opts'] = t[:num].sum()  # position optimal points

            # set the next best as flex
            if pos in ['RB', 'WR', 'TE'] and len(t) > num:
                # if current flex value is greater than previous actual, replace value
                if t[num] > actflex:
                    actflex = t[num]
                    #f = slate.query('Pos == @pos').sort_values(by='Actual', ascending=False).iloc[num,]
        
        # Add flex points to total optimal and position optimal
        pts['opts'] += actflex
        pts[pos+'.opts'] += actflex
        
        data[tmid] = pts
        
    return data

def convert_to_df(d, data):    
    team_map = get_params(d)[7]
    df = pd.DataFrame.from_dict(data, orient='index')
    df['week'] = week
    df['team'] = df.index
    df = df.replace({'team':team_map})
    
    return df


#league_id = 1382012
league_id = 7811678
season = 2020
swid = ''
espn = ''
d = load_data(league_id, season, swid="", espn="")

regular_season_end = get_params(d)[2]
lineup_slots_df = get_params(d)[12]
pos_df = lineup_slots_df[(lineup_slots_df.posID != 20) & (lineup_slots_df.posID != 21)]
posns = pos_df.pos.to_list()
struc = pos_df.limit.to_list()

df_scores = pd.DataFrame()
for week in range(1,regular_season_end+1):
    json = get_matchups(league_id, season, week, swid="", espn="")
    wslate = get_slates(json=json)
    wdata = compute_pts(wslate, posns, struc)
    
    df = convert_to_df(d, wdata)
    df_scores = df_scores.append(df)
    
    
# says week is not defined???
def analyze_lineup(d):
    regular_season_end = get_params(d)[2]
    lineup_slots_df = get_params(d)[12]
    pos_df = lineup_slots_df[(lineup_slots_df.posID != 20) & (lineup_slots_df.posID != 21)]
    posns = pos_df.pos.to_list()
    struc = pos_df.limit.to_list()
    
    df_scores = pd.DataFrame()
    for week in range(1,regular_season_end+1):
        json = get_matchups(league_id, season, week, swid, espn)
        wslate = get_slates(json=json)
        wdata = compute_pts(wslate, posns, struc)
        
        df = convert_to_df(d, wdata)
        df_scores = df_scores.append(df)
analyze_lineup(d)
