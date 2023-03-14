import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import sys
import os
import csv
import xgboost as xgb


from subprocess import check_output

# https://www.markmoog.com/ranking_analysis - good source for ranking analysis

def seed_to_int(seed):
    # Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int

def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000
    x =  preds-labels
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess


def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))


def logit(p):
    return np.log(p / (1 - p))


def make_spread_submission(logloss_sub_fn, rmse_sub_fn):
    df = pd.read_csv(logloss_sub_fn)
    df["Pred"] = 6 * logit(df.Pred)
    df.to_csv(rmse_sub_fn, index=False)


def append_csv():
    data_dir = 'ncaam-tournament-data-2023/'

    # read the first CSV file
    df1 = pd.read_csv(data_dir + 'MMasseyOrdinals_2023_133_only_46systems.csv')

    # read the second CSV file
    df2 = pd.read_csv(data_dir + 'MMasseyOrdinals_thru_Season2023_Day128.csv')

    # append the second CSV file to the first CSV file and reset the index
    # concatenate the two DataFrames vertically
    df_combined = pd.concat([df2, df1], ignore_index=True)
    df_combined.head(10)
    df_combined.tail(10)

    # write the combined DataFrame to file1.csv
    df_combined.to_csv('ncaam-tournament-data-2023/MMasseyOrdinals_thru_Season2023_Day133.csv', index=False)


def main():
    data_dir = 'ncaam-tournament-data-2023/'
    extra_data_dir = 'extra-data-2023/'

    print(check_output(["ls", data_dir]).decode("utf8"))

    df_seeds = pd.read_csv(data_dir + 'MNCAATourneySeeds.csv')
    df_tour = pd.read_csv(data_dir + 'MNCAATourneyCompactResults.csv')
    df_teams = pd.read_csv(data_dir + 'MTeams.csv')
    df_massey = pd.read_csv(data_dir + 'MMasseyOrdinals_thru_Season2023_Day133.csv')

    df_massey_postseason = pd.read_csv(extra_data_dir + 'massey-postseason-data-2023.csv')
    week = 133
    evaluate_postseason = False

    # filter massey according to last day of season
    df_massey = df_massey[df_massey['RankingDayNum'] == 133]
    df_massey = df_massey[df_massey['SystemName'].isin(['POM', 'TRP', 'DOK', 'EBP', 'SAG'])]

    # view frame
    df_seeds.head()
    df_tour.head()
    df_teams.head()
    df_massey.head()

    df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
    df_seeds.drop(labels=['Seed'], inplace=True, axis=1)  # This is the string label
    df_seeds.head()

    df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
    df_tour.head()

    #  summarizes wins & losses along with their corresponding seed differences
    df_winseeds = df_seeds.rename(columns={'TeamID': 'WTeamID', 'seed_int': 'WSeed'})
    df_lossseeds = df_seeds.rename(columns={'TeamID': 'LTeamID', 'seed_int': 'LSeed'})

    df_winseeds_mass = df_massey.rename(columns={'TeamID': 'WTeamID'})
    df_lossseeds_mass = df_massey.rename(columns={'TeamID': 'LTeamID'})

    # Merge POM
    df_temp = df_winseeds_mass[(df_winseeds_mass.SystemName == 'POM')]
    df_concat = pd.merge(df_tour, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

    df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'POM')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

    df_concat = df_concat.rename(columns={'OrdinalRank_x': 'POM_W', 'OrdinalRank_y': 'POM_L'})

    # Merge SAG
    df_temp = df_winseeds_mass[(df_winseeds_mass.SystemName == 'SAG')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

    df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'SAG')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

    df_concat = df_concat.rename(columns={'OrdinalRank_x': 'SAG_W', 'OrdinalRank_y': 'SAG_L'})

    # Merge TRP
    df_temp = df_winseeds_mass[(df_winseeds_mass.SystemName == 'TRP')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

    df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'TRP')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

    df_concat = df_concat.rename(columns={'OrdinalRank_x': 'TRP_W', 'OrdinalRank_y': 'TRP_L'})

    # Merge EBP
    df_temp = df_winseeds_mass[(df_winseeds_mass.SystemName == 'EBP')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

    df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'EBP')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

    df_concat = df_concat.rename(columns={'OrdinalRank_x': 'EBP_W', 'OrdinalRank_y': 'EBP_L'})

    # Merge DOK
    df_temp = df_winseeds_mass[(df_winseeds_mass.SystemName == 'DOK')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

    df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'DOK')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

    df_concat = df_concat.rename(columns={'OrdinalRank_x': 'DOK_W', 'OrdinalRank_y': 'DOK_L'})

    df_concat = pd.merge(left=df_concat, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
    df_concat = pd.merge(left=df_concat, right=df_lossseeds, on=['Season', 'LTeamID'])

    df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed
    df_concat.head()

    df_wins = pd.DataFrame()
    df_wins['POM_W'] = df_concat['POM_W']
    df_wins['POM_L'] = df_concat['POM_L']
    df_wins['SAG_W'] = df_concat['SAG_W']
    df_wins['SAG_L'] = df_concat['SAG_L']
    df_wins['EBP_W'] = df_concat['EBP_W']
    df_wins['EBP_L'] = df_concat['EBP_L']
    df_wins['TRP_W'] = df_concat['TRP_W']
    df_wins['TRP_L'] = df_concat['TRP_L']
    df_wins['DOK_W'] = df_concat['DOK_W']
    df_wins['DOK_L'] = df_concat['DOK_L']

    df_wins['Result'] = 1

    df_losses = pd.DataFrame()
    df_losses['POM_W'] = df_concat['POM_L']
    df_losses['POM_L'] = df_concat['POM_W']
    df_losses['SAG_W'] = df_concat['SAG_L']
    df_losses['SAG_L'] = df_concat['SAG_W']
    df_losses['EBP_W'] = df_concat['EBP_L']
    df_losses['EBP_L'] = df_concat['EBP_W']
    df_losses['TRP_W'] = df_concat['TRP_W']
    df_losses['TRP_L'] = df_concat['TRP_L']
    df_losses['DOK_W'] = df_concat['DOK_W']
    df_losses['DOK_L'] = df_concat['DOK_L']
    df_losses['Result'] = 0

    df_predictions = pd.concat((df_wins, df_losses))
    df_predictions.head()

    X_train = df_predictions[['POM_W', 'POM_L', 'SAG_W', 'SAG_L', 'EBP_W', 'EBP_L', 'TRP_W', 'TRP_L', 'DOK_W', 'DOK_L']].values.reshape(-1, 10)
    y_train = df_predictions.Result.values # train according to a 0 or 1 -> 1 winning and 0 losing
    X_train, y_train = shuffle(X_train, y_train) # shuffle the training data

    param = {
        'max_depth': 3,
        'eta': 0.02,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'subsample': 0.35,
        'colsample_bytree': 0.7,
        'min_child_weight': 40,
        'gamma': 10,
        'num_parallel_tree': 10,
        'silent': 1
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)

    num_round = 3000
    xgb_model = xgb.train(param, dtrain, num_round)

    logreg = LogisticRegression(max_iter=10000)

    # use grid serach to identify the params for the regularization of paramaters
    # use log loss to score the grid search because we are using logistic regression to evaluate a binary outcome
    # TODO: run through a pipeline
    params = {'C': np.logspace(start=-5, stop=5, num=50)}
    clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)

    # train the funcation
    clf.fit(X_train, y_train)
    print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))

    df_sample_sub = pd.read_csv(data_dir + 'MSampleSubmission2023.csv')
    if evaluate_postseason:
        df_sample_sub = pd.read_csv(extra_data_dir + 'MSampleSubmissionStage3.csv')

    n_test_games = len(df_sample_sub)

    # predict the current year
    X_test = np.zeros(shape=(n_test_games, 10))
    t1_arr = []
    t2_arr = []
    # t1 = winning team, t2 = losing team -> prob t1 beats t2

    if evaluate_postseason:
        df_massey = df_massey_postseason

    for ii, row in df_sample_sub.iterrows():
        year, t1, t2 = get_year_t1_t2(row.ID)
        t1_name = df_teams[(df_teams.TeamID == t1)].TeamName.values[0]
        t2_name = df_teams[(df_teams.TeamID == t2)].TeamName.values[0]
        t1_arr.append(t1_name)
        t2_arr.append(t2_name)
        try:
            t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
            t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
            print("Year: " + str(year) + " for teams: " + str(t1_name) + " vs " + str(t2_name))
            POM_t1 = df_massey[(df_massey.RankingDayNum == week) & (df_massey.Season == year) & (df_massey.TeamID == t1) &
                               (df_massey.SystemName == 'POM')].OrdinalRank.values[0]
            POM_t2 = df_massey[(df_massey.RankingDayNum == week) & (df_massey.Season == year) & (df_massey.TeamID == t2) &
                               (df_massey.SystemName == 'POM')].OrdinalRank.values[0]
            SAG_t1 = df_massey[(df_massey.RankingDayNum == week) & (df_massey.Season == year) & (df_massey.TeamID == t1) &
                               (df_massey.SystemName == 'SAG')].OrdinalRank.values[0]
            SAG_t2 = df_massey[(df_massey.RankingDayNum == week) & (df_massey.Season == year) & (df_massey.TeamID == t2) &
                               (df_massey.SystemName == 'SAG')].OrdinalRank.values[0]
            TRX_t1 = df_massey[(df_massey.RankingDayNum == week) & (df_massey.Season == year) & (df_massey.TeamID == t1) &
                               (df_massey.SystemName == 'EBP')].OrdinalRank.values[0]
            TRX_t2 = df_massey[(df_massey.RankingDayNum == week) & (df_massey.Season == year) & (df_massey.TeamID == t2) &
                               (df_massey.SystemName == 'EBP')].OrdinalRank.values[0]

            TRP_t1 = df_massey[(df_massey.RankingDayNum == week) & (df_massey.Season == year) & (df_massey.TeamID == t1) &
                               (df_massey.SystemName == 'TRP')].OrdinalRank.values[0]
            TRP_t2 = df_massey[(df_massey.RankingDayNum == week) & (df_massey.Season == year) & (df_massey.TeamID == t2) &
                               (df_massey.SystemName == 'TRP')].OrdinalRank.values[0]

            DOX_t1 = df_massey[(df_massey.RankingDayNum == week) & (df_massey.Season == year) & (df_massey.TeamID == t1) &
                               (df_massey.SystemName == 'DOK')].OrdinalRank.values[0]
            DOX_t2 = df_massey[(df_massey.RankingDayNum == week) & (df_massey.Season == year) & (df_massey.TeamID == t2) &
                               (df_massey.SystemName == 'DOK')].OrdinalRank.values[0]

            X_test[ii, 0] = POM_t1
            X_test[ii, 1] = POM_t2
            X_test[ii, 2] = SAG_t1
            X_test[ii, 3] = SAG_t2
            X_test[ii, 4] = TRX_t1
            X_test[ii, 5] = TRX_t2

            X_test[ii, 6] = TRP_t1
            X_test[ii, 7] = TRP_t2

            X_test[ii, 8] = DOX_t1
            X_test[ii, 9] = DOX_t2
        except Exception:
            X_test[ii, 0] = 355
            X_test[ii, 1] = 355
            X_test[ii, 2] = 355
            X_test[ii, 3] = 355
            X_test[ii, 4] = 355
            X_test[ii, 5] = 355

            X_test[ii, 6] = 355
            X_test[ii, 7] = 355

            X_test[ii, 8] = 355
            X_test[ii, 9] = 355

    preds = clf.predict_proba(X_test)[:, :]
    # Make predictions on the test data
    dtest = xgb.DMatrix(X_test)
    y_pred = xgb_model.predict(dtest)


    # clip the predictions so do not get infinite log loss - Use XG Boost
    clipped_preds = np.clip(y_pred, 0.025, 0.975)
    actual_preds = clipped_preds
    if len(actual_preds.shape) == 1:
        actual_preds = actual_preds.reshape(-1, 1)
    df_sample_sub['Pred'] = actual_preds[:, 0]
    df_sub = df_sample_sub
    df_sub['t1'] = np.asarray(t1_arr)
    df_sub['t2'] = np.asarray(t2_arr)
    df_sample_sub.head()
    df_sub.head()

    # df_sample_sub.to_csv('preds/2022/apex_builder_v1_submit.csv', index=False)
    team_name_df_sub = 'preds/2023/apex_builder_v1_xg.csv'
    df_sub.to_csv(team_name_df_sub, index=False)
    make_spread_submission(team_name_df_sub, 'preds/2023/rmse_sub_fn_v1_xg.csv')


if __name__ == "__main__":
    main()