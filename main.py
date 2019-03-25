import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import sys

from subprocess import check_output

def seed_to_int(seed):
    # Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int


def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

def main():
    data_dir = '2019/'

    print(check_output(["ls", "2019/"]).decode("utf8"))

    df_seeds = pd.read_csv(data_dir + 'NCAATourneySeeds.csv')
    df_tour = pd.read_csv(data_dir + 'NCAATourneyCompactResults.csv')
    df_teams = pd.read_csv(data_dir + 'Teams.csv')
    df_massey = pd.read_csv(data_dir + 'MasseyOrdinals_thru_2019_day_128.csv')

    # filter massey
    df_massey = df_massey[df_massey['RankingDayNum'] == 128]
    df_massey = df_massey[df_massey['SystemName'].isin(['POM', 'SAG', 'TRP', 'TRK', 'DOK'])]

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
    print(df_temp)
    df_concat = pd.merge(df_concat, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

    df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'TRP')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

    df_concat = df_concat.rename(columns={'OrdinalRank_x': 'TRP_W', 'OrdinalRank_y': 'TRP_L'})

    # Merge TRK
    df_temp = df_winseeds_mass[(df_winseeds_mass.SystemName == 'TRK')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

    df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'TRK')]
    df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

    df_concat = df_concat.rename(columns={'OrdinalRank_x': 'TRK_W', 'OrdinalRank_y': 'TRK_L'})

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
    df_wins['TRK_W'] = df_concat['TRK_W']
    df_wins['TRK_L'] = df_concat['TRK_L']
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
    df_losses['TRK_W'] = df_concat['TRK_L']
    df_losses['TRK_L'] = df_concat['TRK_W']
    df_losses['TRP_W'] = df_concat['TRP_W']
    df_losses['TRP_L'] = df_concat['TRP_L']
    df_losses['DOK_W'] = df_concat['DOK_W']
    df_losses['DOK_L'] = df_concat['DOK_L']
    df_losses['Result'] = 0

    df_predictions = pd.concat((df_wins, df_losses))
    df_predictions.head()

    X_train = df_predictions[['POM_W', 'POM_L', 'SAG_W', 'SAG_L', 'TRK_W', 'TRK_L', 'TRP_W', 'TRP_L', 'DOK_W', 'DOK_L']].values.reshape(-1, 10)
    y_train = df_predictions.Result.values
    X_train, y_train = shuffle(X_train, y_train)

    logreg = LogisticRegression()
    params = {'C': np.logspace(start=-5, stop=3, num=9)}
    clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
    clf.fit(X_train, y_train)
    print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))

    df_sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage2.csv')
    n_test_games = len(df_sample_sub)

    X_test = np.zeros(shape=(n_test_games, 10))
    t1_arr = []
    t2_arr = []
    # t1 = winning team, t2 = losing team -> prob t1 beats t2
    for ii, row in df_sample_sub.iterrows():
        year, t1, t2 = get_year_t1_t2(row.ID)
        t1_name = df_teams[(df_teams.TeamID == t1)].TeamName.values[0]
        t2_name = df_teams[(df_teams.TeamID == t2)].TeamName.values[0]
        t1_arr.append(t1_name)
        t2_arr.append(t2_name)
        t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
        t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
        print("Year: " + str(year) + " for teams: " + str(t1_name) + " vs " + str(t2_name))
        POM_t1 = df_massey[(df_massey.RankingDayNum == 128) & (df_massey.Season == year) & (df_massey.TeamID == t1) &
                           (df_massey.SystemName == 'POM')].OrdinalRank.values[0]
        POM_t2 = df_massey[(df_massey.RankingDayNum == 128) & (df_massey.Season == year) & (df_massey.TeamID == t2) &
                           (df_massey.SystemName == 'POM')].OrdinalRank.values[0]
        SAG_t1 = df_massey[(df_massey.RankingDayNum == 128) & (df_massey.Season == year) & (df_massey.TeamID == t1) &
                           (df_massey.SystemName == 'SAG')].OrdinalRank.values[0]
        SAG_t2 = df_massey[(df_massey.RankingDayNum == 128) & (df_massey.Season == year) & (df_massey.TeamID == t2) &
                           (df_massey.SystemName == 'SAG')].OrdinalRank.values[0]
        TRX_t1 = df_massey[(df_massey.RankingDayNum == 128) & (df_massey.Season == year) & (df_massey.TeamID == t1) &
                           (df_massey.SystemName == 'TRK')].OrdinalRank.values[0]
        TRX_t2 = df_massey[(df_massey.RankingDayNum == 128) & (df_massey.Season == year) & (df_massey.TeamID == t2) &
                           (df_massey.SystemName == 'TRK')].OrdinalRank.values[0]

        TRP_t1 = df_massey[(df_massey.RankingDayNum == 128) & (df_massey.Season == year) & (df_massey.TeamID == t1) &
                           (df_massey.SystemName == 'TRP')].OrdinalRank.values[0]
        TRP_t2 = df_massey[(df_massey.RankingDayNum == 128) & (df_massey.Season == year) & (df_massey.TeamID == t2) &
                           (df_massey.SystemName == 'TRP')].OrdinalRank.values[0]

        DOX_t1 = df_massey[(df_massey.RankingDayNum == 128) & (df_massey.Season == year) & (df_massey.TeamID == t1) &
                           (df_massey.SystemName == 'DOK')].OrdinalRank.values[0]
        DOX_t2 = df_massey[(df_massey.RankingDayNum == 128) & (df_massey.Season == year) & (df_massey.TeamID == t2) &
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

    preds = clf.predict_proba(X_test)[:, :]

    clipped_preds = np.clip(preds, 0.025, 0.975)
    actual_preds = [1] - clipped_preds
    df_sample_sub.Pred = actual_preds
    df_sub = df_sample_sub
    df_sample_sub['t1'] = np.asarray(t1_arr)
    df_sample_sub['t2'] = np.asarray(t2_arr)
    df_sample_sub.head()

    df_sample_sub.to_csv('preds/2019/apex_builder_v1.csv', index=False)
    df_sub.to_csv('preds/2019/apex_builder_v1.csv', index=False)

if __name__ == "__main__":
    main()