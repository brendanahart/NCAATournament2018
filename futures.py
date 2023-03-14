from ncaa_simulator import Data, Submission, Tournament, round_names
import pandas as pd


def main():
    gender = 'M'
    dir = 'ncaam-tournament-data-2023/'
    season = 2023

    # regions -> W, X, Y, Z
    # Games -> R1(Region, game number) ie R1W5 (Round 1 W Region Game #5)
    # Teams and Seed in reverse dicts too (id to team name, etc...)
    ncaa_data = Data(mw=gender, dir=dir)

    df = pd.read_csv(f'preds/{season}/apex_builder_v1_submit.csv')
    submission = Submission(sub_df=df, data=ncaa_data)

    # initiate a tournament object
    tourney = Tournament(data=ncaa_data, submission=submission, season=season)

    print(tourney)

    df_round_prob = pd.DataFrame()
    team_ids = tourney.s_dict_rev.keys()
    seeds = tourney.s_dict_rev.values()
    teams = []
    for team_id in team_ids:
        teams.append(tourney.t_dict[team_id])

    df_round_prob['Team'] = pd.Series(teams)
    df_round_prob['TeamID'] = pd.Series(team_ids)
    df_round_prob['Seed'] = pd.Series(seeds)

    df_round_prob.head()
    # Season Team Name Team ID R1 Advance


if __name__ == '__main__':
    main()
