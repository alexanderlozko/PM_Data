import pandas as pd
from sklearn import preprocessing


class DataLoader:

    """
    Preparing data for the model
    """

    @staticmethod
    def prepare_data(path_reporting, path_wellness, path_calories, path_distance):

        """
        Calls functions to prepare a specific file

        :param path_reporting: Path to reporting file
        :param path_wellness: Path to wellness file
        :param path_calories: Path to calories file
        :param path_distance: Path to distance file
        :return: Main DataFrame
        """

        reporting = DataLoader.prepare_reporting(path_reporting=path_reporting)
        wellness = DataLoader.prepare_wellness(path_wellness=path_wellness)
        calories_distance = DataLoader.prepare_calories_distance(path_calories=path_calories,
                                                                 path_distance=path_distance)

        main = pd.merge(reporting, wellness, on='date', how='inner')
        main = pd.merge(main, calories_distance, on='date', how='inner')
        main = main.drop(['date'], axis=1)

        for columns in main.columns:
            if len(main[columns].unique()) == 1:
                main = main.drop(columns, axis=1)

        return main

    @staticmethod
    def prepare_reporting(path_reporting):

        """
        Prepares data from report file

        :param path_reporting: Path to reporting file
        :return: Reporting DataFrame
        """

        reporting = pd.read_csv(path_reporting)

        reporting.fillna(reporting.weight.mean(), inplace=True)
        reporting = reporting.drop_duplicates(['date'])
        reporting['date'] = reporting.date.astype('datetime64[ns]')
        reporting = reporting.sort_values(by=['date'], ignore_index=True)

        meal = ['Breakfast', 'Lunch', 'Dinner', 'Evening']
        for meal in meal:
            for line in reporting:
                reporting[meal.lower()] = reporting['meals'].apply(lambda x: 1 if meal in x else 0)

        reporting['alcohol_consumed'] = reporting['alcohol_consumed'].apply(lambda x: 1 if x == 'Yes' else 0)

        reporting['yesterday_weight'] = 0
        reporting['yesterday_weight'] = reporting['yesterday_weight'].astype('float64')
        reporting['result'] = 0
        for line in range(len(reporting)):
            line_previous = line
            if line == 0:
                line_previous = 1
            reporting.loc[line, ('yesterday_weight')] = reporting['weight'][line_previous - 1]
            if reporting['yesterday_weight'][line] > reporting['weight'][line]:
                reporting.loc[line, ('result')] = 1
            elif reporting['yesterday_weight'][line] < reporting['weight'][line]:
                reporting.loc[line, ('result')] = 2
            else:
                reporting.loc[line, ('result')] = 0

        reporting = reporting.drop(['timestamp', 'meals', 'weight'], axis=1)
        reporting[['yesterday_weight', 'glasses_of_fluid']] = preprocessing.scale(reporting[['yesterday_weight', 'glasses_of_fluid']])

        return reporting

    @staticmethod
    def prepare_wellness(path_wellness):

        """
        Prepares data from report file

        :param path_wellness: Path to wellness file
        :return: Wellness DataFrame
        """

        wellness = pd.read_csv(path_wellness)

        wellness['date'] = wellness['effective_time_frame'].astype('datetime64[ns]').dt.date.astype('datetime64[ns]')
        wellness = wellness.drop_duplicates(['date'])
        wellness = wellness.sort_values(by=['date'])

        area_list = list()
        for list_area in wellness.soreness_area.unique():
            received_list = list_area.strip('[]').split(', ')
            for area in range(len(received_list)):
                if received_list[area] != '':
                    area_list.append(received_list[area])

        for area in set(area_list):
            wellness[area] = 0

        for line in range(len(wellness)):
            received_list = wellness.soreness_area[line].strip('[]').split(', ')
            if received_list == ['']:
                continue
            for area in range(len(received_list)):
                wellness.loc[line, (received_list[area])] = 1


        wellness['sleep_duration_h'] = wellness['sleep_duration_h'].replace(range(4), 1)
        wellness['sleep_duration_h'] = wellness['sleep_duration_h'].replace(range(4, 6), 2)
        wellness['sleep_duration_h'] = wellness['sleep_duration_h'].replace(range(6, 8), 3)
        wellness['sleep_duration_h'] = wellness['sleep_duration_h'].replace(range(8, 10), 4)
        wellness['sleep_duration_h'] = wellness['sleep_duration_h'].replace(range(10, 25), 5)

        wellness['readiness'] = wellness['readiness'].replace(range(3), 1)
        wellness['readiness'] = wellness['readiness'].replace(range(3, 5), 2)
        wellness['readiness'] = wellness['readiness'].replace(range(5, 7), 3)
        wellness['readiness'] = wellness['readiness'].replace(range(7, 9), 4)
        wellness['readiness'] = wellness['readiness'].replace(range(9, 11), 5)

        wellness = wellness.drop(['effective_time_frame', 'soreness_area'], axis=1)

        return wellness


    @staticmethod
    def prepare_calories(path_calories):

        """
        Prepares data from calories file

        :param path_calories: Path to calories file
        :return: All_calories DataFrame
        """

        calories = pd.read_json(path_calories)
        calories['date'] = calories.dateTime.astype('datetime64[ns]').dt.date
        all_calories = pd.DataFrame({'calories': calories.groupby(['date'], as_index=False).sum()['value'],
                                     'date': calories.groupby(['date'], as_index=False).sum()['date']})
        all_calories = all_calories.drop_duplicates(['date'])
        all_calories = all_calories.sort_values(by=['date'])

        return all_calories


    @staticmethod
    def prepare_distance(path_distance):

        """
        Prepares data from distance file

        :param path_distance: Path to distance file
        :return: All_distance DataFrame
        """

        distance = pd.read_json(path_distance)
        distance['date'] = distance.dateTime.astype('datetime64[ns]').dt.date
        all_distance = pd.DataFrame({'distance': distance.groupby(['date'], as_index=False).sum()['value'],
                                     'date': distance.groupby(['date'], as_index=False).sum()['date']})
        all_distance = all_distance.drop_duplicates(['date'])
        all_distance = all_distance.sort_values(by=['date'])

        return all_distance


    @staticmethod
    def prepare_calories_distance(path_distance, path_calories):

        """
        Continues to prepare data from calories and distance files

        :param path_distance: Path to distance file
        :param path_calories: Path to calories file
        :return: Calories_distance DataFrame
        """

        all_distance = DataLoader.prepare_distance(path_distance)
        all_calories = DataLoader.prepare_calories(path_calories)

        calories_distance = pd.merge(all_distance, all_calories, on='date', how='inner')
        calories_distance['date'] = calories_distance['date'].astype('datetime64[ns]')
        calories_distance['calories_per_step'] = calories_distance['calories'] / (calories_distance['distance'] / 70)

        calories_distance['centimeter_norm'] = 0
        for line in range(len(calories_distance)):
            if calories_distance.iloc[line]['distance'] < 280000 or calories_distance.iloc[line]['distance'] == 'inf':
                calories_distance.loc[line, ('centimeter_norm')] = 1
            elif calories_distance.iloc[line]['distance'] >= 280000 and calories_distance.iloc[line][
                'distance'] < 630000:
                calories_distance.loc[line, ('centimeter_norm')] = 2
            elif calories_distance.iloc[line]['distance'] >= 630000 and calories_distance.iloc[line][
                'distance'] < 770000:
                calories_distance.loc[line, ('centimeter_norm')] = 3
            elif calories_distance.iloc[line]['distance'] >= 770000 and calories_distance.iloc[line][
                'distance'] < 1120000:
                calories_distance.loc[line, ('centimeter_norm')] = 4
            else:
                calories_distance.loc[line, ('centimeter_norm')] = 5

        calories_distance = calories_distance.drop(['distance'], axis=1)

        return calories_distance



