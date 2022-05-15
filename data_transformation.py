#!/usr/bin/python
import pandas as pd
import math
import os
import numpy as np


def transform_data(directory):
    testing_directory=directory


    #### getting the list of files, with the directory
    file_names = np.array(os.listdir(testing_directory))
    file_names = [testing_directory + file for file in file_names]

    ###a list that will conatain patient numbers
    patient_list=[]


    # this dataframe will hold a row corresponding to each patient we want to test. we fill it later.
    testing_dataframe = pd.DataFrame()

    # these are appended to the start of each column in the fixed data to denote how far
    # are we in the patient's history
    div_names = ['33_', '66_', '100_']

    # This creates our fixed test data
    for file in file_names:
        dataframe = pd.read_csv(file, sep='|')
        split_string = file.split(".", 1)
        substring = split_string[0]
        split_string = substring.split("patient_", 1)
        # getting our patient number
        patient_num = split_string[1]
        patient_list.append(patient_num)

        #### We will first remove the ones we said : EtCO2, Bilirubin_direct, Fibrinogen

        dataframe = dataframe.drop(labels=['EtCO2', 'Bilirubin_direct', 'Fibrinogen'], axis=1)
        # we impute each patients table using backwards fill, then forward filling
        dataframe = dataframe.fillna(method='bfill').ffill()

        #### Remove the rows after the first 1.
        # last rows
        try:
            first_sic = dataframe[dataframe['SepsisLabel'] == 1].index[0]
            dataframe = dataframe.loc[0:first_sic]
        except:
            first_sic = -1

        # we now count how many rows we have in the fixed dataframe
        rows_count = dataframe.shape[0] - 1
        # we take the constant values, i.e the ones who don't change between rows and put them into one df
        constant_cols = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'SepsisLabel']
        constant_df = dataframe[constant_cols]
        # we also make a dataframe of the values who change in each column
        series_df = dataframe.drop(constant_cols, axis=1)

        ##We take one row out of the constant data, as it is all the same
        first_constant = constant_df.iloc[-1]

        # We here create a row for each patient, as explained in our report
        dividers = [math.floor(rows_count * i / 3) for i in range(1, 4)]
        first_row = series_df.iloc[0].add_prefix('0_percent')
        for i, div in enumerate(dividers):
            cur_div = div_names[i]
            first_row = pd.concat([first_row, series_df.iloc[div].add_prefix(cur_div + 'percent')])
        # finally adding the constant vales to our changing values
        first_row = pd.concat([first_row, first_constant])
        print(patient_num)

        # adding the new patient to the  dataframe
        testing_dataframe = testing_dataframe.append(first_row, ignore_index=True)

    return testing_dataframe, patient_list
    # testing_dataframe.to_csv('testing_dataframe.csv')`