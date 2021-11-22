# -*- coding:utf-8 -*-
# Python Package import
# importing libraries
import pandas as pd
import os
import time
import pickle
import joblib
import numpy as np
from nptdms import TdmsFile
from pandas.api.types import CategoricalDtype  # category type Lv 설정
from scipy import signal
from scipy.signal import savgol_filter, firwin, filtfilt
from scipy.interpolate import interp1d
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
# import datetime
import string
import warnings
import traceback
import math
import lightgbm as lgb
from lightgbm import LGBMClassifier, plot_importance
import xgboost
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
def file_load(fp, fl):
    df = pd.DataFrame()
    for i in fl:
        df = pd.concat([df, pd.read_csv(fp + i)], axis=0)
    return df
def Y_change_prediction(data):
    data.loc[data['prediction'] == 0, 'prediction'] = '0%'
    data.loc[data['prediction'] == 1, 'prediction'] = '10%'
    data.loc[data['prediction'] == 2, 'prediction'] = '20%'
    data.loc[data['prediction'] == 3, 'prediction'] = '30%'
    # data.loc[data['prediction'] == '40%', 'prediction'] = 4
    # data.loc[data['prediction'] == '50%', 'prediction'] = 5
    data.loc[data['prediction'] == 4, 'prediction'] = '60%'
    data.loc[data['prediction'] == 5, 'prediction'] = '70%'
    data.loc[data['prediction'] == 6, 'prediction'] = '80%'
    # data.loc[data['prediction'] == '90%', 'prediction'] = 9
    data.loc[data['prediction'] == 7, 'prediction'] = '100%'
    data.loc[data['prediction'] == 10, 'prediction'] = 'Tan_1.5mm'
    data.loc[data['prediction'] == 11, 'prediction'] = 'Tan_3.0mm'
    data.loc[data['prediction'] == 8, 'prediction'] = 'Redial_1.0mm'
    data.loc[data['prediction'] == 9, 'prediction'] = 'Redial_2.0mm'
    return data
def Y_change(data):
    data.loc[data['pad_state'] == '0%', 'Y'] = 0
    data.loc[data['pad_state'] == '10%', 'Y'] = 1
    data.loc[data['pad_state'] == '20%', 'Y'] = 2
    data.loc[data['pad_state'] == '30%', 'Y'] = 3
    # data.loc[data['pad_state'] == '40%', 'Y'] = 4
    # data.loc[data['pad_state'] == '50%', 'Y'] = 5
    data.loc[data['pad_state'] == '60%', 'Y'] = 4
    data.loc[data['pad_state'] == '70%', 'Y'] = 5
    data.loc[data['pad_state'] == '80%', 'Y'] = 6
    # data.loc[data['pad_state'] == '90%', 'Y'] = 9
    data.loc[data['pad_state'] == '100%', 'Y'] = 7
    data.loc[data['pad_state'] == 'Tangential 1.5mm', 'Y'] = 8
    data.loc[data['pad_state'] == 'Tangential 3.0mm', 'Y'] = 9
    data.loc[data['pad_state'] == 'Redial 1.0mm', 'Y'] = 10
    data.loc[data['pad_state'] == 'Redial 2.0mm', 'Y'] = 11
    return data
class MODEL_TRAIN:
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    batch_n = ''
    path = ''

    def __init__(self, train_x, train_y, val_x, val_y, batch_n, path, analog_col):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.batch_n = batch_n
        self.path = path
        self.analog_col = analog_col
    def xgb_train(self):

        batch_size = int(len(self.train_x) / self.batch_n)
        xgb_val = xgboost.DMatrix(self.val_x, label=self.val_y)

        parameter = {'objective': ['multi:softprob'],
                     'booster': ['gbtree'],
                     'eval_metric': ['mlogloss'],
                     'num_class': [12],
                     'eta': [0.05],
                     'max_depth': [6, 8],
                     'gamma': [1, 2],
                     'min_child_weight': [0],
                     'colsample_bytree': [0.5, 1],
                     'random_state': [2021]
                     }

        expand_param_list = pd.DataFrame([row for row in product(*parameter.values())],
                                         columns=parameter.keys()).to_dict('index')

        xgb_param_result = pd.DataFrame(index=range(0, len(expand_param_list)), columns=['param_num', 'acc'])
        for param_num in range(0, len(expand_param_list)):
            for batch in range(0, len(self.train_x), batch_size):
                train_x_temp = self.train_x[batch:batch + batch_size]
                train_y_temp = self.train_y[batch:batch + batch_size]

                xgb_train_temp = xgboost.DMatrix(train_x_temp, label=train_y_temp)

                if batch == 0:
                    print('batch == 0')
                    xgb_model = xgboost.train(expand_param_list[param_num], xgb_train_temp)
                else:
                    print('batch != 0')
                    xgb_model = xgboost.train(expand_param_list[param_num], xgb_train_temp, xgb_model=xgb_model)

                xgb_acc = accuracy_score(xgb_model.predict(xgb_val).argmax(axis=1), self.val_y)
                print('param_num: {}/{} & batch: {}/{}'.format(param_num, len(expand_param_list) - 1, batch, range(0, len(self.train_x), batch_size)[-1]))
            pickle.dump(xgb_model, open('{}xgb_model_{}.pkl'.format(self.path, param_num), "wb"))

            xgb_param_result.loc[param_num, 'param_num'] = param_num
            xgb_param_result.loc[param_num, 'acc'] = round(xgb_acc, 5)
            print('---------------------- finish ', str(param_num), ' / ', len(expand_param_list) - 1)
        best_param_num = xgb_param_result.sort_values(by=['acc'], ascending=False).reset_index(drop=True, inplace=False).loc[0, 'param_num']  # 66
        xgb_acc = xgb_param_result.sort_values(by=['acc'], ascending=False).reset_index(drop=True, inplace=False).loc[0, 'acc']  # best accuracy # 0.78754
        xgb_acc = pd.DataFrame({'xgb_acc': [xgb_acc]})
        xgb_acc.to_csv('{}{}_xgb_acc.csv'.format(self.path,self.analog_col))

        xgb_best_param = expand_param_list[best_param_num]  # best parameter
        pickle.dump(xgb_best_param, open('{}_{}_xgb_best_param.pkl'.format(self.path,self.analog_col), "wb"))

        file_xgb = pd.DataFrame(index=range(0, len(expand_param_list)), columns=['param_num', 'xgb_model_path'])
        for param_num in range(0, len(expand_param_list)):
            file_xgb.loc[param_num, 'param_num'] = param_num
            file_xgb.loc[param_num, 'xgb_model_path'] = '{}xgb_model_{}.pkl'.format(self.path, str(param_num))

        file_xgb = file_xgb.drop(file_xgb[file_xgb['param_num'] == best_param_num].index)
        for i in file_xgb['xgb_model_path']:
            os.remove(i)

        os.rename('{}xgb_model_{}.pkl'.format(self.path, best_param_num),
                  '{}xgb_model_{}_{}.pkl'.format(self.path, self.analog_col, datetime.now().strftime("%Y%m%d%H")))

    "SELECT * FROM users WHERE username = %(user)s"
    def lgb_train(self):

        dictionary = {'objective': ['multiclass'],
                      'num_class': [12],
                      'num_threads': [8],  # =n_jobs
                      'metric': ['multi_logloss'],
                      'bagging_fraction': [0.7],
                      'boosting_type': ['gbdt'],
                      'feature_fraction': [0.9],
                      'learning_rate': [0.01],
                      'num_iterations': [250],  # =n_estimators
                      'num_leaves': [10, 15, 17, 20, 30],  # *하나의 트리가 가질 수 있는 최대 리프 수 # 2^(max_depth)보다 작아야 함
                      'max_depth': [6, 8],  # *트리 최대 깊이
                      'random_state': [2021]}

        dic3 = pd.DataFrame([row for row in product(*dictionary.values())],
                            columns=dictionary.keys()).to_dict('index')  # all params df

        param_df = pd.DataFrame()

        batch_size = int(len(self.train_x) / self.batch_n)  # mini batch size
        lgb_test = lgb.Dataset(self.val_x, label=self.val_y, free_raw_data=False)

        for i in range(len(dic3)):
            params = dic3[i]

            for start in range(0, len(self.train_x), batch_size):
                dtrain = lgb.Dataset(self.train_x[start:start + batch_size], self.train_y[start:start + batch_size],
                                     free_raw_data=False)

                if start == 0:
                    model = lgb.train(params, dtrain, valid_sets=lgb_test, early_stopping_rounds=100)
                else:
                    model = lgb.train(params, dtrain, valid_sets=lgb_test, init_model=model, early_stopping_rounds=100)

            pickle.dump(model, open('{}{}_lgb_model.pkl'.format(self.path, str(i)), "wb"))

            y_pr = model.predict(self.val_x)
            accuracy = accuracy_score(self.val_y['Y'], y_pr.argmax(axis=1))  # 계산할 때 타입 맞추기

            df = pd.DataFrame({'param': [i], 'acc': [accuracy]})
            param_df = param_df.append(df, ignore_index=True)

            print('---------------------- params done: {} /{}, batch: {}'.format(i, len(dic3) - 1, start))

        # best params save
        param_id = param_df['acc'].idxmax()
        lgb_params = dic3[param_id]
        # parameter save
        pickle.dump(lgb_params, open('{}lgb_best_param.pkl'.format(self.path), "wb"))

        # param_df acc save
        pd.DataFrame({  # 'model': ['lgb'],
            'lgb_acc': [param_df.loc[param_id, 'acc']]}).to_csv('{}{}'.format(self.path, self.analog_col+'_lgb_acc.csv'))

        # 안쓰는 모델 삭제
        remove_list = param_df[param_df['param'] != param_id]['param'].astype(str).apply(lambda x: x + '_lgb_model.pkl')
        remove_list = remove_list.to_list()

        for file in range(0, len(remove_list)):
            os.remove("{}{}".format(self.path, remove_list[file]))

        # 최종 모델 이름 변경(lgb_연월일.pkl)
        os.rename("{}{}".format(self.path, str(param_id) + '_lgb_model.pkl'),
                  "{}lgb_model_{}_{:%Y%m%d%H}.pkl".format(self.path, self.analog_col, datetime.now()))
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        # 1 이슈
        # 기존꺼 냅두고 재학습
        # 2 이슈
        # node1의 편향될 경우 node별 안에 있는 피쳐에 대해서 가중치를 초기화 해야된다!
        # 입력층에 대해서 편향될 경우 초기화 함
        # 입력층 1 : 1280
        # 입력층 2 : 640
        node_1 = 1280
        node_2 = 640
        node_3 = 320
        node_4 = 128
        # node_5 = 64

        self.layer_1 = nn.Linear(num_feature, node_1)
        self.layer_2 = nn.Linear(node_1, node_2)
        self.layer_3 = nn.Linear(node_2, node_3)
        self.layer_4 = nn.Linear(node_3, node_4)
        self.layer_out = nn.Linear(node_4, num_class)

        self.loss = nn.ReLU6()
        # self.loss = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.001)
        self.batchnorm1 = nn.BatchNorm1d(node_1)
        self.batchnorm2 = nn.BatchNorm1d(node_2)
        self.batchnorm3 = nn.BatchNorm1d(node_3)
        self.batchnorm4 = nn.BatchNorm1d(node_4)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.loss(x)
        # x = self.dropout(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.loss(x)
        # x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.loss(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.loss(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x
# TDMS_file_read
def tdms_load(tdms_dt, group):
    try:
        data = tdms_dt[group].as_dataframe(time_index=True, absolute_time=True).copy()
        data["time"] = [item for item in data.index].copy()
        df = data
    except:
        data = tdms_dt.object(group).as_dataframe().copy()
        time_col = pd.DataFrame({'time': tdms_dt.group_channels(group)[0].time_track(absolute_time=True)}).copy()
        df = pd.concat([time_col, data], axis=1)
    return df
# TDMS_Data Frame(CAN_DATA) 컬럼 선정
class col_select:
    def __init__(self, CAN_dt, col, road_list):
        self.data = CAN_dt
        self.colList = ['time', 'MCyl', 'WHL_FL', 'WHL_FR', 'WHL_RL', 'WHL_RR', 'YAW_RATE', 'SAS_ANG', 'SAS_SPEED',
                        'Pedal_Travel', 'Pedal_effort']
        self.pad = col
        self.road_list = road_list

    def making(self):
        data_1 = self.data
        if 'CYL_PRES' in self.data.columns:
            data_1 = data_1.rename(
                columns={"CYL_PRES": "MCyl", "WHL_SPD_FL": "WHL_FL", "WHL_SPD_FR": "WHL_FR", "WHL_SPD_RL": "WHL_RL",
                         "WHL_SPD_RR": "WHL_RR", "C_59102758_3_t": "Pedal_Travel", "C_59102758_4_e": "Pedal_effort",
                         "SAS_Angle": "SAS_ANG", "SAS_Speed": "SAS_SPEED", "C_56803861_3_PDT":"Pedal_Travel"})

        data_1 = data_1.rename(columns={'Pedal_effort': 'Pedal_Effort'})
        try:
            if 'EMS12.BRAKE_ACT' in self.data.columns:
                CAN_df = data_1[['time', 'MCyl', 'WHL_FL', 'WHL_FR', 'WHL_RL', 'WHL_RR', 'YAW_RATE', 'SAS_ANG', 'SAS_SPEED',
                                 'EMS12.BRAKE_ACT', 'CF_Clu_InhibitN', 'CF_Clu_InhibitP', 'CF_Clu_InhibitR',
                                 'CF_Clu_InhibitD']]
            else:
                CAN_df = data_1[['time', 'MCyl', 'WHL_FL', 'WHL_FR', 'WHL_RL', 'WHL_RR', 'YAW_RATE', 'SAS_ANG', 'SAS_SPEED',
                                 'Pedal_Travel', 'Pedal_Effort']]
        except KeyError:
                CAN_df = data_1[['time', 'MCyl', 'WHL_FL', 'WHL_FR', 'WHL_RL', 'WHL_RR', 'YAW_RATE', 'SAS_ANG', 'SAS_SPEED',
                                 'EMS12.BRAKE_ACT', 'CF_Clu_InhibitN', 'CF_Clu_InhibitP', 'CF_Clu_InhibitR','CF_Clu_InhibitD']]

        if (CAN_df['time'][2] - CAN_df['time'][1]).total_seconds() == 0.01:
            CAN_df = CAN_df.iloc[CAN_df.index % 10 == 0,]
        elif (CAN_df['time'][2] - CAN_df['time'][1]) == 0.01:
            CAN_df = CAN_df.iloc[CAN_df.index % 10 == 0,]
        else:
            pass
        CAN_df_1 = CAN_df.copy()
        CAN_df_1['pad_state'] = self.pad
        CAN_df_1['file_name'] = self.road_list
        return CAN_df_1
### Referee ####
## BrakePoint 조건 정의
# class MakeBrake_Rule:
#     def __init__(self, MCyl, Pedal_Travel):
#         self.Cyl = MCyl
#         self.Pedal = Pedal_Travel
#
#     def making(self):
#         MCyl_rule = self.Cyl >= 1.5  # 일정값
#         # MCyl_stand = (self.Cyl - np.mean(self.Cyl, axis=0)) / np.std(self.Cyl, axis=0)  # 표준화
#         # Travel_stand = (self.Pedal - min(self.Pedal)) / (max(self.Pedal) - min(self.Pedal))  # 표준화
#         Travel_mean = self.Pedal > np.mean(self.Pedal)  # 평균값
#         Travel_maxmin = self.Pedal > self.Pedal.min() + (self.Pedal.max() - self.Pedal.min()) * 0.20  # 최대값-최소값 대비 5%
#
#         make_df = pd.DataFrame({#'MCyl_stand_val': MCyl_stand,
#                                 #'Pedal_Travel_stand': Travel_stand,
#                                 "Pedal_Travel_mean": Travel_mean,
#                                 "MCyl_rule_val": MCyl_rule,
#                                 "Pedal_Travel_min": Travel_maxmin})
#         return make_df
# BrakePoint 1차 생성(끝)
class MakeBrakePoint_First:
    def __init__(self, Can_join, hz = 100):
        self.Can_join = Can_join
        self.Can_join = self.Can_join.reset_index(drop=True)
        self.hz = hz

    def making(self):
        if 'MCyl_rule_val' in self.Can_join.columns:
            self.Can_join.loc[
                 (self.Can_join['MCyl_rule_val'] == True) & (self.Can_join['Pedal_Travel_mean'] == True)
                & (self.Can_join['Pedal_Travel_min'] == True) & (self.Can_join['WHL_FR'] > 0) & (self.Can_join['WHL_FL'] > 0) & (self.Can_join['WHL_RR'] > 0) & (
                        self.Can_join['WHL_RL'] > 0) |  # >>TTT

                 (self.Can_join['MCyl_rule_val'] == True) & (self.Can_join['Pedal_Travel_mean'] == True)
                & (self.Can_join['Pedal_Travel_min'] == True) & (self.Can_join['WHL_FR'] > 0) & (self.Can_join['WHL_FL'] > 0) & (self.Can_join['WHL_RR'] > 0) & (
                        self.Can_join['WHL_RL'] > 0) |  # <>TTT

                 (self.Can_join['MCyl_rule_val'] == True) & (self.Can_join['Pedal_Travel_mean'] == True)
                & (self.Can_join['Pedal_Travel_min'] == True) & (self.Can_join['WHL_FR'] > 0) & (self.Can_join['WHL_FL'] > 0) & (self.Can_join['WHL_RR'] > 0) & (
                        self.Can_join['WHL_RL'] > 0) |  # ><TTT

                 (self.Can_join['MCyl_rule_val'] != True) & (self.Can_join['Pedal_Travel_mean'] == True)
                & (self.Can_join['Pedal_Travel_min'] == True) & (self.Can_join['WHL_FR'] > 0) & (self.Can_join['WHL_FL'] > 0) & (self.Can_join['WHL_RR'] > 0) & (
                        self.Can_join['WHL_RL'] > 0) |  # >>FTT

                 (self.Can_join['MCyl_rule_val'] == True) & (self.Can_join['Pedal_Travel_mean'] != True)
                & (self.Can_join['Pedal_Travel_min'] == True) & (self.Can_join['WHL_FR'] > 0) & (self.Can_join['WHL_FL'] > 0) & (self.Can_join['WHL_RR'] > 0) & (
                        self.Can_join['WHL_RL'] > 0) |  # >>TFT
                 (self.Can_join['MCyl_rule_val'] == True) & (self.Can_join['Pedal_Travel_mean'] == True)
                & (self.Can_join['Pedal_Travel_min'] != True) & (self.Can_join['WHL_FR'] > 0) & (self.Can_join['WHL_FL'] > 0) & (self.Can_join['WHL_RR'] > 0) & (
                        self.Can_join['WHL_RL'] > 0), 'brake_yn'] = 1  # >>TTF
        elif 'EMS12.BRAKE_ACT' in self.Can_join.columns:
            self.Can_join.loc[(self.Can_join['EMS12.BRAKE_ACT'] == 2) & (self.Can_join['WHL_FR'] > 0) & (self.Can_join['WHL_FL'] > 0) & (self.Can_join['WHL_RR'] > 0) &
                              (self.Can_join['WHL_RL'] > 0) & (self.Can_join['CF_Clu_InhibitD'] == 1) & (self.Can_join['MCyl'] >= 1.2), 'brake_yn'] = 1
            self.Can_join.loc[(self.Can_join['CF_Clu_InhibitR'] == 1) | (self.Can_join['CF_Clu_InhibitN'] == 1), 'brake_yn'] = 0
        else:
            self.Can_join.loc[(self.Can_join['DriverBraking'] == 1) & (self.Can_join['WHL_FR'] > 0) & (self.Can_join['WHL_FL'] > 0) & (self.Can_join['WHL_RR'] > 0) &
                              (self.Can_join['WHL_RL'] > 0) & (self.Can_join['MCyl'] >= 1.2), 'brake_yn'] = 1
        # CAN_join = Data raw
        brake_rule = self.Can_join.fillna(0)
        time_label_test = pd.DataFrame(brake_rule.loc[brake_rule['brake_yn'] == 1, ['time', 'brake_yn']]).reset_index(drop=True)
        time_label_test['brake_count'] = None
        try:
            for q in range(0, len(time_label_test)):
                if q == 0:
                    time_label_test.loc[[q], ['brake_count']] = 1
                elif (time_label_test['time'][q] - time_label_test['time'][q-1]).total_seconds() < 0.5:
                    time_label_test.loc[[q], ['brake_count']] = int(time_label_test.loc[[q-1]].brake_count)
                else:
                    # time_label_test.loc[[q], ['brake_count']] = time_label_test['brake_count'].max()
                    time_label_test.loc[[q], ['brake_count']] = int(time_label_test.loc[[q-1]].brake_count) + 1

        except KeyError:
            time_label_test.loc[[q], ['brake_count']] = time_label_test['brake_count'].max()

        brake_point_first = pd.merge(self.Can_join, time_label_test, how='left', on=['time','brake_yn']).fillna(0)
        brake_point_first.loc[(brake_point_first['brake_yn'] == 0), 'brake_count'] = 0
        return brake_point_first
# BrakePoint 특정 속도로 유지되는 구간 제외(끝)
def maintain(df):
    # Brake 끝지점 5이하 제외
    brake_point_re = df.copy()
    brake_point_re.drop(['brake_yn', 'brake_count'], axis=1, inplace=True)
    mantain = pd.DataFrame()

    for m in range(1, int(df['brake_count'].max()) + 1):
        try:
            brake_maintain = df[df['brake_count'] == m].reset_index(drop=True)
            brake_maintain['WHL_mean'] = brake_maintain[['WHL_FL', 'WHL_FR', 'WHL_RL', 'WHL_RR']].mean(axis=1)
            for r in range(0, len(brake_maintain)):
                if brake_maintain['WHL_mean'][r] < 5.1:
                    break
            brake_y = brake_maintain.loc[brake_maintain.time < brake_maintain['time'][r]].reset_index(drop=True).copy()
           # brake_y['brake_count'] = m

            if (brake_y['time'].max() - brake_y['time'].min()).total_seconds() < 1.0:
                brake_y['brake_count'] = 0
                brake_y['brake_yn'] = 0
            mantain = mantain.append(brake_y)
        except:
            pass

    mantain = mantain.reset_index(drop=True)
    # mantain.drop(['WHL_mean'], axis=1, inplace=True)
    result = pd.merge(brake_point_re, mantain, how='left').fillna(0)
    return result
# BrakePoint 2차 생성(끝)
class MakeBrakePoint_Second:
    def __init__(self, brake_point_first):
        self.brake_point_first = brake_point_first

    def making(self):
        # Brake 회차별 최대 휠 속도 20 이하 제외
        for h in range(1, int(self.brake_point_first['brake_count'].max()+1)):
            try:
                brake_detail = self.brake_point_first[self.brake_point_first['brake_count'] == h].reset_index(drop=True)
                brake_detail['WHL_mean'] = brake_detail[['WHL_FL', 'WHL_FR', 'WHL_RL', 'WHL_RR']].mean(axis=1)
                if brake_detail.WHL_mean.max() <= 20:
                    self.brake_point_first.loc[(self.brake_point_first['brake_count'] == h),['brake_count','brake_yn']] = 0
                elif brake_detail.WHL_mean.max() > 150:
                    self.brake_point_first.loc[(self.brake_point_first['brake_count'] == h), ['brake_count', 'brake_yn']] = 0
                else:
                    pass
            except IndexError:
                pass

        # max time에서 min time 빼기
        brake_y = self.brake_point_first[self.brake_point_first['brake_count'] != 0]
        result = (brake_y.groupby('brake_count')['time'].agg('max') - brake_y.groupby('brake_count')['time'].agg('min'))

        sec_df = result.reset_index(level='brake_count').iloc[:, :3]
        sec_df.columns = ['brake_count', 'sec']
        sec_df['sec'] = sec_df.sec.astype('timedelta64[ms]')/1000
        sec_brake = sec_df[sec_df.sec >= 1].reset_index(drop=True)

        brake_point_second = pd.merge(self.brake_point_first, sec_brake, how='left', on=['brake_count'])
        brake_point_second['sec'] = brake_point_second['sec'].fillna(0)
        brake_point_second.loc[(brake_point_second['sec'] == 0), ['brake_count','brake_yn']] = 0
        # brake_point_second.loc[(brake_point_second['sec'] == 0), 'brake_yn'] = 0

        return brake_point_second
# CAN Columns 제외
def can_columns(CAN):
    drop_col = ['MCyl_stand_val', 'Pedal_Travel_stand', 'Pedal_Travel_mean', 'Pedal_Travel_min', 'MCyl_rule_val',
                'brake_yn', 'CF_Clu_InhibitN', 'CF_Clu_InhibitP', 'CF_Clu_InhibitR', 'CF_Clu_InhibitD',
                'Pedal_Effort', 'Pedal_Travel', 'EMS12.BRAKE_ACT', 'brake_yn', 'YAW_RATE','DriverBraking']
    for drop_c in drop_col:
        try:
            CAN.drop(drop_c, axis=1, inplace=True)
        except Exception:
            pass
    return CAN
# 변화율 생성
class Makelag:  # - 데이터 lag shift
    def __init__(self, data, lag, col):
        self.data = data
        self.lag = range(1, lag + 1)
        self.col = col
        self.stand = data[col]

    def addlag(self):  ### 데이터 lag 데이터 생성
        result = self.data
        for lag_time in self.lag:
            new_col = f'{self.col}_lag{lag_time}'
            # diff =  self.stand.shift(lag_time)

            df_diff = pd.DataFrame({new_col: self.stand.shift(lag_time) - self.stand})
            result = pd.concat([result, df_diff], axis=1)
        return result
class MakeSpdDiff:  # - shift 된 데이터에 대해 차이값 / 비율 생성
    def __init__(self, data, col):
        self.data = data
        self.col = col
        self.stand = data.loc[:, col].apply(lambda x: 0.001 if x == 0 else x)

        self.lag1 = data.loc[:, f'{col}_lag1']
        self.lag2 = data.loc[:, f'{col}_lag2']
        self.lag3 = data.loc[:, f'{col}_lag3']

        self.newcol1 = f'{col}_lag1_dif'
        self.newcol2 = f'{col}_lag2_dif'
        self.newcol3 = f'{col}_lag3_dif'
        self.newcol4 = f'{col}_lag1_per'
        self.newcol5 = f'{col}_lag2_per'
        self.newcol6 = f'{col}_lag3_per'

    def addDiff(self):
        df_dif = pd.DataFrame({self.newcol1: self.lag1,
                               self.newcol2: (self.lag1 + self.lag2) / 2,
                               self.newcol3: (self.lag1 + self.lag2 + self.lag3) / 3
                               })

        df_dif2 = pd.DataFrame({self.newcol4: (df_dif.iloc[:, 0] / self.stand),
                                self.newcol5: (df_dif.iloc[:, 1] / self.stand),
                                self.newcol6: (df_dif.iloc[:, 2] / self.stand)
                                })
        # pd.concat([df_dif[[self.newcol1, self.newcol2, self.newcol3]], df_dif2[[self.newcol4, self.newcol5, self.newcol6]]], axis=1)

        return pd.concat([df_dif[self.newcol3], df_dif2[self.newcol6]], axis=1)

# Analog 컬럼 선정
class Analog_col_select:
    def __init__(self, Analog, analog_col_list):
        self.data = Analog
        # self.colList = ['STR_X', 'STR_Y', 'STR_Z']
        self.colList = analog_col_list

    def making(self):
        if 'STR_X' not in self.data.columns:
            self.data = self.data[['time','FR_X','FR_Y','FR_Z']].rename(columns={'FR_X':'STR_X', 'FR_Y':'STR_Y', 'FR_Z':'STR_Z'})
        else:
            self.data = self.data[['time','STR_X','STR_Y','STR_Z']]
        return self.data
# 초 만들기
def make_sec(df):  # - 브레이크별 시간 생성
    group_set = ['file_name', 'brake_count']
    result = df.groupby(group_set).agg('size')
    result = result.reset_index(level=group_set).iloc[:, :3]
    result.columns = ['file_name', 'brake_count', 'sec']
    result = result[result['brake_count'] != 0]
    return result
# CAN 가속도
class MakeAccel:
    def __init__(self, brake):
        self.brake_point = brake
        self.whl_colnm = ['WHL_FL', 'WHL_FR', 'WHL_RL', 'WHL_RR', 'WHL_mean']
    def making(self):
        final_accel = pd.DataFrame()
        all_final_accel = pd.DataFrame()
        for m in range(1, int(self.brake_point['brake_count'].max()) + 2):
            brake_detail_count = self.brake_point[self.brake_point.brake_count == m]
            bra_info = brake_detail_count[
                ['pad_state', 'file_name', 'brake_yn', 'brake_count']].drop_duplicates().reset_index(drop=True)
            make_accel = pd.DataFrame()
            try:
                for whl_list in self.whl_colnm:
                    WHL_max = brake_detail_count[brake_detail_count['time'] == brake_detail_count['time'].max()][
                        whl_list].values.__str__().strip("'[]")
                    WHL_max = pd.to_numeric(WHL_max)
                    WHL_min = brake_detail_count[brake_detail_count['time'] == brake_detail_count['time'].min()][
                        whl_list].values.__str__().strip("'[]")
                    WHL_min = pd.to_numeric(WHL_min)
                    # WHL_max = float(brake_detail_count[brake_detail_count['time'] == brake_detail_count['time'].max()][whl_list].values.__str__().strip("'[]"))
                    # WHL_min = float(brake_detail_count[brake_detail_count['time'] == brake_detail_count['time'].min()][whl_list].values.__str__().strip("'[]"))
                    # time_val = (datetime.strptime(brake_detail_count['time'].max(), '%Y-%m-%d %H:%M:%S.%f') - datetime.strptime(brake_detail_count['time'].min(), '%Y-%m-%d %H:%M:%S.%f')).total_seconds()

                    time_val = (brake_detail_count['time'].max() - brake_detail_count['time'].min()).total_seconds()
                    brake_accel = ((WHL_max - WHL_min) / 3.6) / time_val
                    make_df = pd.DataFrame({f'{whl_list}_accel': [brake_accel]})
                    # make_df[f'{whl_list}_accel'] = ((make_df[f'{whl_list}_accel']) - 0) / (100*1.5 - 0)
                    make_accel = pd.concat([make_accel, make_df], axis=1)
                make_accel = pd.concat([bra_info, make_accel], axis=1)
                final_accel = final_accel.append(make_accel)
            except:
                pass
        all_final_accel = all_final_accel.append(final_accel).dropna()
        return all_final_accel
class MakeAccel_test:
    def __init__(self, brake, cut_labels):
        self.brake_point = brake
        self.whl_colnm = ['WHL_FL', 'WHL_FR', 'WHL_RL', 'WHL_RR', 'WHL_mean']
        self.cut_list = cut_labels
    def making(self):
        final_accel = pd.DataFrame()
        all_final_accel = pd.DataFrame()
        for m in range(1, int(self.brake_point['brake_count'].max()) + 2):
            brake_detail_count = self.brake_point[self.brake_point.brake_count == m]
            bra_info = brake_detail_count[['pad_state', 'file_name', 'brake_yn', 'brake_count']].drop_duplicates().reset_index(drop=True)
            make_accel = pd.DataFrame()
            try:
                for cut in self.cut_list:
                    cut_point = brake_detail_count[brake_detail_count['cut']==cut]
                    for whl_list in self.whl_colnm:
                        WHL_max = cut_point[cut_point['time'] == cut_point['time'].max()][whl_list].values.__str__().strip("'[]")
                        WHL_max = pd.to_numeric(WHL_max)
                        WHL_min = cut_point[cut_point['time'] == cut_point['time'].min()][whl_list].values.__str__().strip("'[]")
                        WHL_min = pd.to_numeric(WHL_min)
                        # WHL_max = float(cut_point[cut_point['time'] == cut_point['time'].max()][whl_list].values.__str__().strip("'[]"))
                        # WHL_min = float(cut_point[cut_point['time'] == cut_point['time'].min()][whl_list].values.__str__().strip("'[]"))
                        # time_val = (datetime.strptime(cut_point['time'].max(), '%Y-%m-%d %H:%M:%S.%f') - datetime.strptime(cut_point['time'].min(), '%Y-%m-%d %H:%M:%S.%f')).total_seconds()
                        time_val = (cut_point['time'].max() - cut_point['time'].min()).total_seconds()
                        brake_accel = ((WHL_max - WHL_min) / 3.6) / time_val
                        make_df = pd.DataFrame({f'{whl_list}_accel': [brake_accel]})
                        # make_df[f'{whl_list}_accel'] = ((make_df[f'{whl_list}_accel']) - 0) / (100*1.5 - 0)
                        make_accel = pd.concat([make_accel, make_df], axis=1)
                    make_accel = pd.concat([bra_info, make_accel], axis=1)
                    final_accel = final_accel.append(make_accel)
            except:
                pass
        all_final_accel = all_final_accel.append(final_accel).dropna()
        return all_final_accel
# CAN 정규화
def data_filter(list_raw):

    list_raw = list_raw[list_raw['MCyl_meancut1'] <= 30]
    list_raw = list_raw[list_raw['MCyl_meancut2'] <= 30]
    list_raw['MCyl_meancut1'] = list_raw['MCyl_meancut1']/30
    list_raw['MCyl_meancut2'] = list_raw['MCyl_meancut2']/30

    list_raw = list_raw[list_raw['WHL_mean_max'] < 150]
    list_raw = list_raw[list_raw['WHL_mean_min'] < 150]

    list_raw = list_raw[list_raw['WHL_mean_accel'] < 0]

    list_raw['WHL_mean_accel'] = abs(list_raw['WHL_mean_accel']).copy()
    list_raw = list_raw[abs(list_raw['SAS_ANG_meancut1']) < 10]
    list_raw = list_raw[abs(list_raw['SAS_ANG_meancut2']) < 10]

    list_raw['SAS_ANG_meancut1'] = list_raw['SAS_ANG_meancut1'] / 10
    list_raw['SAS_ANG_meancut2'] = list_raw['SAS_ANG_meancut2'] / 10

    list_raw['SAS_SPEED_meancut1'] = ((list_raw['SAS_SPEED_meancut1']) - 0) / ((250 * 1.2) - 0)
    list_raw['SAS_SPEED_meancut2'] = ((list_raw['SAS_SPEED_meancut2']) - 0) / ((250 * 1.2) - 0)


    return list_raw
def scaling(list_raw):
    # list_raw['WHL_FL_mean'] = ((list_raw['WHL_FL_mean']) - 0) / (150 - 0)

    list_raw['WHL_FL_meancut1'] = ((list_raw['WHL_FL_meancut1']) - 0) / (150 - 0)
    list_raw['WHL_FR_meancut1'] = ((list_raw['WHL_FR_meancut1']) - 0) / (150 - 0)
    list_raw['WHL_RL_meancut1'] = ((list_raw['WHL_RL_meancut1']) - 0) / (150 - 0)
    list_raw['WHL_RR_meancut1'] = ((list_raw['WHL_RR_meancut1']) - 0) / (150 - 0)
    list_raw['WHL_mean_meancut1'] = ((list_raw['WHL_mean_meancut1']) - 0) / (150 - 0)

    list_raw['WHL_FL_meancut2'] = ((list_raw['WHL_FL_meancut2']) - 0) / (150 - 0)
    list_raw['WHL_FR_meancut2'] = ((list_raw['WHL_FR_meancut2']) - 0) / (150 - 0)
    list_raw['WHL_RL_meancut2'] = ((list_raw['WHL_RL_meancut2']) - 0) / (150 - 0)
    list_raw['WHL_RR_meancut2'] = ((list_raw['WHL_RR_meancut2']) - 0) / (150 - 0)
    list_raw['WHL_mean_meancut2'] = ((list_raw['WHL_mean_meancut2']) - 0) / (150 - 0)

    list_raw['WHL_FL_meancut3'] = ((list_raw['WHL_FL_meancut3']) - 0) / (150 - 0)
    list_raw['WHL_FR_meancut3'] = ((list_raw['WHL_FR_meancut3']) - 0) / (150 - 0)
    list_raw['WHL_RL_meancut3'] = ((list_raw['WHL_RL_meancut3']) - 0) / (150 - 0)
    list_raw['WHL_RR_meancut3'] = ((list_raw['WHL_RR_meancut3']) - 0) / (150 - 0)
    list_raw['WHL_mean_meancut3'] = ((list_raw['WHL_mean_meancut3']) - 0) / (150 - 0)

    list_raw['WHL_FL_meancut4'] = ((list_raw['WHL_FL_meancut4']) - 0) / (150 - 0)
    list_raw['WHL_FR_meancut4'] = ((list_raw['WHL_FR_meancut4']) - 0) / (150 - 0)
    list_raw['WHL_RL_meancut4'] = ((list_raw['WHL_RL_meancut4']) - 0) / (150 - 0)
    list_raw['WHL_RR_meancut4'] = ((list_raw['WHL_RR_meancut4']) - 0) / (150 - 0)
    list_raw['WHL_mean_meancut4'] = ((list_raw['WHL_mean_meancut4']) - 0) / (150 - 0)

    list_raw['WHL_FL_meancut5'] = ((list_raw['WHL_FL_meancut5']) - 0) / (150 - 0)
    list_raw['WHL_FR_meancut5'] = ((list_raw['WHL_FR_meancut5']) - 0) / (150 - 0)
    list_raw['WHL_RL_meancut5'] = ((list_raw['WHL_RL_meancut5']) - 0) / (150 - 0)
    list_raw['WHL_RR_meancut5'] = ((list_raw['WHL_RR_meancut5']) - 0) / (150 - 0)
    list_raw['WHL_mean_meancut5'] = ((list_raw['WHL_mean_meancut5']) - 0) / (150 - 0)

    # 휠합산 평균속도 컬럼 추가
    # list_raw = list_raw.loc[(list_raw['SAS_SPEED_meancut1'] < 250)].copy()  # 250 이상 제외 / 절대값 매겨서 놓고
    # list_raw = list_raw.loc[(list_raw['SAS_SPEED_meancut2'] < 250)].copy()  # 250 이상 제외 / 절대값 매겨서 놓고
    # list_raw = list_raw.loc[(list_raw['SAS_SPEED_meancut3'] < 250)].copy()  # 250 이상 제외 / 절대값 매겨서 놓고
    # list_raw = list_raw.loc[(list_raw['SAS_SPEED_meancut4'] < 250)].copy()  # 250 이상 제외 / 절대값 매겨서 놓고
    # list_raw = list_raw.loc[(list_raw['SAS_SPEED_meancut5'] < 250)].copy()  # 250 이상 제외 / 절대값 매겨서 놓고
    #
    # list_raw['SAS_SPEED_meancut1'] = ((list_raw['SAS_SPEED_meancut1']) - 0) / ((200 * 1.2) - 0)
    # list_raw['SAS_SPEED_meancut2'] = ((list_raw['SAS_SPEED_meancut2']) - 0) / ((200 * 1.2) - 0)
    # list_raw['SAS_SPEED_meancut3'] = ((list_raw['SAS_SPEED_meancut3']) - 0) / ((200 * 1.2) - 0)
    # list_raw['SAS_SPEED_meancut4'] = ((list_raw['SAS_SPEED_meancut4']) - 0) / ((200 * 1.2) - 0)
    # list_raw['SAS_SPEED_meancut5'] = ((list_raw['SAS_SPEED_meancut5']) - 0) / ((200 * 1.2) - 0)

    # list_raw = list_raw[abs(list_raw['SAS_ANG_meancut1']) < 10]
    # list_raw = list_raw[abs(list_raw['SAS_ANG_meancut2']) < 10]
    # list_raw = list_raw[abs(list_raw['SAS_ANG_meancut3']) < 10]
    # list_raw = list_raw[abs(list_raw['SAS_ANG_meancut4']) < 10]
    # list_raw = list_raw[abs(list_raw['SAS_ANG_meancut5']) < 10]

    # list_raw['SAS_ANG_meancut1'] = list_raw['SAS_ANG_meancut1'] / 10
    # list_raw['SAS_ANG_meancut2'] = list_raw['SAS_ANG_meancut2'] / 10
    # list_raw['SAS_ANG_meancut3'] = list_raw['SAS_ANG_meancut3'] / 10
    # list_raw['SAS_ANG_meancut4'] = list_raw['SAS_ANG_meancut4'] / 10
    # list_raw['SAS_ANG_meancut5'] = list_raw['SAS_ANG_meancut5'] / 10

    # list_raw = list_raw.loc[(list_raw['MCyl'] >= 1.2)].copy()  # MCyl 1.2 이상만 가져가기
    ### 50 인지는 체크 --> 기준 값이상은 50으로 변환
    # list_raw['MCyl'] = (list_raw['MCyl']) / 50  # 값 찾아야됨
    # list_raw.loc[(list_raw['SAS_ANG'] > 0), 'SAS_ANG_posi'] = 1  # 양음수 구분자 넣기
    # list_raw.loc[(list_raw['SAS_ANG'] < 0), 'SAS_ANG_nege'] = 1  # 양음수 구분자 넣기
    # list_raw = list_raw.loc[(abs(list_raw['SAS_ANG']) < 10)].copy()  # 10 이상 제외 / 절대값 매겨서 놓고

    # list_raw['SAS_ANG'] = list_raw['SAS_ANG'].abs()
    # list_raw['SAS_ANG'] = ((list_raw['SAS_ANG']) - 0) / (50 - 0)
    return list_raw
def scaling2(list_raw):
    list_raw['WHL_FL'] = ((list_raw['WHL_FL']) - 0) / (150 - 0)
    list_raw['WHL_FR'] = ((list_raw['WHL_FR']) - 0) / (150 - 0)
    list_raw['WHL_RL'] = ((list_raw['WHL_RL']) - 0) / (150 - 0)
    list_raw['WHL_RR'] = ((list_raw['WHL_RR']) - 0) / (150 - 0)
    list_raw['WHL_mean'] = ((list_raw['WHL_mean']) - 0) / (150 - 0)

    # 휠합산 평균속도 컬럼 추가
    # list_raw = list_raw.loc[(list_raw['SAS_SPEED'] < 250)].copy()  # 250 이상 제외 / 절대값 매겨서 놓고
    list_raw['SAS_SPEED'] = ((list_raw['SAS_SPEED']) - 0) / ((200 * 1.2) - 0)

    # list_raw = list_raw[abs(list_raw['SAS_ANG']) < 10]
    list_raw['SAS_ANG'] = list_raw['SAS_ANG'] / 10

    # list_raw = list_raw.loc[(list_raw['MCyl'] >= 1.2)].copy()  # MCyl 1.2 이상만 가져가기
    ### 50 인지는 체크 --> 기준 값이상은 50으로 변환
    # list_raw['MCyl'] = (list_raw['MCyl']) / 50  # 값 찾아야됨
    # list_raw.loc[(list_raw['SAS_ANG'] > 0), 'SAS_ANG_posi'] = 1  # 양음수 구분자 넣기
    # list_raw.loc[(list_raw['SAS_ANG'] < 0), 'SAS_ANG_nege'] = 1  # 양음수 구분자 넣기
    # list_raw = list_raw.loc[(abs(list_raw['SAS_ANG']) < 10)].copy()  # 10 이상 제외 / 절대값 매겨서 놓고

    # list_raw['SAS_ANG'] = list_raw['SAS_ANG'].abs()
    # list_raw['SAS_ANG'] = ((list_raw['SAS_ANG']) - 0) / (50 - 0)
    return list_raw
# CAN 속도 평균
def CAN_mean(lag_dt, sec_df):
    key = ['file_name', 'brake_count']
    lag_dt = pd.merge(lag_dt, sec_df, on=key)

    lag_dt = lag_dt[lag_dt['brake_count'] > 0]
    lag_dt = lag_dt[lag_dt['sec'] > 10]
    lag_dt['row_per'] = (lag_dt.groupby(['file_name', 'brake_count'])['pad_state'].cumcount() + 1) / lag_dt['sec']

    label_nm = ['cut1']
    lag_dt["cut"] = pd.cut(lag_dt["row_per"], 1, labels=label_nm).astype(str)
    lag_dt.drop(['row_per', 'time', 'brake_yn'], axis=1, inplace=True)
    grp_result = lag_dt.groupby(['pad_state', 'file_name', 'brake_count', 'sec', 'cut']).agg({'mean'})
    return grp_result
# Cut 만들기
def group_col(grp_df):
    grp_df = grp_df.reset_index()
    col1 = grp_df.columns.droplevel(1)
    col0 = "_" + grp_df.columns.droplevel(0)  # + "_"+cut
    grp_df.columns = col1 + col0
    change_col = grp_df.columns
    grp_df.rename(columns={change_col[0]: 'pad_state',
                           change_col[1]: 'file_name',
                           change_col[2]: 'brake_count',
                           change_col[3]: 'sec',
                           change_col[4]: 'cut'}, inplace=True)
    return grp_df
# 패드 번호 변환
def Y_change(data):
    data.loc[data['pad_state'] == '0%', 'Y'] = 0
    data.loc[data['pad_state'] == '10%', 'Y'] = 1
    data.loc[data['pad_state'] == '20%', 'Y'] = 2
    data.loc[data['pad_state'] == '30%', 'Y'] = 3
    data.loc[data['pad_state'] == '40%', 'Y'] = 4
    data.loc[data['pad_state'] == '50%', 'Y'] = 5
    data.loc[data['pad_state'] == '60%', 'Y'] = 6
    data.loc[data['pad_state'] == '70%', 'Y'] = 7
    data.loc[data['pad_state'] == '80%', 'Y'] = 8
    data.loc[data['pad_state'] == '90%', 'Y'] = 9
    data.loc[data['pad_state'] == '100%', 'Y'] = 10
    data.loc[data['pad_state'] == 'Tangential 1.5mm', 'Y'] = 11
    data.loc[data['pad_state'] == 'Tangential 3.0mm', 'Y'] = 12
    data.loc[data['pad_state'] == 'Redial 1.0mm', 'Y'] = 13
    data.loc[data['pad_state'] == 'Redial 2.0mm', 'Y'] = 14
    return data
class psd_data:
    def __init__(self, analog, analog_col, brake, scaling = True):
        self.analog = analog
        self.analog_col = analog_col
        self.brake = brake
        self.scaling = scaling

    def make_psd(self, hz, seg, name):
        raw_df = pd.DataFrame()
        psd_fir_result = pd.DataFrame()
        for m in range(1, int(self.brake['brake_count'].max())+1):
            try:
                self.analog = self.analog.fillna(method='ffill')
                brake_detail_count = self.brake[self.brake.brake_count == m].copy()
                analog_test = self.analog.loc[(self.analog.time < brake_detail_count.time.max()) & (self.analog.time > brake_detail_count.time.min())]
                analog_final = analog_test.reset_index(drop=True)
                # print('brake_count'+str(m))
                tmp = analog_final.copy()
                tmp['brake_count'] = m
                tmp['file_name'] = self.brake['file_name'][1]
                tmp['pad_state'] = self.brake['pad_state'][1]
                raw_df = raw_df.append(tmp)
                psd_concat = pd.DataFrame()
                # print('chk'+str(m))
                try:
                    for colnm in self.analog_col:
                        # print(colnm)
                        # if self.scaling == True:
                        #     analog_final[colnm] = 2*((analog_final[colnm] - analog_final[colnm].min()) / (analog_final[colnm].max() - analog_final[colnm].min())-.5)
                        #     # print(analog_final[colnm].describe())
                        # else:
                        #     pass
                        Analog_stft = PSD(analog_final, colnm, hz, seg, name)
                        psd_concat = pd.concat([psd_concat, Analog_stft], axis=1)
                        # print('done:'+str(m))
                    psd_concat['brake_count'] = m
                except ValueError:
                    # print('pass:' )
                    pass
                psd_fir_result = psd_fir_result.append(psd_concat)
                # print('append done:' + str(m))
            except:
                pass
        psd_fir_result['brake_count'] = psd_fir_result['brake_count'].astype(np.float64)
        # print('float done:' + str(m))
        return psd_fir_result, raw_df
class psd_data_test:
    def __init__(self, analog, analog_col, brake, scaling = True):
        self.analog = analog
        self.analog_col = analog_col
        self.brake = brake
        self.scaling = scaling

    def make_psd(self, hz, seg, name):
        raw_df = pd.DataFrame()
        psd_fir_result = pd.DataFrame()
        cut_list = ['cut1', 'cut2', 'cut3', 'cut4', 'cut5']
        for m in range(1, int(self.brake['brake_count'].max())+1):
            brake_detail_count = self.brake[self.brake.brake_count == m].copy()  # CAN
            for cut_name in cut_list:
                try:
                    # self.analog = self.analog.fillna(method='ffill')
                    brake_count_cut = brake_detail_count[brake_detail_count['cut'] == cut_name]
                    analog_test = self.analog.loc[(self.analog.time < brake_count_cut['time_max'][0]) & (self.analog.time > brake_count_cut['time_min'][0])]
                    analog_final = analog_test.reset_index(drop=True)
                    # print('brake_count'+str(m))
                    tmp = analog_final.copy()
                    tmp['brake_count'] = m
                    tmp['file_name'] = self.brake['file_name'][1]
                    tmp['pad_state'] = self.brake['pad_state'][1]
                    raw_df = raw_df.append(tmp)
                    psd_concat = pd.DataFrame()
                    # print('chk'+str(m))
                    try:
                        for colnm in self.analog_col:
                            # print(colnm)
                            if self.scaling == True:
                                analog_final[colnm] = 2*((analog_final[colnm] - analog_final[colnm].min()) / (analog_final[colnm].max() - analog_final[colnm].min())-.5)
                                # print(analog_final[colnm].describe())
                            else:
                                pass
                            Analog_stft = PSD(analog_final, colnm, hz, seg, name)
                            psd_concat = pd.concat([psd_concat, Analog_stft], axis=1)
                            # print('done:'+str(m))
                        psd_concat['brake_count'] = m
                    except ValueError:
                        # print('pass:' )
                        pass
                    psd_fir_result = psd_fir_result.append(psd_concat)
                    # print('append done:' + str(m))
                except:
                    pass
        psd_fir_result['brake_count'] = psd_fir_result['brake_count'].astype(np.float64)
        # print('float done:' + str(m))
        return psd_fir_result, raw_df

def PSD(data, analog_col, hz, seg, name):
    data = data
    analog_col = analog_col
    hz = hz
    seg = seg
    name = name
    # data PSD
    f, Pxx_den = signal.welch(data[analog_col], hz, nperseg=seg, return_onesided=True)

    # psd_spectrum
    stft_df_median = pd.DataFrame(np.abs(Pxx_den)).T
    freq_col_median = pd.Series(f)
    freq_col_median[0] = 0.001
    stft_df_median.columns = freq_col_median
    stft_df_median2 = stft_df_median ** 2

    df2 = pd.DataFrame()
    for i in range(0, int((hz/2)+10), 10):
        log_df_T = stft_df_median2.loc[:, (stft_df_median2.columns < i+1) & (stft_df_median2.columns > (i - 10))].T
        temp = pd.DataFrame({f'{analog_col}_psd_{name}_{i}_sum': log_df_T.sum(axis=0)}).apply(np.sqrt)
        df2 = pd.concat([df2, temp], axis=1)
    df2 = df2.fillna(0)
    return df2
# stft 생성
class make_stft:
    def __init__(self, analog, analog_col, brake, CAN):
        self.analog = analog # 진동 원본
        self.analog_col = analog_col # 컬럼 명칭
        self.brake = brake # 브레이크 회차
        self.CAN = CAN # CAN time

    def make_stft(self, hz, seg):
        stft_piv_result = pd.DataFrame()

        for analog_col_2 in self.analog_col:
            # analog[analog_col_2] = analog[analog_col_2].fillna(method='ffill')
            # self.analog[analog_col_2] = 2 * ((self.analog[analog_col_2] - self.analog[analog_col_2].min()) / (self.analog[analog_col_2].max() - self.analog[analog_col_2].min()) - .5)
            f2, t2, Zxx2 = signal.stft(self.analog[analog_col_2], hz, nperseg=seg, return_onesided=True)
            analog_time = self.analog.iloc[self.analog.index % (hz/2) == 0,].reset_index(drop=True)
            # stft_time = pd.date_range(start=CAN.time.min(), end=CAN.time.max(), freq='0.5S').tolist()
            stft_df_1 = pd.DataFrame(np.abs(Zxx2)).T
            freq_col = pd.Series(f2)
            freq_col[0] = 0.001
            stft_df_1.columns = freq_col
            stft_df_1 = pd.concat([stft_df_1, pd.DataFrame(analog_time['time']).rename(columns={0: 'time'})],axis=1).copy()

            stft_df_1.loc[(stft_df_1['time'].isna() == True), 'time'] = stft_df_1['time'].max() + timedelta(seconds=0.5)
            stft_df_1 = stft_df_1.dropna(axis=0)
            psd_stft_result = pd.DataFrame()

            for m in range(1, int(self.brake['brake_count'].max() + 1)):
                try:
                    brake_count = self.brake[self.brake.brake_count == m].copy()  # CAN
                    analog_test = stft_df_1.loc[
                        (stft_df_1.time < brake_count.time.max()) & (stft_df_1.time > brake_count.time.min())].copy()
                    analog_final = analog_test.reset_index(drop=True)
                    analog_final.drop(['time'], axis=1, inplace=True)
                    stft_concat = pd.DataFrame()

                    Analog_stft = stft(analog_final, analog_col_2, 'stft', hz=hz)
                    stft_concat = pd.concat([stft_concat, Analog_stft], axis=1)
                    stft_concat['brake_count'] = m
                except ValueError:
                    pass
                psd_stft_result = psd_stft_result.append(stft_concat)
                psd_stft_result = psd_stft_result[psd_stft_result[analog_col_2+'_stft_0_sum']!=0].reset_index(drop=True)

            stft_piv_result = pd.concat([stft_piv_result, psd_stft_result], axis=1)
            stft_piv_result = stft_piv_result.loc[:, ~stft_piv_result.columns.duplicated()]
            stft_piv_result = stft_piv_result[stft_piv_result['STR_X_stft_0_sum']!=0]
        return stft_piv_result
class make_stft_test:
    def __init__(self, analog, analog_col, brake, CAN):
        self.analog = analog # 진동 원본
        self.analog_col = analog_col # 컬럼 명칭
        self.brake = brake # 브레이크 회차
        self.CAN = CAN # CAN time

    def make_stft(self, hz, seg):
        stft_piv_result = pd.DataFrame()
        for analog_col_2 in self.analog_col:
            # analog[analog_col_2] = analog[analog_col_2].fillna(method='ffill')
            self.analog[analog_col_2] = 2 * ((self.analog[analog_col_2] - self.analog[analog_col_2].min()) / (self.analog[analog_col_2].max() - self.analog[analog_col_2].min()) - .5)
            f2, t2, Zxx2 = signal.stft(self.analog[analog_col_2], hz, nperseg=seg, return_onesided=True)
            analog_time = self.analog.iloc[self.analog.index % (hz/2) == 0,].reset_index(drop=True)
            # stft_time = pd.date_range(start=CAN.time.min(), end=CAN.time.max(), freq='0.5S').tolist()
            stft_df_1 = pd.DataFrame(np.abs(Zxx2)).T
            freq_col = pd.Series(f2)
            freq_col[0] = 0.001
            stft_df_1.columns = freq_col
            stft_df_1 = pd.concat([stft_df_1, pd.DataFrame(analog_time['time']).rename(columns={0: 'time'})],axis=1).copy()

            stft_df_1.loc[(stft_df_1['time'].isna() == True), 'time'] = stft_df_1['time'].max() + timedelta(seconds=0.5)
            stft_df_1 = stft_df_1.dropna(axis=0)
            psd_stft_result = pd.DataFrame()

            cut_list = ['cut1','cut2','cut3','cut4','cut5']
            for m in range(1, int(self.brake['brake_count'].max() + 1)):
                brake_count = self.brake[self.brake.brake_count == m].copy()  # CAN
                for cut_name in cut_list:
                    try:
                        brake_count_cut = brake_count[brake_count['cut']==cut_name]
                        analog_test = stft_df_1.loc[(stft_df_1.time < brake_count_cut['time_max'][0]) & (stft_df_1.time > brake_count_cut['time_max'][0])].copy()
                        analog_final = analog_test.reset_index(drop=True)
                        analog_final.drop(['time'], axis=1, inplace=True)
                        stft_concat = pd.DataFrame()

                        Analog_stft = stft(analog_final, analog_col_2, 'stft', hz=hz)
                        stft_concat = pd.concat([stft_concat, Analog_stft], axis=1)
                        stft_concat['brake_count'] = m
                    except ValueError:
                        pass
                    psd_stft_result = psd_stft_result.append(stft_concat)
                    psd_stft_result = psd_stft_result[psd_stft_result[analog_col_2+'_stft_0_sum']!=0].reset_index(drop=True)
            stft_piv_result = pd.concat([stft_piv_result, psd_stft_result], axis=1)
            stft_piv_result = stft_piv_result.loc[:, ~stft_piv_result.columns.duplicated()]
            stft_piv_result = stft_piv_result[stft_piv_result['STR_X_stft_0_sum']!=0]
        return stft_piv_result

# class make_stft:
#     def __init__(self, analog, analog_col, brake, CAN):
#         self.analog = analog # 진동 원본
#         self.analog_col = analog_col # 컬럼 명칭
#         self.brake = brake # 브레이크 회차
#         self.CAN = CAN # CAN time
#
#     def make_stft(self, hz, seg):
#         stft_piv_result = pd.DataFrame()
#
#         for analog_col_2 in self.analog_col:
#             # analog[analog_col_2] = analog[analog_col_2].fillna(method='ffill')
#             self.analog[analog_col_2] = 2 * ((self.analog[analog_col_2] - self.analog[analog_col_2].min()) / (self.analog[analog_col_2].max() - self.analog[analog_col_2].min()) - .5)
#             f2, t2, Zxx2 = signal.stft(self.analog[analog_col_2], hz, nperseg=seg, return_onesided=True)
#             analog_time = self.analog.iloc[self.analog.index % hz/2 == 0,].reset_index(drop=True)
#             # stft_time = pd.date_range(start=CAN.time.min(), end=CAN.time.max(), freq='0.5S').tolist()
#             stft_df_1 = pd.DataFrame(np.abs(Zxx2)).T
#             freq_col = pd.Series(f2)
#             freq_col[0] = 0.001
#             stft_df_1.columns = freq_col
#             stft_df_1 = pd.concat([stft_df_1, pd.DataFrame(analog_time['time']).rename(columns={0: 'time'})],axis=1).copy()
#
#             stft_df_1.loc[(stft_df_1['time'].isna() == True), 'time'] = stft_df_1['time'].max() + timedelta(seconds=0.5)
#             stft_df_1 = stft_df_1.dropna(axis=0)
#             psd_stft_result = pd.DataFrame()
#
#             for m in range(1, int(self.brake['brake_count'].max() + 1)):
#                 try:
#                     brake_count = self.brake[self.brake.brake_count == m].copy()  # CAN
#                     analog_test = stft_df_1.loc[
#                         (stft_df_1.time < brake_count.time.max()) & (stft_df_1.time > brake_count.time.min())].copy()
#                     analog_final = analog_test.reset_index(drop=True)
#                     analog_final.drop(['time'], axis=1, inplace=True)
#                     stft_concat = pd.DataFrame()
#
#                     Analog_stft = stft(analog_final, analog_col_2, 'stft', hz=hz)
#                     stft_concat = pd.concat([stft_concat, Analog_stft], axis=1)
#                     stft_concat['brake_count'] = m
#                 except ValueError:
#                     pass
#                 psd_stft_result = psd_stft_result.append(stft_concat)
#                 psd_stft_result = psd_stft_result[psd_stft_result[analog_col_2+'_stft_0_sum']!=0].reset_index(drop=True)
#
#             stft_piv_result = pd.concat([stft_piv_result, psd_stft_result], axis=1)
#             stft_piv_result = stft_piv_result.loc[:, ~stft_piv_result.columns.duplicated()]
#             stft_piv_result = stft_piv_result[stft_piv_result['STR_X_stft_0_sum']!=0]
#         return stft_piv_result
def stft(data, analog_col, name, hz):

    stft_df = data
    analog_col = analog_col
    f = name
    df = pd.DataFrame()
    hz = hz

    for i in range(0, int((hz/2)+10), 10):
        log_df1 = stft_df.loc[:, (stft_df.columns < i+1) & (stft_df.columns > (i - 10))]
        temp = pd.DataFrame({f'{analog_col}_{f}_{i}_sum': log_df1.sum(axis=1)}).sum(axis=0).apply(np.sqrt)
        temp2 = pd.DataFrame(temp).transpose()
        df = pd.concat([df, temp2], axis=1)
    df = df.fillna(0)
    return df
# Data Set 만들기
def pivot_add_pad(df, index, columns, Y, pad):
    state = pad
    df = pd.pivot_table(df, index=['pad_state', 'file_name', 'brake_count', 'sec'], columns='cut').dropna()
    df.reset_index(inplace=True)
    df.columns = df.columns.droplevel(1) + df.columns.droplevel(0)
    pad_stat = CategoricalDtype(categories=state['pad_state'].unique())
    df[Y] = df['pad_state'].astype(pad_stat).cat.codes
    return df
# 예측 데이터 분할
def split(x):

    # file_split = x['file_name'].unique()
    # train_set, test_set = train_test_split(file_split, test_size=0.2,random_state=123)
    # train_set, test_set = train_test_split(file_split, test_size=0.3, random_state=1004)
    file_split = x[['file_name','pad_state']].drop_duplicates()
    train_set, test_set = train_test_split(file_split, test_size=0.25, random_state=1004,stratify=file_split['pad_state'])

    train_set = train_set['file_name'].unique().tolist()
    test_set = test_set['file_name'].unique().tolist()

    train_final = pd.DataFrame()
    for i in range(0, len(train_set)):
        train_data = x[x['file_name'] == train_set[i]]
        train_final = train_final.append(train_data)

    test_final = pd.DataFrame()
    for k in range(0, len(test_set)):
        test_data = x[x['file_name'] == test_set[k]]
        test_final = test_final.append(test_data)

    train_final_Y = train_final[['Y']]
    test_final_Y = test_final[['Y']]
    final_all = train_final.append(test_final)
    final_all_Y = final_all[['Y']]

    file_train = train_final[['file_name']].copy()
    file_test = test_final[['file_name']].copy()
    file_full = final_all[['file_name']].copy()

    x_train = train_final.drop(['pad_state', 'file_name', 'brake_count', 'Y'], axis=1).copy()  # all
    X_test = test_final.drop(['pad_state', 'file_name', 'brake_count', 'Y'], axis=1).copy()  # all
    x = final_all.drop(['pad_state', 'file_name', 'brake_count', 'Y'], axis=1).copy()  # all

    x_train['sec'] = np.log1p(x_train['sec']).copy()
    X_test['sec'] = np.log1p(X_test['sec']).copy()
    x['sec'] = np.log1p(x['sec']).copy()

    return final_all, final_all_Y, x_train, X_test, train_final_Y, test_final_Y, file_full, file_train, file_test, train_final, test_final

def DL(train):

    full = train
    pd.options.display.max_columns = 999
    Y = full.pad_state
    def pad_conv(Y):
        if Y == '0%':
            return 0
        elif Y == '10%':
            return 1
        elif Y == '20%':
            return 2
        elif Y == '30%':
            return 3
        elif Y == '40%':
            return 4
        elif Y == '50%':
            return 5
        elif Y == '60%':
            return 6
        elif Y == '70%':
            return 7
        elif Y == '80%':
            return 8
        elif Y == '90%':
            return 9
        elif Y == '100%':
            return 10
        elif Y == 'Tangential 1.5mm':
            return 11
        elif Y == 'Tangential 3.0mm':
            return 12
        elif Y == 'Redial 1.0mm':
            return 13
        elif Y == 'Redial 2.0mm':
            return 14

    y2 = []
    for i in full.pad_state:
        y2.append(pad_conv(i))

    # Split into train+val and test
    # X_trainval1, X_test1, y_trainval, y_test = train_test_split(full, y2, test_size=0.15, stratify=y2, random_state=69)
    X_train1, X_val1, y_train, y_val = train_test_split(full, y2, test_size=0.20, stratify=y2, random_state=69)
    # Split train into train-val #
    X_train = X_train1.drop(['pad_state',  'file_name', 'brake_count','Y'], axis=1)
    X_val = X_val1.drop(['pad_state',  'file_name', 'brake_count', 'Y'], axis=1)
    # X_test = X_test1.drop(['pad_state', 'file_name', 'brake_count', 'Y'], axis=1)

    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.fit_transform(X_val)
    # X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    # X_test, y_test = np.array(X_test), np.array(y_test)

    class ClassifierDataset(Dataset):

        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__(self):
            return len(self.X_data)

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    # test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    def get_class_distribution(obj):
        count_dict = {
            "0%": 0,
            "10%": 0,
            "20%": 0,
            "30%": 0,
            "40%": 0,
            "50%": 0,
            "60%": 0,
            "70%": 0,
            "80%": 0,
            "90%": 0,
            "100%": 0,
            "Redial 1.0mm": 0,
            "Redial 2.0mm": 0,
            "Tangential 1.5mm": 0,
            "Tangential 3.0mm": 0
        }

        for i in obj:
            if i == 0:
                count_dict['0%'] += 1
            elif i == 1:
                count_dict['10%'] += 1
            elif i == 2:
                count_dict['20%'] += 1
            elif i == 3:
                count_dict['30%'] += 1
            elif i == 4:
                count_dict['40%'] += 1
            elif i == 5:
                count_dict['50%'] += 1
            elif i == 6:
                count_dict['60%'] += 1
            elif i == 7:
                count_dict['70%'] += 1
            elif i == 8:
                count_dict['80%'] += 1
            elif i == 9:
                count_dict['90%'] += 1
            elif i == 10:
                count_dict['100%'] += 1
            elif i == 11:
                count_dict['Tangential 1.5mm'] += 1
            elif i == 12:
                count_dict['Tangential 3.0mm'] += 1
            elif i == 13:
                count_dict['Redial 1.0mm'] += 1
            elif i == 14:
                count_dict['Redial 2.0mm'] += 1
            else:
                print("Check classes.")

        return count_dict

    class_count = [i for i in get_class_distribution(y2).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)

    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    class_weights_all = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True)

    EPOCHS = 150
    BATCH_SIZE = 5
    LEARNING_RATE = 0.0015
    NUM_FEATURES = X_train.shape[1]
    NUM_CLASSES = len(np.unique(y_train))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=weighted_sampler, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, drop_last=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    class MulticlassClassification(nn.Module):
        def __init__(self, num_feature, num_class):
            super(MulticlassClassification, self).__init__()

            # 1 이슈
            # 기존꺼 냅두고 재학습
            #2 이슈
            # node1의 편향될 경우 node별 안에 있는 피쳐에 대해서 가중치를 초기화 해야된다!
            # 입력층에 대해서 편향될 경우 초기화 함
            # 입력층 1 : 1280
            # 입력층 2 : 640
            node_1 = 1280
            node_2 = 640
            node_3 = 320
            node_4 = 128
            # node_5 = 64

            self.layer_1 = nn.Linear(num_feature, node_1)
            self.layer_2 = nn.Linear(node_1, node_2)
            self.layer_3 = nn.Linear(node_2, node_3)
            self.layer_4 = nn.Linear(node_3, node_4)
            self.layer_out = nn.Linear(node_4, num_class)

            self.loss = nn.ReLU6()
            # self.loss = nn.Sigmoid()

            self.dropout = nn.Dropout(p=0.001)
            self.batchnorm1 = nn.BatchNorm1d(node_1)
            self.batchnorm2 = nn.BatchNorm1d(node_2)
            self.batchnorm3 = nn.BatchNorm1d(node_3)
            self.batchnorm4 = nn.BatchNorm1d(node_4)

        def forward(self, x):
            x = self.layer_1(x)
            x = self.batchnorm1(x)
            x = self.loss(x)
            #x = self.dropout(x)

            x = self.layer_2(x)
            x = self.batchnorm2(x)
            x = self.loss(x)
            #x = self.dropout(x)

            x = self.layer_3(x)
            x = self.batchnorm3(x)
            x = self.loss(x)
            x = self.dropout(x)

            x = self.layer_4(x)
            x = self.batchnorm4(x)
            x = self.loss(x)
            x = self.dropout(x)

            x = self.layer_out(x)

            return x

    def multi_acc(y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)

        acc = torch.round(acc) * 100

        return acc

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)
    print(model)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    print("Begin training.")
    # for e in tqdm(range(1, EPOCHS + 1)):

    for e in (range(1, EPOCHS + 1)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()

        for i, data in enumerate(train_loader):
            X_train_batch, y_train_batch = data
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            for j, data1 in enumerate(val_loader):
                X_val_batch, y_val_batch = data1
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        # VALIDATION
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))

        print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')
    return model
def DL_str(pre_datamart,epochs,rate,list_item):
    list_item =list_item
    EPOCHS = epochs
    test_rate = rate
    full = pre_datamart
    pd.options.display.max_columns = 999
    Y = full.pad_state
    def pad_conv(Y):
        if Y == '0%':
            return 0
        elif Y == '10%':
            return 1
        elif Y == '20%':
            return 2
        elif Y == '30%':
            return 3
        # elif Y == '40%':
        #     return 4
        # elif Y == '50%':
        #     return 5
        elif Y == '60%':
            return 4
        elif Y == '70%':
            return 5
        elif Y == '80%':
            return 6
        # elif Y == '90%':
        #     return 9
        elif Y == '100%':
            return 7
        elif Y == 'Redial 1.0mm':
            return 10
        elif Y == 'Redial 2.0mm':
            return 11
        elif Y == 'Tangential 1.5mm':
            return 8
        elif Y == 'Tangential 3.0mm':
            return 9
    y2 = []
    for i in full.pad_state:
        y2.append(pad_conv(i))

    # Split into train+val and test
    # X_trainval1, X_test1, y_trainval, y_test = train_test_split(full, y2, test_size=0.15, stratify=y2, random_state=69)

    # Split train into train-val #
    X_train1, X_val1, y_train, y_val = train_test_split(full, y2, test_size=test_rate, stratify=y2, random_state=69)
    X_train = X_train1.drop(['pad_state',  'file_name', 'brake_count','Y'], axis=1)
    X_train_len = len(X_train.columns)
    X_val = X_val1.drop(['pad_state',  'file_name', 'brake_count', 'Y'], axis=1)
    # X_test = X_test1.drop(['pad_state', 'file_name', 'brake_count', 'Y'], axis=1)

    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.fit_transform(X_val)
    # X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    # X_test, y_test = np.array(X_test), np.array(y_test)

    class ClassifierDataset(Dataset):

        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__(self):
            return len(self.X_data)

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    # test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    def get_class_distribution(obj):
        count_dict = {
            "0%": 0,
            "10%": 0,
            "20%": 0,
            "30%": 0,
            # "40%": 0,
            # "50%": 0,
            "60%": 0,
            "70%": 0,
            "80%": 0,
            # "90%": 0,
            "100%": 0,
            "Redial 1.0mm": 0,
            "Redial 2.0mm": 0,
            "Tangential 1.5mm": 0,
            "Tangential 3.0mm": 0
        }

        for i in obj:
            if i == 0:
                count_dict['0%'] += 1
            elif i == 1:
                count_dict['10%'] += 1
            elif i == 2:
                count_dict['20%'] += 1
            elif i == 3:
                count_dict['30%'] += 1
            # elif i == 4:
            #     count_dict['40%'] += 1
            # elif i == 5:
            #     count_dict['50%'] += 1
            elif i == 4:
                count_dict['60%'] += 1
            elif i == 5:
                count_dict['70%'] += 1
            elif i == 6:
                count_dict['80%'] += 1
            # elif i == 9:
            #     count_dict['90%'] += 1
            elif i == 7:
                count_dict['100%'] += 1
            elif i == 10:
                count_dict['Redial 1.0mm'] += 1
            elif i == 11:
                count_dict['Redial 2.0mm'] += 1
            elif i == 8:
                count_dict['Tangential 1.5mm'] += 1
            elif i == 9:
                count_dict['Tangential 3.0mm'] += 1
            else:
                print("Check classes.")

        return count_dict

    class_count = [i for i in get_class_distribution(y2).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)

    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    class_weights_all = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True)

    # EPOCHS = 250
    BATCH_SIZE = 5
    LEARNING_RATE = 0.0015
    NUM_FEATURES = X_train_len
    NUM_CLASSES = len(np.unique(y_train))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=weighted_sampler, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, drop_last=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    # test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    class MulticlassClassification(nn.Module):
        def __init__(self, num_feature, num_class):
            super(MulticlassClassification, self).__init__()

            node_1 = 1280
            node_2 = 640
            node_3 = 320
            node_4 = 120
            # node_5 = 64

            self.layer_1 = nn.Linear(num_feature, node_1)
            self.layer_2 = nn.Linear(node_1, node_2)
            self.layer_3 = nn.Linear(node_2, node_3)
            self.layer_4 = nn.Linear(node_3,node_4)
            self.layer_out = nn.Linear(node_4, num_class)

            self.loss = nn.ReLU6()
            # self.loss = nn.Sigmoid()

            self.dropout = nn.Dropout(p=0.001)
            self.batchnorm1 = nn.BatchNorm1d(node_1)
            self.batchnorm2 = nn.BatchNorm1d(node_2)
            self.batchnorm3 = nn.BatchNorm1d(node_3)
            self.batchnorm4 = nn.BatchNorm1d(node_4)

        def forward(self, x):
            x = self.layer_1(x)
            x = self.batchnorm1(x)
            x = self.loss(x)
            #x = self.dropout(x)

            x = self.layer_2(x)
            x = self.batchnorm2(x)
            x = self.loss(x)
            #x = self.dropout(x)

            x = self.layer_3(x)
            x = self.batchnorm3(x)
            x = self.loss(x)
            x = self.dropout(x)

            x = self.layer_4(x)
            x = self.batchnorm4(x)
            x = self.loss(x)
            x = self.dropout(x)

            x = self.layer_out(x)

            return x

    def multi_acc(y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)

        acc = torch.round(acc) * 100

        return acc

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)
    print(model)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    print("Begin training.")
    # for e in tqdm(range(1, EPOCHS + 1)):

    for e in (range(1, EPOCHS + 1)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()

        for i, data in enumerate(train_loader):
            X_train_batch, y_train_batch = data
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            for j, data1 in enumerate(val_loader):
                X_val_batch, y_val_batch = data1
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        # VALIDATION
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))

        print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')
    return model

class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

def proba(train_x, train_y, test_x, file_test, dup, train_dup, train, test_y):

    xgb_model = xgboost.XGBClassifier(scale_pos_weight=1, silent=None, booster='gbtree', learning_rate=0.05,
                                      colsample_bytree=0.7, subsample=0.5, objective='binary:logistic',#class_weight='balanced',
                                      n_estimators=350, max_depth=12,random_state=10,
                                      gamma=10, seed=777, nfold=10)

    lgb_model = lgb.LGBMClassifier(objective='multiclass', n_jobs=5, is_unbalance=True, num_threads=8, two_round=True, class_weight='balanced',
                                   bagging_fraction=0.9, bagging_freq=5, boosting_type='gbdt', feature_fraction=0.9, learning_rate=0.05,
                                   min_child_samples=10, min_child_weight=5, min_data_in_leaf=20, min_split_gain=0.0, n_estimators=350, num_leaves=100, reg_alpha=0.0,
                                   reg_lambda=0.0, subsample=1.0,random_state=10)

    xg_pred = xgb_model.fit(train_x, train_y).predict_proba(test_x)
    light_pred = lgb_model.fit(train_x, train_y).predict_proba(test_x)

    # DL
    if len(train.Y.unique()) == 15:
        dl_model = DL(train)
    else:
        dl_model = DL_str(train)

    num = len(test_y.Y.unique())
    test_x_dl, test_y_dl = np.array(test_x), np.array(test_y)

    test_dataset = ClassifierDataset(torch.from_numpy(test_x_dl).float(), torch.from_numpy(test_y_dl).long())
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    # y_pred_list = []
    dl_pred = np.empty((0, num), dtype='float32')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        dl_model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = dl_model(X_batch)
            y_pred_softmax = torch.softmax(y_test_pred, dim=1)
            dl_pred = np.append(dl_pred, y_pred_softmax.numpy(), axis=0)

    hr_pred = (xg_pred + light_pred + dl_pred)/3
    hr_pred = (xg_pred + light_pred)/2
    hr_pred = light_pred
    # hr_pred = dl_pred

    pred_result = pd.DataFrame(pd.DataFrame(hr_pred).idxmax(axis=1)).rename(columns={0:'prediction'})
    pred_result['count'] = '1'
    # pred_result_file = pred_result
    # pred_result_file['file_name'] = '2788_201229'
    pred_result_file = pd.concat([pred_result, file_test.reset_index(drop=True)], axis=1)
    proba_result = pred_result_file.groupby(['file_name','prediction'], as_index=False).count()
    result_2 = proba_result.groupby(['file_name'], as_index=False)['count'].max()
    result_prediction = pd.merge(result_2, proba_result, how='left', on=['file_name','count']).drop(['count'],axis=1).drop_duplicates('file_name',keep='last').merge(dup,how='left',on=['file_name'])

    # proba_result = pd.concat([pd.DataFrame(hr_pred), file_test.reset_index(drop=True)], axis=1)
    # result = proba_result.groupby(['file_name'], as_index=False)
    # result = result.mean()
    # result_file = result[['file_name']]
    # result.drop(['file_name'], axis=1, inplace=True)
    # result_max = pd.DataFrame(result.idxmax(axis=1))
    # result_max.columns = ['prediction']
    # result_max = pd.concat([result_file, result_max], axis=1).copy()
    count = train[['pad_state', 'Y']]
    Y_name = count.drop_duplicates().sort_values(by='Y')

    zxx_report_proba = pd.DataFrame(classification_report(result_prediction.Y, result_prediction.prediction.astype(int), output_dict=True,target_names=Y_name.pad_state)).T
    matrix = pd.DataFrame(confusion_matrix(result_prediction.Y,result_prediction.prediction.astype(int), labels=np.sort(result_prediction.Y.unique())),columns=Y_name.pad_state,index=Y_name.pad_state)
    data_count = pd.concat([pd.DataFrame(train_dup.Y.value_counts(sort=False)).rename(columns={'Y': 'Y_train'}),
                            pd.DataFrame(dup.Y.value_counts(sort=False).astype(str)).rename(columns={'Y': 'Y_test'})],axis=1).fillna(0)

    return zxx_report_proba, result_prediction, result_prediction, data_count, matrix

    pred_result = pd.DataFrame(pd.DataFrame(hr_pred).idxmax(axis=1)).rename(columns={0:'prediction'})
    pred_result['count'] = '1'
    # pred_result_file = pred_result
    # pred_result_file['file_name'] = '2788_201229'
    pred_result_file = pd.concat([pred_result, file_all_full.reset_index(drop=True)], axis=1)
    proba_result = pred_result_file.groupby(['file_name','prediction'], as_index=False).count()
    result_2 = proba_result.groupby(['file_name'], as_index=False)['count'].max()
    result_prediction = pd.merge(result_2, proba_result, how='left', on=['file_name','count']).drop(['count'],axis=1).drop_duplicates('file_name',keep='last').merge(dup,how='left',on=['file_name'])


