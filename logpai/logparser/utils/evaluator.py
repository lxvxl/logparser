#coding=utf-8
"""
Description : This file implements the function to evaluation accuracy of log parsing
Author      : LogPAI team
License     : MIT
"""

import math
import sys
import pandas as pd
from collections import defaultdict
import scipy.special


def evaluate(groundtruth, parsedresult):
    """ Evaluation function to benchmark log parsing accuracy
    
    Arguments
    ---------
        groundtruth : str
            file path of groundtruth structured csv file 
        parsedresult : str
            file path of parsed structured csv file

    Returns
    -------
        f_measure : float
        accuracy : float
    """ 
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult)
    # Remove invalid groundtruth event Ids
    non_empty_log_ids = df_groundtruth[~df_groundtruth['EventId'].isnull()].index
    df_groundtruth = df_groundtruth.loc[non_empty_log_ids]
    df_parsedlog = df_parsedlog.loc[non_empty_log_ids]
    (precision, recall, f_measure, accuracy) = get_accuracy(df_groundtruth['EventId'], df_parsedlog['EventId'])
    print('Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Parsing_Accuracy: %.4f'%(precision, recall, f_measure, accuracy))
    return precision,recall,f_measure, accuracy

def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    """ Compute accuracy metrics between log parsing results and ground truth
    
    Arguments
    ---------
        series_groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        series_parsedlog : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """
    series_groundtruth_valuecounts = series_groundtruth.value_counts()#记录模式编号及出现次数的对应关系
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0 # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:#对于每一种模式
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index#模式名为parsed_eventId的日志序号
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()#模式名为parsed_eventId的日志序号在标准聚类中被聚成的类别
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy

def loss(df_t):
    QL = 0
    for idx, line in df_t.iterrows():
        template = line['EventTemplate']
        template_seq = str.split(template)

        parameter_count = 0

        #去除单独的标点符号
        for i in range(len(template_seq)-1 , -1, -1):
            token = template_seq[i]
            if len(token) == 1 and token[0] in ':,=$@#;_|':
                del template_seq[i]

        for token in template_seq:
            if str.find(token, '<*>') >= 0:
                parameter_count += 1

        QL += (1.0 * parameter_count / len(template_seq)) ** 2
        #print('%.3f'%(1.0 * parameter_count / len(template_seq)) ** 2, parameter_count, template)
        
    print('QL=%.3f'%QL)
    LL = math.log(len(df_t))**1.5
    print('LL=%.3f'%LL)
    print('LOSS=%.3f'%(QL + LL))
    return QL, LL







