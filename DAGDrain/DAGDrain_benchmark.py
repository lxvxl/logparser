#!/usr/bin/env python
import os

import pandas as pd

from DAGDrain import DAGDrain
from utils import evaluator
from utils import LOSS_evaluate
from datetime import datetime


benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'tau': 1,
        'depth': 4,
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'tau': 1,
        'depth': 4,      
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.5,
        'tau': 1,
        'depth': 4,
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'st': 0.6,
        'tau': 1,
        'depth': 4,
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
        'st': 0.4,
        'tau': 1,
        'depth': 4,        
        },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
        'st': 0.6,
        'tau': 1,
        'depth': 4, 
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'tau': 1,
        'depth': 4, 
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
        'st': 0.6,
        'tau': 1,
        'depth': 5,
        },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
        'st': 0.4,
        'tau': 1,
        'depth': 6,
        },

    'Andriod': {
        'log_file': 'Andriod/Andriod_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'st': 0.5,
        'tau': 1,
        'depth': 6,
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
        'st': 0.25,
        'tau': 1,
        'depth': 4,
        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'tau': 1,
        'depth': 4, 
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        'st': 0.5,
        'tau': 0.7,
        'depth': 4,
        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.6,
        'tau': 1,
        'depth': 5, 
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
        'st': 0.5,
        'tau': 1,
        'depth': 6,
        },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.6,
        'tau': 1,
        'depth': 6,
        },
}

input_dir = 'logs'
output_dir = 'DAGDrain_results'
bechmark_result = []
if __name__ == '__main__':
    for dataset in benchmark_settings.keys():
        print('\n=== Evaluation on %s ==='%dataset)
        parser = DAGDrain.LogParser(log_format=benchmark_settings[dataset]['log_format'], 
                         indir=os.path.join(input_dir, os.path.dirname(benchmark_settings[dataset]['log_file'])), 
                         outdir=output_dir,  
                         depth=benchmark_settings[dataset]['depth'], 
                         st=benchmark_settings[dataset]['st'], 
                         rex=benchmark_settings[dataset]['regex'], 
                         tau=benchmark_settings[dataset]['tau'])
        start_time = datetime.now()
        parser.parse(os.path.basename(benchmark_settings[dataset]['log_file']))
        parse_time = (datetime.now() - start_time).total_seconds()
        Precision, Recall, F1_measure, accuracy = evaluator.evaluate( groundtruth=os.path.join(input_dir, benchmark_settings[dataset]['log_file'] + '_structured.csv'),
                    parsedresult=os.path.join(output_dir, dataset + '_2k.log' + '_structured.csv')
                    )    
        QL, LL = LOSS_evaluate.loss(pd.read_csv(os.path.join(output_dir, dataset + '_2k.log' + '_templates.csv')))
        bechmark_result.append([dataset, Precision, Recall, F1_measure, accuracy, parse_time, QL, LL, (QL+LL)])
        print('')

    print('\n=== Overall evaluation results ===')
    df_result = pd.DataFrame(bechmark_result, columns=['Dataset', 'Precision', 'Recall', 'F1_measure', 'Accuracy', 'Time', 'QL', 'LL', 'LOSS'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    df_result.to_csv('DAG_benchmark_result.csv')