#!/usr/bin/env python
import os

import pandas as pd

from LogSer import LogSer
from LogSer.Jaccard import Jaccard
from utils import evaluator
from utils import LOSS_evaluate
from datetime import datetime


benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'tau': 0.7,
        'depth': 4,
        'replaceD': {
            'blk_-?\d+':'(BLK)',
            ':':' : '
        }
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.65,
        'tau': 0.9,
        'depth': 4,      
        'replaceD': {
            ':':' : ',
            #'$':' ',
            '_':' ',
            '@':' '
        }
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.5,
        'tau': 0.8,
        'depth': 4,
        'replaceD': {
            '_':' '
        }
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'st': 0.6,
        'tau': 0.9,
        'depth': 4,
        'replaceD': {
            ':':' : ',
            '$':' '
        }        
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
        'st': 0.5,
        'tau': 0.8,
        'depth': 4,        
        'replaceD': {
            r'0x(\w){8}':'(ADDR)',
            r'core\.\d+':'(CORE)',
            ',':' ',
            ':':' : ',
            '=':' = '
        }
        },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
        'st': 0.6,
        'tau': 0.7,
        'depth': 4, 
        'replaceD': {
            r'HWID=\d+':'(HWID)',
            ':':' : ',
            '=':' = '
        }
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'tau': 0.4,
        'depth': 4,       
        'replaceD': {
            r'\b([0-9a-zA-Z]){14}\b':'(PID)',
            '#':' ',
            '=':' = '
        }
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
        'st': 0.8,
        'tau': 0.7,
        'depth': 5,
        'replaceD': {
            r'0x(\w){8}':'(ADDR)',
            ':':' : ',
            '@':' '
        }
        },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
        'st': 0.4,
        'tau': 0.99,
        'depth': 6,
        'replaceD': {
            #':':' : ',
            ';':' ',
            ',':' ',
            '_':' ',
            '\(':' ( ',
            '\)':' ) ',
            '=': ' = '
        }       
        },

    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'st': 0.6,
        'tau': 0.8,
        'depth': 4,
        'replaceD': {
            ':':' ',
            r'\|':' ',
            ',':' ',
            '@':' ',
            '=':' = '
        }    
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
        'st': 0.65,
        'tau': 1,
        'depth': 4,
        'replaceD': {
            '=': ' = ',
            #'##':' ',
            ':': ' : '
        }
        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'tau': 0.7,
        'depth': 4,
        'replaceD': {
            r'child \d+':'CHILD'
        }        
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        'st': 0.5,
        'tau': 0.7,
        'depth': 4,
        'replaceD': {
        }
        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.6,
        'tau': 1,
        'depth': 6,
        'replaceD': {
            ':':' : ',
            '_':' ',
            '=':' = ',
            '\[': '[ ',
            '\]': ' ]'
        }   
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
        'st': 0.65,
        'tau': 0.8,
        'depth': 6,
        'replaceD': {
            r'[0-9a-z]{8}(-[0-9a-z]{4}){3}-[0-9a-z]{12}':'(INST)',
            #',':' ',
            ':':' ',
            #'=':' = '
        }   
        },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.6,
        'tau': 0.8,
        'depth': 6,
        'replaceD': {
            r'ARPT: \d+.\d+':'(APRT)',
            '::' : ' :: ',
            '_':' '
        }     
        },
}

input_dir = r'E:\日志解析-大修\logparser\data\loghub_2k_corrected'
output_dir = 'LogSer_results_temp'
bechmark_result = []
dataset = 'HPC'
if __name__ == '__main__':
    accuracy_data = []  # 用于存储准确度数据的列表

    for ht in [i/10 for i in range(4, 11)]:  # 以步长为0.1暴力尝试ht参数，范围为0.4-1
        for jt in [i/10 for i in range(5, 11)]:  # 以步长为0.1暴力尝试jt参数，范围为0.4-1
            parser = LogSer.LogParser(log_format=benchmark_settings[dataset]['log_format'], 
                            indir=os.path.join(input_dir, os.path.dirname(benchmark_settings[dataset]['log_file'])), 
                            outdir=output_dir,  
                            depth=benchmark_settings[dataset]['depth'], 
                            ht=ht,  # 使用当前的ht值
                            rex=benchmark_settings[dataset]['regex'], 
                            jt=jt,  # 使用当前的jt值
                            postProcessFunc=Jaccard, 
                            replaceD=benchmark_settings[dataset]['replaceD'])
            parser.parse(os.path.basename(benchmark_settings[dataset]['log_file']))
            Precision, Recall, F1_measure, accuracy = evaluator.evaluate(
                        groundtruth=os.path.join(input_dir, benchmark_settings[dataset]['log_file'] + '_structured_corrected.csv'),
                        parsedresult=os.path.join(output_dir, dataset + '_2k.log' + '_structured.csv')
                        )    
            accuracy_data.append({'ht': ht, 'jt': jt, 'accuracy': accuracy})  # 将准确度数据存储到列表中

    # 将准确度数据整理成表格
    df = pd.DataFrame(accuracy_data)
    df.sort_values(by='accuracy', inplace=True)
    print(df)
    print(dataset)