# -*- coding: UTF-8 -*-
"""
Description : This file implements the Drain algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""
import re
import os
import sys
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
import math

import io


class Logcluster:
    def __init__(self, logTemplate=[], logIDL=None):
        self.logTemplate = logTemplate
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL

class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        if childD is None:
            childD = dict()
        self.childD = childD
        self.depth = depth
        self.digitOrtoken = digitOrtoken

template_regex_map = {}
class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, ht=0.4, 
                 maxChild=100, rex=[], keep_para=True, jt = 1.0, postProcessFunc = None, replaceD = {}):
        self.path = indir
        self.depth = depth - 2
        self.ht = ht
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para
        self.jt = jt
        self.mergeFunc = postProcessFunc
        self.replaceD = replaceD
    
    def level(self, s:str):
        '''measure the posibility that the token is a parameter'''
        l = 0
        for char in s:
            if char.isdigit():
                return 2
            elif char in '#^$\'*+,/<=>@_)|~':
                l = 1
        return l

    def treeSearch(self, rn, seq):
        '''
            若匹配到，则返回对应的LogCluster。否则返回None
        '''
        retLogClust = None

        seqLen = len(seq)
        if seqLen not in rn.childD:
            return retLogClust

        parentn = rn.childD[seqLen] #parent node
        currentDepth = 1
        '''
            若子节点中有相同的token，则优先匹配相同的token。
            若没有相同的token但是有<*>,则匹配<*>
        '''
        i = 0
        j = len(seq) - 1
        while i <= j:
            if currentDepth >= self.depth:
                break

            if self.level(seq[i]) > self.level(seq[j]):
                token = seq[j]
                j = j - 1
            else:
                token = seq[i]
                i = i + 1

            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else:
                return retLogClust #return none
            currentDepth += 1

        logClustL = parentn.childD
        retLogClust = self.fastMatch(logClustL, seq)
        return retLogClust

    def addSeqToPrefixTree(self, rn:Node, logClust:Logcluster):

        #第一层为长度叶节点
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        i = 0
        j = seqLen - 1
        while True:
            if self.level(logClust.logTemplate[i]) > self.level(logClust.logTemplate[j]):
                token = logClust.logTemplate[j]
                j = j - 1
            else:
                token = logClust.logTemplate[i]
                i = i + 1
            #若达到叶节点或模式的尾部
            if currentDepth >= self.depth or currentDepth >= seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            #若当前树中没有token
            if token not in parentn.childD:
                #若token中没有数字
                if self.level(token) < 2:
                    #若能匹配为参数
                    if '<*>' in parentn.childD:
                        #若子节点数量小于最大限制，则创建新节点
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        #若子节点数量达到最大限制，则匹配为参数
                        else:
                            parentn = parentn.childD['<*>']
                    #若不能匹配为参数
                    else:
                        
                        if len(parentn.childD)+1 < self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD)+1 == self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']

                else:
                    if '<*>' not in parentn.childD:
                        newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD['<*>']

            #If the token is matched
            else:
                parentn = parentn.childD[token]

            currentDepth += 1

    #seq1 is template
    def seqDist(self, seq1, seq2):
        '''
            判定seq2是否与seq1匹配
            parameter：
                seq1: 模板
                seq2: 日志文本的单词序列
            returns：
                retval：非参数数量所占的比例
                numOfPar：参数的数量
        '''
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue
            if token1 == token2:
                simTokens += 1 

        retVal = float(simTokens) / len(seq1)

        return retVal, numOfPar


    def fastMatch(self, logClustL:list, seq:list):
        '''
            在logClustL中搜索匹配seq的模板，若未搜索到，则返回None，否则返回对应的LogCluster对象
        '''
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim>maxSim or (curSim==maxSim and curNumOfPara>maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if maxSim >= self.ht:
            retLogClust = maxClust  

        return retLogClust

    def getTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append('<*>')

            i += 1

        return retVal

    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        for logClust in logClustL:
            #print(logClust.logTemplate)
            template_str = ' '.join(logClust.logTemplate)
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])
        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates

        print('开始提取参数')
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
            self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)
            #更节省内存的输出方法
            '''with open(os.path.join(self.savePath, self.logName + '_structured.csv'), 'w') as f:
                f.write(",".join(self.df_log.columns) + ",ParameterList\n")
                for index, row in self.df_log.iterrows():
                    processed_parameter = self.get_parameter_list(row)  # 假设这里是处理每一行数据的方法
                    f.write(row.to_csv() + f",\"{processed_parameter}\"\n")
                    if index % 100000 == 0:
                        print(f'已提取{index}条日志的参数')'''
        else:
            self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)
        print('提取参数完毕')

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.sort_values(by='Occurrences', inplace=True)
        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False, columns=["EventId", "EventTemplate", "Occurrences"])

    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        print('HammingThreshold=%.2f'%self.ht)
        print('JaccardThreshold=%.2f'%self.jt)
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node()
        logCluL = []

        self.load_data()
        print('数据已加载')

        count = 0
        for idx, line in self.df_log.iterrows():
            #print(line['LineId'], line['Content'])
            logID = line['LineId']
            logmessageL = self.preprocess(line['Content']).strip().split()
            matchCluster = self.treeSearch(rootNode, logmessageL)

            #Match no existing log cluster
            if matchCluster is None:
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)

            #Add the new log message to the existing cluster
            else:
                newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                matchCluster.logIDL.append(logID)
                if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate): 
                    matchCluster.logTemplate = newTemplate

            count += 1
            if count % 10000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        print("开始模式整合，共有", len(logCluL), "种模式")
        if not self.mergeFunc == None:
            logCluL = mergeClusters(logCluL, self.jt, self.mergeFunc)
        print("模式整合完毕，共有", len(logCluL), "种模式")


        self.outputResult(logCluL)
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))


    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def get_parameter_list(self, row):
        template = template_regex = row["EventTemplate"]
        if "<*>" not in template_regex: return []
        if template in template_regex_map.keys(): 
            template_regex = template_regex_map[template]
        else:
            template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
            template_regex = re.sub(r'\\ +', ' ?', template_regex)
            template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
            template_regex_map[template] = template_regex
        parameter_list = re.findall(template_regex, re.sub(' +', ' ', row["Content"]))
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return list(filter(None, parameter_list))
    
    def log_to_dataframe(self, log_file, regex, headers:list, logformat):
        """ Function to transform log file to dataframe 
        """
        log_messages = []
        linecount = 0
        print('开始加载日志')
        content_index = headers.index('Content')
        with open(log_file, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    message[content_index] = self.perprocess1(message[content_index])
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
                if linecount % 100000 == 0:
                    print("已加载", linecount, '条日志')
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        print("已加载", linecount, '条日志,加载完毕')
        return logdf

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex
    
    def perprocess1(self, content):
        for key, value in self.replaceD.items():
            content = re.sub(key, value, content)
        return re.sub(' +',' ',content)    

def mergeClusters(logCluL:list, threshold:float, mergeFunc):
    templateCluL = []
    for logCluster in logCluL:
        isMerged = False
        for templateCluster in templateCluL:
            if templateCluster.tryMerge(logCluster, threshold):
                isMerged = True
                break
        if not isMerged:
            templateCluL.append(TemplateCluster(logCluster, mergeFunc))
    newLogClulL = []
    for templateCluster in templateCluL:
        newLogClulL.append(templateCluster.generateLogCluster())
    return newLogClulL

class TemplateCluster:
    def __init__(self, cluster: Logcluster, mergeFunc):
        self.logTemplate = cluster.logTemplate
        self.logClusterL = [cluster]
        self.lenL = [len(cluster.logTemplate)]
        self.mergeFunc = mergeFunc

    def tryMerge(self, newCluster: Logcluster, threshold: float)->bool:
        '''
            尝试将新模式合并到该模式聚类中,若成功则更新该模式聚类的标准模式。
        '''
        
        check, newTemplate = self.mergeFunc(newCluster.logTemplate, self.logTemplate, threshold)
        if not check:
            return False
        self.logTemplate = newTemplate
        self.logClusterL.append(newCluster)
        #self.lenL.append(len(newCluster.logTemplate))
        return True
        
    def generateLogCluster(self)->Logcluster:
        '''
            根据该日志模式的集合生成日志模式
        '''
        logIDL = []
        for logCluster in self.logClusterL:
            logIDL.extend(logCluster.logIDL)
        return Logcluster(self.logTemplate, logIDL)