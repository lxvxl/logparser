from LCS import LCS
from DAGDrain import LogSer
class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, st=0.4, 
                 maxChild=100, rex=[], keep_para=True, tau = 1.0):
        self.parser = LogSer.LogParser(log_format, 
                         indir=indir, 
                         outdir=outdir,  
                         depth=depth, 
                         st=st, 
                         rex=rex, 
                         tau=tau, 
                         postProcessFunc = LCS)        


    def parse(self, logName):
        self.parser.parse(logName)