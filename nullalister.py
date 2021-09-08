#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script is used to get info on non-detections for the EHT SgrA* observations
in 2017. The first approach to try is to inspect alist files that were produced 
by the HOPS pipeline. 
There are tree stages to try as pointed out by Lindy: 
    You could check a few upstream stages in the pipeline,
    step 2: this has all the scans that are not explicitly flagged by telescope/data issues. 
    however, it has a wide fringe search window for every baseline independently, 
    so you will get false fringes at S/N ~ 6 or so.

    /data/2017-april/ce/er6/hops-lo/2.+pcal/data/alist.v6

    step 4: this is ad-hoc phase corrected and delay referenced, so S/N will be 
    more accurate than step 2. however, it only contains station data which overlaps 
    in time and has a detection to the reference station, so some data will be missing. 
    it will still contain false fringes at S/N ~ 6.

    /data/2017-april/ce/er6/hops-lo/4.+delays/data/alist.v6

    step 5: this is from the closed (global) solution. it is what I pointed you 
    to before. it only contains fringe solutions with known delay and delay-rate 
    via the global solution. since there is no more fringe search window for weak 
    fringes, you can trust the S/N all the way down to zero (aside from thermal noise). 
    probably it is a good way to begin inspection of the null.

    /data/2017-april/ce/er6/hops-lo/5.+close/data/alist.v6


Created on Wed Jun 23 11:38:18 2021



@author: mlisakov
"""
__author__ = 'Mikhail Lisakov'
__credits__ = ['Mikhail Lisakov']
__maintainer__ = 'Mikhail Lisakov'
__email__ = 'mlisakov@mpifr-bonn.mpg.de'
__copyright__ = 'Copyright 2021, EHT Non-detections'
__license__ = 'GNU GPL v3.0'
__version__ = '0.5'
__status__ = 'Dev'

import logging
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_logger(obj=None, dest=['stderr'], levels=['INFO']):
    """Creates a logger instance to write log information to STDERR.
    
    Args:
        obj: caller object or function with a __name__ parameter
        
        dest (list, default is ['stderr']): a destination where to write 
        logging information. Possible values: 'stderr', 'filename'. Every 
        non-'stderr' string is treated as a filename
            
        levels (list, default is ['INFO']): logging levels. One value per dest 
            is allowed. A single value will be applied to all dest. 
    
    Returns:
        logging logger object
        
    """
    
    # initialize a named logger    
    try:
        logger = logging.getLogger(obj.__name__)
        try:
            logger = logging.getLogger(obj)  # treat obj as a string
        except:
            pass
    except:
        logger = logging.getLogger('')

    # solve the issue of multiple handlers appearing unexpectedly
    if (logger.hasHandlers()):
        logger.handlers.clear()

    # match dimensions of dest and level
    if isinstance(dest, list):
        num_dest = len(dest)
        
    else:
        num_dest = 1
        dest = [dest]
    
    if isinstance(levels, list):
        num_levels = len(levels)
    else:
        num_levels = 1
        levels = [levels]
    
    if num_dest > num_levels:
        for i in np.arange(0, num_dest - num_levels):
            levels.append(levels[-1])
    
    if num_dest < num_levels:
        levels = levels[:len(num_dest)]


    # add all desired destinations with proper levels
    for i, d in enumerate(dest):
        
        if d.upper() in ['STDERR', 'ERR']:
            handler = logging.StreamHandler()   # stderr
        else:
            handler = logging.FileHandler(d, mode='w') # file
            
        
        level = levels[i]
        # set logging level
        if level.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            original_level = level
            level = 'INFO'
            logger.setLevel(level)
            logger.error('Logging level was not properly set ({}). Fall back to INFO.'.format(original_level))
        else:
            logger.setLevel(level.upper())
        
        logger.addHandler(handler)
    
    return logger


class alist():
    """Class to handle alist objects. Besides obviously the data, there should 
    be some essential additional parameters set like: SNR cutoff, stage number
    in the pipeline etc. 
    The data are stored as a pandas dataframe.
    The logic is still being developed. 
    
    Args:
        logger: A logging object. Default is None
    
    
    Attributes:
        stage (int): pipeline stage of the data. See detainled description in 
        the script source header. In general: 2 - after baseline based fringe fitting;
        4 - after applying delay corrections; 5 - after global fringe fit.
        snr_cutoff (float): SNR cutoff. Any fringes with SNR below this value 
        are considered as non-detections. Should be different for different stages. 
        For stage 2 - around 6-7; for stage 5 - around 0 (above thermal noise). 
        Some playing with values is expected later. 
        
    """


    def __init__(self, file=None, logger=None):
        """Initialize an alist object. Could be read from a file directly during init. 
        """     
        if logger is not None:
            self.logger = logger
            logger.debug("Logger object was passed in the arguments")
        else:
            self.logger = create_logger()
            logger.debug("Logger object was created in the class __init__")

            
        self.stage = 2
        self.snr_cutoff = 7
        self.data = pd.DataFrame()
        if file is not None:
            self.data = self.read(file)
            
   
            
    def read(self, file):
        """Read data from an alist file and form an alist object. 
        
        Args:
            file (string): a filename to read data from
        
        Returns: 
            an alist object
        """
        # logger = create_logger(self)
        self.logger.info('Reading data from file {}'.format(file))
        self.data = pd.read_csv(file, sep='\s+', comment='*', header=None, engine='python')
        # rename only useful columns
        self.data.rename({7:'exp', 8:'scan', 13:'source', 14:'baseline', 
                          17:'pol', 20:'snr', 32:'u', 33:'v', 36:'freq',
                          10: 'year', 11:'doytime'},
                         axis='columns', inplace=True)
        
        return self.data


    def remove_autocorr(self):
        """Delete autocorrelations from the data, e.g. when baseline is XX.
        
        Returns:
            dataframe with autocorr lines deleted
        """
        
        logger.info("before dropping autocorrs there were {} rows".format(self.data.index.size))
        self.data.drop(self.data.loc[self.data.baseline.str[0] == self.data.baseline.str[1]].index, inplace=True)
        logger.info("after dropping autocorrs there were {} rows".format(self.data.index.size))
        return self.data
    
    
    def select_source(self, sources=[None]):
        """Select only specified sources in the data
        
        Args:
            sources: array of source names or a single source name as a string
        
        Returns:
            data with only specified sources
        """
        if isinstance(sources, list):
            pass
        else:
            sources = [sources]
        
        self.data = self.data.loc[self.data.source.isin(sources)]
        
        return self.data
     
    def select_telescopes(self, telescopes=[None]):
        """Select only baselines with specified telescopes included. Telescope 
        codes are given according to the fourfit codes. 
        
        Args:
            telescopes: array of telescopes' one-letter fourfit codes
            
        Returns:
            data with only specified telescopes included
        """
        if isinstance(telescopes, list):
            pass
        else:
            telescopes = [telescopes]
        
        self.data = self.data.loc[(self.data.baseline.str[0].isin(telescopes)) | (self.data.baseline.str[1].isin(telescopes))]
        
        return self.data
        
    def select_baselines(self, baselines=[None]):
        """Select specified baselines only. Telescope codes in the baselines
        are given according to those used in fourfit. Baseline are failproof 
        to the different order of stations, i.e. XY == YX. 
        
        Args:
            baselines: array of two-letter baselines
           
        Returns:
            data with only specified baselines
        """
        if isinstance(baselines, list):
            pass
        else:
            baselines = [baselines]
        
        reversed_baselines = []
        
        for b in baselines:
            reversed_baselines.append(b[::-1])
        
        baselines = baselines + reversed_baselines
        
        print(baselines)
        
        self.data = self.data.loc[self.data.baseline.isin(baselines)]
        
        
        return self.data
             
         
        
     
    def drop_unused_columns(self, columns=None):
        """Drop unused columns. 
        Currently, will leave [exp, scan, source, baseline, u, v, freq]
        
        Args:
            columns: list of columns to keep. If not set explicitly,
            use the default list
            
        Returns:
            data with only specified columns
        """
        
        # run add_columns if it was not done before
        if 'time' not in self.data.columns:
            self.add_columns()
        
        
        if columns is None:
            columns = ['exp', 'scan', 'source', 'baseline', 'u', 'v', 'snr', 'pol',
                       'freq', 'uvdist', 'time']
        
        self.data = self.data.loc[:, columns]
        
        return self.data
        
    
    def add_columns(self):
        """Add columns such as [datetime, uv_dist]  to data.
        
        Returns:
            data with added columns
        """
        
        # time
        self.data.loc[:, 'time'] = self.data.loc[:, 'year'].astype(str) + self.data.loc[:, 'doytime']
        self.data.loc[:, 'time'] = pd.to_datetime(self.data.loc[:, 'time'], format="%Y%j-%H%M%S")
        
        #uv distance
        self.data.loc[:, 'uvdist'] = np.sqrt(np.power(self.data.loc[:, 'u'], 2) + np.power(self.data.loc[:, 'v'], 2))
        
        return self.data
    
    
    def radplot(self, uv_range=None, highlight_low_snr=False):
        """Plot snr vs uv distance.
        
        Args:
            uv_range (list): a list of two values to limit plotted uv distances
            [uv_min:uv_max]
                
                """
        
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.data.uvdist, self.data.snr, 'o', label='{}'.format(self.data.source.unique()))
        if highlight_low_snr is True:
            ax.plot(self.data.loc[self.data.snr < self.snr_cutoff, 'uvdist'],
                    self.data.loc[self.data.snr < self.snr_cutoff, 'snr'], 
                    'v', label='SNR < {}'.format(self.snr_cutoff))
        
        ax.set_ylabel('SNR')
        ax.set_xlabel(r'UV distance (M$\lambda$)')
        ax.set_yscale('log')
        # ax.set_xscale('log')
        if uv_range is not None:
            ax.set_xlim(uv_range)
        
        fig.legend()
        
        return
    
    
    def radplot_low_snr_fraction(self, bin_width=500, label=None, title=None, hardcopy=None,
                                 uv_range=None):
        """Plot ratio of number of low snr points to total number of point in each bin
        of a certain width in uv distance.
        
        Args:
            bin_width: a width of the uvdist bins in megalambdas
        """
        self.data.loc[:, 'uv_binned'], _ = pd.cut(self.data.uvdist.values, bins = np.int(self.data.uvdist.max() / bin_width), retbins=True)
        
        all_snr = pd.pivot_table(self.data, values='snr',  index=['uv_binned'], aggfunc='count')
        low_snr = pd.pivot_table(self.data.loc[self.data.snr < self.snr_cutoff, :], values='snr',  index=['uv_binned'], aggfunc='count')
        snr_ratio = all_snr.join(low_snr, on='uv_binned', how='left', lsuffix='_all', rsuffix='_low')
        
        snr_ratio.loc[:, 'ratio'] = snr_ratio.loc[:, 'snr_low'] / snr_ratio.loc[:, 'snr_all']
        snr_ratio.loc[:, 'mid'] = snr_ratio.index.categories.mid
        fig,ax = plt.subplots(1, 1)
        if uv_range is not None:
            ax.set_xlim(uv_range)

        
        if label is not None:
            # ax.plot(snr_ratio.loc[:, 'mid'], snr_ratio.loc[:, 'ratio'], '-*', label=label)
            ax2 = ax.twinx()
            # ax2.plot(self.data.uvdist, self.data.snr, 'o', label='{}'.format(self.data.source.unique()))
            if uv_range is not None:
                ax2.plot(self.data.loc[self.data.uvdist > uv_range[0], 'uvdist'], 
                         self.data.loc[self.data.uvdist > uv_range[0], 'snr'], 'o', 
                         label='{}'.format(self.data.source.unique()))
            
            ax.bar(snr_ratio.loc[:, 'mid'], snr_ratio.loc[:, 'ratio'], width=bin_width, color='red', alpha=0.5, label=label)
            ax.legend()
        else:
            # ax.plot(snr_ratio.loc[:, 'mid'], snr_ratio.loc[:, 'ratio'], '-*')
            ax.bar(snr_ratio.loc[:, 'mid'], snr_ratio.loc[:, 'ratio'], width=bin_width, color='red', alpha=0.5)
            
        ax.set_xlabel(r'UV distance (M$\lambda$)')
        ax.set_ylabel('Ratio of low SNR fringes')
            

        if title is not None:
            fig.suptitle(title)
    
        if hardcopy is not None:
            fig.savefig(hardcopy)
    
        return snr_ratio.loc[:, ['mid', 'ratio']].values

if __name__ == "__main__":


    # alist_file = '/homes/mlisakov/sci/eht/5.alist.v6'
    # alist_file = '/homes/mlisakov/correlation/ml005/alist_v6.out'
    alist_file = '/homes/mlisakov/data/correlation/ml005/alist.out'
    

    logger = create_logger(dest=['nullalister.log'])
    logger.info('\n============================================\nStart running version {} at {}'.format(__version__, dt.datetime.now()))
    sgra = alist(file=alist_file, logger=logger) # read data
    logger.info("Found sources: {}".format(sgra.data.source.unique()))
    sgra.add_columns() # add datetime and uv distance
    sgra.drop_unused_columns() # remove unused columns
    sgra.select_source('M87') # select specific source by name
    # sgra.select_source('SGRA') # select specific source by name
    # sgra.select_source('3C273') # select specific source by name
    # sgra.select_source('3C279') # select specific source by name
    # sgra.select_source('0716+714') # select specific source by name
    sgra.remove_autocorr()  # remove autocorrelations since they are not analyzed


    sgra.select_telescopes(['B', 'G', 'A', 'P'])      # select only baselines with specific telescopes included
    # sgra.select_baselines(['BG'])      # select only specified baselines



    
    # some specific constraints can be put directly on the data here
    # sgra.data = sgra.data.loc[(sgra.data.pol == 'RR') | (sgra.data.pol == 'LL')] # select only parallels
    sgra.data = sgra.data.loc[(sgra.data.pol == 'RR') ] # select only parallels
    # sgra.data = sgra.data.loc[(sgra.data.pol == 'RL') | (sgra.data.pol == 'LR')] # select only crosses
    # sgra.data = sgra.data.loc[(sgra.data.time > '2017-04-05 00:00:00') & 
    #                           (sgra.data.time < '2017-04-06 00:00:00')] # select specifit time range

    sgra.snr_cutoff = 6.4   # GMVA 2018 ML005
    # sgra.snr_cutoff = 7   # EHT 2017 SGRA*
    
    
        
    
    
    
    # plot SNR vs radial UV distance
    # sgra.radplot(uv_range = [100,3300], highlight_low_snr=True) # plot SNR(uv distance)

    # plot the ratio of the number of low SNR fringes to the number of all fringes 
    # binned over uv distance
    bin_width = 50 # in Mega lambda
    # snr_ratio_data = sgra.radplot_low_snr_fraction(bin_width = bin_width, label=r'RR and LL, bin width = {} M$\lambda$'.format(bin_width), 
    #                               title='{} 2017,  all days\nNumber of low SNR fringes (<{}) to all in the bin'.format(sgra.data.source.unique(), sgra.snr_cutoff),
    #                               hardcopy='nullalister.png',
    #                               uv_range = [100,3300])
    



    # M87 in ML005, 2018 with GMVA
    snr_ratio_data = sgra.radplot_low_snr_fraction(bin_width = bin_width, label=r'RR and LL, bin width = {} M$\lambda$'.format(bin_width), 
                                  title='{} ML005, 2018 with GMVA,  all days\nNumber of low SNR fringes (<{}) to all in the bin\nOnly baselines to GBT, EF, PV, ALMA'.format(sgra.data.source.unique(), sgra.snr_cutoff),
                                  hardcopy='nullalister.png',
                                  uv_range = [100,3300])

    
    # print("SNR ratio data\n{}".format(snr_ratio_data))
    
    logger.info('Finished run at {}'.format(dt.datetime.now()))

        
        
        
      