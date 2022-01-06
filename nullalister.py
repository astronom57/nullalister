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
        self.data.rename({7:'exp', 8:'scan', 13:'source', 14:'baseline', 15:'qe',
                          17:'pol', 20:'snr', 32:'u', 33:'v', 36:'freq',
                          10: 'year', 11:'doytime'},
                         axis='columns', inplace=True)
        
        return self.data


    def remove_autocorr(self):
        """Delete autocorrelations from the data, e.g. when baseline is XX.
        
        Returns:
            dataframe with autocorr lines deleted
        """
        
        self.logger.info("before dropping autocorrs there were {} rows".format(self.data.index.size))
        self.data.drop(self.data.loc[self.data.baseline.str[0] == self.data.baseline.str[1]].index, inplace=True)
        self.logger.info("after dropping autocorrs there were {} rows".format(self.data.index.size))
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
    
    def exclude_rings(self):
        """Exclude M87 and SGRA.
        
        Args:
            None
        
        Returns:
            data without SGRA and M87
            
        """
 
        self.data = self.data.loc[~self.data.source.isin(['M87', 'SGRA'])]
        
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
            columns = ['exp', 'scan', 'source', 'baseline', 'qe', 'u', 'v', 'snr', 'pol',
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
                                 uv_range=None, bins=None, ndata=None, plot_counts=False,
                                 min_num_fringes=0):
        """Plot ratio of number of low snr points to total number of point in each bin
        of a certain width in uv distance.
        
        Args:
            bin_width: 
                a width of the uvdist bins in megalambdas
            plot_counts (Bool):
                print a number of low_snr_fringes (low_snr_fringes_4scale) over 
                total_fringes (total_fringes_4scale) 
            min_num_fringes (int):
                plot only bin with this minimum total number of fringes

            
            
        """
        if bins is None:
            if uv_range is not None:
                bins = np.int((uv_range[1] - uv_range[0]) / bin_width)
            else:
                bins = np.int(self.data.uvdist.max() / bin_width)
            
        self.data.loc[:, 'uv_binned'], _ = pd.cut(self.data.uvdist.values, bins=bins, retbins=True)
        
        all_snr = pd.pivot_table(self.data, values='snr',  index=['uv_binned'], aggfunc='count')
        low_snr = pd.pivot_table(self.data.loc[self.data.snr < self.snr_cutoff, :], values='snr',  index=['uv_binned'], aggfunc='count')
        snr_ratio = all_snr.join(low_snr, on='uv_binned', how='left', lsuffix='_all', rsuffix='_low')
        
        snr_ratio.loc[:, 'ratio'] = snr_ratio.loc[:, 'snr_low'] / snr_ratio.loc[:, 'snr_all']
        snr_ratio.loc[:, 'mid'] = snr_ratio.index.categories.mid
        
        # correcting bad values in snr_ratio. 
        # ratio = nan if snr_all ==0. Change ratio nan->0.0
        snr_ratio.loc[:, 'ratio'].fillna(value=0.0, inplace=True)
        
        
        self.logger.debug('Ratio data (before scaling): {}'.format(snr_ratio.loc[:, ['mid', 'ratio']]))
        self.logger.debug('snr_ratio index = {}'.format(snr_ratio.index))

        if ndata is not None:
            ndata.data.loc[:, 'uv_binned'], _ = pd.cut(ndata.data.uvdist.values, bins=bins, retbins=True)

            all_snr_4scale = pd.pivot_table(ndata.data, values='snr',  index=['uv_binned'], aggfunc='count')
            low_snr_4scale = pd.pivot_table(ndata.data.loc[ndata.data.snr < ndata.snr_cutoff, :], values='snr',  index=['uv_binned'], aggfunc='count')
            snr_ratio_4scale = all_snr_4scale.join(low_snr_4scale, on='uv_binned', how='left', lsuffix='_all', rsuffix='_low')
            
            snr_ratio_4scale.loc[:, 'ratio'] = snr_ratio_4scale.loc[:, 'snr_low'] / snr_ratio_4scale.loc[:, 'snr_all'] # bugfixed: snr_ratio -> snr_ratio_4scale
            snr_ratio_4scale.loc[:, 'mid'] = snr_ratio_4scale.index.categories.mid
            
            # correcting bad values in snr_ratio_4scale. 
            # ratio = nan if snr_all ==0. Change ratio nan->median(non-nan ratios)
            # print(snr_ratio_4scale.loc[:, snr_ratio_4scale.ratio.median()])
            # QUESTIONABLE. Using median undermines all the idea of binning
            # snr_ratio_4scale.loc[snr_ratio_4scale.ratio == 0, 'ratio'] = snr_ratio_4scale.ratio.median()
            # snr_ratio_4scale.loc[:, 'ratio'].fillna(value=snr_ratio_4scale.ratio.median(), inplace=True)
            # this should work. ratio == 0 => ratio =1 
            snr_ratio_4scale.loc[snr_ratio_4scale.ratio == 0, 'ratio'] = 1.0
            snr_ratio_4scale.loc[:, 'ratio'].fillna(value=0.0, inplace=True)

            
            
            
            self.logger.debug('Ratio data (before scaling): {}'.format(snr_ratio.loc[:, ['mid', 'ratio']]))
            self.logger.debug('snr_ratio_4scale index = {}'.format(snr_ratio_4scale.index))    
        
            snr_ratio.loc[:, 'ratio'] = snr_ratio.loc[:, 'ratio'] * snr_ratio_4scale.loc[:, 'ratio']
        
        
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
            
            # bars = ax.bar(snr_ratio.loc[:, 'mid'], snr_ratio.loc[:, 'ratio'], width=bin_width, color='red', alpha=0.5, label=label)
            bars = ax.bar(snr_ratio.loc[snr_ratio.snr_all >= min_num_fringes, 'mid'], snr_ratio.loc[snr_ratio.snr_all >= min_num_fringes, 'ratio'], width=bin_width, color='red', alpha=0.5, label=label) # introduced a min_num_fringes
            # labels_x = np.array([bars.patches[i].xy[0] for i in range(0,len(bars.patches))]) + np.array([bars.patches[i].get_width() / 2  for i in range(0,len(bars.patches))])
            # labels_y = np.array([bars.patches[i].xy[1] for i in range(0,len(bars.patches))]) + np.array([bars.patches[i].get_height() for i in range(0,len(bars.patches))])
            # # labels_text = 
            # print(labels_x)
            # print(labels_y)
            
            if plot_counts is True:
                for i in snr_ratio.index:
                    if ndata is not None:
                        ax.text(snr_ratio.loc[i, 'mid'], snr_ratio.loc[i,'ratio'], '{}({})\n--\n{}({})'.format(snr_ratio.loc[i, 'snr_low'],snr_ratio_4scale.loc[i, 'snr_low'], snr_ratio.loc[i, 'snr_all'], snr_ratio_4scale.loc[i, 'snr_all']), zorder=100)
                    else:
                        ax.text(snr_ratio.loc[i, 'mid'], snr_ratio.loc[i,'ratio'], '{}\n--\n{}'.format(snr_ratio.loc[i, 'snr_low'], snr_ratio.loc[i, 'snr_all']), zorder=100)
            
            
            
            # fig.legend()
        else:
            # ax.plot(snr_ratio.loc[:, 'mid'], snr_ratio.loc[:, 'ratio'], '-*')
            ax.bar(snr_ratio.loc[:, 'mid'], snr_ratio.loc[:, 'ratio'], width=bin_width, color='red', alpha=0.5)
            
        ax.set_xlabel(r'UV distance (M$\lambda$)')
        ax.set_ylabel('Ratio of low SNR fringes')
        ax2.set_ylabel('Fringe SNR')
        ax2.hlines(y=self.snr_cutoff, xmin=uv_range[0], xmax=uv_range[1])

        if title is not None:
            fig.suptitle(title)
    
        if hardcopy is not None:
            fig.savefig(hardcopy)
    
        return snr_ratio.loc[:, ['mid', 'ratio']].values


    def radplot_low_qe_fraction(self, bin_width=500, qe_cutoff=3,  label=None, title=None, hardcopy=None,
                                 uv_range=None):
        """Plot ratio of number of low quality fringes (QE <=qe_cutoff , which means a non-detection) points to total number of point in each bin
        of a certain width in uv distance.
        
        Args:
            bin_width (float): 
                a width of the uvdist bins in megalambdas
            qe_cutoff (int):
                maximum Fringe quality, which is considerted a non-detection
        """
        
        self.data.loc[:, 'uv_binned'], _ = pd.cut(self.data.uvdist.values, bins = np.int(self.data.uvdist.max() / bin_width), retbins=True)
        
        # all_snr = pd.pivot_table(self.data, values='snr',  index=['uv_binned'], aggfunc='count')
        all_qe = pd.pivot_table(self.data, values='qe',  index=['uv_binned'], aggfunc='count')
        # low_snr = pd.pivot_table(self.data.loc[self.data.snr < self.snr_cutoff, :], values='snr',  index=['uv_binned'], aggfunc='count')
        low_qe = pd.pivot_table(self.data.loc[self.data.qe <= qe_cutoff , :], values='qe',  index=['uv_binned'], aggfunc='count')
        
        # snr_ratio = all_snr.join(low_snr, on='uv_binned', how='left', lsuffix='_all', rsuffix='_low')
        qe_ratio = all_qe.join(low_qe, on='uv_binned', how='left', lsuffix='_all', rsuffix='_low')
        
        qe_ratio.loc[:, 'ratio'] = qe_ratio.loc[:, 'qe_low'] / qe_ratio.loc[:, 'qe_all']
        qe_ratio.loc[:, 'mid'] = qe_ratio.index.categories.mid
        fig,ax = plt.subplots(1, 1)
        if uv_range is not None:
            ax.set_xlim(uv_range)
    
        
        if label is not None:
            # ax.plot(snr_ratio.loc[:, 'mid'], snr_ratio.loc[:, 'ratio'], '-*', label=label)
            ax2 = ax.twinx()
            # ax2.plot(self.data.uvdist, self.data.snr, 'o', label='{}'.format(self.data.source.unique()))
            if uv_range is not None:
                ax2.plot(self.data.loc[self.data.uvdist > uv_range[0], 'uvdist'], 
                         self.data.loc[self.data.uvdist > uv_range[0], 'qe'], 'o', 
                         label='{}'.format(self.data.source.unique()))
            
            ax.bar(qe_ratio.loc[:, 'mid'], qe_ratio.loc[:, 'ratio'], width=bin_width, color='red', alpha=0.5, label=label)
            # fig.legend()
        else:
            # ax.plot(snr_ratio.loc[:, 'mid'], snr_ratio.loc[:, 'ratio'], '-*')
            ax.bar(qe_ratio.loc[:, 'mid'], qe_ratio.loc[:, 'ratio'], width=bin_width, color='red', alpha=0.5)
            
        ax.set_xlabel(r'UV distance (M$\lambda$)')
        ax.set_ylabel('Ratio of low QE fringes')
        ax2.set_ylabel('Fringe Quality')
        # ax2.hlines(y=self.snr_cutoff, xmin=uv_range[0], xmax=uv_range[1])
    
        if title is not None:
            fig.suptitle(title)
    
        if hardcopy is not None:
            fig.savefig(hardcopy)
    
        return qe_ratio.loc[:, ['mid', 'ratio']].values
    

    
    
    
def proceed(alist_file=None, source=None, full=False, timerange=None, polar=['RR','LL'], date_str=None,
            plot_counts=False, bin_width=200):
    """ Proceed .
    No scaling is applied. 
    
    Args:
        alist_file (str):
            filename of the alist file to proceed.
        source (str):
            horizon scale source to be selected. SGRA or M87
        full (bool):
            if True, plot uncorrected ratios for a horizon source, corrected, and 
            also ratios for non-horizon sources. If False, plot only corrected ones
        timerange ([timestamp]):
            limit timerange to spesified
        polar ([str]):
            array of polarization correlation products to be analyzed. Choose any 
            combination of RR, LL, RL, LR. Obvious choices are ['RR', 'LL'],
            ['RL', 'LR'], and ['RR', 'LL', 'RL', 'LR']
        date_str (str):
            date identifier to put in the title   
        plot_counts (Bool):
            print a number of low_snr_fringes (low_snr_fringes_4scale) over 
            total_fringes (total_fringes_4scale)
        bin_width (float):
            bin width in mega-lambda
    """
    
    # alist_file = '/homes/mlisakov/data/correlation/ehtALL/ehtALL.alist'
    # alist_file = '/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist'
    # alist_file = '/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev7.alist'
    
    # alist_file = '/homes/mlisakov/data/correlation/eht2018/eht2018_jw_rev3.alist'
    # alist_file = '/homes/mlisakov/data/correlation/eht2018/rev3/e18c21-3-b2.3644.alist'
    # alist_file = '/homes/mlisakov/data/correlation/eht2018/e18c21.alist'
    
    # alist_file = '/homes/mlisakov/data/correlation/eht2021/eht2021_jw_rev0.alist'
    if alist_file is None:
        alist_file = '/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist'
    

    logger = create_logger(dest=['nullalister.log','stderr'])
    logger.info('\n============================================\nStart running version {} at {}'.format(__version__, dt.datetime.now()))
    adata = alist(file=alist_file, logger=logger) # read data
    logger.info("Found sources: {}".format(adata.data.source.unique()))
    adata.add_columns() # add datetime and uv distance
    adata.drop_unused_columns() # remove unused columns
    
    if timerange is not None:
        adata.data = adata.data.loc[(adata.data.time >= timerange[0]) & (adata.data.time <= timerange[1])]


    adata.select_source([source]) # select specific source by name
    adata.remove_autocorr()  # remove autocorrelations since they are not analyzed
    adata.data = adata.data.loc[adata.data.pol.isin(polar)] # select polarizations 
    
    logger.info('Selected polarization products are: {}'.format(adata.data.pol.unique()))
    

    
    # calculate snr cutoff as mean + 2 std for all fringes with snr < 7.0
    adata.snr_cutoff = adata.data.snr[adata.data.snr< 7.0].mean() + 2*adata.data.snr[adata.data.snr< 7.0].std() 
    
    
    
    snr_ratio_data = adata.radplot_low_snr_fraction(bin_width = bin_width, label=r'{}, bin width = {} M$\lambda$'.format(polar, bin_width), 
                                  title='{}, {}, {}\nNumber of low SNR fringes (<{:.1f}) to all in the bin'.format(adata.data.source.unique(), 
                                                                                                                         '' if date_str is None else date_str, 
                                                                                                                         polar, adata.snr_cutoff),
                                  hardcopy='nullalister_snr.png',
                                  uv_range = [100,8200],
                                  plot_counts=plot_counts)
    
    # the same but based on the fringe quality data. Proven to be consistent with the SNR-based estimates 
    # qe_cutoff = 3
    # qe_ratio_data = adata.radplot_low_qe_fraction(bin_width = bin_width, qe_cutoff=qe_cutoff, label=r'RR and LL, bin width = {} M$\lambda$'.format(bin_width), 
    #                               title='{} 2021,  all days\nNumber of low QE fringes (<={}) to all in the bin'.format(adata.data.source.unique(), qe_cutoff),
    #                               hardcopy='nullalister_qe.png',
    #                               uv_range = [100,8200])
    
    logger.info('Finished run at {}'.format(dt.datetime.now()))
    
    
    
    
    return
    
def proceed_no_rings():
    """ Proceed with all 230 GHz data combined into one huge alist file. 
    """
    
    # alist_file = '/homes/mlisakov/data/correlation/ehtALL/ehtALL.alist'
    # alist_file = '/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist'
    # alist_file = '/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev7.alist'
    
    # alist_file = '/homes/mlisakov/data/correlation/eht2018/eht2018_jw_rev3.alist'
    # alist_file = '/homes/mlisakov/data/correlation/eht2018/rev3/e18c21-3-b2.3644.alist'
    alist_file = '/homes/mlisakov/data/correlation/eht2018/e18c21.alist'

    alist_file = '/homes/mlisakov/data/correlation/eht2021/eht2021_jw_rev0.alist'
    
    

    logger = create_logger(dest=['nullalister.log','stderr'])
    logger.info('\n============================================\nStart running version {} at {}'.format(__version__, dt.datetime.now()))
    adata = alist(file=alist_file, logger=logger) # read data
    logger.info("Found sources: {}".format(adatRR and LLa.data.source.unique()))
    adata.add_columns() # add datetime and uv distance
    adata.drop_unused_columns() # remove unused columns
    
    adata.exclude_rings()   # all sources except M87 and SGRA. This will give a baseline for the level lof non-detections
    
    # adata.select_source(['M87', 'SGRA']) # select specific source by name
    # adata.select_source('M87') # select specific source by name
    # adata.select_source('3C273') # select specific source by name
    # adata.select_source('SGRA') # select specific source by name
    # adata.select_telescopes(['B', 'G', 'A', 'P'])
    adata.remove_autocorr()  # remove autocorrelations since they are not analyzed
    adata.data = adata.data.loc[(adata.data.pol == 'RR') | (adata.data.pol == 'LL')] # select only parallels
    # adata.snr_cutoff = 7   # EHT 2017 SGRA*
    
    # calculate snr cutoff as mean + 2 std for all fringes with snr < 7.0
    adata.snr_cutoff = adata.data.snr[adata.data.snr< 7.0].mean() + 2*adata.data.snr[adata.data.snr< 7.0].std() 
    
    
    bin_width = 200 # in Mega lambda
    snr_ratio_data = adata.radplot_low_snr_fraction(bin_width = bin_width, label=r'RR and LL, bin width = {} M$\lambda$'.format(bin_width), 
                                  title='{} 2021,  all days\nNumber of low SNR fringes (<{:.1f}) to all in the bin'.format(adata.data.source.unique(), adata.snr_cutoff),
                                  hardcopy='nullalister_snr.png',
                                  uv_range = [100,8200])
    
    qe_cutoff = 3
    qe_ratio_data = adata.radplot_low_qe_fraction(bin_width = bin_width, qe_cutoff=qe_cutoff, label=r'RR and LL, bin width = {} M$\lambda$'.format(bin_width), 
                                  title='{} 2021,  all days\nNumber of low QE fringes (<={}) to all in the bin'.format(adata.data.source.unique(), qe_cutoff),
                                  hardcopy='nullalister_qe.png',
                                  uv_range = [100,8200])
    
    
    
    
    logger.info('Finished run at {}'.format(dt.datetime.now()))
    
    return

  
def proceed_scale(alist_file=None, source=None, full=False, timerange=None, polar=['RR','LL'], date_str=None,
                  plot_counts=False, bin_width=200, min_num_fringes=0):
    """ Proceed .
    This should be the most accurate method. The difference is that now the ratios 
    of non-detection to detections per bin for the horizon sources (SGRA or M87) are multiplied 
    by the same ratio for the same bins for non-horizon sources. 
    With non-horizon sources we will remove the impact of all technical stuff on the non-detections.
    What should left are just minima in the visibility function due to the ring structure. 
    
    Args:
        alist_file (str):
            filename of the alist file to proceed.
        source (str):
            horizon scale source to be selected. SGRA or M87
        full (bool):
            if True, plot uncorrected ratios for a horizon source, corrected, and 
            also ratios for non-horizon sources. If False, plot only corrected ones
        timerange ([timestamp]):
            limit timerange to spesified
        polar ([str]):
            array of polarization correlation products to be analyzed. Choose any 
            combination of RR, LL, RL, LR. Obvious choices are ['RR', 'LL'],
            ['RL', 'LR'], and ['RR', 'LL', 'RL', 'LR']
        date_str (str):
            date identifier to put in the title
        plot_counts (Bool):
            print a number of low_snr_fringes (low_snr_fringes_4scale) over 
            total_fringes (total_fringes_4scale) 
        bin_width (float):
            bin width in mega-lambda
        min_num_fringes (int):
            plot only bin with this minimum total number of fringes
    """
    
    uv_range = [100,8200] # range of UV distances to consider. [100 cuts very sensitive baselines AA-AP
    qe_cutoff = 3  # fourfit fringe quality cutoff

    if source is None:
        source = 'M87'


    if alist_file is None:
        # alist_file = '/homes/mlisakov/data/correlation/ehtALL/ehtALL.alist'
        # alist_file = '/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist'
        # alist_file = '/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev7.alist'
        
        # alist_file = '/homes/mlisakov/data/correlation/eht2018/eht2018_jw_rev3.alist'
        # alist_file = '/homes/mlisakov/data/correlation/eht2018/rev3/e18c21-3-b2.3644.alist'
        # alist_file = '/homes/mlisakov/data/correlation/eht2018/e18c21.alist'
        alist_file = '/homes/mlisakov/data/correlation/eht2021/eht2021_jw_rev0.alist'    
    

    logger = create_logger(dest=['nullalister.log','stderr'])
    logger.info('\n============================================\nStart running version {} at {}'.format(__version__, dt.datetime.now()))
    
    adata = alist(file=alist_file, logger=logger) # read data. Will leave here only M87 or SGRA
    ndata = alist(file=alist_file, logger=logger) # read data. Will leave here only non-horizon sources
    
    years = adata.data.year.unique()
    logger.info('Data is taken in years {}'.format(years))
    
    logger.info("Found sources: {}".format(adata.data.source.unique()))
    adata.add_columns() # add datetime and uv distance
    ndata.add_columns() # add datetime and uv distance
    adata.drop_unused_columns() # remove unused columns
    ndata.drop_unused_columns() # remove unused columns
    
    if timerange is not None:   # limit timerange if required.
        adata.data = adata.data.loc[(adata.data.time >= timerange[0]) & (adata.data.time <= timerange[1])]
        ndata.data = ndata.data.loc[(ndata.data.time >= timerange[0]) & (ndata.data.time <= timerange[1])]
        
    # source selection
    adata.select_source([source]) # select specific source by name
    ndata.exclude_rings()   # all sources except M87 and SGRA. This will give a baseline for the level lof non-detections
        
    # polarisation selection
    adata.remove_autocorr()  # remove autocorrelations since they are not analyzed
    adata.data = adata.data.loc[adata.data.pol.isin(polar)] # select polarizations 
    logger.info('Selected polarization products are: {}'.format(adata.data.pol.unique()))

    ndata.remove_autocorr()  # remove autocorrelations since they are not analyzed
    ndata.data = ndata.data.loc[ndata.data.pol.isin(polar)] # select polarizations 
    
    
    # calculate bins to have them the same between horizon and non-horizon sources
    _, bins = pd.cut(adata.data.uvdist.values, bins=np.int((uv_range[1] - uv_range[0]) / bin_width), retbins=True)
    
    
    # calculate snr cutoff as mean + 2 std for all fringes with snr < 7.0
    adata.snr_cutoff = adata.data.snr[adata.data.snr< 7.0].mean() + 2*adata.data.snr[adata.data.snr< 7.0].std() 
    ndata.snr_cutoff = ndata.data.snr[ndata.data.snr< 7.0].mean() + 2*ndata.data.snr[ndata.data.snr< 7.0].std() 
    
    
    logger.info('Total non-detection rate for {} is {} / {} = {:.2f}'.format(source, 
                                                                             adata.data.loc[adata.data.snr < adata.snr_cutoff, 'snr'].count(),
                                                                             adata.data.loc[:, 'snr'].count(),
                                                                             adata.data.loc[adata.data.snr < adata.snr_cutoff, 'snr'].count()/adata.data.loc[:, 'snr'].count()
                                                                             ))

    
    
    if full is True:
        
        
        
        snr_ratio_adata = adata.radplot_low_snr_fraction(bin_width = bin_width, label=r'{}, bin width = {} M$\lambda$'.format(polar, bin_width), 
                                      title='{} {},  {}\nNumber of low SNR fringes (<{:.1f}) to all in the bin'.format(adata.data.source.unique(), years, '' if date_str is None else date_str, adata.snr_cutoff),
                                      hardcopy='nullalister_snr.png',
                                      uv_range=uv_range,
                                      bins=bins,
                                      plot_counts=plot_counts)
        
        snr_ratio_adata = adata.radplot_low_snr_fraction(bin_width = bin_width, label=r'{}, bin width = {} M$\lambda$'.format(polar, bin_width), 
                                      title='{} {},  all days\nNumber of low SNR fringes (<{:.1f}) to all in the bin\nscaled by the non-horizon sources non-detections'.format(adata.data.source.unique(), years, '' if date_str is None else date_str, adata.snr_cutoff),
                                      hardcopy='nullalister_snr_scaled.png',
                                      uv_range=uv_range,
                                      bins=bins,
                                      ndata=ndata,
                                      plot_counts=plot_counts)
        
        snr_ratio_ndata = ndata.radplot_low_snr_fraction(bin_width = bin_width, label=r'{}, bin width = {} M$\lambda$'.format(polar, bin_width), 
                                      title='{} {},  all days\nNumber of low SNR fringes (<{:.1f}) to all in the bin'.format(ndata.data.source.unique(), years, '' if date_str is None else date_str, ndata.snr_cutoff),
                                      hardcopy='nullalister_snr.png',
                                      uv_range=uv_range,
                                      bins=bins,
                                      plot_counts=plot_counts) 
    else:
        snr_ratio_adata = adata.radplot_low_snr_fraction(bin_width = bin_width, label=r'{}, bin width = {} M$\lambda$'.format(polar, bin_width), 
                                      title='{} {},  {}, {}\nNumber of low SNR fringes (<{:.1f}) to all in the bin\nscaled by the non-horizon sources non-detections'.format(adata.data.source.unique(), years,'' if date_str is None else date_str, polar,  adata.snr_cutoff),
                                      hardcopy='nullalister_snr_scaled.png',
                                      uv_range=uv_range,
                                      bins=bins,
                                      ndata=ndata,
                                      plot_counts=plot_counts,
                                      min_num_fringes=min_num_fringes) 




    logger.info('Finished run at {}'.format(dt.datetime.now()))
    
    
    
    
    return


if __name__ == "__main__":

    if True:
        # proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='M87', polar=['RR','LL'])
        # proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2018/eht2018_jw_rev3.alist', source='M87', polar=['RR','LL'])
        # proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2021/eht2021_jw_rev0.alist', source='M87', polar=['RR','LL'])
        
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', polar=['RR','LL'], 
                      plot_counts=True, min_num_fringes=5)
        # proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2018/eht2018_jw_rev3.alist', source='SGRA', polar=['RR','LL'])
        # proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2021/eht2021_jw_rev0.alist', source='SGRA', polar=['RR','LL'])
 
    if False:  # compare different pol products on 2017 Apr 5 and 6.
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
                      timerange=[dt.datetime(2017,4,5,0,0,0),dt.datetime(2017,4,6,0,0,0)], polar=['RR', 'LL'])
       
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
                      timerange=[dt.datetime(2017,4,5,0,0,0),dt.datetime(2017,4,6,0,0,0)], polar=['RL', 'LR'])
       
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
                      timerange=[dt.datetime(2017,4,5,0,0,0),dt.datetime(2017,4,6,0,0,0)], polar=['RR','LL','RL', 'LR'])
    
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
                      timerange=[dt.datetime(2017,4,6,0,0,0),dt.datetime(2017,4,7,0,0,0)], polar=['RR', 'LL'])
       
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
                      timerange=[dt.datetime(2017,4,6,0,0,0),dt.datetime(2017,4,7,0,0,0)], polar=['RL', 'LR'])
       
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
                      timerange=[dt.datetime(2017,4,6,0,0,0),dt.datetime(2017,4,7,0,0,0)], polar=['RR','LL','RL', 'LR'])


    if False:
    #     # compare Apr 6th and other days in 2017. With scaling
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
                      timerange=[dt.datetime(2017,4,5,0,0,0),dt.datetime(2017,4,6,0,0,0)], polar=['RR', 'LL'],
                      date_str = 'Apr 5, 2017', plot_counts=False)
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
                      timerange=[dt.datetime(2017,4,6,0,0,0),dt.datetime(2017,4,7,0,0,0)], polar=['RR', 'LL'],
                      date_str = 'Apr 6, 2017', plot_counts=False)
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
                      timerange=[dt.datetime(2017,4,7,0,0,0),dt.datetime(2017,4,8,0,0,0)], polar=['RR', 'LL'],
                      date_str = 'Apr 7, 2017')
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
                      timerange=[dt.datetime(2017,4,10,0,0,0),dt.datetime(2017,4,12,0,0,0)], polar=['RR', 'LL'],
                      date_str = 'Apr 10-11, 2017')
        # proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
        #               timerange=[dt.datetime(2017,4,11,0,0,0),dt.datetime(2017,4,12,0,0,0)], polar=['RR', 'LL'],
        #               date_str = 'Apr 11, 2017')
        proceed_scale(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
                      timerange=[dt.datetime(2017,4,5,0,0,0),dt.datetime(2017,4,12,0,0,0)], polar=['RR', 'LL'],
                      date_str = 'Apr 5-11, 2017')
    
    
    #     # compare Apr 6th and other days in 2017. Without scaling
    
        # proceed(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
        #               timerange=[dt.datetime(2017,4,5,0,0,0),dt.datetime(2017,4,6,0,0,0)], polar=['RR', 'LL'],
        #               date_str = 'Apr 5, 2017', plot_counts=True)
        # proceed(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
        #               timerange=[dt.datetime(2017,4,6,0,0,0),dt.datetime(2017,4,7,0,0,0)], polar=['RR', 'LL'],
        #               date_str = 'Apr 6, 2017', plot_counts=True)
        # proceed(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
        #               timerange=[dt.datetime(2017,4,7,0,0,0),dt.datetime(2017,4,8,0,0,0)], polar=['RR', 'LL'],
        #               date_str = 'Apr 7, 2017')
        # proceed(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
        #               timerange=[dt.datetime(2017,4,10,0,0,0),dt.datetime(2017,4,11,0,0,0)], polar=['RR', 'LL'],
        #               date_str = 'Apr 10, 2017')
        # proceed(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
        #               timerange=[dt.datetime(2017,4,11,0,0,0),dt.datetime(2017,4,12,0,0,0)], polar=['RR', 'LL'],
        #               date_str = 'Apr 11, 2017')
        # proceed(alist_file='/homes/mlisakov/data/correlation/eht2017/eht2017_jw_rev5.alist', source='SGRA', 
        #               timerange=[dt.datetime(2017,4,5,0,0,0),dt.datetime(2017,4,12,0,0,0)], polar=['RR', 'LL'],
        #               date_str = 'Apr 5-11, 2017', bin_width=250)
