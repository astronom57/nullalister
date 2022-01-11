#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:09:11 2022

@author: mikhail
"""
import logging
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import platform  # to get computer name in order to avoid selecting computer name and paths by hand

from nullalister import *

def tplot_low_snr_fraction(self, bin_width=10, label=None, title=None, hardcopy=None,
                             uv_range=None, bins=None, ndata=None, plot_counts=False, 
                             min_num_fringes=0):
    """Plot ratio of number of low snr points to total number of point in each bin
    of a certain width in uv distance.
    
    Args:
        bin_width: 
            a width of time bins in seconds
        plot_counts (Bool):
            print a number of low_snr_fringes (low_snr_fringes_4scale) over 
            total_fringes (total_fringes_4scale)
        min_num_fringes (int):
            plot only bin with this minimum total number of fringes
  
        
        
    """
    print(bin_width)
    
    if bins is None:
        bins = np.int((self.data.time.max() - self.data.time.min()).total_seconds() / bin_width)
    print(bins)
        
    self.data.loc[:, 't_binned'], _ = pd.cut(self.data.time, bins=bins, retbins=True)
    
    all_snr = pd.pivot_table(self.data, values='snr',  index=['t_binned'], aggfunc='count')
    low_snr = pd.pivot_table(self.data.loc[self.data.snr < self.snr_cutoff, :], values='snr',  index=['t_binned'], aggfunc='count')
    snr_ratio = all_snr.join(low_snr, on='t_binned', how='left', lsuffix='_all', rsuffix='_low')
    
    snr_ratio.loc[:, 'ratio'] = snr_ratio.loc[:, 'snr_low'] / snr_ratio.loc[:, 'snr_all']
    snr_ratio.loc[:, 'mid'] = snr_ratio.index.categories.mid
    
    # correcting bad values in snr_ratio. 
    # ratio = nan if snr_all ==0. Change ratio nan->0.0
    snr_ratio.loc[:, 'ratio'].fillna(value=0.0, inplace=True)
    
    
    self.logger.debug('Ratio data (before scaling): {}'.format(snr_ratio.loc[:, ['mid', 'ratio']]))
    self.logger.debug('snr_ratio index = {}'.format(snr_ratio.index))

       
    fig,ax = plt.subplots(1, 1)
    # if uv_range is not None:
        # ax.set_xlim(uv_range)

    
    if label is not None:
        # ax.plot(snr_ratio.loc[:, 'mid'], snr_ratio.loc[:, 'ratio'], '-*', label=label)
        ax2 = ax.twinx()
        # ax2.plot(self.data.uvdist, self.data.snr, 'o', label='{}'.format(self.data.source.unique()))
        ax2.plot(self.data.loc[:, 'time'], 
                 self.data.loc[:, 'snr'], 'o', 
                 label='{}'.format(self.data.source.unique()))
        
        # bars = ax.bar(snr_ratio.loc[:, 'mid'], snr_ratio.loc[:, 'ratio'], width=bin_width, color='red', alpha=0.5, label=label)
        bars = ax.bar(snr_ratio.loc[snr_ratio.snr_all >= min_num_fringes, 'mid'], snr_ratio.loc[snr_ratio.snr_all >= min_num_fringes, 'ratio'], width=dt.timedelta(seconds=bin_width), color='red', alpha=0.5, label=label) # introduced a min_num_fringes
        
    ax.set_xlabel(r'Time')
    ax.set_ylabel('Ratio of low SNR fringes')
    ax2.set_ylabel('Fringe SNR')
    # ax2.hlines(y=self.snr_cutoff, xmin=uv_range[0], xmax=uv_range[1])

    if title is not None:
        fig.suptitle(title)

    if hardcopy is not None:
        fig.savefig(hardcopy)

    return snr_ratio.loc[:, ['mid', 'ratio']].values







def null_var(alist_file, source=None, full=False, timerange=None, polar=['RR','LL'], date_str=None,
                  plot_counts=False, bin_width=200, 
                  exclude_telescopes=None, exclude_baselines=None,
                  select_telescopes=None, select_baselines=None,
                  min_num_fringes=0):
    """Nulls variability. Analyze and plot fringes (non-detections and detections) 
    in a time domain
    
    alist_file (str):
        alist file to analyze
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
    exclude_telescopes ([str]):
        exclude these telescopes data
    exclude_baselines ([str]):
        exclude specific baselines
    select_telescopes ([str]):
        select only these telescopes data
    select_baselines ([str]):
        select only these specific baselines
    min_num_fringes (int):
        plot only bin with this minimum total number of fringes
    """
    
    logger = create_logger(dest=['nullalister_t.log','stderr'])
    adata = alist(file=alist_file, logger=logger) # read data
    logger.info("Found sources: {}".format(adata.data.source.unique()))
    adata.add_columns() # add datetime and uv distance
    adata.drop_unused_columns() # remove unused columns
    
    if timerange is not None:   # limit timerange if required.
        adata.data = adata.data.loc[(adata.data.time >= timerange[0]) & (adata.data.time <= timerange[1])]
        
    if exclude_telescopes is not None:
        adata.exclude_telescopes(exclude_telescopes)

    if exclude_baselines is not None:
        adata.exclude_baselines(exclude_baselines)

    if select_telescopes is not None:
        adata.select_telescopes(select_telescopes)

    if select_baselines is not None:
        adata.select_baselines(select_baselines)

    if source is not None:
        adata.select_source([source]) # select specific source by name
        
    adata.remove_autocorr()  # remove autocorrelations since they are not analyzed
    adata.data = adata.data.loc[adata.data.pol.isin(polar)] # select polarizations 
    logger.info('Selected polarization products are: {}'.format(adata.data.pol.unique()))
    adata.snr_cutoff = adata.data.snr[adata.data.snr< 7.0].mean() + 2*adata.data.snr[adata.data.snr< 7.0].std() 

    years = adata.data.time.dt.year.unique() 

    snr_ratio_adata = adata.tplot_low_snr_fraction(bin_width = bin_width, label=r'{}, bin width = {} M$\lambda$'.format(polar, bin_width), 
                                  title='{} {},  {}, {}\nNumber of low SNR fringes (<{:.1f}) to all in the bin\nscaled by the non-horizon sources non-detections'.format(adata.data.source.unique(), years,'' if date_str is None else date_str, polar,  adata.snr_cutoff),
                                  hardcopy='nullalister_snr_scaled.png',
                                  plot_counts=plot_counts,
                                  min_num_fringes=min_num_fringes)


    


    return adata



if __name__ == "__main__":
    alist.null_var = null_var  # TODO: test if it works <- this added a method to a class imported from nullalister
    alist.tplot_low_snr_fraction = tplot_low_snr_fraction
    
    
    
    
    computer_name = platform.node()
    if computer_name == 'vlb140': # desktop 
        base='/homes/mlisakov/data/correlation/'
    else: # laptop
        base = '/home/mikhail/data/correlation/'

    alist_filename = '4.alist.v6'
    
    a = null_var(alist_file='{}eht2017/{}'.format(base, alist_filename), bin_width=100,
                 source='SGRA', timerange=[dt.datetime(2017,4,4,22,31,0),dt.datetime(2017,4,5,17,7,0)],
                 exclude_baselines=['SR'],
                 select_baselines=['AL', 'XL'])
    
    print(a)
    
    # a = null_var(alist_file='{}eht2017/{}'.format(base, alist_filename), bin_width=100,
    #              source='SGRA', timerange=[dt.datetime(2017,4,6,0,46,0),dt.datetime(2017,4,6,16,14,0)],
    #              exclude_baselines=['SR'],
    #              select_baselines=['AL', 'XL'])
    
    # a = null_var(alist_file='{}eht2017/{}'.format(base, alist_filename), bin_width=100,
    #              source='SGRA', timerange=[dt.datetime(2017,4,6,0,46,0),dt.datetime(2017,4,6,16,14,0)],
    #              exclude_baselines=['SR'],
    #              select_telescopes=['L', 'L'])
    
    
    # a = null_var(alist_file='{}eht2017/{}'.format(base, alist_filename), bin_width=100,
    #              source='SGRA', timerange=[dt.datetime(2017,4,7,4,1,0),dt.datetime(2017,4,7,20,42,0)],
    #              exclude_baselines=['SR'],
    #              select_baselines=['AL', 'XL'])
    
    # a = null_var(alist_file='{}eht2017/{}'.format(base, alist_filename), bin_width=100,
    #              source='SGRA', timerange=[dt.datetime(2017,4,9,23,17,0),dt.datetime(2017,4,11,15,22,0)],
    #              exclude_baselines=['SR'],
    #              select_baselines=['AL', 'XL'])