'''
Based on (but not identical to) the paper:
"On Periodicity Detection and Structural Periodic Similarity" (Vlachos et al, 2005).
http://alumni.cs.ucr.edu/~mvlachos/pubs/sdm05.pdf
'''

import warnings
import math
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft,ifft


def AutoPeriod(timeseries, pmin, pmax):
    """
    Find the most dominant periods in time series.

    :param timeseries: numpy dtype
    :param pmin: mimimal periodicity required (in number of sample points).
    :param pmax: maximal periodicity required (in number of sample points).
    :return  periods : list   (you must mutiply the time interval)
             scores  : list   contain the scores of the periods found.


    """

    # Coeffs for Savitzky-Golay filters: 1st derivative (quadratic).
    # https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter#Tables_of_selected_convolution_coefficients
    sgolcoeffs_1der_wind5 = np.array([2.0,1.0,0.0,-1.0,-2.0]) / 10
    sgolcoeffs_1der_wind7 = np.array([3.0,2.0,1.0,0.0,-1.0,-2.0,-3.0]) / 28

    """
    Parameters validation: check restrains on pmin, pmax:
      pmin has to be at least 4.
      pmax has to be at most half the time series length.
    """

    pmin = max(pmin, 4)
    pmax = min(pmax, int(len(timeseries) / 2) - 2)  
    if pmin >= pmax:
        return -1,-1

    # step1:======================================================================================
    # Calculate the power spectrum (similar to periodogram) and the autocorrelation function (=acf)
    m = timeseries.mean()
    timeseries = timeseries - m  
    length = len(timeseries)
    Fs = 1.0
    Ts = 1.0 / Fs
    xf = np.arange(length)  # freq
    T = length / Fs
    frq = xf / T  # xf * fs / length 
    freq = frq[range(1, int(len(x) / 2))]
    per = 1 / freq   # period

    # the auto correlation function is the inverse fourier of the power spectrum

    pspc = abs(fft(timeseries)) ** 2
    acf = ifft(pspc).real[range(int(len(timeseries) / 2))]  
    acf = (acf - acf.min()) / (acf.max() - acf.min())
    psp = pspc.real[range(1, int(len(timeseries) / 2))]


    # step2:======================================================================================
    """
    Find 'hints' for the periods by finding the local maxima in the power spectrum
    find power spectrum threshold by randomly permuting the time series (see paper)

    """
    tsb = timeseries.copy()
    np.random.seed(0)
    np.random.shuffle(tsb)
    psp_rp = abs(fft(tsb)) ** 2
    thresh = max(psp_rp)

    # set the power of periods bellow threshold to 0 (so we don't find any local maxima there)
    psp = np.where(psp >= thresh, psp, 0)

    # find the local maxima: points that are bigger than their surrounding points
    islocmax = (psp[1:len(psp) - 1] > psp[0:len(psp) - 2]) & (psp[1:len(psp) - 1] > psp[2:len(psp)])
    islocmax = np.insert(islocmax, 0, False)
    islocmax = np.insert(islocmax, len(islocmax), False)
    psp_locmax = np.where(islocmax, psp, 0)
    psp_locmax = np.where(per > pmin, psp_locmax, 0)
    psp_locmax = np.where(per < pmax, psp_locmax, 0)

    # step3:======================================================================================
    """
    Find local maxima in the acf.
    This is done by calculating the first derivative using the Savitzky-Golay filter 
    and finding where it crosses zero.
    In order to allow for both high resolution to find shord periods and robustness to noise,
    we first run a filter of size 5 and search for periods of size <10 and then run a filter
    of size 7 and search for periods of size >=10.
    """
    islocmax1 = np.zeros(len(acf))
    if pmin < 10:
        # The "+/-3" is to compensate for invalid points in the filter convolution result.
        rightLimit = pmax + 3 if (pmax <= 9) else 12
        islocmax1 = PeriodsAcfMaxima(acf, sgolcoeffs_1der_wind5, pmin - 3, rightLimit);
    if pmax >= 10:
        #Start from the minimal required period. The "+/-4" is to compensate for invalid points in the filter convolution result.
        min_lag = pmin - 4 if (pmin > 10) else 10 - 4
        max_lag = pmax + 4
        islocmax2 = PeriodsAcfMaxima(acf, sgolcoeffs_1der_wind7, min_lag, max_lag)

        islocmax_acf = (islocmax1 | islocmax2)
    else:
        islocmax_acf = islocmax1

    # acf_locmax_val equals the the acf value where there is a local maximum
    acf_locmax_val = np.where(islocmax_acf, acf, 0)
    # Make sure that no period less than pmin or larger than pmax will be found
    # acf_locmax_val = np.where(per > pmin, acf_locmax_val, 0)
    # acf_locmax_val = np.where(per < pmax, acf_locmax_val, 0)



    # step4:======================================================================================
    """
    Find the periods by finding a match between a 'hint' from the power spectrum and maximum in the acf.
    Each time, we examine for each time series a new candidate 'hint', which is the one with the strongest spectral intensity.
    """
    # psp_locmax_rew = psp_locmax.copy() 
    # acf_locmax_val = acf.copy()
    periods = []
    scores = []

    while True:
        if max(psp_locmax) == 0:
            break
        psp_index_sort = np.argsort(-psp_locmax)
        psp_index_max = psp_index_sort[0]
        psp_cand_max = max(psp_locmax)

        # lower/upper bounds for searching the match in the acf (see paper)
        lowerb_ind = psp_index_max + 1
        upperb_ind = psp_index_max - 1

        curr_cand_per_lowerb = 0.5 * (per[psp_index_max] + per[lowerb_ind]) - 1
        curr_cand_per_upperb = 0.5 * (per[psp_index_max] + per[upperb_ind]) + 1

        curr_cand_per_lowerb = int(curr_cand_per_lowerb)
        curr_cand_per_upperb = int(math.ceil(curr_cand_per_upperb))

        local_acf_val = acf_locmax_val[curr_cand_per_lowerb:curr_cand_per_upperb + 1]
        local_acfmax_val = max(local_acf_val)
        local_acfmax_index = np.argsort(-local_acf_val)[0] + curr_cand_per_lowerb

        # Update the periods we found.
        # We include the new period if its score is higher than the smallest score found so far.
        if len(periods) == 0:
            periods.append(local_acfmax_index)
            scores.append(local_acfmax_val)
        else:
            if local_acfmax_val > max(scores):
                periods.append(local_acfmax_index)
                scores.append(local_acfmax_val)

        psp_locmax[psp_index_max] = 0

    # sort the periods that were found, by their scores (and, of course, sort the scores too).
    # this procedure is done to avoid looping the rows multiple times and make use of vertorized indexing for efficiency.

    if len(periods) == 0:
        return -1,-1
    # sum_score = sum(scores)
    # scores_per = [each/sum_score for each in scores]
    target = [(score, period) for score, period in zip(scores, periods)]
    target.sort(reverse=True)
    periods = [period for score, period in target]
    scores = [score for score, period in target]
    return periods, scores

def PeriodsAcfMaxima(acf, sgolcoeffs, min_lag, max_lag):
    # Compute the 1st derivative of the ACF using Savitzky-Golay filter.
    # Find local maxima where the sign of 1st derivative changes from + (or 0) to -
    fsize = len(sgolcoeffs)
    delta = int(fsize / 2)  # the required number of points to the left and to the right of a certain point for the filter result to be valid there.
    valid_ind_min = delta
    valid_ind_max = max_lag - delta - min_lag
    id = 0
    acf_1d = np.convolve(acf[min_lag:max_lag + 1], sgolcoeffs, "same")  # length = max_lag - min_lag + 1
    acf_1d_up = acf_1d[valid_ind_min:valid_ind_max] >= 0
    acf_1db_down = acf_1d[valid_ind_min+1:valid_ind_max+1] < 0

    islocmax = acf_1d_up & acf_1db_down
    islocmax = np.insert(islocmax, 0, np.zeros(min_lag + valid_ind_min,int))
    islocmax = np.insert(islocmax, len(islocmax), np.zeros(len(acf)-len(islocmax), int))
    temp = islocmax[1:] + islocmax[:-1]
    islocmax = np.insert(temp, 0, islocmax[0])
    return islocmax



if __name__ == "__main__":
    pmin = 4
    pmax = 500

    # filepath = "Data/seasonal/"
    filepath = "A1benchmark/real_1"
    filename = filepath + ".csv"
    file = pd.read_csv(filename, index_col="timestamp")
    x = file.iloc[:, 0]
    timeseries = np.array(x)
    periods, scores = AutoPeriod(timeseries, pmin, pmax)
    print("periods: ", periods)
    print("scores: ", scores)
