#  Copyright (c) 2021 Robert Lieck

import re
from itertools import product
import os
from warnings import warn
from multiprocessing import Pool

import numpy as np
import pandas as pd

import pitchscapes.reader as rd
import pitchscapes.plotting as pt
from pitchscapes.keyfinding import KeyEstimator

from util import MusicData


def load_scape(file_path=None, prior_counts=None, n_samples=None, pitch_scape=None, normalise_observations=True):
    assert (file_path is None) == (prior_counts is None) == (n_samples is None) != (pitch_scape is None), \
        "Either 'pitch_scape' or 'file_path', 'prior_counts', and 'n_samples' have to be provided"
    # read file into scape
    if pitch_scape is None:
        pitch_scape = rd.get_pitch_scape(file_path, prior_counts=prior_counts)
    # define time grid
    times = np.linspace(pitch_scape.min_time, pitch_scape.max_time, n_samples + 1)
    # get samples from scape
    scape_samples = []
    for start, end in product(times, times):
        if end <= start:
            continue
        scape_samples.append(pitch_scape[start, end])
    scape_samples = np.array(scape_samples)
    scape_samples /= scape_samples.sum(axis=1, keepdims=True)
    # convert to colours
    k = KeyEstimator()
    scores = k.get_score(scape_samples)
    colours = pt.key_scores_to_color(scores)
    # get samples from time intervals
    interval_samples = []
    for start, end in zip(times[:-1], times[1:]):
        interval_samples.append(pitch_scape[start, end])
    interval_samples = np.array(interval_samples)
    if normalise_observations:
        interval_samples /= interval_samples.sum(axis=1, keepdims=True)
    # return
    return interval_samples, scape_samples, colours, times


def load_music(n_samples,
               file_paths,
               prior_counts=1.,
               parallel=False
):
    # sort files
    file_paths = list(sorted(file_paths))
    # load data from files
    if parallel:
        with Pool(len(file_paths)) as p:
            data_list = p.starmap(load_scape, [(file_path, prior_counts, n_samples) for file_path in file_paths])
    else:
        data_list = map(load_scape, file_paths, [prior_counts] * len(file_paths), [n_samples] * len(file_paths))
    # combine data from different files into arrays
    (observations,
     scapes,
     colours,
     times) = [np.moveaxis(np.array(data), source=0, destination=1) for data in zip(*data_list)]
    # get file name (same order as data)
    collected_file_names = [os.path.basename(p) for p in file_paths]
    # return as music data object
    return MusicData(observations=observations, scapes=scapes, colours=colours, times=times, names=collected_file_names)
