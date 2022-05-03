# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:47:24 2020

@author: finup
"""

import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class time_consume:
    def __init__(self):
        self._start_time = None
    
    def start(self):
        if self._start_time:
            raise TimeoutError("Timer is running.")            
        self._start_time = time.perf_counter()
    
    def stop(self):
        if not self._start_time:
            raise TimeoutError("Timer is not running.")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print('Elapsed time: {0:.2f} seconds.'.format(elapsed_time))



