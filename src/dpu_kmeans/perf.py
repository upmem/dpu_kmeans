# -*- coding: utf-8 -*-

# Authors: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

host_reduction_timer = 0
host_quantize_timer = 0
host_unquantize_timer = 0
host_average_timer = 0
host_c_timer = 0
alloc_dealloc_timer = 0


def reset_timers():
    global host_reduction_timer, host_quantize_timer, host_unquantize_timer, host_average_timer, host_c_timer, alloc_dealloc_timer, host_reduction_timer
    host_quantize_timer = 0
    host_unquantize_timer = 0
    host_average_timer = 0
    host_c_timer = 0
    alloc_dealloc_timer = 0
