# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 17:34:24 2016

@author: loves
"""

from pycuda import driver as drv

drv.init()
dev = drv.Device(0)

print "nVIDIA CUDA Device Name:", dev.name()
print "Total Memory:", dev.total_memory()/(1024.**3), "GiB"

for key, value in drv.device_attribute.values.iteritems():
    try:
        print "%03d\t%40s\t%s\t" % (key, value, dev.get_attribute(value))
    except:
        pass
