import subprocess as sp
import sys
import numpy as np

filename  = sys.argv[1]
#exe_times = sys.argv[2]
#exe_times = int(exe_times)

wvsrt = int(sys.argv[2])
wvend = int(sys.argv[3])
interval = int(sys.argv[4])
tsteps = int(sys.argv[5])

def exe_wait(cmd):
    out, err = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
    if err != None: print(err)

    return out

if __name__ == '__main__':

    import datetime as dt

    """
    print("{0} will be executed {1} times" .format(filename,exe_times))

    for i in range(exe_times):
        out = exe_wait('mpirun.openmpi -host yboot,y205,y206,y207,y208 python3 %s' %(filename))
        if out == None: raise Exception
        else : print("{0}/{1} finished" .format(i+1,exe_times))
    """

    wvlens = np.arange(wvsrt, wvend+1, interval)

    print("{} will be executed from {} nm to {} nm." .format(filename, wvlens[0], wvlens[-1]))

    for i, value in enumerate(wvlens):

        #out = exe_wait('mpirun.openmpi -host yboot,y205,y206,y207,y208 python3 %s' %(filename))
        out = exe_wait('python3 {} {} {}' .format(filename, value, tsteps))
        if out == None: raise Exception
        else : print("{0}/{1} finished" .format(i+1, len(wvlens)))
