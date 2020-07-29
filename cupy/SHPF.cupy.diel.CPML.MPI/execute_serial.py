import subprocess as sp
import sys

filename  = sys.argv[1]
exe_times = sys.argv[2]

exe_times = int(exe_times)

def exe_wait(cmd):
	out, err = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
	if err != None: print(err)

	return out

if __name__ == '__main__':

	import datetime as dt

	print("{0} will be executed {1} times" .format(filename,exe_times))
	for i in range(exe_times):
		out = exe_wait('mpirun.openmpi -host yboot,y205,y206,y207,y208 python3 %s' %(filename))
		if out == None: raise Exception
		else : print("{0}/{1} finished" .format(i+1,exe_times))
