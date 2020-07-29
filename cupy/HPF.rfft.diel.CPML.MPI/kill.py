import os

procname = "python3"
#cmd = "ps -elf | grep '%s' | awk '{print $4}' | xargs kill"

cmd = "kill -9 `ps -ef | grep %s | grep -v grep | awk '{print $2}'`"

os.system(cmd % procname)
