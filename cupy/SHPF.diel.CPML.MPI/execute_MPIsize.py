import os, sys

target_file	= sys.argv[1]
node_num_str= int(sys.argv[2]) - 1
node_num_end= int(sys.argv[3]) 

node_names = ["y205","y206","y207","y208","y210","y211","y212"]

for node_num in range(node_num_str, node_num_end):

	cal_nodes = "y213"

	for i in range(node_num):

		cal_nodes += "," + node_names[i]

	cmd = "mpirun.openmpi -host %s python3 %s" %(cal_nodes, target_file)

	print(cmd)
	os.system(cmd)
