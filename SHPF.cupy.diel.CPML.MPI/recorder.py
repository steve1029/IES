import os, psutil, datetime
import space

class Recorder:

    def __init__(self, space, start_time, savedir):

        self.savedir = savedir

        # Simulation finished time
        finished_time = datetime.datetime.now()

        # Record simulation size and operation time
        if not os.path.exists(self.savedir) : os.mkdir(self.savedir)
        record_path = self.savedir+"record_%s.txt" %(datetime.date.today())

        if not os.path.exists(record_path):
            f = open( record_path,'a')
            f.write("{:4}\t{:4}\t{:4}\t{:4}\t{:4}\t\t{:4}\t\t{:4}\t\t{:8}\t{:4}\t\t\t\t{:6}\t{:12}\t{:12}\n\n" \
                .format("Node","Nx","Ny","Nz","dx","dy","dz","tsteps","Time","Method", "VM/Node(GB)","RM/Node(GB)"))
            f.close()

        me = psutil.Process(os.getpid())
        me_rssmem_GB = float(me.memory_info().rss)/1024/1024/1024
        me_vmsmem_GB = float(me.memory_info().vms)/1024/1024/1024

        cal_time = finished_time - start_time
        f = open( record_path,'a')
        f.write("{:2d}\t\t{:04d}\t{:04d}\t{:04d}\t{:5.2e}\t{:5.2e}\t{:5.2e}\t{:06d}\t\t{}\t\t{:>6}\t\t{:06.3f}\t\t\t{:06.3f}\n" \
                    .format(space.MPIsize, space.Nx, space.Ny, space.Nz,\
                        space.dx, space.dy, space.dz, space.tsteps, cal_time, space.method, me_vmsmem_GB, me_rssmem_GB))
        f.close()
        
        print("Simulation specifications are recorded. {}".format(datetime.datetime.now()))
