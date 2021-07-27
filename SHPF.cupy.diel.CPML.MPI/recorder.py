import os, psutil, datetime
import space

class Recorder:

    def __init__(self, space, start_time, savedir):

        if space.MPIrank == 0:

            self.savedir = savedir
            self.space = space

            # Simulation finished time
            finished_time = datetime.datetime.now()

            # Record simulation size and operation time
            if not os.path.exists(self.savedir) : os.makedirs(self.savedir)
            record_path = self.savedir+"record/record_%s.txt" %(datetime.date.today())

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

            if space.dimension == 3:
                f.write("{:2d}\t\t{:04d}\t{:04d}\t{:04d}\t{:5.2e}\t{:5.2e}\t{:5.2e}\t{:06d}\t\t{}\t\t{:>6}\t\t{:06.3f}\t\t\t{:06.3f}\n" \
                            .format(space.MPIsize, space.Nx, space.Ny, space.Nz,\
                                space.dx, space.dy, space.dz, space.tsteps, cal_time, space.method, me_vmsmem_GB, me_rssmem_GB))
            if space.dimension == 2:
                f.write("{:2d}\t\t{:04d}\t{:04d}\t{:4}\t{:5.2e}\t{:5.2e}\t{:>8}\t{:06d}\t\t{}\t\t{:>6}\t\t{:06.3f}\t\t\t{:06.3f}\n" \
                            .format(space.MPIsize, space.Nx, space.Ny, 'None',\
                                space.dx, space.dy, 'None', space.tsteps, cal_time, space.method, me_vmsmem_GB, me_rssmem_GB))
            f.close()
            
            print("Simulation specifications are recorded. {}".format(datetime.datetime.now()))

        else: pass # The other nodes do nothing.


class History:

    def __init__(self, space, savedir):
        """Record simulation info and progress

        Parameters
        ----------
        space: space object.

        savedir: str.

        Returns
        -------
        None
        """

        if space.MPIrank == 0:

            space = self.space
            self.savedir = savedir
            today = datetime.datetime.now().strftime('%Y%m%d')
            start_time = today.strftime('%H%M%S')
            #finished_time = finished_time.strftime('%H%M%S')
            history_path = self.savedir+f'history/{today}_{start_time}_{name}.txt'

            if os.path.exists(history_path) == False: os.makedirs(history_path)

            else:
                history_path = self.savedir+f'history/{today}_{start_time}_{finished_time}_{name}_2.txt'


            f = open(history_path,'a')
            f.write(f"VOLUME of the space: {space.VOLUME:.2e}")
            f.write(f"Size of the space: {int(space.Lx/space.nm):04d} x {int(space.Ly/space.nm):04d} x {int(space.Lz/space.nm):04d}")
            f.write(f"Number of grid points: {space.Nx:5d} x {space.Ny:5d} x {space.Nz:5d}")
            f.write(f"Grid spacing: {space.dx/space.nm:.3f} nm, {space.dy/space.nm:.3f} nm, {space.dz/space.nm:.3f} nm\n")
            f.close()

        else: pass # The other nodes do nothing.
