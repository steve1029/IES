FROM cupy/cupy:latest

ENTRYPOINT ["echo", "hello"]
RUN apt update
RUN apt install python3-numpy python3-psutils python3-matplotlib python3-scipy python3-h5py ipython3 python3-pandas python3-pip python3-mpi4py git vim
CMD /bin/bash
