import numpy as np
import os
import struct
import h5py
from mesh_data import mesh


class gamma_source_mesh_data(mesh):
    def __init__(self, filepath, which):
        super().__init__(which)

        """
        arr units: emissions/second (source INTENSITY)
        energy: MeV

        For ORCS, point to the directory with the
        `msx_source_g` files


        For Shift, point to the HDF5 used for 4D source sampling
        """

        # needed for orcs binary file read/write
        # integer and double
        self.dsize = {'i': 4, 'd': 8}

        if which == 'orcs':
            self.read_orcs(filepath)
        if which == 'shift':
            self.read_shift(filepath)




    def read_orcs(self, workdir):
        # ORCS reports emissions/second-cc

        files = os.listdir(workdir)
        files = [q for q in files if 'msx_source_g' in q]
        files = [q for q in files if '.zip' not in q]
        # check if the files are all there
        nums = [int(q.replace('msx_source_g', '')) for q in files]
        self.particle_type = 'p'
        self.meta_dict['ng'] = max(nums) # + 1
        # check if each file exists
        for g in range(self.meta_dict['ng']):
            assert('msx_source_g'+str(g) in files)

        initialized = False
        print(self.meta_dict['ng'])

        for g in range(self.meta_dict['ng']+1):
            with open(os.path.join(workdir, 'msx_source_g'+str(g)), 'rb') as f:
                # has to be read sequentially cause it's a binary file

                # read dimensions
                # this hardcoded cause the order is different :(
                keys = ['g', 'ng', 'nx', 'ny', 'nz']
                dt = 'i'
                for indx, val in enumerate(keys):
                    b = f.read(self.dsize[dt])
                    tmp = struct.unpack(dt, b)[0]
                    if val != 'g':
                        self.meta_dict[val] = tmp
                    else:
                        assert(tmp == g)

                if not initialized:
                    self.arr = np.zeros((self.meta_dict['ng'],
                                        self.meta_dict['nz'],
                                        self.meta_dict['ny'],
                                        self.meta_dict['nx'])
                                        )
                    initialized = True
                # now read mesh bounds
                keys = ['mesh_g', 'mesh_x', 'mesh_y', 'mesh_z']
                dt = 'd'
                for indx, val in enumerate(keys):
                    len_ = self.meta_dict[val.replace('mesh_', 'n')] + 1 # cause its mesh bounds
                    temp_arr = []
                    for i in range(len_):
                        b = f.read(self.dsize[dt])
                        tmp = struct.unpack(dt, b)[0]
                        temp_arr.append(tmp)
                    self.meta_dict[val] = np.array(temp_arr)

                self.get_num_vals_from_mesh()
                self.get_shape()

                # now read the source
                for i in range(self.shape3d[0]):
                    for j in range(self.shape3d[1]):
                        for k in range(self.shape3d[2]):
                            b = f.read(self.dsize[dt])
                            tmp = struct.unpack(dt, b)[0]
                            self.arr[g,i,j,k] = tmp

        # multiply by mesh volumes
        vol_arr = self.get_mesh_volumes()
        for gindx in range(self.arr.shape[0]):
            self.arr[gindx] = np.multiply(self.arr[gindx], vol_arr)

        print('Finished reading ORCS file from %s.' %workdir)


    def read_shift(self, filepath):
        f = h5py.File(filepath, 'r')
        # eV to MeV
        self.particle_type = 'p'
        self.meta_dict['mesh_g'] = np.array(f['group_bounds']) * 1e-6
        self.meta_dict['ng'] = len(self.meta_dict['mesh_g']) - 1
        for d in ['x','y','z']:
            self.meta_dict['mesh_%s'%d] = np.array(f['mesh_%s' %d])
            self.meta_dict['n%s'%d] = len(self.meta_dict['mesh_%s' %d]) - 1
        self.arr = np.array(f['pdf']) # it's already source intensity

        print('Finished reading Shift HDF5 file from %s.' %filepath)



    def write_orcs(self, workdir):
        self.check_value_shapes()
        # make sure there are no existing files
        ld = os.listdir(workdir)
        for g in range(self.meta_dict['ng']):
            filename = 'msx_source_g'+str(g)
            if filename in ld:
                raise ValueError('File %s already exists in path %s.' %(filename, workdir))
        order = ['g', 'x', 'y', 'z']

        vol_arr = self.get_mesh_volumes()

        for g in range(self.meta_dict['ng']):
            f = open(os.path.join(workdir, 'msx_source_g'+str(g)), 'wb')
            dt = 'i'
            f.write(struct.pack(dt, g))
            # write number of values
            for n in order:
                f.write(struct.pack(dt, self.meta_dict['n'+n]))
            # write mesh bounds
            dt = 'd'
            for n in order:
                for v in self.meta_dict['mesh_%s' %n]:
                    f.write(struct.pack(dt, v))
            # now write array
            for i in range(self.meta_dict['nz']):
                for j in range(self.meta_dict['ny']):
                    for k in range(self.meta_dict['nx']):
                        # divide by volume
                        f.write(struct.pack(dt, self.arr[g,i,j,k] / vol_arr[i,j,k]))
            f.close()

        # write total source intensity
        with open(os.path.join(workdir, 'total_source_intensity.txt'), 'w') as f:
            f.write('{:.7e}'.format((sum(self.arr.flatten()))))
            f.write(' photons/s')

        print('Finished writing ORCS file to %s.' %workdir)


    def write_shift(self, filepath):
        self.check_value_shapes()
        f = h5py.File(filepath, 'w')
        dt = float
        f.create_dataset('group_bounds', data=self.meta_dict['mesh_g']*1e6, dtype=dt)
        for d in ['x', 'y', 'z']:
            f.create_dataset('mesh_%s' %d, data=self.meta_dict['mesh_%s' %d], dtype=dt)
        # 1 is gamma
        f.create_dataset('particle_type', data=1)
        f.create_dataset('pdf', data=self.arr, dtype=dt)
        f.close()

        # write total source intensity
        with open(os.path.join(os.path.dirname(filepath), 'total_source_intensity.txt'), 'w') as f:
            f.write('{:.7e}'.format((sum(self.arr.flatten()))))
            f.write(' photons/s')

        print('Finished writing Shift HDF5 file to %s.' %filepath)


#obj = gamma_source_mesh_data('/home/4ib/git/r2s_orcs_pyne_scripts/workdir/shift_gamma_source.h5',
#                              'shift')
#obj.write_shift('/home/4ib/git/r2s_orcs_pyne_scripts/workdir/reproduced_shift.h5')
#obj.write_orcs('/home/4ib/git/r2s_orcs_pyne_scripts/workdir')
# obj = gamma_source_mesh_data('/home/4ib/git/r2s_orcs_pyne_scripts/workdir/', 'orcs')
# obj2 = gamma_source_mesh_data('/home/4ib/git/r2s_orcs_pyne_scripts/workdir/reproduced_shift.h5', 'shift')
# obj3 = gamma_source_mesh_data('/home/4ib/git/r2s_orcs_pyne_scripts/workdir/shift_gamma_source.h5',
#                               'shift')
