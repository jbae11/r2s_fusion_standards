import numpy as np
import h5py
from mesh_data import mesh

class neutron_mesh_tally_data(mesh):

    def __init__(self, filepath, which, mesh_tally_name='mesh_tally'):

        super().__init__(which)
        
        """
        arr units: n/(cm2-sourceparticle)
        energy units: MeV

        """

        if which == 'orcs':
            self.read_orcs(filepath)
        elif which == 'shift':
            self.read_shift(filepath, mesh_tally_name)

        
    def read_orcs(self, filepath):
        f = h5py.File(filepath, 'r')
        key = list(f.keys())
        assert(len(key) == 1)
        key = key[0]
        f = f[key]

        # check file 
        for ds in ['enbin', 'total', 'total_re', 'xrbin', 'yzbin', 'ztbin']:
            assert(ds in f.keys())
        #! could be cartesian, but lets not think about it for now
        self.meta_dict = {'mesh_g': np.array(f['enbin']),
                          'mesh_x': np.array(f['xrbin']),
                          'mesh_y': np.array(f['yzbin']),
                          'mesh_z': np.array(f['ztbin'])
                        }
        # get nps
        self.nps = int(f.attrs['nps'])
        self.particle_type = 'n'
        self.get_num_vals_from_mesh()
        self.get_shape()


        # build 4d value array
        self.arr = np.zeros(self.shape)
        self.re = np.zeros_like(self.arr)
        for num in range(self.ng - 1):
            numstr = self.get_num_zero_string(num)
            assert('mean_%s' %numstr in f.keys())
            assert('re_%s' %numstr in f.keys())
            self.arr[num, :, :, :] = np.array(f['mean_%s' %numstr])
            self.re[num, :, :, :] = np.array(f['re_%s' %numstr])
    
        print('Finished reading ORCS file %s.' %filepath)
        return

    def read_shift(self, filepath, mesh_tally_name):
        # shift energy is in eV
        f = h5py.File(filepath, 'r')
        # get nps
        self.nps = f['shift']['db']['SOURCE']['Np']
        f = f['tally']
        assert(mesh_tally_name in f.keys())
        f = f[mesh_tally_name]
        self.meta_dict = {'mesh_g': np.array(f['group_bounds_n']) * 1e-6,
                          'mesh_x': np.array(f['mesh_x']),
                          'mesh_y': np.array(f['mesh_y']),
                          'mesh_z': np.array(f['mesh_z'])
                         }
        self.particle_type = 'n'
        self.get_num_vals_from_mesh()
        self.get_shape()

        # get 4d value array
        self.arr = np.array(f['binned'])[:,:,:,:,0,0]
        # shift errors are variance
        self.re = np.sqrt(np.array(f['binned'])[:,:,:,:,0,1]) / self.arr
        print('Finished reading Shift file %s.' %filepath)
        return



    def write_orcs(self, outpath):
        self.check_value_shapes()
        f = h5py.File(outpath, 'w')
        dt = float
        grp = f.create_group('444')
        # write attributes
        grp.attrs['icrd'] = 1
        grp.attrs['id'] = 444
        grp.attrs['nenb'] = self.meta_dict['ng'] + 1
        grp.attrs['nps'] = self.nps
        grp.attrs['ipt'] = 1
        grp.attrs['nxrb'] = self.meta_dict['nx'] + 1
        grp.attrs['nyzb'] = self.meta_dict['ny'] + 1
        grp.attrs['nztb'] = self.meta_dict['nz'] + 1
        grp.attrs['tot_energy_bin'] = 1
        print('Dumping all values into tally number 444')
        grp.create_dataset('enbin', data=self.meta_dict['mesh_g'][::-1], dtype=dt)
        grp.create_dataset('xrbin', data=self.meta_dict['mesh_x'], dtype=dt)
        grp.create_dataset('yzbin', data=self.meta_dict['mesh_y'], dtype=dt)
        grp.create_dataset('ztbin', data=self.meta_dict['mesh_z'], dtype=dt)
        grp.create_dataset('total', data=np.sum(self.arr, axis=0), dtype=dt)
        #! wrong
        weighted_re = np.sum(self.re * self.arr, axis=0) / np.sum(self.arr, axis=0)         
        grp.create_dataset('total_re', data=weighted_re, dtype=dt)
        for num in range(self.ng):
            numstr = self.get_num_zero_string(num)
            grp.create_dataset('mean_%s' %numstr, data=self.arr[num, :,:,:], dtype=dt)
            grp.create_dataset('re_%s' %numstr, data=self.re[num, :,:,:], dtype=dt)
        # that's it folks!
        f.close()
        print('Finished writing ORCS HDF5 file to %s.' %outpath)

    
    def write_shift(self, outpath):
        self.check_value_shapes()
        f = h5py.File(outpath, 'w')
        dt = float
        grp = f.create_group('tally')
        subgrp = grp.create_group('mesh_tally')
        binned = np.zeros(self.shape + (1, 2))
        binned[:,:,:,:,0,0] = self.arr
        binned[:,:,:,:,0,1] = (self.re * self.arr)**2
        subgrp.create_dataset('binned', data=binned, dtype=dt)
        subgrp.create_dataset('mesh_x', data=self.meta_dict['mesh_x'], dtype=dt)
        subgrp.create_dataset('mesh_y', data=self.meta_dict['mesh_y'], dtype=dt)
        subgrp.create_dataset('mesh_z', data=self.meta_dict['mesh_z'], dtype=dt)
        subgrp.create_dataset('group_bounds_n', data=self.meta_dict['mesh_g']*1e6, dtype=dt)
        # other bits, we don't really need for R2S
        # that's it folks!
        f.close()
        print('Finished writing Shift HDF5 file to %s.' %outpath)



    # aux functions
    def get_num_zero_string(self, num, length=3):
        num = str(num)
        assert(len(num) <= length)
        return '0'*(length-len(num)) + num
