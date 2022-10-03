import numpy as np
import os
import struct
import h5py
from mesh_data import mesh

# STILL UNDER DEVELOPMENT

class weight_window_mesh_data(mesh):
    def __init__(self, filepath, which):
        super().__init__(which)

        """
        arr units: lower bound weight windows

        For ORCS, point to the `wwinp` file

        For Shift, point to the HDF5 output (.out.h5)
        """

        # needed for orcs binary file read/write
        # integer and double
        self.dsize = {'i': 4, 'd': 8}

        if which == 'orcs':
            self.read_orcs(filepath)
        if which == 'shift':
            self.read_shift(filepath)

        # weight window specific values
        self.beta = None
        self.norm_factor = None


    def num_line_split(self, line, dtype=float):
        return [dtype(q) for q in line.strip().split()]


    def read_orcs(self, wwinp_path):
        assert(os.path.exists(wwinp_path))
        lines = open(wwinp_path, 'r').readlines()
        lines_it = iter(lines) # iterator, not ITER
        # hard coded ascii stuff here
        line1 = self.num_line_split(next(lines_it), int)
        # third element should be 2 - for two species in wwinp file
        assert(line1[2] == 2)
        # fourth element should be 10 - for cartesian mesh
        assert(line1[3] == 10)

        # now reading number of energy groups
        line2 = self.num_line_split(next(lines_it), int)
        # make sure one of it is zero
        assert(len(line2) == 2)
        assert(line2[0] * line2[1] == 0)
        assert(sum(line2) != 0)
        # get num_groups
        if line2[0] != 0:
            assert(line2[1] == 0)

            self.particle_type = 'n'
        else:
            self.meta_dict['ng'] = line2[1]
            self.particle_type = 'p'


        # now mesh numbers
        line3 = self.num_line_split(next(lines_it), float)
        for indx, i in enumerate(['nx', 'ny', 'nz']):
            self.meta_dict[i] = int(line3[indx])

        # another mesh number?
        line4 = self.num_line_split(next(lines_it), float)
        for indx, i in enumerate(['nx', 'ny', 'nz']):
            # just check?
            assert(self.meta_dict[i] == int(line4[indx]))


        val_per_line = 6
        # now it's block 2
        # mesh grids
        # it's always 1 in between mesh values
        for dim in ['x', 'y', 'z']:
            mesh_grids = []
            expected_num = self.meta_dict['n'+dim] * 3 + 1
            num_lines = int(np.ceil(expected_num/val_per_line))
            for i in range(num_lines):
                tmp = self.num_line_split(next(lines_it), float)
                mesh_grids.extend(tmp)
                if len(mesh_grids) == expected_num:
                    assert(i == num_lines-1)
            # now filter out them ones
            mesh_grids = [mesh_grids[0]] + mesh_grids[2::3]
            assert(len(mesh_grids) == self.meta_dict['n'+dim] + 1)
            self.meta_dict['mesh_'+dim] = np.array(mesh_grids)

        # now it's block 3
        # values
        expected_num = self.meta_dict['ng']
        e_bounds = []
        num_lines = int(np.ceil(expected_num/val_per_line))
        for i in range(num_lines):
            e_bounds.extend(self.num_line_split(next(lines_it), float))
        # it should be this, but one is missing.
        e_bounds = [0] + e_bounds
        # perhaps an upper bound energy?
        assert(len(e_bounds) == self.meta_dict['ng'] + 1) 
        self.meta_dict['mesh_g'] = np.array(e_bounds)

        # now read that data
        shape = self.get_shape()

        # 2d array to begin with
        tmp_2d = np.zeros((shape[0], shape[1]*shape[2]*shape[3]))
        num_lines = int(np.ceil(tmp_2d.shape[1]/val_per_line))
        for g in range(tmp_2d.shape[0]):
            tmp = []
            for l in range(num_lines):
                tmp.extend(self.num_line_split(next(lines_it), float))
            assert(len(tmp) == tmp_2d.shape[1])
            tmp_2d[g,:] = tmp


        # now rearrange to 4d
        print(shape)
        self.arr = np.zeros(shape)
        for g in range(shape[0]):
            for i in range(shape[1]):
                for j in range(shape[2]):
                    for k in range(shape[3]):
                        indx = k + (shape[3] * j) + (shape[2] * shape[3] * i)
                        self.arr[g,i,j,k] = tmp_2d[g, indx]


        # if we did everything right,
        # another next() should throw an error:
        try:
            next(lines_it)
            # if that worked, we in the wrong
            raise ValueError('File was not read completely.')
        except StopIteration:
            # we're good
            print('Finished reading WWINP file at %s.' %wwinp_path)



    def read_shift(self, filepath):
        f = h5py.File(filepath, 'r')
        assert('hybrid' in f.keys())
        hybrid = f['hybrid']['ww']
        self.meta_dict = {'mesh_g': np.array(hybrid['group_bounds_n'])}
        for dim in ['x','y','z']:
            self.meta_dict['mesh_'+dim] = np.array(hybrid['mesh_'+dim])
        for dim in ['g', 'x', 'y', 'z']:
            self.meta_dict['n'+dim] = len(self.meta_dict['mesh_'+dim]) - 1

        #! 
        lower_factor = float(np.array(hybrid['lower_factor']))
        upper_factor = float(np.array(hybrid['upper_factor']))
        self.arr = np.array(hybrid['adj_flux']) * lower_factor

        self.norm_factor = float(np.array(hybrid['normalization']))
        self.beta = upper_factor / lower_factor
        # get particle type
        physics_keys = [q for q in f.keys() if 'physics-' in q]
        # mg and ce physics
        assert(len(physics_keys) in [1,2])
        particle_type = []
        for i in physics_keys:
            mode = str(f[i]['db']['mode'].asstr()[...])
            particle_type.append(mode)
        # make sure they all say the same thing
        assert(len(set(particle_type)) == 1)
        self.particle_type = particle_type[0]


        print('Finished reading Shift HDF5 file from %s.' %filepath)


    # auxiliary functions 
    def list_of_float_to_string(self, l):
        val_per_line = 6
        strl = ['{:.5e}'.format(q) for q in l]
        is_neg = [q<0 for q in l]
        rs = ''
        for indx, val in enumerate(strl):
            if is_neg[indx]:
                rs += ' ' + val
            else:
                rs += '  ' + val
            if (indx+1)%val_per_line == 0:
                # skip if we are at the end of the list
                if indx == len(strl)-1:
                    continue
                rs += '\n'
        return rs


    def list_of_int_to_string(self, l):
        assert(len(l) < 5)
        strl = [str(int(q)) for q in l]
        rs = ''
        for val in strl:
            spaces = 10 - len(val)
            rs += ' '*spaces + val
        return rs


    def write_orcs(self, outfile_path):
        self.check_value_shapes()
        assert not (os.path.exists(outfile_path))
        f = open(outfile_path, 'w')
        # hardcoded madness
        # line 1: if iv ni nr
        # if is always 1
        # iv is not used (what)
        # ni is the number of species (n, g, ...) in the wwinp file
        # nr is 10 for rectangular meshes
        f.write(self.list_of_int_to_string([1, 1, 2, 10]) + '\n')
        if self.particle_type == 'n':
            l = [self.meta_dict['ng'], 0]
        else:
            l = [0, self.meta_dict['ng']]
        # line 2: number of groups
        f.write(self.list_of_int_to_string(l) + '\n')

        # line 3: nfx, nfy, nfz, x0, y0, z0
        l = []
        for dim in ['x','y','z']:
            l.append(self.meta_dict['n'+dim])
        for dim in ['x','y','z']:
            l.append(self.meta_dict['mesh_'+dim][0])
        assert(len(l) == 6)
        f.write(self.list_of_float_to_string(l) + '\n')

        # line 4: ncx, ncy, ncz, nwg
        # nwg = 1 for cartesian mesh
        l = [] 
        for dim in ['x','y','z']:
            l.append(self.meta_dict['n'+dim])
        l.append(1.0)
        f.write(self.list_of_float_to_string(l) + '\n')

        # block 2
        # for some ungodly reason there's a 1 between
        # each mesh boundaries
        for dim in ['x', 'y', 'z']:
            l = [self.meta_dict['mesh_'+dim][0]]
            for val in self.meta_dict['mesh_'+dim][1:]:
                l.extend([1.0, val, 1.0])
            f.write(self.list_of_float_to_string(l) + '\n')

        # block 3 
        # group bounds and array
        f.write(self.list_of_float_to_string(self.meta_dict['mesh_g'][1:]) + '\n')
        # then flatten out the 4d array to write
        shape = self.get_shape()
        for g in range(shape[0]):
            num_voxels = shape[1] * shape[2] * shape[3]
            l = np.zeros(num_voxels)
            for i in range(shape[1]):
                for j in range(shape[2]):
                    for k in range(shape[3]):
                        indx = k + (shape[3] * j) + (shape[2] * shape[3] * i)
                        l[indx] = self.arr[g,i,j,k]
            # write
            f.write(self.list_of_float_to_string(l) + '\n')


        print('Finished writing WWINP file to %s.' %outfile_path)
        print(self.beta)
        print(self.norm_factor)
        if self.beta and self.norm_factor:
            print('Identified normalization factor and beta')
            print('wwp:%s %s j 100 j -1 0 %s' %(self.particle_type, self.beta, self.norm_factor))
        else:
            print('No normalization factor and beta reported, edit from:')
            print('wwp:%s 5.0 j 100 j -1 0 1.0')



    def write_shift(self, filepath):
        dt = float
        self.check_value_shapes()
        assert not (os.path.exists(filepath))
        f = h5py.File(filepath, 'w')
        grp = f.create_group('hybrid')
        grp2 = grp.create_group('ww')
        #! 
        grp2.create_dataset('adj_flux', data=self.arr / 0.5, dtype=dt)
        grp2.create_dataset('group_bounds_n', data=self.meta_dict['mesh_g'], dtype=dt)
        # ratio of WW upper bound to lower bound
        beta = 5
        
        for dim in ['x','y','z']:
            grp2.create_dataset('mesh_'+dim, data=self.meta_dict['mesh_'+dim], dtype=dt)
        #! 
        grp2.create_dataset('lower_factor', data=0.5, dtype=dt)
        grp2.create_dataset('upper_factor', data=2.5, dtype=dt)
        grp2.create_dataset('normalization', data=1, dtype=dt)
        f.close()

        print('Finished writing Shift HDF5 file to %s.' %filepath)


#obj = weight_window_mesh_data('./orcs_files/mscadis_wwinp', 'orcs')
#print(obj.meta_dict)
