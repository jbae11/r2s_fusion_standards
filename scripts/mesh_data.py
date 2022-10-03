import copy
import numpy as np


class mesh:
    def __init__(self, which):
        """
        Strictly follows the g,z,y,x convention for cartesian mesh

        units:
        g: MeV
        dimensions: cm
        """

        self.supported_which = ['orcs', 'shift']
        if which not in self.supported_which:
            raise ValueError('Format %s not supported. Supported formats are: [%s]' %','.join(self.supported_which))

        self.dims = ['g', 'z', 'y', 'x']
        # standardized data format for no confusion
        # i hope.
        self.meta_dict = {}
        for dim in self.dims:
            self.meta_dict['n%s' %dim] = 0
            self.meta_dict['mesh_%s' %dim] = []
        self.arr = np.zeros((1,1,1,1))
        self.re = np.zeros_like(self.arr)
        self.particle_type = ''


    def get_shape(self):
        """
        Returns the shape tuple with number of mesh groups (g,z,y,x)
        """
        self.check_meta_dict()
        self.shape = (self.meta_dict['ng'], self.meta_dict['nz'],
                      self.meta_dict['ny'], self.meta_dict['nx'])
        # for convenience
        self.shape3d = self.shape[1:]
        # get them as attributes too
        for k,v in self.meta_dict.items():
            setattr(self, k, v)

        return self.shape


    def get_num_vals_from_mesh(self):
        for i in self.dims:
            mesh = self.meta_dict['mesh_%s' %i]
            assert(len(mesh) != 0)
            self.meta_dict['n%s' %i] = len(mesh) - 1

    def get_mesh_volumes(self):
        self.check_meta_dict()
        vol_arr = np.zeros(self.arr.shape[1:])
        diff_dict = {}
        for dim in self.dims[1:]:
            diff_dict[dim] = np.diff(self.meta_dict['mesh_%s' %dim])
        # take a slice
        for zindx in range(vol_arr.shape[0]):
            vol_arr[zindx, :, :] = diff_dict['z'][zindx] * np.multiply(diff_dict['x'], diff_dict['y'])
        self.vol_arr = vol_arr
        return vol_arr


    def check_meta_dict(self):
        for dim in self.dims:
            assert(self.meta_dict['n%s' %dim] != 0)
            assert(len(self.meta_dict['mesh_%s' %dim]) != 0)
            assert(len(self.meta_dict['mesh_%s' %dim]) == self.meta_dict['n'+dim]+1)
        assert(self.particle_type in ['n', 'p'])


    def check_value_shapes(self):
        # check if initialized:
        assert(self.arr.shape != (1,1,1,1))
        self.check_meta_dict()

        for indx, val in enumerate(self.dims):
            assert(self.arr.shape[indx] == self.meta_dict['n%s' %val])

        for i in self.dims[1:]:
            assert(len(self.meta_dict['mesh_%s' %i]) == self.meta_dict['n%s' %i]+1)

        return True


    def export_vtk(self, outfile_path, title='', group_collapse=True):
        """
        Brute force way to write a vtk file
        """

        f = open(outfile_path, 'w')
        f.write('# vtk DataFile Version 2.0\n')
        if title:
            f.write(title + '\n')
        else:
            title = 'mesh'
            f.write('%s Mesh data \n' %(outfile_path.replace('.vtk', '')))
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')
        
        # now points
        num_points = 1
        for dim in self.dims[1:]:
            num_points = num_points * len(self.meta_dict['mesh_'+dim])
        print('Writing %s Points..' %int(num_points))
        f.write('POINTS %s double\n' %int(num_points))
        # list all them points
        # increase x first, then y then z
        point_indx_arr = np.zeros((len(self.meta_dict['mesh_x'])+1,
                                     len(self.meta_dict['mesh_y'])+1,
                                     len(self.meta_dict['mesh_z'])+1,
                                     ))
        point_list = []
        counter = 0
        for zindx, zval in enumerate(self.meta_dict['mesh_z']):
            for yindx, yval in enumerate(self.meta_dict['mesh_y']):
                for xindx, xval in enumerate(self.meta_dict['mesh_x']):
                    point = [xval, yval, zval]
                    point_list.append(point)
                    f.write(' '.join([str(q) for q in point]) + '\n')
                    point_indx_arr[xindx, yindx, zindx] = int(counter)
                    counter += 1

        # now cells
        num_cells = 1
        for dim in self.dims[1:]:
            num_cells = num_cells * self.meta_dict['n'+dim]
        print('Writing %s Cells..' %int(num_cells))
        # multiplied by nine because
        # there's 9 values in a line?? #!
        f.write('CELLS %s %s\n' %(int(num_cells), int(num_cells*9)))
        # for each mesh cell
        # shorter
        mx = self.meta_dict['mesh_x']
        my = self.meta_dict['mesh_y']
        mz = self.meta_dict['mesh_z']
        for zindx_, zval in enumerate(mz[:-1]):
            for yindx_, yval in enumerate(my[:-1]):
                for xindx_, xval in enumerate(mx[:-1]):
                    ref_point = [xval, yval, zval]
                    # get all points to make that cell
                    # move +x, +y, -x, -y+z, +x, +y, -x
                    xindx = copy.deepcopy(xindx_)
                    yindx = copy.deepcopy(yindx_)
                    zindx = copy.deepcopy(zindx_)  
                    points = [point_indx_arr[xindx, yindx, zindx]]
                    xindx += 1
                    points.append(point_indx_arr[xindx, yindx, zindx])
                    yindx += 1
                    points.append(point_indx_arr[xindx, yindx, zindx])
                    xindx -= 1
                    points.append(point_indx_arr[xindx, yindx, zindx])
                    yindx -= 1
                    zindx += 1
                    points.append(point_indx_arr[xindx, yindx, zindx])
                    xindx += 1
                    points.append(point_indx_arr[xindx, yindx, zindx])
                    yindx += 1
                    points.append(point_indx_arr[xindx, yindx, zindx])
                    xindx -= 1
                    points.append(point_indx_arr[xindx, yindx, zindx])
                    f.write('8 ' + ' '.join([str(int(q)) for q in points]) + '\n')


        # cell values
        print('Writing %s Cell Data' %(int(num_cells)))
        f.write('CELL_TYPES %s\n' %(int(num_cells)))
        for i in range(int(num_cells)):
            f.write('12\n')
        f.write('CELL_DATA %s\n' %(int(num_cells)))
        collapsed_arr = np.sum(self.arr, axis=0)
        f.write('SCALARS %s_total double 1\n' %title)
        f.write('LOOKUP_TABLE default\n')
        for zindx_, zval in enumerate(mz[:-1]):
            for yindx_, yval in enumerate(my[:-1]):
                for xindx_, xval in enumerate(mx[:-1]):
                    val = collapsed_arr[zindx_, yindx_, xindx_]
                    f.write(str(val) + '\n')

        # write them energy-dependent ones
        if not group_collapse:
            for g in range(self.arr.shape[0]):
                print('Writing energy group %s/%s' %(g, self.arr.shape[0]))
                f.write('SCALARS %s_g_%s double 1\n' %(title,g))
                f.write('LOOKUP_TABLE default\n')
                for zindx_, zval in enumerate(mz[:-1]):
                    for yindx_, yval in enumerate(my[:-1]):
                        for xindx_, xval in enumerate(mx[:-1]):
                            val = self.arr[g, zindx_, yindx_, xindx_]
                            f.write(str(val) + '\n')

        print('Finished writing vtk file at %s' %outfile_path)    


    def compare_arrs(self, l1, l2):
        # flatten just in case
        l1 = np.array(l1).flatten()
        l2 = np.array(l2).flatten()
        if len(l1) != len(l2):
            return False
        for indx, val in enumerate(l1):
            if val != l2[indx]:
                return False

        return True


    def compare(self, mesh_data_):
        for k in self.meta_dict.keys():
            assert(self.compare_arrs(self.meta_dict[k], mesh_data_.meta_dict[k]))
        assert(self.compare_arrs(self.arr, mesh_data_.arr))
        return True