import numpy as np
import os
from mesh_data import mesh
obj = mesh('orcs')
x = np.linspace(0, 12, 4)
y = np.linspace(-12, 0, 4)
z = np.linspace(20, 32, 4)
print(x)
print(y)
print(z)
g = np.linspace(0, 3, 4)
val2 = np.zeros((len(g)-1, len(z)-1, len(y)-1, len(x)-1, 2, 2))
#for g_ in range(g):
for i in range(val2.shape[0]):
    for j in range(val2.shape[1]):
        for k in range(val2.shape[2]):
            # val[g_, i,j,k] = k + 10*j + 100*i + 1000*g
            for n in range(2):
                for n2 in range(2):
                    val2[0, i,j,k, n, n] = k + 10*j + 100*i
obj.meta_dict['mesh_x'] = x
obj.meta_dict['mesh_y'] = y
obj.meta_dict['mesh_z'] = z
obj.meta_dict['nx'] = len(x) - 1
obj.meta_dict['ny'] = len(y) - 1
obj.meta_dict['nz'] = len(z) - 1

obj.arr = val2[:, :,:,:, 0, 0]
obj.export_vtk('test.vtk')

from neutron_mesh_tally import neutron_mesh_tally_data
obj = neutron_mesh_tally_data('orcs_files/merged_meshtal.h5', 'orcs')
obj.export_vtk('orcs_n_meshtal.vtk', title='neutron_flux', group_collapse=False)