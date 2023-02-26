#import libraries

import scipy
from scipy.integrate import solve_ivp
import numpy as np
import cmath as cm
import h5py
from numpy.linalg import multi_dot
from scipy.linalg import logm
from scipy.special import factorial
from scipy.special import *
from scipy.sparse import csr_matrix
from numpy.linalg import eig
from scipy.linalg import eig as sceig
import math
import time
#from math import comb
from sympy.physics.quantum.cg import CG
from sympy import S
import collections
import numpy.polynomial.polynomial as poly
from scipy import sparse
from scipy.sparse import csr_matrix

import sys
argv=sys.argv

if len(argv) < 2:
    #Default
    run_id=1
else:
    try:
        run_id = int(argv[1])
        Natoms = int(argv[2])
        cluster_size = int(argv[3])
        print("Cluster size = " + str(cluster_size), flush = True)
        #det_val_input = float(argv[4])

    except:
        print ("Input error")
        run_id=1

# some definitions 

fe = 0
fg = 0

fixed_param = 1 # 0: L = 20, R = 0.5; 1: mean density; 2: optical depth along L.

# some definitions (do not change!)

e0 = np.array([0, 0, 1])
ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
eplus = -(ex + 1j*ey)/np.sqrt(2)
eminus = (ex - 1j*ey)/np.sqrt(2)
single_decay = 1.0 # single atom decay rate

direc = '/data/rey/saag4275/data_files/'   # directory for saving data

# parameter setting box (may change)

#ratio = 0.1 # distance between atoms in units of lambda, transition/incident wavelength
#r_axis = np.array([1,0, 0]) # orientation of the distance between atoms, 3 vector
#r_axis = r_axis/np.linalg.norm(r_axis)

realization_list = np.array([1,2]) #np.arange(1,11,1)
rabi_val_list = np.array([4.0]) #np.array([0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0])
tfin_list = np.array([10]) #np.array([20]*3 + [6]*int(len(rabi_val_list)-3))
real_id_list = np.arange(0, len(realization_list), 1)
rabi_id_list = np.arange(0, len(rabi_val_list), 1)

# generate 2D array of realisation x rabi --> 

param_grid_real, param_grid_rabi = np.meshgrid(real_id_list, rabi_id_list)
param_grid_real_list = param_grid_real.flatten()
param_grid_rabi_list = param_grid_rabi.flatten()

real_id = param_grid_real_list[run_id-1]
rabi_id = param_grid_rabi_list[run_id-1]

real_val = realization_list[real_id]
rabi_val = rabi_val_list[rabi_id] #rabi_val_list[run_id-1] 
t_final_input = tfin_list[rabi_id] 


eL = np.array([0, 0, 1]) # polarisation of laser, can be expressed in terms of the vectors defined above
detuning_list = np.array([0.0*single_decay]) # detuning of laser from transition
det_set = 0
del_ze = 0.0 # magnetic field, i.e., Zeeman splitting of excited state manifold
del_zg = 0.0 # magnetic field, i.e., Zeeman splitting of ground state manifold
rabi = rabi_val*single_decay


#interactions turned off
turn_off_list = ['incoherent','coherent']
turn_off = [] #[turn_off_list[0], turn_off_list[1]] # leave turn_off = [], if nothing is to be turned off


turn_off_txt = ''
if turn_off != []:
    turn_off_txt += '_no_int_'
    for item in turn_off:
        turn_off_txt += '_'+ item

add_txt_in_params = turn_off_txt


num_pts_dr = int(1e2)

t_initial_dr = 0.0
t_final_dr = t_final_input 
t_range_dr = [t_initial_dr, t_final_dr]
t_vals_dr = np.linspace(t_initial_dr, t_final_dr, num_pts_dr) 

e0_desired = eL


# more definitions and functions (do not change!)

wavelength = 1 # wavelength of incident laser
k0 = 2*np.pi/wavelength
kvec = k0*np.array([1, 0, 0]) # k vector of incident laser

    
def rotation_matrix_a_to_b(va, vb): #only works upto 1e15-ish precision
    ua = va/np.linalg.norm(va)
    ub = vb/np.linalg.norm(vb)
    if np.dot(ua, ub) == 1:
        return np.identity(3)
    elif np.dot(ua, ub) == -1: #changing z->-z changes y->-y, thus preserving x->x, which is the array direction (doesn't really matter though!)
        return -np.identity(3)
    uv = np.cross(ua,ub)
    c = np.dot(ua,ub)
    v_mat = np.zeros((3,3))
    ux = np.array([1,0,0])
    uy = np.array([0,1,0])
    uz = np.array([0,0,1])
    v_mat[:,0] = np.cross(uv, ux)
    v_mat[:,1] = np.cross(uv, uy)
    v_mat[:,2] = np.cross(uv, uz)
    matrix = np.identity(3) + v_mat + (v_mat@v_mat)*1.0/(1.0+c)
    return matrix

 
if np.abs(np.conj(e0)@e0_desired) < 1.0:
    rmat = rotation_matrix_a_to_b(e0,e0_desired)
    eplus = rmat@eplus
    eminus = rmat@eminus
    ex = rmat@ex
    ey = rmat@ey
    e0 = e0_desired

print('kL = '+str(kvec/np.linalg.norm(kvec)), flush=True)
print('e0 = '+str(e0), flush=True)
print('ex = '+str(ex), flush=True)
print('ey = '+str(ey), flush=True)

HSsize = int(2*fg + 1 + 2*fe + 1) # Hilbert space size of each atom
HSsize_tot = int(HSsize**Natoms) # size of total Hilbert space

adde = fe
addg = fg

# polarisation basis vectors
evec = {0: e0, 1:eplus, -1: eminus}
evec = collections.defaultdict(lambda : [0,0,0], evec) 
   
def sort_lists_simultaneously_cols(a, b): #a -list to be sorted, b - 2d array whose columns are to be sorted according to indices of a
    inds = a.argsort()
    sortedb = b[:,inds]
    return sortedb

# levels
deg_e = int(2*fe + 1)
deg_g = int(2*fg + 1)

if (deg_e == 1 and deg_g == 1):
    qmax = 0
else:
    qmax = 1


# dictionaries


# dictionaries




# Clebsch Gordan coeff
cnq = {}
arrcnq = np.zeros((deg_g, 2*qmax+1), complex)
if (deg_e == 1 and deg_g ==1):
    cnq[0, 0] = 1
    arrcnq[0, 0] =  1
else:
    for i in range(0, deg_g):
        mg = i-fg
        for q in range(-qmax, qmax+1):
            if np.abs(mg + q) <= fe:
                cnq[mg, q] =  np.float(CG(S(fg), S(mg), S(qmax), S(q), S(fe), S(mg+q)).doit())
                arrcnq[i, q+qmax] = cnq[mg, q]
cnq = collections.defaultdict(lambda : 0, cnq) 

# Dipole moment

dsph = {}
if (deg_e == 1 and deg_g ==1):
    dsph[0, 0] = np.conjugate(evec[0])
else:
    for i in range(0, deg_e):
        me = i-fe
        for j in range(0, deg_g):
            mg = j-fg
            dsph[me, mg] = (np.conjugate(evec[me-mg])*cnq[mg, me-mg])

dsph = collections.defaultdict(lambda : np.array([0,0,0]), dsph) 



# normalise vector
def hat_op(v):
    return (v/np.linalg.norm(v))


transition_wavelength = 1.0
R_perp_given = 0.5*transition_wavelength # radial std in units of lambda, experimental value for system
L_given = 20.0*transition_wavelength # axial std in units of lambda, experimental value for system
aspect_ratio = L_given/R_perp_given # = axial/radial std
N_given = 2000 # experimental value for system
k = 2*np.pi/transition_wavelength
OD_x_given = 3*N_given/(2*(k*R_perp_given)**2)
Volume_cloud_given = 2*np.pi*(R_perp_given**2)*L_given
mean_density_given = N_given/Volume_cloud_given

def f_cloud_dims_fixed_OD(N_output):
    # Since OD_x = 3*N_given/(2*(k*R_perp_given)**2)
    R_perp_output = R_perp_given*np.sqrt(N_output/N_given*1.0) # = np.sqrt(3*N_output/(2*OD_x*(k**2)))
    L_axial_output = R_perp_output*aspect_ratio

    return [R_perp_output, L_axial_output]


def f_cloud_dims_fixed_mean_density(N_output):
    Volume_cloud_output = N_output/mean_density_given
    R_perp_output = (Volume_cloud_output/(2*np.pi*aspect_ratio))**(1/3.0)
    L_axial_output = R_perp_output*aspect_ratio

    return [R_perp_output, L_axial_output]

if fixed_param == 0:
    std_list = [R_perp_given, L_given]
    fixed_text = '_fixed_size'
elif fixed_param == 1:
    std_list = f_cloud_dims_fixed_mean_density(Natoms)
    fixed_text = '_fixed_mean_density'
elif fixed_param == 2:
    std_list = f_cloud_dims_fixed_OD(Natoms)
    fixed_text = '_fixed_OD_ax'
    
std_rad = std_list[0]
std_ax = std_list[1]
dims_text = '_std_rad_'+str(np.round(std_rad,2)).replace('.',',') + '_std_ax_'+str(np.round(std_ax,2)).replace('.',',')


# plot properties

levels = int(deg_e + deg_g)
rdir_fig = '_3D_gas'+fixed_text+dims_text

eLdir_fig = '_eL_along_'
dirs = ['x','y','z']
temp_add = 0
for i in range(0,3):
    if eL[i]!=0:
        if temp_add == 0:
            eLdir_fig += dirs[i]
        else:
            eLdir_fig += '_and_'+ dirs[i]
        temp_add += 1

kdir_fig = '_k_along_'
dirs = ['x','y','z']
temp_add = 0
for i in range(0,3):
    if kvec[i]!=0:
        if temp_add == 0:
            kdir_fig += dirs[i]
        else:
            kdir_fig += '_and_'+ dirs[i]
        temp_add += 1

rabi_add = '_rabi_'+str((rabi)).replace('.',',')
rabi_add += '_cluster_size_'+str(cluster_size)


h5_title = str(levels)+'_level_'+str(Natoms)+'_atoms'+rdir_fig+kdir_fig+eLdir_fig+'_real_id_'+str(int(real_val))+'.h5'

h5_title_dr = str(levels)+'_level_'+str(Natoms)+'_atoms'+rdir_fig+kdir_fig+eLdir_fig+rabi_add+'_tfin_'+str(int(t_final_input))+'_real_id_'+str(int(real_val))+add_txt_in_params+'.h5'


# try to load data for positions, if no data available, then generate samples



try:
    #hf = h5py.File(direc+'MF_positions_Greens_fn_phases_3D_gas_fixed_size_'+h5_title, 'r')
    hf = h5py.File(direc+'Atomic_positions_Greens_fn_phases_3D_gas_'+h5_title, 'r')

    rvecall = hf['rvecall'][()]
    arrGij = hf['arrGij'][()]
    arrGijtilde = hf['arrGijtilde'][()]
    arrIij = hf['arrIij'][()]
    phase_array = hf['forward_phase_array'][()]
    hf.close()
    
    print("Data for positions loaded from file!", flush=True)
    
    
    #print("Incoherent interactions turned off!", flush=True)
    
    '''
    fac_inc = 1.0
    fac_coh = 1.0
    if turn_off!=[]:
        for item in turn_off:
            if item == 'incoherent':
                fac_inc = 0
            if item == 'coherent':
                fac_coh = 0
                
    if fac_coh != 1.0 and fac_inc == 1.0:
        arrRij = arrGij - 1j*arrIij
        arrGij = arrRij*fac_coh + arrIij*1j
        arrGijtilde = arrRij*fac_coh - arrIij*1j
        
        print("Coherent interactions turned off!", flush=True)
    if fac_inc != 1.0:
        arrIij = np.reshape(np.diag(np.diag(np.reshape(arrIij, (Natoms*deg_e*deg_g, Natoms*deg_e*deg_g)))), (Natoms, Natoms, deg_e, deg_g, deg_e, deg_g)) # removing non-diagonal interacting terms from the dissipative part
        arrRij = arrGij - 1j*arrIij
        arrGij = arrRij*fac_coh + arrIij*1j
        arrGijtilde = arrRij*fac_coh - arrIij*1j
        
        if fac_coh == 1.0:
            print("Incoherent interactions turned off!", flush=True)
        else:
            print("Coherent AND incoherent interactions turned off!", flush=True)
            
    if fac_coh != 1.0 or fac_inc != 1.0:
        hf = h5py.File(direc+'MF_positions_Greens_fn_phases_3D_gas_fixed_size_'+h5_title_turn_off, 'w')
        hf.create_dataset('rvecall', data=rvecall, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGij', data=arrGij, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGijtilde', data=arrGijtilde, compression="gzip", compression_opts=9)
        hf.create_dataset('arrIij', data=arrIij, compression="gzip", compression_opts=9)
        hf.create_dataset('forward_phase_array', data=phase_array, compression="gzip", compression_opts=9)
        hf.close()
        
    '''
    
except:
#if True:
    print("No data for positions found, will generate sampling of positions now and save it!", flush=True)

    temp = np.random.default_rng(real_val)

    r_sampled_raw = temp.normal(0, std_ax, Natoms*1000) # get axial x position
    r_sampled = temp.choice(r_sampled_raw, Natoms, replace=False) # to prevent multiple atoms from being at exactly the same positions
    r_array_x = np.sort(r_sampled)

    r_sampled_rad_z = temp.normal(0, std_rad, Natoms)  # get radial z position
    r_array_z = r_sampled_rad_z 

    r_sampled_rad_y = temp.normal(0, std_rad, Natoms)  # get radial y position
    r_array_y = r_sampled_rad_y 

    r_array_xyz = np.array([r_array_x, r_array_y, r_array_z])
    rvecall = r_array_xyz.T

    r_nn_spacing = np.sqrt(np.einsum('ab->b', (r_array_xyz[:,1:] - r_array_xyz[:,:-1])**2))

    r_nn_spacing_avg = np.mean(r_nn_spacing) # 3d avg spacing for simulation
    r_min = np.min(r_nn_spacing) # 3d min spacing for simulation

    print('3D gas properties:' , flush=True)
    print('std axial = ' + str(std_ax))
    print('std radial = ' + str(std_rad), flush=True)
    print('mean nearest-neighbor spacing = ' + str(r_nn_spacing_avg), flush=True)
    print('minimum nearest-neighbor spacing = ' + str(r_min), flush=True)
    
    
    # phase_array
    
    phase_array = np.zeros((Natoms, Natoms), complex)
    for i in range(0, Natoms):
        for j in range(0, Natoms):
            temp_phase = np.dot(kvec, (rvecall[i] - rvecall[j]))
            phase_array[i, j] = np.exp(1j*temp_phase)

    # Green's function
    def funcG(r):
        tempcoef = 3*single_decay/4.0
        temp1 = (np.identity(3) - np.outer(hat_op(r), hat_op(r)))*np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r)) 
        temp2 = (np.identity(3) - 3*np.outer(hat_op(r), hat_op(r)))*((1j*np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r))**2) - np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r))**3)
        return (tempcoef*(temp1 + temp2))

    def funcGij(i, j):
        return (funcG(rvecall[i] - rvecall[j]))

    fac_inc = 1.0
    fac_coh = 1.0
    if turn_off!=[]:
        for item in turn_off:
            if item == 'incoherent':
                fac_inc = 0
            if item == 'coherent':
                fac_coh = 0


    taD = time.time()

    dictRij = {}
    dictIij = {}
    dictGij = {}
    dictGijtilde = {}

    for i in range(0, Natoms):
        for j in range(0, Natoms):
            for q1 in range(-qmax,qmax+1):
                for q2 in range(-qmax,qmax+1):
                    if i!=j:
                        tempRij = fac_coh*np.conjugate(evec[q1])@np.real(funcGij(i, j))@evec[q2]
                        tempIij = fac_inc*np.conjugate(evec[q1])@np.imag(funcGij(i, j))@evec[q2]

                    else:
                        tempRij = 0
                        tempIij = (single_decay/2.0)*np.dot(np.conjugate(evec[q1]),evec[q2])
                    dictRij[i, j, q1, q2] = tempRij
                    dictIij[i, j, q1, q2] = tempIij
                    dictGij[i, j, q1, q2] = tempRij + 1j*tempIij
                    dictGijtilde[i, j, q1, q2] = tempRij - 1j*tempIij
                    #arrGij[i, j, q1+qmax, q2+qmax] = tempRij + 1j*tempIij
                    #arrGijtilde[i, j, q1+qmax, q2+qmax] = tempRij - 1j*tempIij

    dictRij = collections.defaultdict(lambda : 0, dictRij) 
    dictIij = collections.defaultdict(lambda : 0, dictIij) 
    dictGij = collections.defaultdict(lambda : 0, dictGij) 
    dictGijtilde = collections.defaultdict(lambda : 0, dictGijtilde) 

    tbD = time.time()
    print("time to assign Rij, Iij dict: "+str(tbD-taD), flush=True)

    taG = time.time()

    arrGij = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
    arrGijtilde = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
    arrIij = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
    for i in range(0, Natoms):
        for j in range(0, Natoms):
            for ima in range(0, deg_e):
                ma = ima - fe
                for ina in range(0, deg_g):
                    na = ina - fg
                    for imb in range(0, deg_e):
                        mb = imb - fe
                        for inb in range(0, deg_g):
                            nb = inb - fg
                            arrGij[i, j, ima, ina, imb, inb] = dictGij[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]
                            arrGijtilde[i, j, ima, ina, imb, inb] = dictGijtilde[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]
                            arrIij[i, j, ima, ina, imb, inb] = dictIij[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]

    tbG = time.time()
    print("time to assign Gij matrix: "+str(tbG-taG), flush=True)



    # save position and Gij data for future reference
    
    if fac_coh == 1.0 and fac_inc == 1.0:
        hf = h5py.File(direc+'Atomic_positions_Greens_fn_phases_3D_gas_'+h5_title, 'w')
        hf.create_dataset('rvecall', data=rvecall, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGij', data=arrGij, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGijtilde', data=arrGijtilde, compression="gzip", compression_opts=9)
        hf.create_dataset('arrIij', data=arrIij, compression="gzip", compression_opts=9)
        hf.create_dataset('forward_phase_array', data=phase_array, compression="gzip", compression_opts=9)
        hf.close()
        
    else:
        hf = h5py.File(direc+'Atomic_positions_Greens_fn_phases_3D_gas_'+h5_title_turn_off, 'w')
        hf.create_dataset('rvecall', data=rvecall, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGij', data=arrGij, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGijtilde', data=arrGijtilde, compression="gzip", compression_opts=9)
        hf.create_dataset('arrIij', data=arrIij, compression="gzip", compression_opts=9)
        hf.create_dataset('forward_phase_array', data=phase_array, compression="gzip", compression_opts=9)
        hf.close()
    
    if fac_coh != 1.0 and fac_inc == 1.0:
            print("Coherent interactions turned off!", flush=True)
    elif fac_coh == 1.0 and fac_inc != 1.0:
            print("Incoherent interactions turned off!", flush=True)
    elif fac_coh != 1.0 and fac_inc != 1.0:
            print("Coherent AND incoherent interactions turned off!", flush=True)


# redifining to keep q = 0 only

arrGij = arrGij[:,:,0,0,0,0]
arrGijtilde = arrGijtilde[:,:,0,0,0,0]
arrIij = arrIij[:,:,0,0,0,0]
    
# choosing cluster based on largest coherent interaction couplings

cluster_list = np.zeros((Natoms, cluster_size), int)
if cluster_size > 1:
    for i in range(0, Natoms):
        #indR = np.argpartition(np.abs(arrGij.real[i]), -(cluster_size-1))[-(cluster_size-1):]
        indR = np.argpartition(np.abs(arrGij[i]), -(cluster_size-1))[-(cluster_size-1):]
        cluster_list[i,0] = i
        cluster_list[i,1:] = np.sort(indR)
else:
    for i in range(0, Natoms):
        cluster_list[i,0] = i

# term saying whether j belongs in cluster of atom i or not; = 0 if yes, else = 1

theta_ij_list = np.zeros((Natoms, Natoms))
for i in range(0, Natoms):
    for j in range(0, Natoms):
        if j not in cluster_list[i]:
            theta_ij_list[i,j] = 1.0
        else:
            theta_ij_list[i,j] = 0
        
            
# arrGij for mean-field average over atoms outside the cluster

taG = time.time()

arrGij_MF_C = np.zeros((Natoms, cluster_size, Natoms), complex)
arrGijtilde_MF_C = np.zeros((Natoms, cluster_size, Natoms), complex)
arrIij_MF_C = np.zeros((Natoms, cluster_size, Natoms), complex)

for ic in range(0, Natoms): # ic is cluster_index
    for i in range(0, cluster_size):
        i_actual = cluster_list[ic, i]
        for j in range(0, Natoms):
            if j in cluster_list[ic]: # keeping coeffs zero for atoms within cluster
                continue
            arrGij_MF_C[ic, i, j] = arrGij[i_actual, j]
            arrGijtilde_MF_C[ic, i, j] = arrGijtilde[i_actual, j]
            arrIij_MF_C[ic, i, j] = arrIij[i_actual, j]

tbG = time.time()
print("time to assign Gij matrix (outside cluster): "+str(tbG-taG), flush=True)


#Rabi frequency for each atom

omega_atom = np.zeros(Natoms, complex)
for n in range(0, Natoms):
    omega_atom[n] = (rabi*np.dot(dsph[0, 0],eL)*np.exp(1j*np.dot(kvec, rvecall[n]))) 

# defining ops for ED part

HSsize = int(2*fg + 1 + 2*fe + 1) # Hilbert space size of each atom
HSsize_tot = int(HSsize**(cluster_size)) # size of cluster Hilbert space

adde = fe
addg = fg

# polarisation basis vectors
evec = {0: e0, 1:eplus, -1: eminus}
evec = collections.defaultdict(lambda : [0,0,0], evec) 


def f_dsph(me, mg):
    return (np.conjugate(evec[me-mg])*cnq[mg, me-mg])

def f_omega_atom(k):
    return (rabi*np.dot(f_dsph(0, 0),eL)*np.exp(1j*np.dot(kvec, rvecall[k])))

   
def f_sort_lists_simultaneously_cols(a, b): #a -list to be sorted, b - 2d array whose columns are to be sorted according to indices of a
    inds = a.argsort()
    sortedb = b[:,inds]
    return sortedb
    
# more functions

def ketEm(me):
    temp = np.zeros(HSsize)
    temp[int(me + adde)] = 1
    return temp


def ketGn(mg):
    temp = np.zeros(HSsize)
    temp[int(mg + addg + 2*fe + 1)] = 1
    return temp


def sigma_emgn(me, mg):
    return np.outer(ketEm(me), ketGn(mg))

def sigma_emem(me1, me2):
    return np.outer(ketEm(me1), ketEm(me2))

def sigma_gngn(mg1, mg2):
    return np.outer(ketGn(mg1), ketGn(mg2))

def sigma_gnem(mg, me):
    return np.outer(ketGn(mg), ketEm(me))

def commutator(A, B):
    return (np.dot(A,B)-np.dot(B,A))

def anticommutator(A, B):
    return (np.dot(A,B)+np.dot(B,A))

def funcOp(kinput, A):
    k = kinput+1
    if (k> cluster_size or k<1):
        return "error"
    elif k == 1:
        temp = csr_matrix(A)
        for i in range(1, cluster_size):
            temp = csr_matrix(sparse.kron(temp, np.identity(HSsize)))
        return temp
    else:
        temp = csr_matrix(np.identity(HSsize))
        for i in range(2, k):
            temp = csr_matrix(sparse.kron(temp, np.identity(HSsize)))
        temp = csr_matrix(sparse.kron(temp, A))
        for i in range(k, cluster_size):
            temp = csr_matrix(sparse.kron(temp, np.identity(HSsize)))
        return temp
    
def Dminus(k, q):
    #temp = np.zeros((HSsize_tot,HSsize_tot), complex)
    #for i in range(0, int(2*fe+1)):
    #    me = i-fe
    #    if np.abs(me-q) <= fg:
    temp = funcOp(k, sigma_gnem(0, 0))
    #else:
    #continue
    return temp

def Dplus(k, q):
    #temp = np.zeros((HSsize_tot,HSsize_tot), complex)
    #for i in range(0, int(2*fe+1)):
    #    me = i-fe
    #    if np.abs(me-q) <= fg:
    temp = funcOp(k, sigma_emgn(0, 0))
    #else:
    #continue
    return temp

def sparse_trace(A):
    return A.diagonal().sum()



indgg = int(deg_g*deg_g)
indee = int(deg_e*deg_e)
indeg = int(deg_e*deg_g)
total_num = indgg+indee+indeg
# gs states of each atom

gs_states = np.zeros(deg_g)
for i in range(0, deg_g):
    gs_states[i] = i-fg
    
# es states of each atom

es_states = np.zeros(deg_e)
for i in range(0, deg_e):
    es_states[i] = i-fe

# single atom operators' index in the list of all ops for all atoms

dict_ops = {}
index = 0
for n in range(0, Natoms):
    dict_ops['ee', n] = index
    index += 1

    dict_ops['eg', n] = index
    index += 1
        
dict_ops = collections.defaultdict(lambda : 'None', dict_ops)



def f_trace(sig_list):
    trace = 0+0*1j
    for n in range(0, Natoms):
        trace += sig_list[dict_ops['gg', n]]
        trace += sig_list[dict_ops['ee', n]]
    return trace/Natoms

# funcOp(k, sigma_emgn) and h.c. operator list for a cluster

sigma_emgn_cluster_list = []
sigma_gnem_cluster_list = []
for k in range(0, cluster_size):
    sigma_emgn_cluster_list.append(funcOp(k, sigma_emgn(0, 0)).toarray())
    sigma_gnem_cluster_list.append(funcOp(k, sigma_gnem(0, 0)).toarray())
    


##  USING SPARSE MATRICES
# vectorised master eqn, i.e., matrix

def funcL(A):
    return sparse.kron(A, np.identity(HSsize_tot))

def funcR(A):
    return sparse.kron(np.identity(HSsize_tot), np.transpose(A))

def rho_dot_atom_V(ind_del, driven, cluster_index): # driven = 0 or 1
    temp1 = 0 + 0*1j
    for k in range(0, cluster_size):
        k_actual = cluster_list[cluster_index,k]
        temp1 += (-detuning_list[ind_del])*funcOp(k, sigma_emem(0, 0))
        temp1 += -((f_omega_atom(k_actual)*csr_matrix(funcOp(k, sigma_emgn(0, 0)), dtype = complex))+(np.conjugate(f_omega_atom(k_actual))*csr_matrix(funcOp(k, sigma_gnem(0, 0)), dtype = complex)))*driven
    return (-1j*(funcL(temp1)-funcR(temp1)))

def rho_dot_H_int(cluster_index):
    temp = 0+0*1j
    for i in range(0, cluster_size):
        i_actual = cluster_list[cluster_index,i]
        for j in range(0, cluster_size):
            j_actual = cluster_list[cluster_index,j]
            temp += (arrGij[i_actual, j_actual].real)*csr_matrix(sparse.csr_matrix.dot(Dplus(i, 0),Dminus(j, 0)), dtype = complex)
    return (1j*funcL(temp)-1j*funcR(temp))

def rho_dot_L_rho(cluster_index):
    temp1 = 0+0*1j
    temp2 = 0+0*1j
    for i in range(0, cluster_size):
        i_actual = cluster_list[cluster_index,i]
        for j in range(0, cluster_size):
            j_actual = cluster_list[cluster_index,j]
            temp1 += (arrGij[i_actual, j_actual].imag)*sparse.csr_matrix.dot(Dplus(i, 0),Dminus(j, 0))
            temp2 += -2*(arrGij[i_actual, j_actual].imag)*sparse.csr_matrix.dot(funcL(Dminus(j, 0)),funcR(Dplus(i, 0)))
    return -((funcL(temp1)+funcR(temp1)) + temp2)

def rho_dot_full_V(driven, ind_del, cluster_index):
    return (rho_dot_atom_V(ind_del, driven, cluster_index) + rho_dot_H_int(cluster_index) + rho_dot_L_rho(cluster_index))

# function for calculating partial trace of a system of N atoms, in which N-1 atoms are traced over


def f_partial_trace(rho_input, N_input, nstates_input, not_traced_input):
    shape_tuple = []
    for i in range(0, 2*N_input):
        shape_tuple.append(nstates_input)
    rho_proc = np.reshape(rho_input, shape_tuple)
    RDM = rho_proc
    n_remain = N_input
    del_n = 0
    for iN in range(0, N_input):
        if iN != not_traced_input:
            ax1 = iN - del_n 
            ax2 = iN - del_n + n_remain
            RDM = np.trace(RDM, axis1 = ax1, axis2 = ax2)
            n_remain -= 1
            del_n += 1        
    return np.reshape(RDM, (nstates_input, nstates_input))


# create superoperator array for all atoms

M_array = [] #(np.zeros((Natoms, HSsize_tot**2, HSsize_tot**2), complex))
for i in range(0, Natoms):
    #M_array[i] = (rho_dot_full_V(1, det_set, i)) 
    M_array.append(rho_dot_full_V(1, det_set, i))

###################################################################################
###################################################################################

# final EOM function


def f_rho_dot_vec(t, DM_cluster_list):
    DM_cluster_mat = np.reshape(DM_cluster_list, (Natoms, HSsize_tot**2))
    
    
    # cluster ED part of dynamics
    
    DM_dot_cluster_mat = sparse.csr_matrix(np.zeros((Natoms, HSsize_tot**2), complex))
    #DM_dot_cluster_mat = (np.zeros((Natoms, HSsize_tot**2), complex))
    #DM_atom_mat = (np.zeros((Natoms, HSsize, HSsize), complex))
    sig_eg = np.zeros(Natoms, complex)
    for i in range(0, Natoms):
        DM_dot_cluster_mat[i] = (sparse.csr_matrix.dot(M_array[i],DM_cluster_mat[i]))
        #DM_dot_cluster_mat[i] = ((M_array[i]@DM_cluster_mat[i]))
        DM_atom_mat = f_partial_trace(np.reshape(DM_cluster_mat[i], (HSsize_tot, HSsize_tot)), cluster_size, HSsize, 0)
        sig_eg[i] = DM_atom_mat[1,0]
        
    # MF part of dynamics

    #sig_eg = DM_atom_mat[:,1,0] # important line!!!!!! rho_eg = sig_ge and vice versa!!!!!!!
    
    MF_sum_full_cluster = 1j*np.einsum('nik,k,iuv->nuv', arrGij_MF_C, np.conj(sig_eg), np.array(sigma_emgn_cluster_list))
    MF_sum_full_cluster += 1j*np.einsum('nik,k,iuv->nuv', arrGijtilde_MF_C, sig_eg, np.array(sigma_gnem_cluster_list))
    # i = index of atoms in a cluster n
    # k = actual index of atoms, going from 0...N-1
    
    

    for i in range(0, Natoms):
        DM_dot_cluster_mat[i] += (MF_sum_full_cluster[i]@np.reshape(DM_cluster_mat[i],(HSsize_tot, HSsize_tot)) - np.reshape(DM_cluster_mat[i],(HSsize_tot, HSsize_tot))@MF_sum_full_cluster[i]).flatten() #(funcL(MF_sum_full_cluster[i]) - funcR(MF_sum_full_cluster[i]))@DM_cluster_mat[i]
        
    return (DM_dot_cluster_mat.toarray()).flatten()
    #return (DM_dot_cluster_mat).flatten()



###################################################################################
###################################################################################

# initial condition

temp0 = ketGn(0) + 0*1j 
temp = temp0
for n in range(1, cluster_size):
    temp = np.array(np.kron(temp,temp0), complex)

initial_state = temp
initial_rho = np.outer(np.conjugate(initial_state), initial_state)
initial_vec = initial_rho.flatten() # IC of each cluster

initial_sig_mat = np.zeros((Natoms, HSsize_tot**2), complex)
for i in range(0, Natoms):
    initial_sig_mat[i] = initial_vec

initial_sig_vec = initial_sig_mat.flatten()

print('trace = '+str(np.trace(initial_rho)), flush=True)

###################################################################################
###################################################################################


# driven evolutiion from a ground state superposition to get to the steady state

initial_sig_vec_current = initial_sig_vec

num_single_particle_ops = int(2*Natoms)

SpSm_op_cluster = 0
SpSm_op_cluster_with_phases_list = []

for j in range(1, cluster_size):
    SpSm_op_cluster += sparse.csr_matrix.dot(funcOp(0, sigma_emgn(0, 0)), funcOp(j, sigma_gnem(0, 0)))
    SpSm_op_cluster += sparse.csr_matrix.dot(funcOp(j, sigma_emgn(0, 0)), funcOp(0, sigma_gnem(0, 0)))

for k in range(0, Natoms):
    temp_SpSm_Op_phases = 0
    for j in range(1, cluster_size):
        j_actual = cluster_list[k,j]
        temp_SpSm_Op_phases += (phase_array[k,j_actual])*sparse.csr_matrix.dot(funcOp(0, sigma_emgn(0, 0)), funcOp(j, sigma_gnem(0, 0))) + (phase_array[j_actual,k])*sparse.csr_matrix.dot(funcOp(j, sigma_emgn(0, 0)), funcOp(0, sigma_gnem(0, 0)))
    SpSm_op_cluster_with_phases_list.append(temp_SpSm_Op_phases)
    
total_exc_dr = np.zeros(len(t_vals_dr))
total_gs_dr = np.zeros(len(t_vals_dr))
total_Sx_dr = np.zeros(len(t_vals_dr))
total_Sy_dr = np.zeros(len(t_vals_dr))
total_SpSm_dr = np.zeros(len(t_vals_dr))
forward_intensity_dr = np.zeros(len(t_vals_dr))

single_atom_exc_dr = np.zeros((Natoms,len(t_vals_dr)))
single_atom_gs_dr = np.zeros((Natoms,len(t_vals_dr)))
single_atom_Sx_dr = np.zeros((Natoms,len(t_vals_dr)))
single_atom_Sy_dr = np.zeros((Natoms,len(t_vals_dr)))
single_atom_SpSm_connected_part_dr = np.zeros((Natoms, len(t_vals_dr)))
# we save connected part separately; if the connected part becomes constant after increasing cluster size beyond a point,
# then it means that the dynamics have converged with respect to cluster size.

# input expectation values for initial state at t = 0


for k in range(0, Natoms):
    
    rho_sol_dr = csr_matrix(np.reshape(initial_sig_vec_current,(Natoms,HSsize_tot,HSsize_tot))[k,:,:])

    temp_Sp_k = sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,funcOp(0, sigma_emgn(0, 0))))
    
    single_atom_exc_dr[k,0] = np.real(sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,funcOp(0, sigma_emem(0, 0)))))
    single_atom_gs_dr[k,0] = np.real(sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,funcOp(0, sigma_gngn(0, 0)))))
    single_atom_Sx_dr[k,0] = np.real(temp_Sp_k + np.conj(temp_Sp_k))
    single_atom_Sy_dr[k,0] = np.real(-1j*(temp_Sp_k - np.conj(temp_Sp_k)))
    
    # get SpSm from inside cluster
    single_atom_SpSm_connected_part_dr[k,0] = sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,SpSm_op_cluster))
    forward_intensity_dr[0] += sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,SpSm_op_cluster_with_phases_list[k]))
    

ta1 = time.time()


for i_step in range(0,len(t_vals_dr)-1):
    
    sol = solve_ivp(f_rho_dot_vec, [t_vals_dr[i_step], t_vals_dr[i_step+1]], initial_sig_vec_current, method='RK45', t_eval=[t_vals_dr[i_step+1]], dense_output=False, events=None, atol = 10**(-7), rtol = 10**(-6))
    initial_sig_vec_current = sol.y[:,-1]
    
    for k in range(0, Natoms):
        
        rho_sol_dr = csr_matrix(np.reshape(initial_sig_vec_current,(Natoms,HSsize_tot,HSsize_tot))[k,:,:])

        temp_Sp_k = sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,funcOp(0, sigma_emgn(0, 0))))

        single_atom_exc_dr[k,i_step] = np.real(sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,funcOp(0, sigma_emem(0, 0)))))
        single_atom_gs_dr[k,i_step] = np.real(sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,funcOp(0, sigma_gngn(0, 0)))))
        single_atom_Sx_dr[k,i_step] = np.real(temp_Sp_k + np.conj(temp_Sp_k))
        single_atom_Sy_dr[k,i_step] = np.real(-1j*(temp_Sp_k - np.conj(temp_Sp_k)))

        # get SpSm from inside cluster
        single_atom_SpSm_connected_part_dr[k,i_step] = sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,SpSm_op_cluster))
        forward_intensity_dr[i_step] += sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,SpSm_op_cluster_with_phases_list[k]))
    
    print('Time step ' + str(i_step) + ' done.', flush = True)
total_exc_dr = np.einsum('kt->t', single_atom_exc_dr)
total_gs_dr = np.einsum('kt->t', single_atom_gs_dr)
total_Sx_dr = np.einsum('kt->t', single_atom_Sx_dr)
total_Sy_dr = np.einsum('kt->t', single_atom_Sy_dr)


# get SpSm (MF, disconnected part) from outside cluster

single_atom_Sp = (single_atom_Sx_dr + 1j*single_atom_Sy_dr)/2.0
single_atom_Sm = (single_atom_Sx_dr - 1j*single_atom_Sy_dr)/2.0

total_SpSm_dr = (np.einsum('kt->t', single_atom_SpSm_connected_part_dr) + np.real(np.einsum('kj,kt,jt->t', theta_ij_list, single_atom_Sp, single_atom_Sm) + np.einsum('kj,jt,kt->t', theta_ij_list, single_atom_Sp, single_atom_Sm)))/2.0

forward_intensity_dr = forward_intensity_dr/2.0 + (np.real(np.einsum('kj,kt,jt,kj->t', theta_ij_list, single_atom_Sp, single_atom_Sm, phase_array) + np.einsum('kj,jt,kt,jk->t', theta_ij_list, single_atom_Sp, single_atom_Sm, phase_array)))/2.0 + total_exc_dr

tb1 = time.time()
runtime1 = tb1-ta1

print("Runtime for time evolution and data calculation: " + str(runtime1), flush=True)

###################################################################################
###################################################################################

                
# save data

hf = h5py.File(direc+'Data_cluster-MF_dynamics_to_equil_'+h5_title_dr, 'w')

hf.create_dataset('total_exc', data=total_exc_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sx', data=total_Sx_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sy', data=total_Sy_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_gs', data=total_gs_dr, compression="gzip", compression_opts=9)
hf.create_dataset('t_vals_dr', data=t_vals_dr, compression="gzip", compression_opts=9)
hf.create_dataset('single_atom_exc_dr', data=single_atom_exc_dr, compression="gzip", compression_opts=9)
hf.create_dataset('single_atom_gs_dr', data=single_atom_gs_dr, compression="gzip", compression_opts=9)
hf.create_dataset('single_atom_Sx_dr', data=single_atom_Sx_dr, compression="gzip", compression_opts=9)
hf.create_dataset('single_atom_Sy_dr', data=single_atom_Sy_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_SpSm_dr', data=total_SpSm_dr, compression="gzip", compression_opts=9)
hf.create_dataset('single_atom_SpSm_connected_part_dr', data=single_atom_SpSm_connected_part_dr, compression="gzip", compression_opts=9)
hf.create_dataset('forward_intensity_dr', data=forward_intensity_dr, compression="gzip", compression_opts=9)

hf.create_dataset('theta_ij_list', data=theta_ij_list, compression="gzip", compression_opts=9)

hf.close()

print("Data saved to h5 file.", flush=True)

print("All runs done. Did not run decay, only equil. May all your codes run this well! :)", flush=True)
