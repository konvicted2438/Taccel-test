import os
import numpy as np
import pyvista as pv
from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
#import pypardiso

from utils.mesh_utils import *
from examples.example_utils import init_robot_demo_SDTO

def get_next_design_iter_decaying_volfrac(current_design_iter, volfrac) -> str:
    prefix = ''.join(filter(str.isalpha, current_design_iter))
    number = ''.join(filter(str.isdigit, current_design_iter))
    return f"{prefix}{int(number) + 1}_volfrac={volfrac}"

def get_next_design_iter(current_design_iter) -> str:
    prefix = ''.join(filter(str.isalpha, current_design_iter))
    number = ''.join(filter(str.isdigit, current_design_iter))
    return f"{prefix}{int(number) + 1}"

def create_soft_gripper_benchmarkTO(nelx, nely, nelz, volfrac, penal, rmin, ft, threshold, OUT_DIR, max_loops = 50):

    # =========== MATERIAL/TO PROPERTIES
    E0 = 1
    Emin = 1e-9
    nu = 0.4

    # =========== PREPARE FEA
    KE = lk(E = E0, nu = nu) #shape = (8,8)
    edofMat = np.zeros((nelx * nely, 8), dtype=int) 
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely       # element number reading column-wise left to right, starting from 0 (in Python, 1 in MATLAB)
            n1 = (nely + 1) * elx + ely # n1 is lower left node of element, el, and n2 is lower right node. Different from 99-line paper which grabbed upper left/upper right.
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array([2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 
                                       2 * n2 + 3, 2 * n2, 2 * n2 + 1, 
                                       2 * n1, 2 * n1 + 1])
    iK = np.kron(edofMat, np.ones((8, 1))).flatten(order = "C") # row indices of non-zero global K entries. NOTE: order = "C" by default i.e. row-major flattening
    jK = np.kron(edofMat, np.ones((1, 8))).flatten(order = "C") # col indices of non-zero global K entries. 
    F = np.zeros((2 * (nely + 1) * (nelx + 1), 2)) # Shape = (nDOFS, 1)
    U = np.zeros((2 * (nely + 1) * (nelx + 1), 2)) # Shape = (nDOFS, 1)

    dinx =  2*(nely+1)*nelx     # x-DOF of top R corner (0-based Python indexing) 
    diny =  2*(nely+1)*nelx + 1 # y-DOF of top R corner (0-based Python indexing) 
    #doutx =  2*(nely+1) - 2     # x-DOF of bottom L corner (0-based Python indexing) 
    douty =  2*(nely+1) - 1     # y-DOF of bottom L corner (0-based Python indexing)  

    F[dinx,0] = -np.sqrt(0.5)
    F[diny,0] = -np.sqrt(0.5)
    F[douty,1] = -1

    # BOTTOM-RIGHT CORNER/FIXED PORT OF COMPLIANT GRIPPER
    j = nely - 1            # 0-based y index for bottom right element of compliant gripper
    i = -j + nelx - 1       # 0-based x index for bottom right element of compliant gripper
    el = j + i * nely       # 0-based Python indexing
    n1 = (nely + 1) * i + j # n1 is upper left node of element, el, and n2 is upper right node. 
    n2 = (nely + 1) * (i + 1) + j
    fixedport_DOFs = [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 
                                2 * n2 + 3, 2 * n2, 2 * n2 + 1, 
                                2 * n1, 2 * n1 + 1]
    alldofs = np.arange(2 * (nely + 1) * (nelx + 1)) 
    freedofs = np.setdiff1d(alldofs, fixedport_DOFs) 

    # =========== PREPARE FILTER
    nfilter = int(nelx * nely * ((2 * np.ceil(rmin) - 1) ** 2)) #Max number of elements in H. If r=2.5, H is N x 5*5, where N is number of elements.
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    k = 0
    for i in range(nelx):
        for j in range(nely):
            e1 = i * nely + j #element, e, in H_ei, where e = 0,...,N-1 (N elements)
            for i2 in range(max(i - int(np.ceil(rmin)) + 1, 0), min(i + int(np.ceil(rmin)), nelx)):
                for j2 in range(max(j - int(np.ceil(rmin)) + 1, 0), min(j + int(np.ceil(rmin)), nely)):
                    e2 = i2 * nely + j2 #element, i, in H_ei, where i = 0,...,N-1 (N elements)
                    iH[k] = e1 # row of value sH[k] placed in H
                    jH[k] = e2 # col of value sH[k] placed in H
                    sH[k] = max(0, rmin - np.sqrt((i - i2) ** 2 + (j - j2) ** 2)) # value
                    k += 1
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc() #type(H) = <class 'scipy.sparse._csc.csc_matrix'> | Doesn't have shape (N,N) for operations w/ numpy arrays!
    Hs = np.sum(H, axis=1) # shape = (N, 1)

    # =========== INITIALISE PASSIVE MASK
    x = np.ones(nely * nelx, dtype = float) # shape = (N, )
    xFilt = x.copy() #I'm using xFilt rather than xPhys in the paper. shape = (N, )
    loop = 0
    change = 1
    passive = np.zeros((nely, nelx), dtype=int)

    # Iterate over each element in the mesh
    for i in range(nelx):
        for j in range(nely):
            # Check if the element (i, j) is within the triangular void region
            if j >= -i + nelx:
                passive[j, i] = 1  # Set to 1 for void (non-design) region
            # Check if the element (i,j) is along the bottom boundary
            if j == nely - 1 and j < -i + nelx:
                passive[j, i] = 2  # Set to 2 for solid (non-design) region

    x.reshape(nely, nelx, order='F')[passive == 1] = 0
    #x.reshape(nely, nelx, order='F')[passive == 2] = 1

    # =========== START ITERATION
    while change > 0.01 and loop < max_loops:
        loop += 1
        sK = (KE.reshape((KE.shape[0]*KE.shape[1], 1), order = "C") * (Emin + xFilt.reshape(1, -1) ** penal * (E0 - Emin))).flatten(order = "F") # shape = (N*8*8, )
        K = coo_matrix((sK, (iK, jK)), shape=(2 * (nely + 1) * (nelx + 1), 2 * (nely + 1) * (nelx + 1))).tocsc() # shape = (nDOFs, nDOFs)
        K[dinx,dinx] += 0.1 
        K[diny,diny] += 0.1 
        K[douty,douty] += 0.1 
        U[freedofs, :] = spsolve(K[freedofs, :][:, freedofs], F[freedofs, :])

        # Objective function and sensitivity analysis
        U_in = U[:, 0]
        U_out = U[:, 1]

        MSE = (U_out.T @ K @ U_in).item() # These are correct as they include PERTURBED stiffness matrices K
        SE = (U_out.T @ K @ U_out).item()  # These are correct as they include PERTURBED stiffness matrices K
        obj = MSE/(SE)

        u = (U_out.T @ K @ U_in).item() 
        v = (U_out.T @ K @ U_out).item()
        u_dash = (penal * xFilt ** (penal - 1) * (E0 - Emin)) * ((U_out[edofMat].reshape(nelx * nely, 8) @ KE) * U_in[edofMat].reshape(nelx * nely, 8)).sum(axis=1)
        v_dash = (penal * xFilt ** (penal - 1) * (E0 - Emin)) * ((U_out[edofMat].reshape(nelx * nely, 8) @ KE) * U_out[edofMat].reshape(nelx * nely, 8)).sum(axis=1)
        dc = (u_dash*v - u*v_dash)/(v**2)
        dv = np.ones(nely * nelx) #shape = (N,)

        # Filtering/modification of sensitivities
        if ft == 1:
            dc = np.asarray((H * (xFilt * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x) 
        elif ft == 2:
            dc = np.asarray(H * (dc.reshape(nelx*nely,1) / Hs))[:, 0] # shape = (N, )
            dv = np.asarray(H * (dv.reshape(nelx*nely,1) / Hs))[:, 0] # shape = (N, )

        # Optimality criteria update of design variables and physical densities
        l1, l2, move = 0, 100000, 0.1
        while (l2 - l1)/(l2 + l1) > 1e-4 and l2 > 1e-40:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.maximum(1e-10, -dc/lmid)**0.3))))
            xnew.reshape(nely, nelx, order='F')[passive == 1] = 0
            if ft == 1:
                xFilt = xnew # shape = (N, )
            elif ft == 2:
                xFilt = np.asarray(H * xnew[np.newaxis].T / Hs)[:, 0] #new: density filtering | shape = (N, )
            # Bisection algorithm to update lagrangian multiplier for volume constraint
            if np.sum(xFilt) > volfrac * nelx * nely:
                l1 = lmid
            else:
                l2 = lmid

        change = np.max(np.abs(xnew - x))
        x = xnew # shape = (N, )
        plot_densities(x.reshape((nely,nelx), order = 'F'),loop)
        print(f' Iter.: {loop:4d} | Obj.: {obj:10.4f} | Vol.: {np.mean(x):6.3f} | change.: {change:6.3f}')
    # =========== END OF TOPOLOGY OPTIMIZATION

    # =========== SAVING VARIOUS RELEVANT DENSITY ARRAYS
    density_1D_design = x.copy()
    density_1D_filepath = os.path.join(OUT_DIR,f"benchmark_1D_density.npy")
    np.save(density_1D_filepath, density_1D_design)  

    # Thresholding 1D design and saving
    x[x < threshold] = 0
    x[x >= threshold] = 1
    binary_1D_design = x.copy()
    binary_1D_design_filepath = os.path.join(OUT_DIR, f"benchmark_1D_binary.npy")
    np.save(binary_1D_design_filepath, binary_1D_design)  

    # TILING ALONG THIRD DIMENSION FOR VOXEL-BASED STL
    binary_3D_design= np.tile(binary_1D_design.reshape((nely,nelx,1), order = 'F'), (1, 1, nelz))
    binary_3D_design_filepath = os.path.join(OUT_DIR, f"benchmark_3D_binary.npy")
    np.save(binary_3D_design_filepath, binary_3D_design)  

    assert isinstance(binary_3D_design, np.ndarray) and binary_3D_design.shape == (nely, nelx, nelz), "3D binary design is not a np.ndarray or doesn't have the right shape!"
    return binary_3D_design


def lk(E = 1.0, nu = 0.2):

    k = [1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
        -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8]
    
    KE = E / (1 - nu ** 2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    
    return KE # shape = (8,8)

def plot_densities(x, iteration, CONVERGED = False):

    # Using a specific figure number ensures that you are plotting 
    # in the same window each time the function is called.
    plt.figure(1, figsize = (8*2, 6*2))

    # This command clears the current figure (figure 1) of all its 
    # contents, including axes, labels, and titles. It prepares the 
    # figure for a fresh plot without creating a new figure window.
    plt.clf()

    # This sets the colormap to grayscale, where low values are dark and high values are bright.
    # Since rho(x) = 1 indicates material, we input -x so that material/high values are dark.
    plt.imshow(-x, cmap='gray', interpolation='none')
    plt.title(r'$\rho(\mathbf{{x}})$ at iteration {}'.format(iteration))

    # Draw and flush the plot to update quickly.
    plt.draw()

    if CONVERGED:
        plt.show()
    else:
        plt.pause(0.0000001)  # Short pause to allow the GUI to catch up

class TopologyOptimizer2D(object):
    
    """
    Used for performing 2.5D-TO in an automated sim-TO-sim-TO-...-sim-TO pipeline applied to the optimization
    of a soft gripper finger. For 3DTO, you'll need to create a 'TopologyOptimizer3D' class which uses
    the pypardiso solver for an acceptable speedup.
    """

    def __init__(self, nelx: int, nely: int, nelz: int, 
                 cps2maxforces_3D: dict, OUT_DIR: str,
                 current_design_iter: str, 
                 current_design_mesh_filepath: str,
                 animation_force_mag: float,
                 HEADLESS: bool, VERBOSE: bool):
        """
        nelx, nely, nelz: Dimensions of the 3D design domain (nelz used for extruding 2.5D-TO designs)

        cps2maxforces_3D: Gained after a simulation of the 'current design/mesh' OR None

        OUT_DIR: Crucial files e.g. cps2maxforces_2D, D0_density.npy, STLs saved here instantly upon creation

        current_design_iter: This equals "D0" if we are about to do SDTO on D0 and produce D1

        current_design_mesh_filepath: This = path/to/f"{current_design_iter}_leftfinger_vol.msh"

        HEADLESS: Boolean flag which, when True, means no GUIs are activated (e.g. live plotters/images)

        VERBOSE: Boolean flag which, when True, prints output of topology optimization to the terminal
        """

        assert isinstance(cps2maxforces_3D, dict) or cps2maxforces_3D is None, "Must be a dictionary or None, in which case it's loaded."
        
        self.nelx, self.nely, self.nelz = nelx, nely, nelz
        self.HEADLESS = HEADLESS
        self.VERBOSE = VERBOSE
        self.OUT_DIR= OUT_DIR
        self.animation_force_mag = animation_force_mag
        self.current_design_iter = current_design_iter
        self.next_design_iter = get_next_design_iter(current_design_iter)

        self.current_design_mesh_filepath = current_design_mesh_filepath

        self.cps2maxforces_3D = cps2maxforces_3D
        if cps2maxforces_3D is not None:
            self.save_cps2maxforces(dictionary = self.cps2maxforces_3D,
                                    npy_filename = f"cps2maxforces_3D_{self.current_design_iter}.npy")
            
        self.cps2maxforces_2D = self.get_cps2maxforces_2D_from_3D()
        self.save_cps2maxforces(dictionary = self.cps2maxforces_2D, 
                                npy_filename = f"cps2maxforces_2D_{self.current_design_iter}.npy")


    def save_cps2maxforces(self, dictionary: dict, npy_filename: str):
        output_file_path = os.path.join(self.OUT_DIR, f"{npy_filename}")
        np.save(output_file_path, dictionary)  # Save

    def load_cps2maxforces(self, npy_filename: str = "cps2maxforces_2D_D0.npy"):
        file_path = os.path.join(self.OUT_DIR, npy_filename)
        dictionary = np.load(file_path, allow_pickle=True).item()
        return dictionary

    def get_cps2maxforces_2D_from_3D(self):

            # Load cps2maxforces_3D from filepath if cself.ps2maxforces_3D is None
            if self.cps2maxforces_3D is None:
                self.cps2maxforces_3D = self.load_cps2maxforces(npy_filename =  f"cps2maxforces_3D_{self.current_design_iter}.npy")

            # CPs_array/forces_array:(n_contacts, 3) 
            CPs_array = np.array(list(self.cps2maxforces_3D.keys()))
            forces_array = np.array(list(self.cps2maxforces_3D.values()))
            assert len(CPs_array) != 0 or len(forces_array) != 0, "Your cps2maxforces_3D dictionary contains no forces and/or CPs (likely due to a bad design)."
            
            self.plot_mesh_and_CPs(title = f"{self.current_design_iter}: 3D Forces", CPs = CPs_array, 
                                    forces = forces_array, mesh_filepath = self.current_design_mesh_filepath,
                                    save_filepath = f"{self.OUT_DIR}/{self.current_design_iter}_3D_Forces.png")

            """
            Computing a single average force vector for vectors along the z-axis/extruded axis. 
            This will reduce the number of forces intelligently, and also produce forces with 
            only x-y components which are needed for 2.5D-TO with automated/simulated forces.
            """

            # Dictionary to accumulate a list of forces for each unique x position
            force_accumulator = defaultdict(list)
            for CP, force in self.cps2maxforces_3D.items():
                # In simulator, x, y, and z might have too many decimal places
                # so this rounding is crucial for SDTO to work
                CP = (round(CP[0], 3) , round(CP[1], 2), round(CP[2],3))  # change y-value
                force_accumulator[CP[0]].append(force)

            # Compute averaged forces at new (x, 0, 0.015) positions
            cps2maxforces_2D = {
                # Chosen 0.015 for rendering/visualing forces along a single 2d line for 2d automated TO
                (x, 0.0, 0.015): np.mean(np.array(list_of_forces), axis=0) for x, list_of_forces in force_accumulator.items()
            }

            CPs_array_2D = np.array(list(cps2maxforces_2D.keys()))
            forces_array_2D = np.array(list(cps2maxforces_2D.values()))
            forces_array_2D[:, -1] = 0 # Zero out z-direction for 2D forces
            self. plot_mesh_and_CPs(title = f"{self.current_design_iter}: 2D Forces", CPs = CPs_array_2D, 
                                    forces = forces_array_2D, mesh_filepath = self.current_design_mesh_filepath,
                                    save_filepath = f"{self.OUT_DIR}/{self.current_design_iter}_2D_Forces.png")


            # Bundling condensed 2D forces back into a dictionary, whereby they have z-component = 0 for 2D-TO
            cps2maxforces_2D = {
                # Chosen 0.015 for rendering/visualing forces along a single 2d line for 2d automated TO
                tuple(CPs_array_2D[i]): forces_array_2D[i] for i in range(len(forces_array_2D))
            }

            return cps2maxforces_2D

    def plot_mesh_and_CPs(self, title: str, CPs: np.ndarray, forces: np.ndarray, 
                          mesh_filepath: str, save_filepath: str = None):

        pv_mesh = pv.read(mesh_filepath)

        pl = pv.Plotter(off_screen=True)  # Off-screen rendering needed for saving PNG
        pl.add_axes()
        pl.add_title(title, font_size=20)
        pl.add_mesh(pv_mesh, color='green', opacity=0.2, show_edges=True, 
                    edge_color='black', line_width=0.5)
        pl.add_points(CPs, color='red', point_size=10)
        pl.add_arrows(CPs, forces, mag=self.animation_force_mag, show_scalar_bar=False, color='purple')


        if save_filepath:
            pl.screenshot(save_filepath)  # Save as PNG
        else:
            pl.show()
        return

    def SDTO_2D(self, volfrac, penal, rmin, ft, threshold, max_loops = 50, angle = 45, E = 1):
            

            # =========== MATERIAL/TO PROPERTIES
            E0 = 1.0/E #normalize by simulated stiffness
            Emin = 1e-9
            nu = 0.4
            # n_forces = input_load + len(cps2maxforces) is the number of forces/displacement vectors needed for SDTO
            n_forces = 1 + len(self.cps2maxforces_2D)

            # =========== DEBUGGING
            if self.VERBOSE:
                print("\nNumber of contact forces = ", len(self.cps2maxforces_2D), "\n")
                for key, val in self.cps2maxforces_2D.items():
                    print("CP: ", key, " | Force: ", val)

            # # x-coord of CPs is in mm, so multiply xyz[0] by 1000 to convert to elx/voxel number.
            # elx = int(xyz[0] * 1000)
            # assert elx <= 81, "Contact points with elx > 81 are outside of your defined design domain. These should have been excluded when you build cps2maxforces."
            # #el = (nely-1) + elx*nely
            # n1 = (self.nely + 1) * elx + (self.nely-1) 
            # # LOWER LEFT x and y DOF (nodes in edofMat are enumerated CCW starting from bottom left node)
            # lowerleft_xDOF, lowerleft_yDOF = (2 * n1 + 2), (2 * n1 + 3)
            # return lowerleft_xDOF, lowerleft_yDOF

            # =========== PREPARE FEA
            KE = self.lk(E = E0, nu = nu) #shape = (8,8)
            edofMat = np.zeros((self.nelx * self.nely, 8), dtype=int) 
            for elx in range(self.nelx):
                for ely in range(self.nely):
                    # element number reading column-wise left to right, starting from 0 (in Python, 1 in MATLAB)
                    el = ely + elx * self.nely
                    # n1 is lower left node of element, el, and n2 is lower right node. 
                    # Different from 99-line paper which grabbed upper left/upper right.
                    n1 = (self.nely + 1) * elx + ely 
                    n2 = (self.nely + 1) * (elx + 1) + ely
                    # Global dofs are enumerated CCW startinf from lower-left node per el
                    edofMat[el, :] = np.array([2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 
                                            2 * n2 + 3, 2 * n2, 2 * n2 + 1, 
                                            2 * n1, 2 * n1 + 1])
            iK = np.kron(edofMat, np.ones((8, 1))).flatten(order = "C") # row indices of non-zero global K entries.
                                                                        # NOTE: order = "C" by default i.e. row-major flattening
            jK = np.kron(edofMat, np.ones((1, 8))).flatten(order = "C") # col indices of non-zero global K entries. 
            F = np.zeros((2 * (self.nely + 1) * (self.nelx + 1), n_forces)) # Shape = (nDOFS, 1)
            U = np.zeros((2 * (self.nely + 1) * (self.nelx + 1), n_forces)) # Shape = (nDOFS, 1)

            # Generate a list containing x and y nodal DOFs for input force and output forces,
            # where len(forces_xyDOFs) = n_forces
            forces_xyDOFs = []
            # Top-right corner x- and y-DOFs for input load
            dinx =  2*(self.nely+1)*self.nelx 
            diny =  2*(self.nely+1)*self.nelx+1 
            forces_xyDOFs.append((dinx, diny)) 
            for cp in self.cps2maxforces_2D.keys():
                Fi_xyDOF = self.XYtoDOF(xyz = cp)
                forces_xyDOFs.append(Fi_xyDOF)

            contact_forces_list = list(self.cps2maxforces_2D.values())

            for i in range(n_forces):

                # Handling the input load case (manually determined input loads)
                if i == 0:
                    dinx, diny = forces_xyDOFs[i]
                    F[dinx,i] = np.cos(np.radians(180 + angle)) # angle = 45 is default arrangement/shouldn't affect previous implementations
                    F[diny,i] = np.sin(np.radians(180 + angle))
                    # print(f"F[dinx,{i}] = ", F[dinx,i])
                    # print(f"F[diny,{i}] = ", F[diny,i])
                else:
                    doutx, douty= forces_xyDOFs[i]
                    # i - 1 is CORRECT because when i = 0, the if statement above triggers,
                    # but when i = 1, we will grab the first contact force, which is the second element in forces_xyDOFs
                    F_contact = contact_forces_list[i-1] # 1D np array w/ 3 elements of the form (Fx, Fy, Fz), where Fz = 0
                    F[doutx,i] = F_contact[0]
                    F[douty,i] = F_contact[1]

            # ==================== Normalizing F by E i.e. simulated stiffness of soft finger
            F /= E
            # ==================== Normalizing F by E i.e. simulated stiffness of soft finger

            # BOTTOM-RIGHT CORNER/FIXED PORT OF COMPLIANT GRIPPER
            j = self.nely - 1            # 0-based y index for bottom right element of compliant gripper
            i = -j + self.nelx - 1       # 0-based x index for bottom right element of compliant gripper
            el = j + i * self.nely       # 0-based Python indexing
            n1 = (self.nely + 1) * i + j # n1 is upper left node of element, el, and n2 is upper right node. 
            n2 = (self.nely + 1) * (i + 1) + j
            fixedport_DOFs = [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 
                                        2 * n2 + 3, 2 * n2, 2 * n2 + 1, 
                                        2 * n1, 2 * n1 + 1]
            alldofs = np.arange(2 * (self.nely + 1) * (self.nelx + 1)) 
            freedofs = np.setdiff1d(alldofs, fixedport_DOFs) 

            # =========== PREPARE FILTER
            # Max number of elements in H. If r=2.5, H is N x 5*5, where N is number of elements.
            nfilter = int(self.nelx * self.nely * ((2 * np.ceil(rmin) - 1) ** 2))
            iH = np.zeros(nfilter)
            jH = np.zeros(nfilter)
            sH = np.zeros(nfilter)
            k = 0
            for i in range(self.nelx):
                for j in range(self.nely):
                    e1 = i * self.nely + j #element, e, in H_ei, where e = 0,...,N-1 (N elements)
                    for i2 in range(max(i - int(np.ceil(rmin)) + 1, 0), min(i + int(np.ceil(rmin)), self.nelx)):
                        for j2 in range(max(j - int(np.ceil(rmin)) + 1, 0), min(j + int(np.ceil(rmin)), self.nely)):
                            e2 = i2 * self.nely + j2 #element, i, in H_ei, where i = 0,...,N-1 (N elements)
                            iH[k] = e1 # row of value sH[k] placed in H
                            jH[k] = e2 # col of value sH[k] placed in H
                            sH[k] = max(0, rmin - np.sqrt((i - i2) ** 2 + (j - j2) ** 2)) # value
                            k += 1
            H = coo_matrix((sH, (iH, jH)), shape=(self.nelx * self.nely, self.nelx * self.nely)).tocsc() #type(H) = <class 'scipy.sparse._csc.csc_matrix'> | Doesn't have shape (N,N) for operations w/ numpy arrays!
            Hs = np.sum(H, axis=1) # shape = (N, 1)

            # =========== INITIALISE PASSIVE MASK
            passive = np.load(os.path.join(self.OUT_DIR, "passive.npy"))
            binary_1D_filepath = os.path.join(self.OUT_DIR,f"{self.current_design_iter}_1D_binary.npy")
            x = np.load(binary_1D_filepath)
            xFilt = x.copy() #I'm using xFilt rather than xPhys in the paper. shape = (N, )
            loop = 0
            change = 1
            
            # Saving initial design domain, regardless of headless or not
            # NOTE: Only shows plot if headless is True - accounted for internally
            self.plot_densities(x.reshape(self.nely,self.nelx, order = 'F'), loop, 
                                save_density = True, animate_densities = None, show_plot = True,
                                threshold = None)

            # =========== START ITERATION
            if self.VERBOSE:
                print(f"\nStarting 2.5D-TO: {self.current_design_iter} --> {self.next_design_iter}\n")
            while change > 0.01 and loop < max_loops:
                loop += 1
                sK = (KE.reshape((KE.shape[0]*KE.shape[1], 1), order = "C") * (Emin + xFilt.reshape(1, -1) ** penal * (E0 - Emin))).flatten(order = "F") # shape = (N*8*8, )
                K = coo_matrix((sK, (iK, jK)), shape=(2 * (self.nely + 1) * (self.nelx + 1), 2 * (self.nely + 1) * (self.nelx + 1))).tocsc() # shape = (nDOFs, nDOFs)

                # Adding numerical springs for input loads + contact forces, all of which have non-zero xyDOFs
                for dofx, dofy in forces_xyDOFs:
                    K[dofx, dofx] += 0.1
                    K[dofy, dofy] += 0.1

                U[freedofs, :] = spsolve(K[freedofs, :][:, freedofs], F[freedofs, :])
                c, dc = self.get_c_and_dc(U, xFilt, penal, K, E0, Emin, KE, edofMat)
                dv = np.ones(self.nely * self.nelx) #shape = (N,)

                # Filtering/modification of sensitivities
                if ft == 1:
                    dc = np.asarray((H * (xFilt * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x) 
                elif ft == 2:
                    dc = np.asarray(H * (dc.reshape(self.nelx*self.nely,1) / Hs))[:, 0] # shape = (N, )
                    dv = np.asarray(H * (dv.reshape(self.nelx*self.nely,1) / Hs))[:, 0] # shape = (N, )

                # Optimality criteria update of design variables and physical densities
                l1, l2, move = 0, 100000, 0.1
                while (l2 - l1)/(l2 + l1) > 1e-4 and l2 > 1e-40:
                    lmid = 0.5 * (l2 + l1)
                    xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.maximum(1e-10, -dc/lmid)**0.3))))
                    xnew.reshape(self.nely, self.nelx, order='F')[passive == 1] = 0
                    #xnew.reshape(nely, nelx, order='F')[passive == 2] = 1 --- ENFORCING MATERIAL DOMAIN
                    if ft == 1:
                        xFilt = xnew # shape = (N, )
                    elif ft == 2:
                        xFilt = np.asarray(H * xnew[np.newaxis].T / Hs)[:, 0]
                    # Bisection algorithm to update lagrangian multiplier for volume constraint
                    if np.sum(xFilt) > volfrac * self.nelx * self.nely:
                        l1 = lmid
                    else:
                        l2 = lmid

                change = np.max(np.abs(xnew - x))
                x = xnew # shape = (N, )

                if not self.HEADLESS:
                    self.plot_densities(x.reshape((self.nely,self.nelx), order = 'F'),loop, save_density = False,
                                        animate_densities=True, threshold = None)
                if self.VERBOSE:
                    print(f' Iter.: {loop:4d} | Obj.: {c:10.4f} | Vol.: {np.mean(x):6.3f} | change.: {change:6.3f}')
            # =========== END OF TOPOLOGY OPTIMIZATION

            # =========== SAVING VARIOUS RELEVANT DENSITY ARRAYS
            self.density_1D_design = x.copy()
            density_1D_filepath = os.path.join(self.OUT_DIR,f"{self.next_design_iter}_1D_density.npy")
            np.save(density_1D_filepath, self.density_1D_design)  

            # Thresholding 1D design and saving
            x[x < threshold] = 0
            x[x >= threshold] = 1
            self.binary_1D_design = x.copy()
            binary_1D_design_filepath = os.path.join(self.OUT_DIR, f"{self.next_design_iter}_1D_binary.npy")
            np.save(binary_1D_design_filepath, self.binary_1D_design)  
            self.plot_densities(x.reshape(self.nely,self.nelx, order = 'F'), loop, 
                                save_density = True, animate_densities = None, show_plot = True, threshold = threshold)

            # TILING ALONG THIRD DIMENSION FOR VOXEL-BASED STL
            self.binary_3D_design= np.tile(self.binary_1D_design.reshape((self.nely,self.nelx,1), order = 'F'), (1, 1, self.nelz))
            binary_3D_design_filepath = os.path.join(self.OUT_DIR, f"{self.next_design_iter}_3D_binary.npy")
            np.save(binary_3D_design_filepath, self.binary_3D_design)  

            assert isinstance(self.binary_3D_design, np.ndarray) and self.binary_3D_design.shape == (self.nely, self.nelx, self.nelz), "3D binary design is not a np.ndarray or doesn't have the right shape!"
            return self.binary_3D_design


    def lk(self, E = 1.0, nu = 0.4):

        k = [1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
            -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8]
        
        KE = E / (1 - nu ** 2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
        
        return KE # shape = (8,8)

    def XYtoDOF(self, xyz: tuple):
        """
        xyz: CP of the form (np.float64(x),np.float64(x),np.float64(x))
        nely: # of y-elements
        """

        # x-coord of CPs is in mm, so multiply xyz[0] by 1000 to convert to elx/voxel number.
        elx = int(xyz[0] * 1000)
        assert elx <= 81, "Contact points with elx > 81 are outside of your defined design domain. These should have been excluded when you build cps2maxforces."
        #el = (nely-1) + elx*nely
        n1 = (self.nely + 1) * elx + (self.nely-1) 
        # LOWER LEFT x and y DOF (nodes in edofMat are enumerated CCW starting from bottom left node)
        lowerleft_xDOF, lowerleft_yDOF = (2 * n1 + 2), (2 * n1 + 3)
        return lowerleft_xDOF, lowerleft_yDOF

    def plot_densities(self, x, iteration, save_density = None, animate_densities = None, show_plot = None,
                       threshold = None):

        # Using a specific figure number ensures that you are plotting 
        # in the same window each time the function is called.
        plt.figure(1, figsize = (8*2, 6*2))

        # This command clears the current figure (figure 1) of all its 
        # contents, including axes, labels, and titles. It prepares the 
        # figure for a fresh plot without creating a new figure window.
        plt.clf()

        # This sets the colormap to grayscale, where low values are dark and high values are bright.
        # Since rho(x) = 1 indicates material, we input -x so that material/high values are dark.
        plt.imshow(-x, cmap='gray', interpolation='none')

        if threshold is None:
            title = r'$\rho(\mathbf{{x}})$ at iteration {}'.format(iteration)
            title += f" for {self.next_design_iter}"
            plt.title(title)
        else:
            title = r'$\rho(\mathbf{{x}})$ at iteration {}'.format(iteration)
            title += f" for {self.next_design_iter} (threshold = {threshold})"
            plt.title(title)
            
        # Draw and flush the plot to update quickly.
        plt.draw()

        if animate_densities:
            plt.pause(0.0000001)  # Short pause to allow the GUI to catch up

        if save_density:
            plt.savefig(os.path.join(self.OUT_DIR, f"{self.next_design_iter}_iteration_{iteration}.png"))

        # SHOW first and/or final plot if not in headless mode 
        if show_plot and not self.HEADLESS: 
            plt.show()

    """
    This approach is not an approach weighted by force magnitude!
    """
    def get_c_and_dc(self, U, xFilt, penal, K, E0, Emin, KE, edofMat):
            # Calculating both c and dc within a for loop
            c = 0
            dc = 0

            U_in = U[:, 0]

            contact_forces_list = list(self.cps2maxforces_2D.values())
            # Compute the L2 norm of each force vector
            force_norms = np.array([np.linalg.norm(f) for f in contact_forces_list])
            # Compute the softmax distribution
            softmax_weights = np.exp(force_norms) / np.sum(np.exp(force_norms))

            # Iterating from 1, ..., n_contactforces. We start
            # from 1 to correctly index into the U vector.
            for i in np.arange(len(self.cps2maxforces_2D)) + 1:

                U_out = U[:, i]

                # weights = contact_forces_list[i-1]/np.linalg.norm

                # Quantities for c
                MSE = (U_out.T @ K @ U_in).item() # These are correct as they include PERTURBED stiffness matrices K
                SE = (U_out.T @ K @ U_out).item()  # These are correct as they include PERTURBED stiffness matrices K
                c += MSE/(SE)

                # 2. WEIGHTED MSE/SE
                u = (U_out.T @ K @ U_in).item() 
                v = (U_out.T @ K @ U_out).item()

                # Derivative terms below are evaluated in terms of KE matrices, as they are rates of change w.r.t rho, not K.
                u_dash = (penal * xFilt ** (penal - 1) * (E0 - Emin)) * ((U_out[edofMat].reshape(self.nelx * self.nely, 8) @ KE) * U_in[edofMat].reshape(self.nelx * self.nely, 8)).sum(axis=1)
                v_dash = (penal * xFilt ** (penal - 1) * (E0 - Emin)) * ((U_out[edofMat].reshape(self.nelx * self.nely, 8) @ KE) * U_out[edofMat].reshape(self.nelx * self.nely, 8)).sum(axis=1)

                # We have not weighted the terms by corresponding force magnitudes, which is a reasonable idea if this fails
                # dc += (u_dash*v - u*v_dash)/(v**2)
                # dc += softmax_weights[i-1] * (u_dash*v - u*v_dash)/(v**2)
                # dc += force_norms[i-1] * (u_dash*v - u*v_dash)/(v**2) --- #PREVIOUSLY WORKING IMPLEMENTATION
                dc +=  force_norms[i-1] * (u_dash*v - u*v_dash)/(v**2) # Attempt to match benchmark approach
            return c, dc


def main():
    ## =============================== TESTING SIM-TO-SIM-TO
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=1, type=int) # must equal 1 for SDTO current implementation
    parser.add_argument("--fps", default=50, type = int)
    parser.add_argument("--num_frames", default=200, type = int)
    parser.add_argument("--sim_substeps",default=4, type=int)
    parser.add_argument("--dhat",default=1e-6, type=float)
    parser.add_argument("--compress_distance",default=0.085, type=float)
    parser.add_argument("--lift_distance",default=0.02, type=float) 
    parser.add_argument("--TO_volfrac",default=0.3, type=float)
    parser.add_argument("--TO_threshold",default=0.5, type=float)
    parser.add_argument("--TO_max_loops",default=100, type=int) 
    args = parser.parse_args()

    current_design_iter = "D0"
    next_design_iter = get_next_design_iter(current_design_iter)
    OUT_DIR, MESH_DIR = init_robot_demo_SDTO(args, "soft_gripper_SDTO", "box_rho=0.25*1000", current_design_iter = current_design_iter)

    topology_optimizer = TopologyOptimizer2D(nelx = 150, nely = 70, nelz = 30, 
                                            cps2maxforces_3D = None, OUT_DIR = OUT_DIR, 
                                            current_design_iter = current_design_iter,
                                            current_design_mesh_filepath =  os.path.join(MESH_DIR, f"{current_design_iter}_leftfinger.msh"),
                                            animation_force_mag=0.01,
                                            HEADLESS = True, VERBOSE = True)

    # =============================== PERFORMING 2.5DTO & CONVERTING THE EXTRUDED 3D DESIGN (np.ndarray) ---> STL ---> GMSH
    binary_3D_design = topology_optimizer.SDTO_2D(volfrac = args.TO_volfrac, penal = 3, rmin = 7, ft = 1, threshold = args.TO_threshold, max_loops = args.TO_max_loops,
                                                  angle = 45, E = 1)
    # voxel_3D_binary_array_to_stl(binary_3D_design, voxel_size=0.001, 
    #                             save_filepath= os.path.join(MESH_DIR, f"{next_design_iter}_leftfinger.stl"))
    # run_fTetWild(os.path.join(MESH_DIR, f"{next_design_iter}_leftfinger.stl"), os.path.join(MESH_DIR, f"{next_design_iter}_leftfinger.msh"), 
    #             executable_path="../fTetWild/build/FloatTetwild_bin")
    # translate_fTetWild_msh(os.path.join(MESH_DIR, f"{next_design_iter}_leftfinger.msh"))

    ## =============================== CREATING BENCHMARK-TO DESIGN
    # OUT_DIR = os.path.join(os.getcwd(), "assets", "objects", "soft_gripper_benchmarkTO")
    # binary_3D_design =create_soft_gripper_benchmarkTO(nelx = 150,nely = 70,nelz = 30, volfrac = 0.3, penal = 3, rmin = 7, ft = 1, 
    #                                 threshold = 0.5, OUT_DIR=OUT_DIR, max_loops=50)
    
    # voxel_3D_binary_array_to_stl(binary_3D_design, voxel_size=0.001, 
    #                             save_filepath= os.path.join(OUT_DIR, f"benchmark_leftfinger.stl"))
    # run_fTetWild(os.path.join(OUT_DIR, f"benchmark_leftfinger.stl"), os.path.join(OUT_DIR, f"benchmark_leftfinger.msh"), 
    #             executable_path="../fTetWild/build/FloatTetwild_bin")
    # # Offset .msh generated by fTetWild by [0, 0.07, 0.03] to ensure we have our desired Cartesian coordinate system for TO and simulation/kabsch.
    # translate_fTetWild_msh(os.path.join(OUT_DIR, f"benchmark_leftfinger.msh"))
if __name__ == "__main__":
    main()