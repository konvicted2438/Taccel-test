import os
import sys
import numpy as np
import pyvista as pv

class MeshContactForcesAnimator(object):

    # Class attributes:
    # --- fps = frames per sec
    # --- contact list = contains (a) None if no contacts at a given rendered frame, or
    #     (b) (positions, forces), where positions is a (n_contacts,3) array and forces is also,
    #     containing the corresponding forces
    # --- n_frames = len(video_frames) - MuJoCo simulation. (!= n_steps)

    def __init__(self, mesh_filepath, contact_list, video_dir = None, fps = 10, n_frames = 30,
                 output_filename = "contacts_animation.mp4"):

        self.fps = fps
        self.n_frames = n_frames
        self.contact_list = contact_list
        self.mesh = pv.read(mesh_filepath)

        # Initialising save_dir
        self.video_dir = video_dir
        self.output_abspath = os.path.join(video_dir, output_filename)

    def animate(self, force_mag: float):

        pl = pv.Plotter(off_screen = True) #optional headless rendering --- TOGGLE False when you want to manipulate the animation live
        pl.open_movie(self.output_abspath, framerate = self.fps)
        pl.add_axes()
        pl.show_bounds()

        # Set up animation
        for ifrm in range(self.n_frames):

            pl.clear()

            # Elements in contact_list are either: (a) None if no contacts, or 
            # (b) (positions, forces), where positions is a (n_contacts,3) array and forces is also,
            # containing the corresponding forces
            positions_and_forces = self.contact_list[ifrm]
            if positions_and_forces is not None:
                positions = positions_and_forces[0]
                forces = positions_and_forces[1]
                pl.add_title(f"t = {ifrm * 1/self.fps:.2f}s", font_size=20)
                pl.add_mesh(self.mesh, color='green', opacity = 0.2, show_edges=True, 
                            edge_color = 'black', line_width = 0.5)
                pl.add_points(positions, color = 'red', point_size = 10)
                pl.add_arrows(positions, forces, mag = force_mag, show_scalar_bar = False, color = 'purple')

            else:
                # Plot only mesh
                pl.add_title(f"t = {ifrm * 1/self.fps:.2f}s", font_size=20)
                pl.add_mesh(self.mesh, color='green', opacity = 0.2, show_edges=True,
                            edge_color = 'black', line_width = 0.5)


            # Adjust frame rate (optional)
            pl.write_frame()

        # Save animation (optional)
        pl.close()

class MeshContactPointsAnimator(object):

    # Class attributes:
    # --- fps = frames per sec
    # --- contact list = contains (a) None if no contacts at a given rendered frame, or
    #     (b) (positions), where positions is a (n_contacts,3) array 
    # --- n_frames = len(video_frames) - MuJoCo simulation. (!= n_steps)

    def __init__(self, mesh_filepath, contact_list, video_dir = None, fps = 10, n_frames = 30,
                 output_filename = "contacts_animation.mp4"):

        self.fps = fps
        self.n_frames = n_frames
        self.contact_list = contact_list
        self.mesh = pv.read(mesh_filepath)

        # Initialising save_dir
        self.video_dir = video_dir
        self.output_abspath = os.path.join(video_dir, output_filename)

    def animate(self, force_mag: float):

        pl = pv.Plotter(off_screen = False) #optional headless rendering --- TOGGLE False when you want to manipulate the animation live
        
        pl.open_movie("test.mp4", framerate = self.fps)
        pl.add_axes()
        pl.show_bounds()

        # Set up animation
        for ifrm in range(self.n_frames):

            pl.clear()

            # Elements in contact_list are either: (a) None if no contacts, or 
            # (b) positions: a (n_contacts,3) array 
            positions = self.contact_list[ifrm]
            if positions is not None:
                pl.add_title(f"t = {ifrm * 1/self.fps}s", font_size = 20)
                pl.add_mesh(self.mesh, color='green', opacity = 0.2, show_edges=True, 
                            edge_color = 'black', line_width = 0.5)
                pl.add_points(positions, color = 'red', point_size = 10)
            else:
                pl.add_title(f"t = {ifrm * 1/self.fps}s", font_size = 20)
                # Plot only mesh
                pl.add_mesh(self.mesh, color='green', opacity = 0.2, show_edges=True,
                            edge_color = 'black', line_width = 0.5)


            # Adjust frame rate (optional)
            pl.write_frame()

        # Save animation (optional)
        pl.close()


def main():


    left_softfinger_filepath = "assets/objects/soft_gripper/D0_leftfinger_vol.msh"
    left_softfinger = pv.read(left_softfinger_filepath)
    left_softfinger_nodeids = np.arange(left_softfinger.n_points, dtype=np.int32)
    left_softfinger_points_restframe = np.array(left_softfinger.copy().points)
    gripper_ids2xyz = {id: xyz for id, xyz in enumerate(left_softfinger_points_restframe)}

    # Plot all nodes in the mesh at every frame
    CPS = [gripper_ids2xyz[id] for id in np.arange(len(gripper_ids2xyz))]
    CPS_list = [np.array(CPS) for _ in range(300)]
    CPS_animator = MeshContactPointsAnimator(left_softfinger_filepath , CPS_list, video_dir = os.getcwd(), fps = 50, 
                                                n_frames = 300, output_filename = "test.mp4")
    CPS_animator.animate(force_mag = None) 



if __name__ == "__main__":
    main()
