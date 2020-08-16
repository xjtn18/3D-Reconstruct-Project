import numpy as np
import camutils
import visutils
import pickle
from scipy.spatial import Delaunay

def writeply(X,color,tri,filename):
    """
    Save out a triangulated mesh to a ply file
    
    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        vertex coordinates shape (3,Nvert)
        
    color : 2D numpy.array (dtype=float)
        vertex colors shape (3,Nvert)
        should be float in range (0..1)
        
    tri : 2D numpy.array (dtype=float)
        triangular faces shape (Ntri,3)
        
    filename : string
        filename to save to    
    """
    f = open(filename,"w");
    f.write('ply\n');
    f.write('format ascii 1.0\n');
    f.write('element vertex %i\n' % X.shape[1]);
    f.write('property float x\n');
    f.write('property float y\n');
    f.write('property float z\n');
    f.write('property uchar red\n');
    f.write('property uchar green\n');
    f.write('property uchar blue\n');
    f.write('element face %d\n' % tri.shape[0]);
    f.write('property list uchar int vertex_indices\n');
    f.write('end_header\n');

    C = (255*color).astype('uint8')
    
    for i in range(X.shape[1]):
        f.write('%f %f %f %i %i %i\n' % (X[0,i],X[1,i],X[2,i],C[0,i],C[1,i],C[2,i]));
    
    for t in range(tri.shape[0]):
        f.write('3 %d %d %d\n' % (tri[t,1],tri[t,0],tri[t,2]))

    f.close();


def prune(pts2L, pts2R, pts3, colors, trithresh):
    # Mesh cleanup parameters

    # Specify limits along the x,y and z axis of a box containing the object
    # we will prune out triangulated points outside these limits
    boxlimits = np.array([-12,24,-5,25,-25,-8])

    # Specify a longest allowed edge that can appear in the mesh. Remove triangles
    # from the final mesh that have edges longer than this value


    #
    # bounding box pruning
    #

    p3 = pts3

    dx = np.where( \
        (p3[0] < boxlimits[0]) \
        | (p3[0] > boxlimits[1]) \
        | (p3[1] < boxlimits[2]) \
        | (p3[1] > boxlimits[3]) \
        | (p3[2] < boxlimits[4]) \
        | (p3[2] > boxlimits[5]) )

    p3 = np.delete(pts3, dx, axis=1)
    p2L = np.delete(pts2L, dx, axis=1)
    p2R = np.delete(pts2R, dx, axis=1)
    colors = np.delete(colors, dx, axis=1)


    #
    # triangulate the 2D points to get the surface mesh
    #

    tri = Delaunay(p2L.T).simplices

    #
    # triangle pruning
    #

    bad_tri = set( )
    for x, tr in enumerate(tri):
        for i in range(3):
            v1_idx = tr[i]
            v2_idx = tr[(i+1)%3] #wrap around

            p0 = p3[:, v1_idx]
            p1 = p3[:, v2_idx]

            if np.linalg.norm(p1-p0) > trithresh: # if this triangle has an edge that is too long
                bad_tri.add(x) # remove this triangle
                break

    tri_prune = np.delete(tri, list(bad_tri), axis=0)


    #
    # remove any points which are not refenced in any triangle
    #

    tokeep = np.unique(tri_prune)
    p3 = p3[:, tokeep]
    colors_prune = colors[:, tokeep] # keep the colors array in sync with the points array

    remap = np.negative(np.ones(len(tri_prune)))
    remap[tokeep] = np.arange(len(tokeep))

    tri_prune = remap[tri_prune]

    return p3, tri_prune, colors_prune



def create_mesh(imprefixL, imprefixR, camL, camR, code_thresh, tri_thresh, grab_number, fid):
    # creates a mesh given using the given image paths and stores them into resultfile using pickle
    pts2L,pts2R,pts3,colors = camutils.reconstruct(imprefixL, imprefixR, code_thresh, camL, camR, grab_number) # reconstruct the points to get their 3D locations
    p3, tri_prune, colors_prune = prune(pts2L, pts2R, pts3, colors, tri_thresh) # prune out larg triangles and unneccesary points
    mesh_data = {'points' : p3, 'triangles' : tri_prune, 'colors' : colors_prune} # creates a dictionary to store the mesh data
    visutils.vis_scene(camL, camR, p3) # visualizes the point cloud in 2d and 3d (for the jupyter notebook)
    pickle.dump(mesh_data,fid) # writes the mesh to the giben pickle file



def smooth_mesh(p3, tri, repeat = 1):
    """smooth the points of the mesh so they arent jaggedy
    p3 : the points we want to smooth
    tri : the triangles of the mesh; uses to get the neighbors of each point
    repeat: int representing the number of times it should run the smoothing algorithm
    """
    for r in range(repeat):
        for x in range(p3.shape[1]):
            # we need to search for "x" in the triangles list
            # all triangles that have "x" will also contiain the point index to point "x"'s neighbors
            x_tri = tri[np.any(tri==x,axis=1),:] # get a subarray of all triangles that have point "x"
            x_tri = x_tri.astype(np.int)
            neighbors = p3[:, np.unique(x_tri)]
            avg = np.mean(neighbors, axis=1) # mean location
            p3[:,x] = avg # update current point to be that mean location
    return p3
