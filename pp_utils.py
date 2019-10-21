import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import transform
from config import config

MIN_BOUND = int(config['pp_normalize_min'])
MAX_BOUND = int(config['pp_normalize_max'])
PIXEL_MEAN = float(config['pp_center_pixel_mean'])
MARGIN = int(config['pp_box_margin'])
MODE = config['mode']

def worldToVoxelCoord(worldCoord, origin, old_spacing, new_spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin) / old_spacing
    spacing = np.array(old_spacing, dtype=np.float32)
    resize_factor = spacing / new_spacing
    voxelCoord = stretchedVoxelCoord * resize_factor
    return voxelCoord

def get_box_from_mask(mask):
    xx,yy,zz= np.where(mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = np.floor(box).astype('int')
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-MARGIN],0),np.min([mask.shape,box[:,1]+2*MARGIN],axis=0).T]).T
    extendbox = extendbox.astype('int')
    return extendbox
    
def display(img, ind, lib='', roi=[]):
    if MODE != 'dev':
        return
    plt.figure(figsize=(8,8))
    plt.title = str(img.shape) + ' slice: ' + str(ind)
    plt.text(0.2, 0.2, str(img.shape) + ' slice: ' + str(ind))
    plt.imshow(img, cmap=plt.cm.gray)
    if roi != []:
        plt.plot(roi[2], roi[1], 'or')
    plt.show()
    
def plot_3d(image, threshold=0):
    if MODE != 'dev':
        return
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
    
def resample(image, old_spacing, new_spacing):
    spacing = np.array(old_spacing, dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape).astype(np.int16)

    # Avec skimage
    image =  transform.resize(image.astype(float), new_shape, order=1, clip=True, mode='edge', preserve_range=True) > 0

    return image, new_spacing

def load_itk_image(filename):
    # coords in z,y,x order
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage).astype(np.int16)
    if isflip:
        print("Scan/mask %s is flip!" % filename)
        numpyImage = numpyImage[:,::-1,::-1]
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return (image*255).astype('uint8')

def zero_center(image):
    image = image - PIXEL_MEAN
    return image