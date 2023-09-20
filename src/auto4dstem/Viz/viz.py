import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
from tqdm import tqdm
import torch
from scipy.linalg import polar
import scipy as sp
from dataclasses import dataclass, field

# plt.rcParams['axes.titlesize'] = 20
# #plt.rcParams['image.camp'] = 'viridis'
# plt.rcParams['axes.labelsize'] = 15
# plt.rcParams['xtick.labelsize'] = 20
# plt.rcParams['ytick.labelsize'] = 20
# plt.rcParams['figure.titlesize'] = 8
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['xtick.top'] = True
# plt.rcParams['ytick.right'] = True
# plt.rcParams['lines.markersize'] = .5
# plt.rcParams['axes.grid'] = False
# plt.rcParams['lines.linewidth'] = .5
# plt.rcParams['axes.linewidth'] = .5
# plt.rcParams['legend.fontsize'] = 5
# plt.rcParams['legend.loc'] = "upper left"
# plt.rcParams['legend.frameon'] = False



def basis2probe(rotation_,
                scale_shear_):
    """_summary_

    Args:
        rotation_ (numpy.array): rotation matrix [batch, cos, sin]
        scale_shear_ (numpy.array): strain matrix [batch,]

    Returns:
        _type_: _description_
    """

    M = []
    #    G_ref_inv = np.linalg.inv(G_ref)
    for i in tqdm(range(65536),leave=True,total=65536):

        theta = np.arctan2(rotation_[i][1], rotation_[i][0]) 

        xx = scale_shear_[i][0]
        yy = scale_shear_[i][3]

        xy = scale_shear_[i][1]
        yx = scale_shear_[i][2]


        r = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
            ])
        t = np.array([
            [xx,xy],
            [yx,yy]
            ])
        m = np.linalg.inv(t) @ np.linalg.inv(r)

        M.append(m)

    M  = np.array(M)
    return M


#  rotation comparison
def compare_rotation(strain_map,
                        rotation_,
                        noise_intensity=0,
                        angle_shift=0
                    ):
    """_summary_

    Args:
        strain_map (_type_): _description_
        rotation_ (_type_): _description_
        noise_intensity (int, optional): _description_. Defaults to 0.
        angle_shift (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    noise_format = format(noise_intensity,'.2f')
    theta_Colin = np.mod(np.rad2deg(strain_map[:,:,3]),60)

    theta_Shuyu = np.mod( angle_shift + \
        1*np.rad2deg(np.arctan2(
            rotation_[:,1].reshape(256,256),
            rotation_[:,0].reshape(256,256))),
        60.0
        )


    fig,ax = plt.subplots(2,2, figsize = (12,12))

    fig.suptitle('Rotation Comparison on '+noise_format+' Background Noise', fontsize=30)

    ax[0,0].title.set_text('Rotation: Py4dstem')
    ax[0,0].set_xticklabels('')
    ax[0,0].set_yticklabels('')

    ax[0,1].title.set_text('Neural Network')
    ax[0,1].set_xticklabels('')
    ax[0,1].set_yticklabels('')

    ax[0,0].imshow(
        theta_Colin,
        cmap='RdBu_r',
        vmin=0,
        vmax=60.0,
    )

    ax[1,0].hist(theta_Colin.reshape(-1),200,range=[0,60]);
    ax[0,1].imshow(
        theta_Shuyu,
        cmap='RdBu_r',
        vmin=0,
        vmax=60.0,
    )

    ax[1,1].hist(theta_Shuyu.reshape(-1),200,range=[0,60]);

    fig.tight_layout()

    return theta_Colin, theta_Shuyu


# Solve for strain tensor
def strain_tensor(M_init,
                im_size):
    """_summary_

    Args:
        M_init (_type_): _description_
        im_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    M_ref = np.median(M_init[30:60,10:40],axis=(0,1))


    exx_Shuyu = np.zeros((im_size[0],im_size[1]))
    eyy_Shuyu = np.zeros((im_size[0],im_size[1]))
    exy_Shuyu = np.zeros((im_size[0],im_size[1]))

    for rx in range(im_size[0]):
        for ry in range(im_size[1]):

            T = M_init[rx,ry] @ np.linalg.inv(M_ref)
            u, p = sp.linalg.polar(T, side='left')

            transformation = np.array([
                [p[0,0] - 1, p[0,1]],
                [p[0,1], p[1,1] - 1],
            ])

            exx_Shuyu[rx,ry] = transformation[1,1]
            eyy_Shuyu[rx,ry] = transformation[0,0]
            exy_Shuyu[rx,ry] = transformation[0,1]

    return exx_Shuyu,eyy_Shuyu,exy_Shuyu


#  strain comparisons
def Strain_Compare(diff_list,
                    diff_range=[-0.03,0.03],
                    rotation_range=[-40,30],
                    noise_intensity=0,
                    data_index = None):
    """_summary_

    Args:
        diff_list (_type_): _description_
        diff_range (list, optional): _description_. Defaults to [-0.03,0.03].
        rotation_range (list, optional): _description_. Defaults to [-40,30].
        noise_intensity (int, optional): _description_. Defaults to 0.
        data_index (_type_, optional): _description_. Defaults to None.
    """

    fig,ax = plt.subplots(4,4, figsize = (24,24))

    noise_format = format(noise_intensity,'.2f')
    fig.suptitle('Performance Comparison on '+noise_format+' Background Noise', fontsize=25)
    ax[0,0].title.set_text('Py4dstem: Strain X')
    ax[1,0].title.set_text('Strain Y')
    ax[2,0].title.set_text('Shear')
    ax[3,0].title.set_text('Rotation')
    ax[0,2].title.set_text('Neural Network: Strain X')
    ax[1,2].title.set_text('Strain Y')
    ax[2,2].title.set_text('Shear')
    ax[3,2].title.set_text('Rotation')

    for i in range(8):
        if int(i/4)==0:
            row = i
            col = 0
        else:
            row = i-4
            col=2

        if i%4==3:
            value_range = rotation_range
        else:
            value_range = diff_range

        ax[row,col].set_xticklabels('')
        ax[row,col].set_yticklabels('')

        ax[row,col].imshow(
            diff_list[i],
            cmap = 'RdBu_r',
            clim = value_range
        )

        if data_index == None:  

            mae_ = np.mean(abs(diff_list[i].reshape(-1)))
            ax[row,col+1].hist(diff_list[i].reshape(-1),200,range=value_range);

        else: 
            mae_ = np.mean(abs(diff_list[i].reshape(-1)[data_index]))
            ax[row,col+1].hist(diff_list[i].reshape(-1)[data_index],200,range=value_range);

#    fig.tight_layout()


    plt.savefig('Strain_Map_'+noise_format+'Percent_BKG'+'.svg')

def cal_diff(exx_Colin,eyy_Colin,exy_Colin,theta_Colin,
                exx_Shuyu,eyy_Shuyu,exy_Shuyu,theta_Shuyu,
                label_xx,label_yy,label_xy,label_rotation):
    """_summary_

    Args:
        exx_Colin (_type_): _description_
        eyy_Colin (_type_): _description_
        exy_Colin (_type_): _description_
        theta_Colin (_type_): _description_
        exx_Shuyu (_type_): _description_
        eyy_Shuyu (_type_): _description_
        exy_Shuyu (_type_): _description_
        theta_Shuyu (_type_): _description_
        label_xx (_type_): _description_
        label_yy (_type_): _description_
        label_xy (_type_): _description_
        label_rotation (_type_): _description_

    Returns:
        _type_: _description_
    """
    dif_colin_xx = exx_Colin - label_xx
    dif_colin_yy = eyy_Colin - label_yy
    dif_colin_xy = exy_Colin - label_xy
    dif_colin_rotation = theta_Colin - label_rotation

    dif_shuyu_xx = exx_Shuyu - label_xx
    dif_shuyu_yy = eyy_Shuyu - label_yy
    dif_shuyu_xy = exy_Shuyu - label_xy
    dif_shuyu_rotation = theta_Shuyu - label_rotation

    return [dif_colin_xx,dif_colin_yy,dif_colin_xy,dif_colin_rotation,dif_shuyu_xx,dif_shuyu_yy,dif_shuyu_xy,dif_shuyu_rotation]


def MAE_diff_with_Label(diff_list,
                        diff_range,
                        rotation_range,
                        noise_intensity=0,
                        data_index = None
                        ):
    """_summary_

    Args:
        diff_list (_type_): _description_
        diff_range (_type_): _description_
        rotation_range (_type_): _description_
        noise_intensity (int, optional): _description_. Defaults to 0.
        data_index (_type_, optional): _description_. Defaults to None.
    """

    fig,ax = plt.subplots(4,4, figsize = (24,24))
    noise_format = format(noise_intensity,'.2f')
    fig.suptitle('MAE Comparison on '+noise_format+' Background Noise', fontsize=25)

    ax[0,0].title.set_text('Py4dstem: Strain X')
    ax[1,0].title.set_text('Strain Y')
    ax[2,0].title.set_text('Shear')
    ax[3,0].title.set_text('Rotation')
    ax[0,2].title.set_text('Neural Network: Strain X')
    ax[1,2].title.set_text('Strain Y')
    ax[2,2].title.set_text('Shear')
    ax[3,2].title.set_text('Rotation') 

    for i in range(8):

        if int(i/4)==0:
            row = i
            col = 0
        else:
            row = i-4
            col=2

        if i%4==3:
            value_range = rotation_range
        else:
            value_range = diff_range

        ax[row,col].set_xticklabels('')
        ax[row,col].set_yticklabels('')

        ax[row,col].imshow(
            diff_list[i],
            cmap = 'RdBu_r',
            clim = value_range
        )

        if data_index == None:  

            mae_ = np.mean(abs(diff_list[i].reshape(-1)))
            ax[row,col+1].hist(diff_list[i].reshape(-1),200,range=value_range);

        else: 
            mae_ = np.mean(abs(diff_list[i].reshape(-1)[data_index]))
            ax[row,col+1].hist(diff_list[i].reshape(-1)[data_index],200,range=value_range);

        ax[row,col+1].title.set_text('MAE: '+format(mae_,'.4f'))

#    fig.tight_layout()

    plt.savefig('Performance_Comparison_'+noise_format+'Percent_BKG'+'.svg')
    
    
@dataclass
class visualize_result:
    rotation: any 
    scale_shear: any
    file_py4DSTEM: str
    label_rotation_path: str = 'rotation_label_2.npy'
    label_xx_path: str = 'Label_strain_xx.npy'
    label_yy_path: str = 'Label_strain_yy.npy'
    label_xy_path: str = 'Label_shear_xy.npy'
    noise_intensity: float = 0.0
    angle_shift: float = 0
    im_size: any = (256,256)
    strain_diff_range: list[float] = field(default_factory=list)
    strain_rotation_range: list[float] = field(default_factory=list)
    mae_diff_range: list[float] = field(default_factory=list)
    mae_rotation_range: list[float] = field(default_factory=list)
    
    def __post_init__(self):
        
        label_rotation = np.load(self.label_rotation_path)
        label_rotation = np.rad2deg(label_rotation)
        label_ref_rotation = np.mean(label_rotation[30:60,10:40])
        self.label_rotation = label_rotation - label_ref_rotation
        self.label_xx = np.load(self.label_xx_path)
        self.label_yy = np.load(self.label_yy_path)
        self.label_xy = np.load(self.label_xy_path)
        
        f= h5py.File(self.file_py4DSTEM)
        self.strain_map = f['4DSTEM_experiment']['data']['realslices']['strain_map']['data'][:]
        
        self.theta_Colin,self.theta_Shuyu = compare_rotation(self.strain_map,
                                                            self.rotation,
                                                            noise_intensity=self.noise_intensity,
                                                            angle_shift=self.angle_shift)
        
        self.theta_ref_Colin = np.mean(self.theta_Colin[30:60,10:40])
        self.theta_Colin = self.theta_Colin - self.theta_ref_Colin
        self.theta_ref_Shuyu = np.mean(self.theta_Shuyu[30:60,10:40])
        self.theta_Shuyu = self.theta_Shuyu - self.theta_ref_Shuyu
        
        self.M_init = basis2probe(self.rotation,self.scale_shear).reshape(self.im_size[0],self.im_size[1],2,2)
        
    def reset_baseline(self):
        
        f= h5py.File(self.file_py4DSTEM)
        self.strain_map = f['4DSTEM_experiment']['data']['realslices']['strain_map']['data'][:]
        
    def reset_angle(self,angle_shift):
        self.angle_shift = angle_shift
        
        self.theta_Colin,self.theta_Shuyu = compare_rotation(self.strain_map,
                                                            self.rotation,
                                                            noise_intensity=self.noise_intensity,
                                                            angle_shift=self.angle_shift)
        
        self.theta_ref_Colin = np.mean(self.theta_Colin[30:60,10:40])
        self.theta_Colin = self.theta_Colin - self.theta_ref_Colin
        self.theta_ref_Shuyu = np.mean(self.theta_Shuyu[30:60,10:40])
        self.theta_Shuyu = self.theta_Shuyu - self.theta_ref_Shuyu
        
    def reset_polar_matrix(self):
        
        self.M_init = basis2probe(self.rotation,self.scale_shear).reshape(self.im_size[0],self.im_size[1],2,2)
    
    def visual_strain(self):
        self.exx_Shuyu,self.eyy_Shuyu,self.exy_Shuyu = strain_tensor(self.M_init,self.im_size)
        self.exx_Colin = self.strain_map[:,:,0]
        self.eyy_Colin = self.strain_map[:,:,1]
        self.exy_Colin = self.strain_map[:,:,2]
        strain_list = [self.exx_Colin,self.eyy_Colin,self.exy_Colin,self.theta_Colin,
                        self.exx_Shuyu,self.eyy_Shuyu,self.exy_Shuyu,self.theta_Shuyu]
        Strain_Compare(strain_list,
                        diff_range=self.strain_diff_range,
                        rotation_range=self.strain_rotation_range,
                        noise_intensity=self.noise_intensity)
        
    def visual_diff(self):
        
        diff_list = cal_diff(self.exx_Colin,self.eyy_Colin,self.exy_Colin,self.theta_Colin,
                            self.exx_Shuyu,self.eyy_Shuyu,self.exy_Shuyu,self.theta_Shuyu,
                            self.label_xx,self.label_yy,self.label_xy,self.label_rotation)

        
        MAE_diff_with_Label(diff_list,
                            diff_range=self.mae_diff_range,
                            rotation_range=self.mae_rotation_range,
                            noise_intensity=self.noise_intensity,
                            data_index = None)
        
    