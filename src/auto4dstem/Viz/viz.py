import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
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
    for i in tqdm(range(rotation_.shape[0]),leave=True,total=rotation_.shape[0]):

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

# visualize model trained rotation
def visual_rotation(rotation_,
                    classification = None,
                    bkg_index = None,
                    sample_index = None,
                    title_name='WS2WSe2',
                    angle_shift=0,
                    img_size = (256,256),
                    clim = [0,60]
                    ):
    """_summary_

    Args:
        rotation_ (_type_): _description_
        classification (_type_, optional): _description_. Defaults to None.
        bkg_index (_type_, optional): _description_. Defaults to None.
        sample_index (_type_, optional): _description_. Defaults to None.
        title_name (str, optional): _description_. Defaults to 'WS2WSe2'.
        angle_shift (int, optional): _description_. Defaults to 0.
        img_size (tuple, optional): _description_. Defaults to (256,256).
        clim (list, optional): _description_. Defaults to [0,60].

    Returns:
        _type_: _description_
    """
    
    if type(title_name) == str:
        name_ = title_name
    else:
        name_ = format(title_name,'.2f')+' Background Noise'
        
    temp_ae = np.mod( angle_shift + \
                        1*np.rad2deg(np.arctan2(
                            rotation_[:,1].reshape(-1),
                            rotation_[:,0].reshape(-1))),
                        60.0
                        )
    if classification is not None:
        
        theta_ae = np.zeros([img_size[0]*img_size[1]])
        theta_ae[sample_index] = temp_ae
    else:
        
        theta_ae = np.copy(temp_ae)
    
    theta_ae = theta_ae.reshape(img_size)
        

    fig,ax = plt.subplots(1,2, figsize = (10,5))

    fig.suptitle('Rotation on '+ name_, fontsize=30)


    ax[0].title.set_text('Neural Network')
    ax[0].set_xticklabels('')
    ax[0].set_yticklabels('')
    
    ax[0].imshow(
        theta_ae,
        cmap='RdBu_r',
        vmin=clim[0],
        vmax=clim[1],
    )

    ax[1].hist(theta_ae.reshape(-1),200,range=clim);

    fig.tight_layout()

    return theta_ae


#  rotation comparison
def compare_rotation(strain_map,
                    rotation_,
                    classification = None,
                    bkg_index = None,
                    sample_index = None,
                    title_name='WS2WSe2',
                    angle_shift=0,
                    shift_ref = 0,
                    img_size = (256,256),
                    clim = [0,60]
                    ):
    """_summary_

    Args:
        strain_map (_type_): _description_
        rotation_ (_type_): _description_
        classification (_type_, optional): _description_. Defaults to None.
        bkg_index (_type_, optional): _description_. Defaults to None.
        sample_index (_type_, optional): _description_. Defaults to None.
        title_name (str, optional): _description_. Defaults to 'WS2WSe2'.
        angle_shift (int, optional): _description_. Defaults to 0.
        shift_ref (int, optional): _description_. Defaults to 0.
        img_size (tuple, optional): _description_. Defaults to (256,256).
        clim (list, optional): _description_. Defaults to [0,60].

    Returns:
        _type_: _description_
    """
    
    if type(title_name) == str:
        name_ = title_name
    else:
        name_ = format(title_name,'.2f')+' Background Noise'
        
    theta_correlation = np.mod(shift_ref+np.rad2deg(strain_map[:,:,3]),60).reshape(-1)
    temp_ae = np.mod( angle_shift + \
                        1*np.rad2deg(np.arctan2(
                            rotation_[:,1].reshape(-1),
                            rotation_[:,0].reshape(-1))),
                        60.0
                        )
    if classification is not None:
        
        theta_correlation[bkg_index] = 0
        theta_ae = np.zeros([img_size[0]*img_size[1]])
        theta_ae[sample_index] = temp_ae
    else:
        
        theta_ae = np.copy(temp_ae)
    
    theta_correlation = theta_correlation.reshape(img_size)
    theta_ae = theta_ae.reshape(img_size)
        

    fig,ax = plt.subplots(2,2, figsize = (12,12))

    fig.suptitle('Rotation Comparison on '+ name_, fontsize=30)

    ax[0,0].title.set_text('Rotation: Py4dstem')
    ax[0,0].set_xticklabels('')
    ax[0,0].set_yticklabels('')

    ax[0,1].title.set_text('Neural Network')
    ax[0,1].set_xticklabels('')
    ax[0,1].set_yticklabels('')

    ax[0,0].imshow(
        theta_correlation,
        cmap='RdBu_r',
        vmin=clim[0],
        vmax=clim[1],
    )

    ax[1,0].hist(theta_correlation.reshape(-1),200,range=clim);
    ax[0,1].imshow(
        theta_ae,
        cmap='RdBu_r',
        vmin=clim[0],
        vmax=clim[1],
    )

    ax[1,1].hist(theta_ae.reshape(-1),200,range=clim);

    fig.tight_layout()

    return theta_correlation, theta_ae


def strain_tensor_for_real(M_init,
                        im_size,
                        sample_index = None,
                        ):
    """_summary_

    Args:
        M_init (_type_): _description_
        im_size (_type_): _description_
        sample_index (_type_):
    Returns:
        _type_: _description_
    """
    exx_ae = np.zeros([im_size[0]*im_size[1]])-1
    eyy_ae = np.zeros([im_size[0]*im_size[1]])-1
    exy_ae = np.zeros([im_size[0]*im_size[1]])-1

    exx_ = np.zeros([M_init.shape[0]])
    eyy_ = np.zeros([M_init.shape[0]])
    exy_ = np.zeros([M_init.shape[0]])

    for i in tqdm(range(M_init.shape[0])):

        T = M_init[i]
        u, p = sp.linalg.polar(T, side='left')

        transformation = np.array([
            [p[0,0] - 1, p[0,1]],
            [p[0,1], p[1,1] - 1],
        ])

    #       transformation = u @ transformation @ u.T

        exx_[i] = transformation[1,1]
        eyy_[i] = transformation[0,0]
        exy_[i] = transformation[0,1]
        
    if M_init.shape[0] == im_size[0]*im_size[1]:
        
        exx_ae = np.copy(exx_)
        eyy_ae = np.copy(eyy_)
        exy_ae = np.copy(exy_)
        
    else:
        
        exx_ae[sample_index] = exx_
        eyy_ae[sample_index] = eyy_
        exy_ae[sample_index] = exy_
        
    
    exx_ae = exx_ae.reshape(im_size)
    eyy_ae = eyy_ae.reshape(im_size)
    exy_ae = exy_ae.reshape(im_size)
    
    

    return exx_ae,eyy_ae,exy_ae

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


    exx_ae = np.zeros((im_size[0],im_size[1]))
    eyy_ae = np.zeros((im_size[0],im_size[1]))
    exy_ae = np.zeros((im_size[0],im_size[1]))

    for rx in range(im_size[0]):
        for ry in range(im_size[1]):

            T = M_init[rx,ry] @ np.linalg.inv(M_ref)
            u, p = sp.linalg.polar(T, side='left')

            transformation = np.array([
                [p[0,0] - 1, p[0,1]],
                [p[0,1], p[1,1] - 1],
            ])

            exx_ae[rx,ry] = transformation[1,1]
            eyy_ae[rx,ry] = transformation[0,0]
            exy_ae[rx,ry] = transformation[0,1]

    return exx_ae,eyy_ae,exy_ae

def Real_Strain_Viz(diff_list,
                    ae_xx_diff_range,
                    ae_yy_diff_range,
                    ae_xy_diff_range,
                    rotation_range=[-40,30],
                    data_index = None):
    """_summary_

    Args:
        diff_list (_type_): _description_
        ae_xx_diff_range (_type_): _description_
        ae_yy_diff_range (_type_): _description_
        ae_xy_diff_range (_type_): _description_
        rotation_range (list, optional): _description_. Defaults to [-40,30].
        data_index (_type_, optional): _description_. Defaults to None.
    """


    fig,ax = plt.subplots(4,2, figsize = (12,24))

    fig.suptitle('Strain Map of Experimental 4DSTEM', fontsize=25)
    ax[0,0].title.set_text('Neural Network: Strain X')
    ax[1,0].title.set_text('Strain Y')
    ax[2,0].title.set_text('Shear')
    ax[3,0].title.set_text('Rotation')

    
    diff_range_list = [ae_xx_diff_range, ae_yy_diff_range, ae_xy_diff_range, rotation_range]

    for i in range(4):

        value_range = diff_range_list[i]


        ax[i,0].set_xticklabels('')
        ax[i,0].set_yticklabels('')

        ax[i,0].imshow(
            diff_list[i],
            cmap = 'RdBu_r',
            clim = value_range
        )

        if data_index == None:  

            mae_ = np.mean(abs(diff_list[i].reshape(-1)))
            ax[i,1].hist(diff_list[i].reshape(-1),200,range=value_range);

        else: 
            mae_ = np.mean(abs(diff_list[i].reshape(-1)[data_index]))
            ax[i,1].hist(diff_list[i].reshape(-1)[data_index],200,range=value_range);

#    fig.tight_layout()


    plt.savefig('Strain_Map_of_Experimental_4DSTEM'+'.svg')


#  strain comparisons
def Strain_Compare(diff_list,
                    ae_xx_diff_range,
                    ae_yy_diff_range,
                    ae_xy_diff_range,
                    cross_xx_diff_range,
                    cross_yy_diff_range,
                    cross_xy_diff_range,
                    rotation_range=[-40,30],
                    title_name=0,
                    data_index = None):
    """_summary_

    Args:
        diff_list (_type_): _description_
        ae_xx_diff_range (_type_): _description_
        ae_yy_diff_range (_type_): _description_
        ae_xy_diff_range (_type_): _description_
        cross_xx_diff_range (_type_): _description_
        cross_yy_diff_range (_type_): _description_
        cross_xy_diff_range (_type_): _description_
        rotation_range (list, optional): _description_. Defaults to [-40,30].
        title_name (int, optional): _description_. Defaults to 0.
        data_index (_type_, optional): _description_. Defaults to None.
    """

    fig,ax = plt.subplots(4,4, figsize = (24,24))

    if type(title_name) == str:
        title_ = title_name
    else:
        title_ = format(title_name,'.2f')+'_Background_Noise'
    
    fig.suptitle('Performance Comparison on '+title_, fontsize=25)
    ax[0,0].title.set_text('Py4dstem: Strain X')
    ax[1,0].title.set_text('Strain Y')
    ax[2,0].title.set_text('Shear')
    ax[3,0].title.set_text('Rotation')
    ax[0,2].title.set_text('Neural Network: Strain X')
    ax[1,2].title.set_text('Strain Y')
    ax[2,2].title.set_text('Shear')
    ax[3,2].title.set_text('Rotation')
    
    diff_range_list = [cross_xx_diff_range, cross_yy_diff_range, cross_xy_diff_range, rotation_range,\
                        ae_xx_diff_range, ae_yy_diff_range, ae_xy_diff_range, rotation_range]

    for i in range(8):
        if int(i/4)==0:
            row = i
            col = 0
        else:
            row = i-4
            col=2

        value_range = diff_range_list[i]


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


    plt.savefig('Strain_Map_'+title_+'.svg')
    
def visual_strain_magnitude(s_xx,
                            s_yy,
                            sample_index = None,
                            ref_xx = None,
                            ref_yy = None,
                            strain_range = [-3,3],
                            ref_range = [-3,3],
                            img_size = (256,256),
                            only_real = False
                            ):
    """_summary_

    Args:
        s_xx (_type_): _description_
        s_yy (_type_): _description_
        sample_index (_type_, optional): _description_. Defaults to None.
        ref_xx (_type_, optional): _description_. Defaults to None.
        ref_yy (_type_, optional): _description_. Defaults to None.
        strain_range (list, optional): _description_. Defaults to [-3,3].
        ref_range (list, optional): _description_. Defaults to [-3,3].
        img_size (tuple, optional): _description_. Defaults to (256,256).
        only_real (bool, optional): _description_. Defaults to False.
    """
    
    right_tri = np.sqrt((s_xx+1)**2+(s_yy+1)**2)
    
    if sample_index is not None:
        mean_tri  = np.mean(right_tri.reshape(-1)[sample_index])
    else:
        mean_tri  = np.mean(right_tri.reshape(-1))
        
    unscale_tri = 100.*right_tri/mean_tri-100
    
    if ref_xx is not None and ref_yy is not None and not only_real:
        
        coef_tri = np.sqrt((ref_xx+1)**2+(ref_yy+1)**2)
    
        if sample_index is not None:
            mean_coef_tri  = np.mean(coef_tri.reshape(-1)[sample_index])
        else:
            mean_coef_tri  = np.mean(coef_tri.reshape(-1))

        unscale_coef_tri = 100.*coef_tri/mean_coef_tri-100
        
        fig, ax = plt.subplots(2,2,figsize=(10,10))
        ax[0,0].set_xticklabels('')
        ax[0,0].set_yticklabels('')
        ax[0,0].title.set_text('py4DSTEM')
        ax[0,1].set_xticklabels('')
        ax[0,1].set_yticklabels('')
        ax[0,1].title.set_text('Neural Network')
        ax[0,0].imshow(unscale_coef_tri.reshape(img_size),clim=ref_range)
        ax[1,0].hist(unscale_coef_tri.reshape(-1),200,range=ref_range);
        ax[0,1].imshow(unscale_tri.reshape(img_size),clim=strain_range)
        ax[1,1].hist(unscale_tri.reshape(-1),200,range=strain_range);
        
        plt.savefig('Strain_Magnitude_Comparison.svg')
    
    else:
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].set_xticklabels('')
        ax[0].set_yticklabels('')
        ax[0].title.set_text('Neural Network')
        ax[0].imshow(unscale_tri.reshape(img_size),clim=strain_range)
        ax[1].hist(unscale_tri.reshape(-1),200,range=strain_range);
        
        plt.savefig('Strain_Magnitude_Performance.svg')
        

def cal_diff(exx_correlation,eyy_correlation,exy_correlation,theta_correlation,
                exx_ae,eyy_ae,exy_ae,theta_ae,
                label_xx,label_yy,label_xy,label_rotation):
    """_summary_

    Args:
        exx_correlation (_type_): _description_
        eyy_correlation (_type_): _description_
        exy_correlation (_type_): _description_
        theta_correlation (_type_): _description_
        exx_ae (_type_): _description_
        eyy_ae (_type_): _description_
        exy_ae (_type_): _description_
        theta_ae (_type_): _description_
        label_xx (_type_): _description_
        label_yy (_type_): _description_
        label_xy (_type_): _description_
        label_rotation (_type_): _description_

    Returns:
        _type_: _description_
    """
    dif_correlation_xx = exx_correlation - label_xx
    dif_correlation_yy = eyy_correlation - label_yy
    dif_correlation_xy = exy_correlation - label_xy
    dif_correlation_rotation = theta_correlation - label_rotation

    dif_ae_xx = exx_ae - label_xx
    dif_ae_yy = eyy_ae - label_yy
    dif_ae_xy = exy_ae - label_xy
    dif_ae_rotation = theta_ae - label_rotation

    return [dif_correlation_xx,dif_correlation_yy,dif_correlation_xy,dif_correlation_rotation,dif_ae_xx,dif_ae_yy,dif_ae_xy,dif_ae_rotation]


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
class visualize_simulate_result:
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
        
        self.theta_correlation,self.theta_ae = compare_rotation(self.strain_map,
                                                                self.rotation,
                                                                title_name=self.noise_intensity,
                                                                angle_shift=self.angle_shift,
                                                                clim = [0,60])
        
        self.theta_ref_correlation = np.mean(self.theta_correlation[30:60,10:40])
        self.theta_correlation = self.theta_correlation - self.theta_ref_correlation
        self.theta_ref_ae = np.mean(self.theta_ae[30:60,10:40])
        self.theta_ae = self.theta_ae - self.theta_ref_ae
        
        self.M_init = basis2probe(self.rotation,self.scale_shear).reshape(self.im_size[0],self.im_size[1],2,2)
        
    def reset_baseline(self):
        
        f= h5py.File(self.file_py4DSTEM)
        self.strain_map = f['4DSTEM_experiment']['data']['realslices']['strain_map']['data'][:]
        
    def reset_angle(self,angle_shift):
        self.angle_shift = angle_shift
        
        self.theta_correlation,self.theta_ae = compare_rotation(self.strain_map,
                                                                self.rotation,
                                                                title_name=self.noise_intensity,
                                                                angle_shift=self.angle_shift,
                                                                clim = [0,60])
        
        self.theta_ref_correlation = np.mean(self.theta_correlation[30:60,10:40])
        self.theta_correlation = self.theta_correlation - self.theta_ref_correlation
        self.theta_ref_ae = np.mean(self.theta_ae[30:60,10:40])
        self.theta_ae = self.theta_ae - self.theta_ref_ae
        
    def reset_polar_matrix(self):
        
        self.M_init = basis2probe(self.rotation,self.scale_shear).reshape(self.im_size[0],self.im_size[1],2,2)
    
    def visual_strain(self):
        self.exx_ae,self.eyy_ae,self.exy_ae = strain_tensor(self.M_init,self.im_size)
        self.exx_correlation = self.strain_map[:,:,0]
        self.eyy_correlation = self.strain_map[:,:,1]
        self.exy_correlation = self.strain_map[:,:,2]
        strain_list = [self.exx_correlation,self.eyy_correlation,self.exy_correlation,self.theta_correlation,
                        self.exx_ae,self.eyy_ae,self.exy_ae,self.theta_ae]
        
        Strain_Compare(strain_list,
                        ae_xx_diff_range = self.strain_diff_range,
                        ae_yy_diff_range = self.strain_diff_range,
                        ae_xy_diff_range = self.strain_diff_range,
                        cross_xx_diff_range = self.strain_diff_range,
                        cross_yy_diff_range = self.strain_diff_range,
                        cross_xy_diff_range = self.strain_diff_range,
                        rotation_range = self.strain_rotation_range,
                        title_name=self.noise_intensity)
        
    def visual_diff(self):
        
        diff_list = cal_diff(self.exx_correlation,self.eyy_correlation,self.exy_correlation,self.theta_correlation,
                            self.exx_ae,self.eyy_ae,self.exy_ae,self.theta_ae,
                            self.label_xx,self.label_yy,self.label_xy,self.label_rotation)

        
        MAE_diff_with_Label(diff_list,
                            diff_range=self.mae_diff_range,
                            rotation_range=self.mae_rotation_range,
                            noise_intensity=self.noise_intensity,
                            data_index = None)
        
        
@dataclass
class visualize_real_4dstem:
    rotation: any 
    scale_shear: any
    file_py4DSTEM: any = None
    angle_shift: float = 0
    shift_ref: float = 0
    im_size: any = (256,256)
    bkg_position: int = 0
    classification: any = None
    title_name: str = 'WS2WSe2'
    rotation_range: list[float] = field(default_factory=list)



    
    def __post_init__(self):
        
        
        if self.classification is not None:
            sample_position = int(1 - self.bkg_position)
            self.bkg_index = np.where(self.classification[:,self.bkg_position]==1)[0]
            self.sample_index = np.where(self.classification[:,sample_position]==1)[0]
            
        else:
            self.bkg_index = None
            self.sample_index = None
        
        if self.file_py4DSTEM is not None:
            f= h5py.File(self.file_py4DSTEM)
            self.strain_map = f['4DSTEM_experiment']['data']['realslices']['strain_map']['data'][:]
            self.exx_correlation = self.strain_map[:,:,0].reshape(-1)
            self.eyy_correlation = self.strain_map[:,:,1].reshape(-1)
            self.exy_correlation = self.strain_map[:,:,2].reshape(-1)
        
            if self.classification is not None:
                self.exx_correlation[self.bkg_index]=-1
                self.eyy_correlation[self.bkg_index]=-1
                self.exy_correlation[self.bkg_index]=-1

            self.exx_correlation = self.exx_correlation.reshape(self.im_size)
            self.eyy_correlation = self.eyy_correlation.reshape(self.im_size)
            self.exy_correlation = self.exy_correlation.reshape(self.im_size)
            
            self.theta_correlation,self.theta_ae = compare_rotation(self.strain_map,
                                                                    self.rotation,
                                                                    classification = self.classification,
                                                                    bkg_index = self.bkg_index,
                                                                    sample_index = self.sample_index,
                                                                    title_name = self.title_name,
                                                                    angle_shift=self.angle_shift,
                                                                    shift_ref = self.shift_ref,
                                                                    img_size = self.im_size,
                                                                    clim = self.rotation_range
                                                                    )

        else:
            self.theta_ae = visual_rotation(self.rotation,
                                            classification = self.classification,
                                            bkg_index = self.bkg_index,
                                            sample_index = self.sample_index,
                                            title_name = self.title_name,
                                            angle_shift=self.angle_shift,
                                            img_size = self.im_size,
                                            clim = self.rotation_range
                                            )
            
        self.M_init = basis2probe(self.rotation,self.scale_shear).reshape(-1,2,2)
        self.exx_ae,self.eyy_ae,self.exy_ae = strain_tensor_for_real(self.M_init,
                                                                    self.im_size,
                                                                    sample_index = self.sample_index)
        
        
    def reset_angle(self,
                    angle_shift,
                    rotation_range = [0.01,60],
                    shift_ref = 0):
        
        self.angle_shift = angle_shift
        self.rotation_range = rotation_range
        self.shift_ref = shift_ref
        
        if self.file_py4DSTEM is not None:
            self.theta_correlation,self.theta_ae = compare_rotation(self.strain_map,
                                                                    self.rotation,
                                                                    classification = self.classification,
                                                                    bkg_index = self.bkg_index,
                                                                    sample_index = self.sample_index,
                                                                    title_name = self.title_name,
                                                                    angle_shift=self.angle_shift,
                                                                    shift_ref = self.shift_ref,
                                                                    img_size = self.im_size,
                                                                    clim = self.rotation_range
                                                                    )
        else:
            
            self.theta_ae = visual_rotation(self.rotation,
                                            classification = self.classification,
                                            bkg_index = self.bkg_index,
                                            sample_index = self.sample_index,
                                            title_name = self.title_name,
                                            angle_shift=self.angle_shift,
                                            img_size = self.im_size,
                                            clim = self.rotation_range
                                            )
        
    def reset_baseline(self):
        
        f= h5py.File(self.file_py4DSTEM)
        self.strain_map = f['4DSTEM_experiment']['data']['realslices']['strain_map']['data'][:]
        self.exx_correlation = self.strain_map[:,:,0].reshape(-1)
        self.eyy_correlation = self.strain_map[:,:,1].reshape(-1)
        self.exy_correlation = self.strain_map[:,:,2].reshape(-1)

        
        if self.classification is not None:
            self.exx_correlation[self.bkg_index]=-1
            self.eyy_correlation[self.bkg_index]=-1
            self.exy_correlation[self.bkg_index]=-1
        
        self.exx_correlation = self.exx_correlation.reshape(self.im_size)
        self.eyy_correlation = self.eyy_correlation.reshape(self.im_size)
        self.exy_correlation = self.exy_correlation.reshape(self.im_size)
    
    
    def reset_polar_matrix(self):
        
        self.M_init = basis2probe(self.rotation,self.scale_shear).reshape(-1,2,2)
        self.exx_ae,self.eyy_ae,self.exy_ae = strain_tensor_for_real(self.M_init,
                                                                    self.im_size,
                                                                    sample_index = self.sample_index)
        
    def visual_strain(self,
                    strain_range_xx_ae=[-0.03,0.015],
                    strain_range_yy_ae=[-0.035,0.005],
                    strain_range_xy_ae=[-0.012,0.02],
                    strain_range_xx_cross=[0.05,0.15],
                    strain_range_yy_cross=[0.05,0.15],
                    strain_range_xy_cross=[-0.05,0.05],
                    title_name ='WS2WSe2'
                    ):
        
        
        self.strain_range_xx_ae = strain_range_xx_ae
        self.strain_range_yy_ae = strain_range_yy_ae
        self.strain_range_xy_ae = strain_range_xy_ae
        self.strain_range_xx_cross = strain_range_xx_cross
        self.strain_range_yy_cross = strain_range_yy_cross
        self.strain_range_xy_cross = strain_range_xy_cross
        
        strain_list = [self.exx_correlation,self.eyy_correlation,self.exy_correlation,self.theta_correlation,
                self.exx_ae,self.eyy_ae,self.exy_ae,self.theta_ae]
        
        Strain_Compare(strain_list,
                        ae_xx_diff_range=self.strain_range_xx_ae,
                        ae_yy_diff_range=self.strain_range_yy_ae,
                        ae_xy_diff_range=self.strain_range_xy_ae,
                        cross_xx_diff_range=self.strain_range_xx_cross,
                        cross_yy_diff_range=self.strain_range_yy_cross,
                        cross_xy_diff_range=self.strain_range_xy_cross,
                        rotation_range=self.rotation_range,
                        title_name=title_name
                        )
    
    def visual_real_strain(self,                        
                            strain_range_xx_ae=[-0.03,0.015],
                            strain_range_yy_ae=[-0.035,0.005],
                            strain_range_xy_ae=[-0.012,0.02],
                            ):
        
        self.strain_range_xx_ae = strain_range_xx_ae
        self.strain_range_yy_ae = strain_range_yy_ae
        self.strain_range_xy_ae = strain_range_xy_ae

        strain_list = [self.exx_ae,self.eyy_ae,self.exy_ae,self.theta_ae]
        
        Real_Strain_Viz(strain_list,
                        ae_xx_diff_range=self.strain_range_xx_ae,
                        ae_yy_diff_range=self.strain_range_yy_ae,
                        ae_xy_diff_range=self.strain_range_xy_ae,
                        rotation_range=self.rotation_range,
                        )

    def visual_magnitude_of_strain(self,
                                    strain_range = [-2.5,2.5],
                                    ref_range = [-3,3],
                                    only_real = False):
        
        
        self.strain_range = strain_range
        self.ref_range = ref_range
        self.only_real = only_real
        
        if self.file_py4DSTEM is not None:
            
            ref_xx = self.exx_correlation
            ref_yy = self.eyy_correlation
        else:
            ref_xx = None
            ref_yy = None
        
        visual_strain_magnitude(self.exx_ae,
                                self.eyy_ae,
                                sample_index = self.sample_index,
                                ref_xx = ref_xx,
                                ref_yy = ref_yy,
                                strain_range = self.strain_range,
                                ref_range = self.ref_range,
                                img_size = self.im_size,
                                only_real = self.only_real
                                )
        
    