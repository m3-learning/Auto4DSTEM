import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import morphology
from skimage.morphology import binary_erosion
import scipy as sp
from dataclasses import dataclass, field
from m3util.util.IO import make_folder


def set_format_Auto4D(**kwargs):
    """function to set visualization format

    Returns:
        dictionary: format of the visualization
    """
    
    params = {'axes.titlesize':20,
            'xtick.direction': 'in' ,
            'ytick.direction' : 'in',
            'xtick.top' : True,
            'ytick.right' : True,
            'ytick.labelsize':16,
            'xtick.labelsize' : 16
            }
    
    params.update(kwargs)
    
    return params

def add_colorbar(im,
                ax,
                size="5%",
                pad=0.05
                ):
    """function to add colorbar to subplots

    Args:
        im (matplotlib image): image that colorbar comes from
        ax (matplotlib subplot ax): where to attach the colorbar
        size (str, optional): size of the colorbar. Defaults to "5%".
        pad (float, optional): pad of the colorbar. Defaults to 0.05.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    plt.colorbar(im, cax=cax)

def visual_performance_plot(x_list,
                            auto,
                            py4d,
                            add_x = None,
                            add_y = None,
                            auto_yerr = None,
                            py4d_yerr = None,
                            fill_between = True,
                            errorbar = False,
                            ylim = [0,1e-2],
                            xlim = None,
                            color_list = ['blue','red','green'],
                            marker_list = ['o','H','s'],
                            auto_alpha = 0.3,
                            py4d_alpha = 0.3,
                            markersize = 10,
                            linestyle = ':',
                            title = '',
                            xlabel = '',
                            ylabel ='',
                            fontsize_x = 16,
                            fontsize_y = 18,
                            fontsize_title = 20,
                            direction = 'in',
                            save_figure = True,
                            folder_path = '',
                            figsize = (10,6)
                            ):
    """function to plot MAE and error bar with py4dstem and auto4dstem

    Args:
        x_list (list): list of noise level
        auto (list): list of auto4dstem results (mae)
        py4d (list): list of py4dstem results (mae)
        auto_yerr (list, optional): list of auto4dstem results (std). Defaults to None.
        py4d_yerr (list, optional): list of py4dstem results (std). Defaults to None.
        fill_between (bool, optional): determine if need to show fill_between plot. Defaults to True.
        errorbar (bool, optional): determine if need to show errorbar. Defaults to True.
        ylim (list, optional): y axis limit . Defaults to [0,5].
        auto_col (str, optional): color of auto4dstem line. Defaults to 'blue'.
        py4d_col (str, optional): color of py4dstem line. Defaults to 'red'.
        auto_marker (str, optional): shape of marker. Defaults to 'o'.
        py4d_marker (str, optional): shape of marker. Defaults to 'H'.
        auto_alpha (float, optional): alpha value in plot. Defaults to '0.3'.
        py4d_alpha (float, optional): alpha value in plot. Defaults to '0.3'.
        markersize (int, optional): size of marker. Defaults to 10.
        linestyle (str, optional): linestyle of the line. Defaults to ':'.
        title (str, optional): title of the figure. Defaults to ''.
        xlabel (str, optional): title of the x axis. Defaults to ''.
        ylabel (str, optional): title of the y axis. Defaults to ''.
        fontsize_x (int, optional): fontsize of x label. Defaults to 16.
        fontsize_y (int, optional): fontsize of y label. Defaults to 18.
        fontsize_title (int, optional): fontsize of title. Defaults to 20.
        direction (str, optional): direction of tick. Defaults to 'in'.
        save_figure (bool, optional): determine if save the figure or not. Defaults to True.
        folder_path (str, optional): folder path of saved figure. Defaults to ''.
    """
    # plot the results with fill between version if set to True 
    plt.figure(figsize=figsize)
    
    if fill_between:
        errorbar = False
        plt.fill_between(x_list, auto+auto_yerr, auto-auto_yerr, \
                        alpha=auto_alpha, color= color_list[0])

        
        plt.fill_between(x_list, py4d+py4d_yerr, py4d-py4d_yerr, \
                        alpha=py4d_alpha ,color = color_list[1])

    # plot the results with errorbar if set to True
    if errorbar:
        plt.errorbar(x_list, auto, yerr=auto_yerr, color = color_list[0], marker = marker_list[0],\
                    markersize = markersize, linestyle = linestyle)
        
        plt.errorbar(x_list, py4d, yerr=py4d_yerr, color = color_list[1], marker = marker_list[1],\
                    markersize = markersize, linestyle = linestyle)
    # plot the results without errorbar if set to False
    else:
        plt.plot(x_list, auto, color = color_list[0], marker = marker_list[0],\
                    markersize = markersize, linestyle = linestyle)
        
        plt.plot(x_list, py4d, color = color_list[1], marker = marker_list[1],\
                    markersize = markersize, linestyle = linestyle)
    # add additional plot is necessary
        if add_x is not None and add_y is not None:
            plt.plot(add_x,add_y, color = color_list[2], marker = marker_list[2],\
                    markersize = markersize, linestyle = linestyle)
    # set parameters of the plot
    plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.title(title,fontsize = fontsize_title)
    plt.xlabel(xlabel,fontsize =fontsize_x)
    plt.ylabel(ylabel,fontsize =fontsize_y)
    plt.tick_params(direction=direction)
    # save plot
    if save_figure:
        plt.savefig(f'{folder_path}{title}_.svg')
        
def normalized_strain_matrices(list_of_strain_tuple,
                            color_list = ['red','green','blue','orange','grey','purple'],
                            strain_range = [-0.03,0.03],
                            rotation_range = [-40,30],
                            file_name = '',
                            save_figure = True
                            ):
    """function to generate normalized strain histogram for scale, shear and rotation

    Args:
        list_of_strain_tuple (list of tuple): strain values
        color_list (list, optional): list of color. Defaults to ['red','green','blue','orange','grey','purple'].
        strain_range (list, optional): strain range. Defaults to [-0.03,0.03].
        rotation_range (list, optional): rotation range. Defaults to [-40,30].
        file_name (str, optional): file name. Defaults to ''.
        save_figure (bool, optional): determine if save needed. Defaults to True.
    """
    if len(color_list)<len(list_of_strain_tuple):
        return ("not enough color for show")
    fig,ax = plt.subplots(1,4,figsize = (20,5))
    for i in range(len(list_of_strain_tuple)):
        hist_plotter(ax[0],list_of_strain_tuple[i][0],color=color_list[i],clim=strain_range)
        hist_plotter(ax[1],list_of_strain_tuple[i][1],color=color_list[i],clim=strain_range)
        hist_plotter(ax[2],list_of_strain_tuple[i][2],color=color_list[i],clim=strain_range)
        hist_plotter(ax[3],list_of_strain_tuple[i][3],color=color_list[i],clim=rotation_range)
    fig.tight_layout()
    if save_figure:
        plt.savefig(f'{file_name}_normalized_strain_histogram.svg')
    

def basis2probe(rotation_,
                scale_shear_):
    """function to turn affine parameters into affine matrix 

    Args:
        rotation_ (numpy.array): rotation matrix [batch, cos, sin]
        scale_shear_ (numpy.array): strain matrix [batch,]

    Returns:
        np.array: combined affine matrix of input data
    """

    M = []
    
    for i in tqdm(range(rotation_.shape[0]),leave=True,total=rotation_.shape[0]):
        
        # switch cosine and sine into radius
        theta = np.arctan2(rotation_[i][1], rotation_[i][0]) 
        
        # generate scale transformation matrix and rotation transformation matrix with parameters from corresponding input index
        xx = scale_shear_[i][0]
        yy = scale_shear_[i][3]
        xy = scale_shear_[i][1]
        yx = scale_shear_[i][2]

        # generate rotation matrix with cosine and sine value
        r = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
            ])
        
        # generate scale and shear with input value 
        t = np.array([
            [xx,xy],
            [yx,yy]
            ])
        
        # matrix multiplication 
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
                    folder_name = '',
                    cmap = 'RdBu_r',
                    angle_shift=0,
                    img_size = (256,256),
                    clim = [0,60]
                    ):
    """function to visualize rotation map of input data

    Args:
        rotation_ (numpy.array): rotation matrix generated by neural network
        classification (numpy.array, optional): If exists, classification map generated by neural network. Defaults to None.
        bkg_index (numpy.array, optional): if classification is not None, index of background map. Defaults to None.
        sample_index (numpy.array, optional): if classification is not None, index of sample map. Defaults to None.
        title_name (str, optional): name of the figure. Defaults to 'WS2WSe2'.
        folder_name (str): folder to save the figure.
        cmap (str): color map of plt.imshow.
        angle_shift (float, optional): angle degree shift on rotation map. Defaults to 0.
        img_size (tuple, optional): size of the rotation map. Defaults to (256,256).
        clim (list, optional): visualization range of rotation value. Defaults to [0,60].

    Returns:
        numpy.array: adjusted rotation value of the input images 
    """
    
    # set name of the figure
    if type(title_name) == str:
        name_ = title_name
    else:
        name_ = format(title_name,'.2f')+' Background Noise'
    
    # switch format to degree for each rotation value, mod by 60 to make it distributed in (0, 60)
    temp_ae = np.mod( angle_shift + \
                        1*np.rad2deg(np.arctan2(
                            rotation_[:,1].reshape(-1),
                            rotation_[:,0].reshape(-1))),
                        60.0
                        )
    
    # make rotation map region where belongs to background become 0, keep sample region value.
    if classification is not None:
        
        theta_ae = np.zeros([img_size[0]*img_size[1]])
        theta_ae[sample_index] = temp_ae
    else:
        
        theta_ae = np.copy(temp_ae)
    
    theta_ae = theta_ae.reshape(img_size)
        
    # visualize the rotation map and histogram
    fig,ax = plt.subplots(1,2, figsize = (10,5))

    ax[0].title.set_text('Auto4DSTEM')
    ax[0].set_xticklabels('')
    ax[0].set_yticklabels('')
    
    im = ax[0].imshow(
        theta_ae,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
    )
    add_colorbar(im,ax[0])

    ax[1].hist(theta_ae.reshape(-1),200,range=clim);

    fig.tight_layout()
    # save figure
    plt.savefig(f'{folder_name}/Rotation_map_on_{name_}.svg')
    return theta_ae


def compare_rotation(strain_map,
                    rotation_,
                    classification = None,
                    bkg_index = None,
                    sample_index = None,
                    title_name='WS2WSe2',
                    folder_name = '',
                    cmap = 'RdBu_r',
                    angle_shift=0,
                    shift_ref = 0,
                    img_size = (256,256),
                    clim = [0,60],
                    ref_clim = None,
                    supplementary_angle = False
                    ):
    """function to compare rotation map between results of neural network and py4DSTEM

    Args:
        strain_map (numpy.array): strain map results generated by py4DSTEM
        rotation_ (numpy.array): rotation matrix generated by neural network
        classification (numpy.array, optional): If exists, classification map generated by neural network. Defaults to None.
        bkg_index (numpy.array, optional): if classification is not None, index of background map. Defaults to None.
        sample_index (numpy.array, optional): if classification is not None, index of sample map. Defaults to None.
        title_name (str, optional): name of the figure. Defaults to 'WS2WSe2'.
        folder_name (str): folder to save the figure.
        cmap (str): color map of plt.imshow.
        angle_shift (float, optional): angle degree shift on rotation map neural network. Defaults to 0.
        shift_ref (float, optional): angle degree shift on rotation map py4DSTEM. Defaults to 0.
        img_size (tuple, optional): size of the rotation map. Defaults to (256,256).
        clim (list, optional): visualization range of rotation value. Defaults to [0,60].
        

    Returns:
        numpy.array: adjusted of rotation value for py4DSTEM and neural network
    """
    # set name of the figure
    if type(title_name) == str:
        name_ = title_name
    else:
        name_ = format(title_name,'.2f')+' Background Noise'
        
    # set ref_clim if it is None
    if ref_clim is None:
        ref_clim = clim
    # switch format to degree for each rotation value, mod by 60 to make it distributed in (0, 60)
    theta_correlation = np.mod(shift_ref+np.rad2deg(strain_map[3,:,:]),60).reshape(-1)
    if supplementary_angle:
        temp_ae = np.mod( angle_shift + 60 -\
                    1*np.rad2deg(np.arctan2(
                        rotation_[:,1].reshape(-1),
                        rotation_[:,0].reshape(-1))),
                    60.0
                    )
    else:
        temp_ae = np.mod( angle_shift + \
                            1*np.rad2deg(np.arctan2(
                                rotation_[:,1].reshape(-1),
                                rotation_[:,0].reshape(-1))),
                            60.0
                            )
    if classification is not None:
    # make rotation map region where belongs to background become 0, keep sample region value.
        theta_correlation[bkg_index] = 0
        theta_ae = np.zeros([img_size[0]*img_size[1]])
        theta_ae[sample_index] = temp_ae
    else:
        
        theta_ae = np.copy(temp_ae)
    
    # reshape the rotation map into 2D image size
    theta_correlation = theta_correlation.reshape(img_size)
    theta_ae = theta_ae.reshape(img_size)
        
    # visualize the rotation map and histogram
    fig,ax = plt.subplots(2,2, figsize = (11,10))

    ax[0,0].title.set_text('Rotation: Py4DSTEM')
    ax[0,0].set_xticklabels('')
    ax[0,0].set_yticklabels('')

    ax[0,1].title.set_text('Auto4DSTEM')
    ax[0,1].set_xticklabels('')
    ax[0,1].set_yticklabels('')

    im1 = ax[0,0].imshow(
            theta_correlation,
            cmap=cmap,
            vmin=ref_clim[0],
            vmax=ref_clim[1],
        )
    add_colorbar(im1,ax[0,0])

    ax[1,0].hist(theta_correlation.reshape(-1),200,range=ref_clim);
    im2 = ax[0,1].imshow(
            theta_ae,
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
        )
    add_colorbar(im2,ax[0,1])

    ax[1,1].hist(theta_ae.reshape(-1),200,range=clim);

    fig.tight_layout()
    # save figure
    plt.savefig(f'{folder_name}/Rotation_comparison_on_{name_}.svg')

    return theta_correlation, theta_ae


def strain_tensor_for_real(M_init,
                        im_size,
                        sample_index = None,
                        ):
    """function for polar decomposition 

    Args:
        M_init (numpy.array): combined affine matrix of input data
        im_size (tuple): size of the rotation map
        sample_index (numpy.array, optional): index of sample map. Defaults to None.
    Returns:
        numpy.array: scale and shear parameter after polar decomposition
    """
    # generate image size array with value equal to -1
    exx_ae = np.zeros([im_size[0]*im_size[1]])-1
    eyy_ae = np.zeros([im_size[0]*im_size[1]])-1
    exy_ae = np.zeros([im_size[0]*im_size[1]])-1

    # generate numpy array of size M_init with zeros
    exx_ = np.zeros([M_init.shape[0]])
    eyy_ = np.zeros([M_init.shape[0]])
    exy_ = np.zeros([M_init.shape[0]])

    for i in tqdm(range(M_init.shape[0])):

        T = M_init[i]
        # polar decomposition
        u, p = sp.linalg.polar(T, side='left')
        
        # shear xy = shear yx, symmetric properties
        transformation = np.array([
            [p[0,0] - 1, p[0,1]],
            [p[0,1], p[1,1] - 1],
        ])

        # insert scale and shear value into output array
        exx_[i] = transformation[1,1]
        eyy_[i] = transformation[0,0]
        exy_[i] = transformation[0,1]
        
    if M_init.shape[0] == im_size[0]*im_size[1]:
        
        exx_ae = np.copy(exx_)
        eyy_ae = np.copy(eyy_)
        exy_ae = np.copy(exy_)
        
    else:
        # insert scale and shear value into output array 
        exx_ae[sample_index] = exx_
        eyy_ae[sample_index] = eyy_
        exy_ae[sample_index] = exy_
        
    # reshape output into the same as image size
    exx_ae = exx_ae.reshape(im_size)
    eyy_ae = eyy_ae.reshape(im_size)
    exy_ae = exy_ae.reshape(im_size)
    
    

    return exx_ae,eyy_ae,exy_ae

def hist_plotter(ax, 
                image, 
                color="blue",
                alpha=1,
                clim =[-0.03,0.03]):
    """function to create normalized plot

    Args:
        ax (matplotlib): figure
        image (numpy): image showed in figure
        color (str, optional): color of the histogram. Defaults to "blue".
        alpha (float, optional): transparency of the histogram. Defaults to 1.
        clim (list, optional): color range of the histogram. Defaults to [-0.03,0.03].
    """
    # Compute histogram data to find the maximum bin count
    counts, bin_edges = np.histogram(image.reshape(-1), bins=200, range = clim)
    max_count = counts.max()

    # Calculate weights for each data point
    weights = np.ones_like(image.reshape(-1)) / max_count

    # Plot the histogram using the calculated weights
    ax.hist(image.reshape(-1), bins=200, weights=weights, color=color,range=clim,alpha=alpha);


def strain_tensor(M_init,
                im_size,
                ref_region = (30,60,10,40),
                ):
    """function for polar decomposition for simulated 4dstem

    Args:
        M_init (numpy.array): combined affine matrix of input data
        im_size (tuple): size of the rotation map
        ref_region (tuple, optional): region in image to be considered as reference, (x_start, x_end, y_start, y_end).
    Returns:
        numpy.array: scale and shear parameter after polar decomposition
    """
    
    # calculate mean value of affine parameter for reference region
    M_ref = np.median(M_init[ref_region[0]:ref_region[1],ref_region[2]:ref_region[3]],axis=(0,1))

    # initialize output affine parameter
    exx_ae = np.zeros((im_size[0],im_size[1]))
    eyy_ae = np.zeros((im_size[0],im_size[1]))
    exy_ae = np.zeros((im_size[0],im_size[1]))

    for rx in range(im_size[0]):
        for ry in range(im_size[1]):
            
            # generate updated affine matrix based on M_ref
            T = M_init[rx,ry] @ np.linalg.inv(M_ref)
            # polar decomposition 
            u, p = sp.linalg.polar(T, side='left')
            # shear xy = shear yx, symmetric properties
            transformation = np.array([
                [p[0,0] - 1, p[0,1]],
                [p[0,1], p[1,1] - 1],
            ])
            # insert scale and shear value into output array
            exx_ae[rx,ry] = transformation[1,1]
            eyy_ae[rx,ry] = transformation[0,0]
            exy_ae[rx,ry] = transformation[0,1]

    return exx_ae,eyy_ae,exy_ae

def real_strain_viz(diff_list,
                    title_name, 
                    folder_name = '',
                    cmap_strain = 'RdBu_r',
                    cmap_rotation = 'viridis',
                    ae_xx_diff_range = [-0.03,0.03],
                    ae_yy_diff_range = [-0.03,0.03],
                    ae_xy_diff_range = [-0.03,0.03],
                    rotation_range=[-40,30],
                    data_index = None):
    """function to visualize strain map of the 4dstem

    Args:
        diff_list (list): list of numpy.array of affine parameter: strain x, strain y, shear and rotation
        title_name (str): title of the figure
        folder_name (str): folder to save the figure
        cmap (str): color map of plt.imshow
        ae_xx_diff_range (list): visualization range of strain x
        ae_yy_diff_range (list):  visualization range of strain y
        ae_xy_diff_range (list):  visualization range of shear 
        rotation_range (list, optional): visualization range of rotation. Defaults to [-40,30].
        data_index (numpy.array, optional): index of pixel in histogram. Defaults to None.
    """


    fig,ax = plt.subplots(4,2, figsize = (10,20))

    # set title text
    ax[0,0].title.set_text('Auto4DSTEM: Strain X')
    ax[1,0].title.set_text('Strain Y')
    ax[2,0].title.set_text('Shear')
    ax[3,0].title.set_text('Rotation')

    # put the scale, shear and rotation range value into list
    diff_range_list = [ae_xx_diff_range, ae_yy_diff_range, ae_xy_diff_range, rotation_range]

    for i in range(4):
        # set color map
        if i ==3:
            cmap = cmap_rotation
        else:
            cmap = cmap_strain
        # set color range
        value_range = diff_range_list[i]
        # hide x,y axis 
        ax[i,0].set_xticklabels('')
        ax[i,0].set_yticklabels('')

        im = ax[i,0].imshow(
            diff_list[i],
            cmap = cmap,
            clim = value_range
        )
        add_colorbar(im,ax[i,0])

        if data_index == None:  
            ax[i,1].hist(diff_list[i].reshape(-1),200,range=value_range);

        else: 
            ax[i,1].hist(diff_list[i].reshape(-1)[data_index],200,range=value_range);
    
    fig.tight_layout()
    # save figure
    plt.savefig(f'{folder_name}/{title_name}_Strain_Map_of_Experimental_4DSTEM.svg')


#  strain comparisons
def Strain_Compare(diff_list,
                    ae_xx_diff_range,
                    ae_yy_diff_range,
                    ae_xy_diff_range,
                    cross_xx_diff_range,
                    cross_yy_diff_range,
                    cross_xy_diff_range,
                    rotation_range=[-40,30],
                    ref_rotation_range = None,
                    title_name=0,
                    folder_name = '',
                    cmap_strain = 'RdBu_r',
                    cmap_rotation = 'viridis',
                    data_index = None):
    """function to visualize and compare strain map generated by py4DSTEM and neural network

    Args:
        diff_list (list): list of numpy.array of affine parameter: strain x, strain y, shear and rotation
        ae_xx_diff_range (list): visualization range of strain x of neural network
        ae_yy_diff_range (list): visualization range of strain y of neural network
        ae_xy_diff_range (list): visualization range of shear of neural network
        cross_xx_diff_range (list): visualization range of strain x of py4DSTEM
        cross_yy_diff_range (list): visualization range of strain y of py4DSTEM
        cross_xy_diff_range (list): visualization range of shear of py4DSTEM
        rotation_range (list, optional): visualization range of rotation. Defaults to [-40,30]
        ref_rotation_range (list, optional): visualization range of rotation. Defaults to None
        title_name (float or str, optional): set title name of the figure. Defaults to 0
        folder_name (str): folder to save the figure
        cmap (str): color map of plt.imshow
        data_index (numpy.array, optional): index of pixel in histogram. Defaults to None
    """

    # make ref_rotation_range = rotation_range if it is None
    if ref_rotation_range is None:
        ref_rotation_range = rotation_range
        
    fig,ax = plt.subplots(4,4, figsize = (22,20))
    
    # generate figure title and subtitles 
    if type(title_name) == str:
        title_ = title_name
    else:
        title_ = format(title_name,'.2f')+'_Background_Noise'
    # add subtitle to each subplot
    ax[0,0].title.set_text('Py4DSTEM: Strain X')
    ax[1,0].title.set_text('Strain Y')
    ax[2,0].title.set_text('Shear')
    ax[3,0].title.set_text('Rotation')
    ax[0,2].title.set_text('Auto4DSTEM: Strain X')
    ax[1,2].title.set_text('Strain Y')
    ax[2,2].title.set_text('Shear')
    ax[3,2].title.set_text('Rotation')
    
    # generate range list by given variables
    diff_range_list = [cross_xx_diff_range, cross_yy_diff_range, cross_xy_diff_range, ref_rotation_range,\
                        ae_xx_diff_range, ae_yy_diff_range, ae_xy_diff_range, rotation_range]

    for i in range(8):
        # get row and column value
        if int(i/4)==0:
            row = i
            col = 0
        else:
            row = i-4
            col=2
        # determine color map
        if row ==3:
            cmap = cmap_rotation
        else:
            cmap = cmap_strain

        value_range = diff_range_list[i]
        # delete x,y axis 
        ax[row,col].set_xticklabels('')
        ax[row,col].set_yticklabels('')

        im = ax[row,col].imshow(
            diff_list[i],
            cmap = cmap,
            clim = value_range
        )
        add_colorbar(im,ax[row,col])
        if data_index == None:  
            ax[row,col+1].hist(diff_list[i].reshape(-1),200,range=value_range);

        else: 
            ax[row,col+1].hist(diff_list[i].reshape(-1)[data_index],200,range=value_range);
    fig.tight_layout()
    # save figure
    plt.savefig(f'{folder_name}/Strain_Map_{title_}.svg')
    
def visual_strain_magnitude(s_xx,
                            s_yy,
                            title_name,
                            folder_name ='',
                            cmap = 'RdBu_r',
                            sample_index = None,
                            ref_xx = None,
                            ref_yy = None,
                            strain_range = [-3,3],
                            ref_range = [-3,3],
                            img_size = (256,256),
                            only_real = False
                            ):
    """function to generate and visualize strain magnitude 

    Args:
        s_xx (numpy.array): scale x of neural network
        s_yy (numpy.array): scale y of neural network
        title_name (str): title of the figure
        folder_name (str): folder to save the figure
        cmap (str): color map of plt.imshow
        sample_index (numpy.array, optional): index of sample map. Defaults to None.
        ref_xx (numpy.array, optional): scale x of py4DSTEM. Defaults to None.
        ref_yy (numpy.array, optional): scale y of py4DSTEM. Defaults to None.
        strain_range (list, optional): list of strain range on neural network results by percent. Defaults to [-3,3].
        ref_range (list, optional): list of strain range on py4DSTEM results by percent.. Defaults to [-3,3].
        img_size (tuple, optional): _description_. Defaults to (256,256).
        only_real (bool, optional): _description_. Defaults to False.
    """
    
    # calculate strain magnitude of neural network
    right_tri = np.sqrt((s_xx+1)**2+(s_yy+1)**2)
    
    if sample_index is not None:
        # only keep the value in region of sample index
        mean_tri  = np.mean(right_tri.reshape(-1)[sample_index])
    else:
        mean_tri  = np.mean(right_tri.reshape(-1))
    
    # use percentage to represent scale magnitude
    unscale_tri = 1.*right_tri/mean_tri-1
    
    # calculate strain magnitude of py4DSTEM result
    if ref_xx is not None and ref_yy is not None and not only_real:
        
        coef_tri = np.sqrt((ref_xx+1)**2+(ref_yy+1)**2)

        # only keep strain magnitude value for sample region
        if sample_index is not None:
            mean_coef_tri  = np.mean(coef_tri.reshape(-1)[sample_index])
        else:
            mean_coef_tri  = np.mean(coef_tri.reshape(-1))

        # use percentage to represent scale magnitude
        unscale_coef_tri = 1.*coef_tri/mean_coef_tri-1
        
        # generate figure with py4DSTEM
        fig, ax = plt.subplots(2,2,figsize=(11,10))
        ax[0,0].set_xticklabels('')
        ax[0,0].set_yticklabels('')
        ax[0,0].title.set_text('py4DSTEM')
        ax[0,1].set_xticklabels('')
        ax[0,1].set_yticklabels('')
        ax[0,1].title.set_text('Auto4DSTEM')
        im1 = ax[0,0].imshow(unscale_coef_tri.reshape(img_size),cmap = cmap, clim=ref_range)
        add_colorbar(im1,ax[0,0])
        ax[1,0].hist(unscale_coef_tri.reshape(-1),200,range=ref_range);
        im2 = ax[0,1].imshow(unscale_tri.reshape(img_size),cmap = cmap, clim=strain_range)
        add_colorbar(im2,ax[0,1])
        ax[1,1].hist(unscale_tri.reshape(-1),200,range=strain_range);
        
        fig.tight_layout()
        plt.savefig(f'{folder_name}/{title_name}_Strain_Magnitude_Comparison.svg')
    
    else:
        # generate figure only for neural network
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].set_xticklabels('')
        ax[0].set_yticklabels('')
        ax[0].title.set_text('Auto4DSTEM')
        im = ax[0].imshow(unscale_tri.reshape(img_size), cmap =cmap, clim=strain_range)
        add_colorbar(im,ax[0])
        ax[1].hist(unscale_tri.reshape(-1),200,range=strain_range);
        fig.tight_layout()
        plt.savefig(f'{folder_name}/{title_name}_Strain_Magnitude_Performance.svg')
        

def cal_diff(exx_correlation,eyy_correlation,exy_correlation,theta_correlation,
                exx_ae,eyy_ae,exy_ae,theta_ae,
                label_xx,label_yy,label_xy,label_rotation):
    """function to calculate difference between label and model generated results

    Args:
        exx_correlation (numpy.array): strain x of py4DSTEM
        eyy_correlation (numpy.array): strain y of py4DSTEM
        exy_correlation (numpy.array): shear of py4DSTEM
        theta_correlation (numpy.array): rotation of py4DSTEM
        exx_ae (numpy.array): strain x of neural network
        eyy_ae (numpy.array): strain y of neural network
        exy_ae (numpy.array): shear of neural network
        theta_ae (numpy.array): rotation of neural network
        label_xx (numpy.array): label of strain x
        label_yy (numpy.array): label of strain y
        label_xy (numpy.array): label of shear
        label_rotation (numpy.array): label of rotation

    Returns:
        list: list of difference between label and generated results
    """
    # generate difference between py4DSTEM and label
    dif_correlation_xx = exx_correlation - label_xx
    dif_correlation_yy = eyy_correlation - label_yy
    dif_correlation_xy = exy_correlation - label_xy
    dif_correlation_rotation = theta_correlation - label_rotation
    
    # generate difference between neural network and label
    dif_ae_xx = exx_ae - label_xx
    dif_ae_yy = eyy_ae - label_yy
    dif_ae_xy = exy_ae - label_xy
    dif_ae_rotation = theta_ae - label_rotation
    
    # return list with sequence
    return [dif_correlation_xx,dif_correlation_yy,dif_correlation_xy,dif_correlation_rotation,dif_ae_xx,dif_ae_yy,dif_ae_xy,dif_ae_rotation]


def MAE_diff_with_Label(diff_list,
                        diff_range,
                        rotation_range,
                        noise_intensity=0,
                        folder_name = '',
                        cmap = 'RdBu_r',
                        data_index = None
                        ):
    """function to visualize difference between label and model generated results

    Args:
        diff_list (list): list of difference between label and generated results
        diff_range (list): range of strain and shear difference in visualization
        rotation_range (list): range of rotation difference in visualization
        noise_intensity (float, optional): background intensity of simulated data. Defaults to 0
        folder_name (str): folder to save the figure
        cmap (str): color map of plt.imshow
        data_index (index, optional): index to be calculated. Defaults to None
    """

    fig,ax = plt.subplots(4,4, figsize = (22,20))
    noise_format = format(noise_intensity,'.2f')

    # add subtitles of the figure
    ax[0,0].title.set_text('Py4DSTEM: Strain X')
    ax[1,0].title.set_text('Strain Y')
    ax[2,0].title.set_text('Shear')
    ax[3,0].title.set_text('Rotation')
    ax[0,2].title.set_text('Auto4DSTEM: Strain X')
    ax[1,2].title.set_text('Strain Y')
    ax[2,2].title.set_text('Shear')
    ax[3,2].title.set_text('Rotation') 

    for i in range(8):
        # generate row and column value
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
        # delete x and y axis
        ax[row,col].set_xticklabels('')
        ax[row,col].set_yticklabels('')

        im = ax[row,col].imshow(
            diff_list[i],
            cmap = cmap,
            clim = value_range
        )
        add_colorbar(im,ax[row,col])
        if data_index == None:  
        # calculate MAE of the difference histogram
            mae_ = np.mean(abs(diff_list[i].reshape(-1)))
            ax[row,col+1].hist(diff_list[i].reshape(-1),200,range=value_range);

        else: 
        # calculate MAE of the difference histogram for particular index 
            mae_ = np.mean(abs(diff_list[i].reshape(-1)[data_index]))
            ax[row,col+1].hist(diff_list[i].reshape(-1)[data_index],200,range=value_range);

        ax[row,col+1].title.set_text('MAE: '+format(mae_,'.4f'))
    # save the figure
    fig.tight_layout()
    plt.savefig(f'{folder_name}/Performance_Comparison_{noise_format}Percent_BKG.svg')
    

    
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
    folder_name: str = 'save_figures'
    cmap_strain: str = 'RdBu_r'
    cmap_rotation: str = 'twilight'
    cmap_mae: str = 'viridis'
    angle_shift: float = 0
    im_size: any = (256,256)
    ref_region: any = (30,60,10,40)
    strain_diff_range: list[float] = field(default_factory=list)
    strain_rotation_range: list[float] = field(default_factory=list)
    mae_diff_range: list[float] = field(default_factory=list)
    mae_rotation_range: list[float] = field(default_factory=list)
    
    """ class for visualizing and comparing 
    
    Args:
        rotation (numpy.array): rotation value generated by neural network
        scale_shear (numpy.array): scale and shear value generated by neural network
        file_py4DSTEM (str): hdf5 file of affine parameter generated by py4DSTEM
        label_rotation_path (str):  directory of label rotation  
        label_xx_path (str): directory of label scale x  
        label_yy_path (str): directory of label scale y 
        label_xy_path (str): directory of label shear  
        noise_intensity (float): determine the background noise intensity of the simulated 4dstem
        folder_name (str): save the generated figure to folder
        cmap_strain (str): color map of plt.imshow
        cmap_rotation (str): color map of plt.imshow
        cmap_mae (str): color map of plt.imshow
        angle_shift (float): determine the additional degree add to the rotation for visualization
        im_size (tuple): size of the strain map
        ref_region (tuple): region in image to be considered as reference, (x_start, x_end, y_start, y_end).
        strain_diff_range (list): range of strain value for visualization
        strain_rotation_range (list): range of rotation value for visualization
        mae_diff_range (list): range of strain and shear difference in visualization
        mae_rotation_range (list): range of rotation difference in visualization
        
    """
    

    
    def __post_init__(self):
        
        # create folder to save results
        make_folder(self.folder_name)
        # load various labels of affine parameter
        label_rotation = np.load(self.label_rotation_path)
        label_rotation = np.rad2deg(label_rotation)
        # calculate mean value of label rotation in reference region
        label_ref_rotation = np.mean(label_rotation[self.ref_region[0]:self.ref_region[1],
                                                    self.ref_region[2]:self.ref_region[3]])
        # calculate corresponding rotation based on reference 
        self.label_rotation = label_rotation - label_ref_rotation
        self.label_xx = np.load(self.label_xx_path)
        self.label_yy = np.load(self.label_yy_path)
        self.label_xy = np.load(self.label_xy_path)
        
        # load the h5 file of py4DSTEM results
        f= h5py.File(self.file_py4DSTEM)
        self.strain_map = f['strain_map_root']['strain_map']['data'][:]
        
        # compare performance of rotation value and visualize it
        self.theta_correlation,self.theta_ae = compare_rotation(self.strain_map,
                                                                self.rotation,
                                                                title_name=self.noise_intensity,
                                                                folder_name=self.folder_name,
                                                                cmap = self.cmap_rotation,
                                                                angle_shift=self.angle_shift,
                                                                )
        
        # calculate mean value of py4DSTEM rotation in reference region
        self.theta_ref_correlation = np.mean(self.theta_correlation[self.ref_region[0]:self.ref_region[1],
                                                                    self.ref_region[2]:self.ref_region[3]])
        # calculate corresponding rotation based on reference 
        self.theta_correlation = self.theta_correlation - self.theta_ref_correlation
        # calculate mean value of neural network rotation in reference region
        self.theta_ref_ae = np.mean(self.theta_ae[self.ref_region[0]:self.ref_region[1],
                                                    self.ref_region[2]:self.ref_region[3]])
        # calculate corresponding rotation based on reference 
        self.theta_ae = self.theta_ae - self.theta_ref_ae
        
        # generate affine matrices with affine parameters 
        self.M_init = basis2probe(self.rotation,self.scale_shear).reshape(self.im_size[0],self.im_size[1],2,2)
        
        # set additional angle dictionary for rotation comparison
        self.add_angle_shift = {'00':25, '05':-7, '10':-5, '15':-8, '20':-7, '25':-9, '30':-9, 
                                '35':-6, '40':-8, '45':-7, '50':-6, '60':-9, '70':-8 }
        # initial list of dictionary to future plots
        self.list_of_dic = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
        
    def reset_baseline(self):
        """
            function to reload the h5 file of py4DSTEM results
        """
        
        f= h5py.File(self.file_py4DSTEM)
        self.strain_map = f['strain_map_root']['strain_map']['data'][:]
    
    def add_dictionary(self,
                       **kwargs):
        """function to add key value pairs to dictionary
        """
        self.add_angle_shift.update(kwargs)
        
    def reset_angle(self,
                    angle_shift):
        """function to compare performance of rotation value and visualize it

        Args:
            angle_shift (float): additional degree value added to the rotation of neural network
        """
        self.angle_shift = angle_shift
        # compare performance of rotation value and visualize it
        self.theta_correlation,self.theta_ae = compare_rotation(self.strain_map,
                                                                self.rotation,
                                                                title_name=self.noise_intensity,
                                                                folder_name=self.folder_name,
                                                                cmap = self.cmap_rotation,
                                                                angle_shift=self.angle_shift,
                                                                )
        # calculate mean value of py4DSTEM rotation in reference region
        self.theta_ref_correlation = np.mean(self.theta_correlation[self.ref_region[0]:self.ref_region[1],
                                                                    self.ref_region[2]:self.ref_region[3]])
        # calculate corresponding rotation based on reference 
        self.theta_correlation = self.theta_correlation - self.theta_ref_correlation
        # calculate mean value of neural network rotation in reference region
        self.theta_ref_ae = np.mean(self.theta_ae[self.ref_region[0]:self.ref_region[1],
                                                    self.ref_region[2]:self.ref_region[3]])
        # calculate corresponding rotation based on reference 
        self.theta_ae = self.theta_ae - self.theta_ref_ae
        
    def reset_polar_matrix(self):
        """
            function to reload affine matrices with affine parameters 
        """
        
        self.M_init = basis2probe(self.rotation,self.scale_shear).reshape(self.im_size[0],self.im_size[1],2,2)
    
    def visual_strain(self):
        """
            function to visualize strain comparison between results of py4DSTEM and neural network 
        """
        # get strain parameters of neural network from strain_tensor function
        self.exx_ae,self.eyy_ae,self.exy_ae = strain_tensor(self.M_init,self.im_size,self.ref_region)
        # load strain parameters of py4DSTEM from strain map file
        self.exx_correlation = self.strain_map[0,:,:]
        self.eyy_correlation = self.strain_map[1,:,:]
        self.exy_correlation = self.strain_map[2,:,:]
        # create strain list to be the input of the Strain_Compare function
        self.strain_list = [self.exx_correlation,self.eyy_correlation,self.exy_correlation,self.theta_correlation,
                        self.exx_ae,self.eyy_ae,self.exy_ae,self.theta_ae]
        # visualize strain comparison between results of py4DSTEM and neural network
        Strain_Compare(self.strain_list,
                    ae_xx_diff_range = self.strain_diff_range,
                    ae_yy_diff_range = self.strain_diff_range,
                    ae_xy_diff_range = self.strain_diff_range,
                    cross_xx_diff_range = self.strain_diff_range,
                    cross_yy_diff_range = self.strain_diff_range,
                    cross_xy_diff_range = self.strain_diff_range,
                    rotation_range = self.strain_rotation_range,
                    title_name = self.noise_intensity,
                    folder_name = self.folder_name,
                    cmap_strain = self.cmap_strain,
                    cmap_rotation = self.cmap_rotation)
        
    def visual_diff(self):
        """
            function to visualize difference between label and model generated results
        """
        #  calculate difference between label and model generated results
        self.list_of_difference = cal_diff(self.exx_correlation,self.eyy_correlation,self.exy_correlation,self.theta_correlation,
                            self.exx_ae,self.eyy_ae,self.exy_ae,self.theta_ae,
                            self.label_xx,self.label_yy,self.label_xy,self.label_rotation)

        # visualize difference between label and model generated results
        MAE_diff_with_Label(diff_list=self.list_of_difference,
                            diff_range=self.mae_diff_range,
                            rotation_range=self.mae_rotation_range,
                            noise_intensity=self.noise_intensity,
                            folder_name=self.folder_name,
                            cmap= self.cmap_mae,
                            data_index = None)
    
    def record_performance(self,
                            ste_dic = False,
                            data_index_path = '',
                            data_index = False,
                            width = 2,
                            show_index_map = False
                            ):
        """function to extract standard deviation of both py4dstem and auto4dstem

        Args:
            ste_dic (bool, optional): determine if save standard error. Defaults to False.
            data_index_path (str, optional): path of the data index. Defaults to ''.
            data_index (bool, optional): determine if calculate std only on sample region. Defaults to False.
            width (int, optional): scale of boundary size. Defaults to 2.
            show_index_map (bool, optional): determine if need to show the sample image. Defaults to False.
        """
        # calculate the size of sample
        num_index = int(self.im_size[0]*self.im_size[1])
        # calculate the number of sample and create sample index according to insert sample index 
        if data_index:
            # load data
            load_index = np.load(data_index_path).reshape(-1)
            img = np.zeros([self.im_size[0]*self.im_size[1]])
            img[load_index]=1
            # calculate sample index
            sq = morphology.square(width=width)
            img = binary_erosion(img.reshape(self.im_size),sq)
            # show sample index map
            if show_index_map:
                plt.imshow(img)
                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
            self.sample_index_4_std = img.reshape(-1)
            # calculate the size of sample
            num_index = int(np.sum(self.sample_index_4_std))
        # initial key name for dictionary
        self.key_name = ['exx_correlation','eyy_correlation','exy_correlation','theta_correlation',\
                                'exx_ae','eyy_ae','exy_ae','theta_ae']
        # initial dictionary
        self.mae_dictionary={}
        self.std_dictionary={}
        # initial ste if set to True
        if ste_dic:
            self.ste_dictionary={}
        # update dictionary by key value pair
        for i in range(len(self.list_of_difference)):
            # calculate mae, std and ste
            mae_ = np.mean(abs(self.list_of_difference[i].reshape(-1)))
            if data_index:
                std_ = np.std((self.list_of_difference[i].reshape(-1))[self.sample_index_4_std])
            else:
                std_ = np.std(self.list_of_difference[i].reshape(-1))
            # update key value pair into dictionary
            self.mae_dictionary.update({self.key_name[i]:mae_})
            self.std_dictionary.update({self.key_name[i]:std_})
            if ste_dic:
                ste_ = std_/np.sqrt(num_index)
                self.ste_dictionary.update({self.key_name[i]:ste_})
                
    def add_data_2_plot(self,
                        ste_dic = False
                        ):
        """function to insert values into different dictionaries

        Args:
            ste_dic (bool, optional): determine use std dictionary or ste dictionary. Defaults to False.
        """
        # generate key value
        bkg_str = format(int(self.noise_intensity*100),'02d')
        # determine target dictionary
        dic1 = self.mae_dictionary
        if ste_dic:
            dic2 = self.ste_dictionary
        else:
            dic2 = self.std_dictionary
        # extract key value pairs
        ict_1 = list(dic1.items())
        ict_2 = list(dic2.items())
        # insert value to each dictionary
        for i in range(8):
            self.list_of_dic[i].update({bkg_str:ict_1[i][1]})
        for j in range(8,16):
            self.list_of_dic[j].update({bkg_str:ict_2[j-8][1]})
        
    def extract_ele_from_dic(self,
                            num
                            ):
        """function to generate x,y pairs to plot

        Args:
            num (int): index of the list of dictionary

        Returns:
            np.array, np.array: x, y pair
        """
        x_ = []
        y_ = []
        x_list = sorted(self.list_of_dic[num].keys())
        for i in x_list:
            y_.append(self.list_of_dic[num][i])
            x_.append(int(i)/100)
        x_ = np.array(x_)
        y_ = np.array(y_)
        
        return x_, y_
        
    
    def visual_label_map(self,
                        save_figure = True,
                        cmap_strain = 'viridis',
                        cmap_rotation = 'viridis'):
        """function to visualize strain map of label

        Args:
            shrink (float, optional): control the size of colorbar. Defaults to 0.7.
            save_figure (bool, optional): determine if saving figure needed. Defaults to True.
            cmap (str): color map of plt.imshow. Default to 'viridis'.
        """
        fig,ax = plt.subplots(2,2,figsize=(10,10))
        # set title of each label
        ax[0,0].title.set_text('Strain X')
        ax[0,1].title.set_text('Strain Y')
        ax[1,0].title.set_text('Shear')
        ax[1,1].title.set_text('Rotation')
        # create list of data and corresponding color range
        label_list = [self.label_xx,self.label_xy,self.label_yy,self.label_rotation]
        clim_list = [self.strain_diff_range,self.strain_diff_range,self.strain_diff_range,self.strain_rotation_range]
        # determine the row and colum value
        for i in range(4):
            # generate color map
            if i ==3:
                cmap = cmap_rotation
            else:
                cmap = cmap_strain
            # generate row and colum 
            if int(i/2)==0:
                row = i
                col = 0
            else:
                row = i-2
                col = 1
            # plot image show strain map and color bar.
            im = ax[row,col].imshow(label_list[i],cmap = cmap,clim = clim_list[i])
            add_colorbar(im,ax[row,col])
            ax[row,col].set_xticklabels('')
            ax[row,col].set_yticklabels('')
        fig.tight_layout()
        # save figure
        if save_figure:
            plt.savefig('Strain_Map_of_Label.svg')

    
    def show_normalized_comparison_results(self,
                                            noise_intensity = [0,0.15,0.70],
                                            color_list = ['red','green','blue','orange','grey','purple'],
                                            ):
        """function to show normalized comparison of py4dstem and auto4dstem

        Args:
            noise_intensity (list, optional): _description_. Defaults to [0,0.15,0.70].
            color_list (list, optional): _description_. Defaults to ['red','green','blue','orange','grey','purple'].
        """
        # set color represents each noise intensity
        if len(color_list)<len(noise_intensity):
            return('not enough color to represent noise level')
        # initial file pre
        file_name = 'Noise_Intensity_'
        # sort noise intensity list
        noise_intensity.sort(reverse=True)
        # generate figure 
        fig,ax = plt.subplots(4,3,figsize=(15,20))
        # add title to each subplot
        ax[0,0].title.set_text('Strain X:  Label')
        ax[0,1].title.set_text('Py4DSTEM')
        ax[0,2].title.set_text('Auto4DSTEM')
        ax[1,0].title.set_text('Strain Y:  Label')
        ax[1,1].title.set_text('Py4DSTEM')
        ax[1,2].title.set_text('Auto4DSTEM')
        ax[2,0].title.set_text('Shear:  Label')
        ax[2,1].title.set_text('Py4DSTEM')
        ax[2,2].title.set_text('Auto4DSTEM')
        ax[3,0].title.set_text('Rotation:  Label')
        ax[3,1].title.set_text('Py4DSTEM')
        ax[3,2].title.set_text('Auto4DSTEM')
        
        #  generate label histograms
        hist_plotter(ax[0,0], self.label_xx,'blue',clim=self.strain_diff_range)
        hist_plotter(ax[1,0], self.label_yy,'blue',clim=self.strain_diff_range)
        hist_plotter(ax[2,0], self.label_xy,'blue',clim=self.strain_diff_range)
        hist_plotter(ax[3,0], self.label_rotation,'blue',clim=self.strain_rotation_range)
        
        for i, bkg_intensity in enumerate(noise_intensity):
            # complete file name
            bkg_str = format(int(bkg_intensity*100),'02d')
            file_name += f'{bkg_str}_'
            # load py4dstem results
            py4dstem_path = f'{self.folder_name}/analysis_bg{bkg_str}per_1e5counts__strain.h5'
            f = h5py.File(py4dstem_path)
            strain_map = f['strain_map_root']['strain_map']['data'][:]
            # extract strain value
            exx_correlation = strain_map[0,:,:]
            eyy_correlation = strain_map[1,:,:]
            exy_correlation = strain_map[2,:,:]
            # load auto4dstem results
            rotation_path = f'{self.folder_name}/{bkg_str}Per_2_train_process_rotation.npy'
            strain_path = f'{self.folder_name}/{bkg_str}Per_2_train_process_scale_shear.npy'
            rotation_ = np.load(rotation_path)
            scale_shear_ = np.load(strain_path)
            # determine the value of additional angle added to angle shift
            if bkg_str not in self.add_angle_shift:
                angle_shift = 0
            else:
                angle_shift = self.add_angle_shift[bkg_str]
            # compare performance of rotation value and visualize it
            theta_correlation = np.mod(np.rad2deg(strain_map[3,:,:]),60).reshape(self.im_size)

            theta_ae = np.mod( angle_shift + \
                                1*np.rad2deg(np.arctan2(
                                    rotation_[:,1].reshape(-1),
                                    rotation_[:,0].reshape(-1))),
                                60.0
                                ).reshape(self.im_size)
            # calculate mean value of py4DSTEM rotation in reference region
            theta_ref_correlation = np.mean(theta_correlation[self.ref_region[0]:self.ref_region[1],
                                                            self.ref_region[2]:self.ref_region[3]])
            # calculate mean value of neural network rotation in reference region
            theta_ref_ae = np.mean(theta_ae[self.ref_region[0]:self.ref_region[1],
                                                        self.ref_region[2]:self.ref_region[3]])
            # calculate corresponding rotation based on reference 
            theta_correlation = theta_correlation - theta_ref_correlation
            # calculate corresponding rotation based on reference 
            theta_ae = theta_ae - theta_ref_ae
            # generate strain value
            M_init = basis2probe(rotation_,scale_shear_).reshape(self.im_size[0],self.im_size[1],2,2)
            exx_ae,eyy_ae,exy_ae = strain_tensor(M_init,self.im_size)
            # generate py4dstem and auto4dstem histograms
            hist_plotter(ax[0,1], exx_correlation, color=color_list[i], clim=self.strain_diff_range)
            hist_plotter(ax[0,2], exx_ae, color=color_list[i], clim=self.strain_diff_range)
            hist_plotter(ax[1,1], eyy_correlation, color=color_list[i], clim=self.strain_diff_range)
            hist_plotter(ax[1,2], eyy_ae, color=color_list[i], clim=self.strain_diff_range)
            hist_plotter(ax[2,1], exy_correlation, color=color_list[i], clim=self.strain_diff_range)
            hist_plotter(ax[2,2], exy_ae, color=color_list[i], clim=self.strain_diff_range)
            hist_plotter(ax[3,1], theta_correlation, color=color_list[i], clim=self.strain_rotation_range)
            hist_plotter(ax[3,2], theta_ae, color=color_list[i], clim=self.strain_rotation_range)
        fig.tight_layout()
        # save figure
        plt.savefig(f'{self.folder_name}/{file_name}compare_performance.svg')
        
    def show_normalized_results(self,
                                strain = None,
                                rotation = None,
                                color = 'blue',
                                file_name = '',
                                angle_shift = 0
                                ):
        """function to show normalized results

        Args:
            color_list (list, optional): _description_. Defaults to ['red','green','blue','orange','grey','purple'].
            im_size (tuple, optional): _description_. Defaults to (256,256).
            clim (list, optional): _description_. Defaults to [-0.03,0.03].
            file_name (str, optional): set file name of saved figure. Defaults to ''.
        """
        # set color represents each noise intensity
        if strain is None:
            strain = self.scale_shear
        if rotation is None:
            rotation = self.rotation
        # generate figure 
        fig,ax = plt.subplots(1,4,figsize=(20,5))
        # add title to each subplot
        ax[0].title.set_text('Strain X')
        ax[1].title.set_text('Strain Y')
        ax[2].title.set_text('Shear')
        ax[3].title.set_text('Rotation')


        theta_ae = np.mod( angle_shift + \
                            1*np.rad2deg(np.arctan2(
                                rotation[:,1].reshape(-1),
                                rotation[:,0].reshape(-1))),
                            60.0
                            ).reshape(self.im_size)

        # calculate mean value of neural network rotation in reference region
        theta_ref_ae = np.mean(theta_ae[self.ref_region[0]:self.ref_region[1],
                                                    self.ref_region[2]:self.ref_region[3]])
        # calculate corresponding rotation based on reference 
        theta_ae = theta_ae - theta_ref_ae
        # generate strain value
        M_init = basis2probe(rotation,strain).reshape(self.im_size[0],self.im_size[1],2,2)
        exx_ae,eyy_ae,exy_ae = strain_tensor(M_init,self.im_size)
        # generate py4dstem and auto4dstem histograms
        hist_plotter(ax[0], exx_ae, color=color, clim=self.strain_diff_range)
        hist_plotter(ax[1], eyy_ae, color=color, clim=self.strain_diff_range)
        hist_plotter(ax[2], exy_ae, color=color, clim=self.strain_diff_range)
        hist_plotter(ax[3], theta_ae, color=color, clim=self.strain_rotation_range)
        fig.tight_layout()
        # save figure
        plt.savefig(f'{self.folder_name}/{file_name}_normalized_strain_results.svg')
        
        return exx_ae, eyy_ae, exy_ae, theta_ae
        
        
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
    folder_name: str = 'save_figures'
    cmap_strain: str = 'RdBu_r'
    cmap_rotation: str = 'twilight'
    supplementary_angle: bool = False
    rotation_range: list[float] = field(default_factory=list)
    ref_rotation_range: list[float] = field(default_factory=list)
    """ class for visualizing and comparing 
    
    Args:
        rotation (numpy.array): rotation value generated by neural network
        scale_shear (numpy.array): scale and shear value generated by neural network
        file_py4DSTEM (str): hdf5 file of affine parameter generated by py4DSTEM
        angle_shift (float): determine the additional degree added to the rotation for visualization, neural network
        shift_ref (float): determine the additional degree added to the rotation for visualization, py4DSTEM
        im_size (tuple): size of the strain map
        bkg_position (int): index of background position in classification
        classification (numpy.array, optional): classification generated by neural network. Default to None.
        title_name (str): title of the figure
        folder_name (str): folder to save the figure
        cmap_strain (str): color map of plt.imshow
        cmap_rotation (str): color map of plt.imshow
        strain_rotation_range (list): range of rotation value for visualization   
    """
    
    def __post_init__(self):
        
        # create folder to save results
        make_folder(self.folder_name)
        # create background index and sample index according to classification matrix
        if self.classification is not None:
            # generate index of sample position 
            sample_position = int(1 - self.bkg_position)
            # create background index and sample index
            self.bkg_index = np.where(self.classification[:,self.bkg_position]==1)[0]
            self.sample_index = np.where(self.classification[:,sample_position]==1)[0]
            
        else:
            # if no classification, set background index and sample index to be None
            self.bkg_index = None
            self.sample_index = None
        
        # load the h5 file of py4DSTEM results
        if self.file_py4DSTEM is not None:
            f= h5py.File(self.file_py4DSTEM)
            self.strain_map = f['strain_map_root']['strain_map']['data'][:]
            # load strain parameters of py4DSTEM from strain map file
            self.exx_correlation = self.strain_map[0,:,:].reshape(-1)
            self.eyy_correlation = self.strain_map[1,:,:].reshape(-1)
            self.exy_correlation = self.strain_map[2,:,:].reshape(-1)
            
            # set value in background position to be -1
            if self.classification is not None:
                self.exx_correlation[self.bkg_index]=-1
                self.eyy_correlation[self.bkg_index]=-1
                self.exy_correlation[self.bkg_index]=-1
                
            # reshape affine parameter into image size
            self.exx_correlation = self.exx_correlation.reshape(self.im_size)
            self.eyy_correlation = self.eyy_correlation.reshape(self.im_size)
            self.exy_correlation = self.exy_correlation.reshape(self.im_size)
            
            # compare performance of rotation value and visualize it
            self.theta_correlation,self.theta_ae = compare_rotation(self.strain_map,
                                                                    self.rotation,
                                                                    classification = self.classification,
                                                                    bkg_index = self.bkg_index,
                                                                    sample_index = self.sample_index,
                                                                    title_name = self.title_name,
                                                                    folder_name = self.folder_name,
                                                                    cmap = self.cmap_rotation,
                                                                    angle_shift=self.angle_shift,
                                                                    shift_ref = self.shift_ref,
                                                                    img_size = self.im_size,
                                                                    clim = self.rotation_range,
                                                                    ref_clim = self.ref_rotation_range,
                                                                    supplementary_angle = self.supplementary_angle
                                                                    )

        else:
            # visualize the performance of rotation value
            self.theta_ae = visual_rotation(self.rotation,
                                            classification = self.classification,
                                            bkg_index = self.bkg_index,
                                            sample_index = self.sample_index,
                                            title_name = self.title_name,
                                            folder_name = self.folder_name,
                                            cmap=self.cmap_rotation,
                                            angle_shift=self.angle_shift,
                                            img_size = self.im_size,
                                            clim = self.rotation_range
                                            )
        
        # generate affine matrices by affine parameters
        self.M_init = basis2probe(self.rotation,self.scale_shear).reshape(-1,2,2)
        # get strain parameters of neural network from strain_tensor function
        self.exx_ae,self.eyy_ae,self.exy_ae = strain_tensor_for_real(self.M_init,
                                                                    self.im_size,
                                                                    sample_index = self.sample_index)
        
        
    def reset_angle(self,
                    angle_shift,
                    rotation_range = [0.01,60],
                    ref_rotation_range = [0.01,60],
                    shift_ref = 0,
                    supplementary_angle = False):
        """function to compare performance of rotation value and visualize it

        Args:
            angle_shift (float): additional degree value added to the rotation of neural network
            rotation_range (list, optional): range of rotation value for visualization. Defaults to [0.01,60].
            shift_ref (float, optional): additional degree value added to the rotation of py4DSTEM. Defaults to 0.
        """
        
        self.angle_shift = angle_shift
        self.rotation_range = rotation_range
        self.ref_rotation_range = ref_rotation_range
        self.shift_ref = shift_ref
        self.supplementary_angle = supplementary_angle
        # compare performance of rotation value and visualize it
        if self.file_py4DSTEM is not None:
            self.theta_correlation,self.theta_ae = compare_rotation(self.strain_map,
                                                                    self.rotation,
                                                                    classification = self.classification,
                                                                    bkg_index = self.bkg_index,
                                                                    sample_index = self.sample_index,
                                                                    title_name = self.title_name,
                                                                    folder_name = self.folder_name,
                                                                    cmap = self.cmap_rotation,
                                                                    angle_shift = self.angle_shift,
                                                                    shift_ref = self.shift_ref,
                                                                    img_size = self.im_size,
                                                                    clim = self.rotation_range,
                                                                    ref_clim = self.ref_rotation_range,
                                                                    supplementary_angle = self.supplementary_angle
                                                                    )
        else:
            # visualize the performance of rotation value
            self.theta_ae = visual_rotation(self.rotation,
                                            classification = self.classification,
                                            bkg_index = self.bkg_index,
                                            sample_index = self.sample_index,
                                            title_name = self.title_name,
                                            folder_name = self.folder_name,
                                            cmap = self.cmap_rotation,
                                            angle_shift = self.angle_shift,
                                            img_size = self.im_size,
                                            clim = self.rotation_range
                                            )
        
    def reset_baseline(self):
        """
            function to reload the h5 file of py4DSTEM results
        """
        
        # load the h5 file of py4DSTEM results
        f= h5py.File(self.file_py4DSTEM)
        self.strain_map = f['strain_map_root']['strain_map']['data'][:]
        self.exx_correlation = self.strain_map[0,:,:].reshape(-1)
        self.eyy_correlation = self.strain_map[1,:,:].reshape(-1)
        self.exy_correlation = self.strain_map[2,:,:].reshape(-1)

        # set value in background position to be -1
        if self.classification is not None:
            self.exx_correlation[self.bkg_index]=-1
            self.eyy_correlation[self.bkg_index]=-1
            self.exy_correlation[self.bkg_index]=-1
        # reshape affine parameter into image size
        self.exx_correlation = self.exx_correlation.reshape(self.im_size)
        self.eyy_correlation = self.eyy_correlation.reshape(self.im_size)
        self.exy_correlation = self.exy_correlation.reshape(self.im_size)
    
    
    def reset_polar_matrix(self):
        """
            function to reload affine matrices with affine parameters 
        """
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
                    ):
        """function to visualize strain performance between results of py4DSTEM and neural network 

        Args:
            strain_range_xx_ae (list, optional): visualization range of strain x of neural network. Defaults to [-0.03,0.015].
            strain_range_yy_ae (list, optional): visualization range of strain y of neural network. Defaults to [-0.035,0.005].
            strain_range_xy_ae (list, optional): visualization range of shear of neural network. Defaults to [-0.012,0.02].
            strain_range_xx_cross (list, optional): visualization range of strain x of py4DSTEM. Defaults to [0.05,0.15].
            strain_range_yy_cross (list, optional): visualization range of strain y of py4DSTEM. Defaults to [0.05,0.15].
            strain_range_xy_cross (list, optional): visualization range of shear of py4DSTEM. Defaults to [-0.05,0.05].
            title_name (str, optional): title of the figure. Defaults to 'WS2WSe2'.
        """
        
        
        self.strain_range_xx_ae = strain_range_xx_ae
        self.strain_range_yy_ae = strain_range_yy_ae
        self.strain_range_xy_ae = strain_range_xy_ae
        self.strain_range_xx_cross = strain_range_xx_cross
        self.strain_range_yy_cross = strain_range_yy_cross
        self.strain_range_xy_cross = strain_range_xy_cross 
        
        # generate list of color range by initialized parameters
        self.strain_list = [self.exx_correlation,self.eyy_correlation,self.exy_correlation,self.theta_correlation,
                self.exx_ae,self.eyy_ae,self.exy_ae,self.theta_ae]
        
        # visualize strain comparison between results of py4DSTEM and neural network
        Strain_Compare(self.strain_list,
                        ae_xx_diff_range=self.strain_range_xx_ae,
                        ae_yy_diff_range=self.strain_range_yy_ae,
                        ae_xy_diff_range=self.strain_range_xy_ae,
                        cross_xx_diff_range=self.strain_range_xx_cross,
                        cross_yy_diff_range=self.strain_range_yy_cross,
                        cross_xy_diff_range=self.strain_range_xy_cross,
                        rotation_range=self.rotation_range,
                        ref_rotation_range=self.ref_rotation_range,
                        title_name=self.title_name,
                        folder_name=self.folder_name,
                        cmap_strain=self.cmap_strain,
                        cmap_rotation=self.cmap_rotation
                        )
    
    def visual_real_strain(self,                        
                            strain_range_xx_ae=[-0.03,0.015],
                            strain_range_yy_ae=[-0.035,0.005],
                            strain_range_xy_ae=[-0.012,0.02],
                            ):
        """function to visualize strain performance of neural network results

        Args:
            strain_range_xx_ae (list, optional): visualization range of strain x of neural network. Defaults to [-0.03,0.015].
            strain_range_yy_ae (list, optional): visualization range of strain y of neural network. Defaults to [-0.035,0.005].
            strain_range_xy_ae (list, optional): visualization range of shear of neural network. Defaults to [-0.012,0.02].
        """
        self.strain_range_xx_ae = strain_range_xx_ae
        self.strain_range_yy_ae = strain_range_yy_ae
        self.strain_range_xy_ae = strain_range_xy_ae
        # generate list of color range by initialized parameters
        self.strain_list = [self.exx_ae,self.eyy_ae,self.exy_ae,self.theta_ae]
        # visualize strain performance of neural network results
        real_strain_viz(self.strain_list,
                        title_name=self.title_name,
                        folder_name=self.folder_name,
                        cmap_strain=self.cmap_strain,
                        cmap_rotation=self.cmap_rotation,
                        ae_xx_diff_range=self.strain_range_xx_ae,
                        ae_yy_diff_range=self.strain_range_yy_ae,
                        ae_xy_diff_range=self.strain_range_xy_ae,
                        rotation_range=self.rotation_range,
                        )

    def visual_magnitude_of_strain(self,
                                    strain_range = [-2.5,2.5],
                                    ref_range = [-3,3],
                                    only_real = False):
        """function to visualize strain magnitude 

        Args:
            strain_range (list, optional): visualization range of neural network strain magnitude. Defaults to [-2.5,2.5].
            ref_range (list, optional): visualization range of py4DSTEM strain magnitude. Defaults to [-3,3].
            only_real (bool, optional): determine if only neural network results to be visualized. Defaults to False.
        """
        
        self.strain_range = strain_range
        self.ref_range = ref_range
        self.only_real = only_real
        
        # set strain x and strain y of py4DSTEM if file_py4DSTEM is True
        if self.file_py4DSTEM is not None:
            
            ref_xx = self.exx_correlation
            ref_yy = self.eyy_correlation
        else:
            ref_xx = None
            ref_yy = None
        
        # visualize strain magnitude 
        visual_strain_magnitude(self.exx_ae,
                                self.eyy_ae,
                                title_name=self.title_name,
                                folder_name=self.folder_name,
                                cmap=self.cmap_strain,
                                sample_index = self.sample_index,
                                ref_xx = ref_xx,
                                ref_yy = ref_yy,
                                strain_range = self.strain_range,
                                ref_range = self.ref_range,
                                img_size = self.im_size,
                                only_real = self.only_real
                                )