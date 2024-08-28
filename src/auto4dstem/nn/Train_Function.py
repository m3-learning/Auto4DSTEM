import torch
import os
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
from ..Data.DataProcess import STEM4D_DataSet
from ..Viz.util import (
    make_folder,
    inverse_base,
    Show_Process,
    add_disturb,
    upsample_single_mask,
)
from ..Viz.viz import add_colorbar
from .CC_ST_AE import make_model_fn
from .Loss_Function import AcumulatedLoss
from dataclasses import dataclass, field


@dataclass
class TrainClass:
    data_dir: str
    device: torch.device = torch.device("cpu")
    seed: int = 42
    crop: tuple = ((28, 228), (28, 228))
    transpose: tuple = (2, 3, 0, 1)
    background_weight: float = 0.2
    learned_rotation: any = None  # Specify the data type as required
    adjust_learned_rotation: int = 0
    background_intensity: bool = True
    counts_per_probe: float = 1e5
    intensity_coefficient: float = 1e5 / 4
    standard_scale: Optional[float] = None
    up_threshold: float = 1000
    down_threshold: float = 0
    boundary_filter: bool = False
    norm_order: int = 1
    radius: int = 45
    learning_rate: float = 3e-5
    en_original_step_size: list = field(default_factory=lambda: [200, 200])
    de_original_step_size: list = field(default_factory=lambda: [5, 5])
    pool_list: list = field(default_factory=lambda: [5, 4, 2])
    up_list: list = field(default_factory=lambda: [2, 4, 5])
    conv_size: int = 128
    scale: bool = True
    shear: bool = True
    rotation: bool = True
    rotate_clockwise: bool = True
    translation: bool = False
    Symmetric: bool = True
    mask_intensity: bool = True
    num_base: int = 1
    up_size: int = 800
    scale_limit: float = 0.05
    scale_penalty: float = 0.03
    shear_limit: float = 0.1
    shear_penalty: float = 0.03
    rotation_limit: float = 0.1
    trans_limit: float = 0.15
    adj_mask_para: float = 0
    crop_radius: int = 60
    sub_avg_coef: float = 1.5
    reduced_size: int = 20
    interpolate_mode: str = "bicubic"
    affine_mode: str = "bicubic"
    num_mask: int = 1
    fixed_mask: any = None  # Specify the data type as required
    check_mask: any = None  # Specify the data type as required
    interpolate: bool = True
    revise_affine: bool = True
    soft_threshold: float = 1.5
    hard_threshold: float = 3
    con_div: int = 15
    max_rate: float = 2e-4
    reg_coef: float = 1e-6
    scale_coef: float = 10
    shear_coef: float = 1
    batch_para: int = 1
    step_size_up: int = 20
    set_scheduler: bool = False
    weighted_mse: bool = True
    reverse_mse: bool = True
    weight_coef: int = 5
    lr_decay: bool = True
    lr_circle: bool = False
    batch_size: int = 4
    epochs: int = 20
    epoch_start_compare: int = 0
    epoch_start_save: int = 0
    epoch_start_update: int = 0
    epoch_end_update: int = 100
    folder_path: str = "save_weight"
    save_every_weights: bool = True
    dynamic_mask_region: bool = True
    cycle_consistent: bool = True

    """class of the training process, including load and preprocess the dataset and initialize loss class.

    Attributes:
        data_dir (string): directory of the dataset
        device (torch.device): set the device to run the model. Defaults to torch.device('cpu')
        seed (int): set the seed to make the training reproducible. Defaults to 42.
        crop (tuple, optional): the range of index for image cropping. Defaults to ((28,228),(28,228)).
        background_weight (float, optional): set the intensity of background noise for simulated dataset. Defaults to 0.2.
        learned_rotation (numpy array / string, optional): The numpy array/ directory of rotation weights represents pretrained rotation value if exists. Defaults to None.
        adjust_learned_rotation (int): The rotation degree added to learned_rotation if exists. Defaults to 0.
        background_intensity (bool): determine if the input dataset is simulated data or not. Defaults to True.
        counts_per_probe (float, optional): Counts per probe, can be None or float, defaulting to 1e5.
        intensity_coefficient (float): The intensity coefficient for scaling the noise, defaulting to 1e5/4.
        standard_scale (float, optional): determine if the input dataset needs standard scale or not, the value can determine the scale in data processing. Defaults to None.
        up_threshold (float): determine the value of up threshold of dataset. Defaults to 1000.
        down_threshold (float): determine the value of down threshold of dataset. Default to 0.
        boundary_filter (bool): determine if the dataset needs to be preprocessed with sobel filter. Defaults to False.
        norm_order (float): set the value of parameter multiplied by l norm. Defaults to 1.
        radius (int): set the radius of the small mask circle. Defaults to 45.
        learning_rate (float): set the learning rate for ADAM optimization. Defaults to 3e-5.
        en_original_step_size (list of integer): list of input image size to encoder. Defaults to [200,200].
        de_original_step_size (list of integer, optional): list of image size to decoder before reconstruction. Defaults to [5,5].
        pool_list (list of int): the list of parameter for each 2D MaxPool layer. Defaults to [5,4,2].
        up_list (list of int): the list of parameter for each 2D Upsample layer. Defaults to [2,4,5].
        conv_size (int): the value of filters number goes to each block. Defaults to 128.
        scale (bool): set to True if the model include scale affine transform
        shear (bool): set to True if the model include shear affine transform
        rotation (bool): set to True if the model include rotation affine transform
        rotate_clockwise (bool): set to True if the image should be rotated along one direction 
        translation (bool): set to True if the model include translation affine transform
        Symmetric (bool): set to True if the shear affine transform is symmetric
        mask_intensity (bool): set to True if the intensity of the mask region is learnable 
        num_base(int): the value for number of base. Defaults to 2.
        up_size (int, optional): the size of image to set for calculating MSE loss. Defaults to 800.
        scale_limit (float): set the range of scale. Defaults to 0.05.
        scale_penalty (float): set the scale limitation where to start adding regularization. Defaults to 0.04.
        shear_limit (float): set the range of shear. Defaults to 0.1.
        shear_penalty (float): set the shear limitation where to start adding regularization. Defaults to 0.03.
        rotation_limit (float): set the range of shear. Defaults to 0.1.
        trans_limit (float): set the range of translation. Defaults to 0.15.
        adj_mask_para (float): set the range of learnable parameter used to adjust pixel value in mask region. Defaults to 0.  
        crop_radius (int): set the radius of small square image for cropping. Defaults to 60.
        sub_avg_coef (float, optional): set the threshold for COM operation. Defaults to 1.5.
        reduced_size (int, optional): set the input length of K-top layer. Defaults to 20.
        interpolate_mode (str, optional): set the mode of interpolate function. Defaults to 'bicubic'.
        affine_mode (str, optional): set the affine mode to function F.affine_grid(). Defaults to 'bicubic'.
        num_mask (int): the value for number of mask. Defaults to len(fixed_mask).
        fixed_mask (list of tensor, optional): The list of tensor with binary type. Defaults to None.
        check_mask (list of tensor, optional): The list of tensor with binary type used for mask list updating. Defaults to None.
        interpolate (bool, optional): turn up grid version when inserting images into loss function. Defaults to True.
        revise_affine (bool): set to determine if need to add revise affine to image with affine transformation. Default to True.
        soft_threshold (float): set the value of threshold where using MAE replace MSE. Defaults to 1.5.
        hard_threshold (float): set the value of threshold where using hard threshold replace MAE. Defaults to 3.
        con_div (int): set the value of parameter divided by loss value. Defaults to 15.
        max_rate (float): maximum learning rate in the training cycle. Defaults to 2e-4.
        reg_coef (float): coefficient of l norm regularization. Defaults to 1e-6.
        scale_coef (float): coefficient of scale regularization. Defaults to 10.
        shear_coef (float): coefficient of shear regularization. Defaults to 1.
        batch_para (int):  set the value of parameter multiplied by batch size. Defaults to 1.
        step_size_up (int): the step size of half cycle. Defaults to 20.
        set_scheduler (bool): determine whether using torch.optim.lr_scheduler.CyclicLR function generate learning rate. Defaults to False.
        weighted_mse (bool): determine whether using weighted MSE in loss function. Defaults to True.
        reverse_mse (bool): determine the sequence of weighted MSE in loss function. Defaults to True.
        weight_coef (int):set the value of weight when using weighted MSE as loss function. Defaults to 2.
        lr_decay (bool): determine whether using learning rate decay after each epoch training. Defaults to True.
        lr_circle (bool): determine whether using lr_circular() function generate learning rate after each epoch. Defaults to False.
        batch_size (int): mini-batch value. Defaults to 4.
        epochs (int): determine the number of training epochs. Defaults to 20.
        epoch_start_compare (int): index of epoch to record and save training loss. Defaults to 0.
        epoch_start_save (int): index of epoch to start save pretrained weights. Defaults to 0.
        epoch_start_update (int): index of epoch to start update dynamic mask. Defaults to 0.
        epoch_end_update (int): index of epoch to end update dynamic mask. Defaults to 0.
        folder_path (str): folder dictionary to save pretrained weights. Defaults to ''.
        save_every_weights (bool): determine whether to save every pretrained weights. Defaults to True.
        dynamic_mask_region (bool): determine whether use dynamic mask list when computing loss. Defaults to True.
        cycle_consistent (bool): determine whether computing loss cycle consistently. Defaults to True.
    """

    def __post_init__(self):
        """replace __init__(), load dataset for initialization"""

        self.reset_dataset()

    def reset_dataset(self):
        """function for generating dataset"""

        # load pretrained rotation weights if learned_rotation is directory
        if type(self.learned_rotation) == str:
            self.learned_rotation = np.load(self.learned_rotation)

        # add adjust rotation degree to learned rotation
        if self.learned_rotation is not None:
            self.learned_rotation = add_disturb(
                self.learned_rotation, self.adjust_learned_rotation
            )

        # fix seed to reproduce results
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        # create dataset with or without rotation using updated or initialized parameter
        self.data_class = STEM4D_DataSet(
            self.data_dir,
            self.background_weight,
            crop=self.crop,
            transpose=self.transpose,
            background_intensity=self.background_intensity,
            counts_per_probe=self.counts_per_probe,
            intensity_coefficient=self.intensity_coefficient,
            rotation=self.learned_rotation,
            standard_scale=self.standard_scale,
            up_threshold=self.up_threshold,
            down_threshold=self.down_threshold,
            boundary_filter=self.boundary_filter,
        )

        # return the stem dataset
        self.data_set = self.data_class.stem4d_data
        # set initial value of real space domain
        self.mean_real_space_domain = None
        # pair each stem image with pretrained rotation
        if self.learned_rotation is not None:
            self.rotate_data = self.data_class.stem4d_rotation

    def crop_one_image(self, index=0, clim=[0, 1], cmap="viridis"):
        """function to pick one image for visualization

        Args:
            index (int, optional): index of image to pick. Defaults to 0.
            clim (list, optional): color range of the plt.imshow. Defaults to [0,1].
            cmap (str, optional): color map of imshow. Defaults to 'viridis'.
        """
        # load the dataset
        if self.data_dir.endswith(".h5") or self.data_dir.endswith(".mat"):
            print(self.data_dir)  # Printing the data directory for logging purposes
            with h5py.File(self.data_dir, "r") as f:  # Open the file in read mode
                stem4d_data = f["output4D"][:]  # Extract the data
            # Check if the data directory ends with '.npy' extension
        elif self.data_dir.endswith(".npy"):
            print(self.data_dir)
            stem4d_data = np.load(self.data_dir)  # Load the data using NumPy
        else:
            print("no correct format of input")
        # transpose and reshape the dataset
        stem4d_data = np.transpose(stem4d_data, self.transpose)
        stem4d_data = stem4d_data.reshape(
            -1, stem4d_data.shape[-2], stem4d_data.shape[-1]
        )
        # pick up image
        self.pick_1_image = stem4d_data[index][:]
        # visualize image
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.imshow(self.pick_1_image, cmap=cmap, clim=clim)
        # delete the generated data to clean the memory
        del stem4d_data

    def visual_noise(self, noise_level=[0], clim=[0, 1], file_name="", cmap="viridis"):
        """function to visualize poisson noise scaling images

        Args:
            noise_level (list, optional): list of noise level. Defaults to [0].
            clim (list, optional): color range of plot. Defaults to [0,1].
            file_name (str, optional): name of saved figure. Defaults to ''.
            cmap (str, optional): color map of imshow. Defaults to '1'.
        """
        # create figure
        fig, ax = plt.subplots(1, len(noise_level), figsize=(5 * len(noise_level), 5))
        # add poisson noise on image
        for i, background_weight in enumerate(noise_level):
            # generate string of noise
            bkg_str = format(int(background_weight * 100), "02d")
            test_img = np.copy(self.pick_1_image)
            # add poisson noise
            qx = np.fft.fftfreq(self.pick_1_image.shape[0], d=1)
            qy = np.fft.fftfreq(self.pick_1_image.shape[1], d=1)
            qya, qxa = np.meshgrid(qy, qx)
            qxa = np.fft.fftshift(qxa)
            qya = np.fft.fftshift(qya)
            qra2 = qxa**2 + qya**2
            im_bg = 1.0 / (1 + qra2 / 1e-2**2)
            im_bg = im_bg / np.sum(im_bg)
            # generate noisy image
            int_comb = test_img * (1 - background_weight) + im_bg * background_weight
            int_noisy = (
                np.random.poisson(int_comb * self.counts_per_probe)
                / self.counts_per_probe
            )
            if background_weight == 0:
                int_noisy = self.pick_1_image * self.intensity_coefficient
            else:
                int_noisy = int_noisy * self.intensity_coefficient
            # add title to each image
            ax[i].title.set_text(f"{bkg_str} Percent")
            ax[i].imshow(int_noisy, cmap=cmap, clim=clim)
        # clean x,y tick labels
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        fig.tight_layout()
        # save figure
        plt.savefig(
            f"{self.folder_path}/{file_name}_generated_different_level_noise.svg"
        )

    def lr_circular(
        self,
        epoch,
        step_size_up=20,
        min_rate=3e-5,
        max_rate=2e-4,
    ):
        """function for custom learning rate decay

        Args:
            epoch (int): the current epoch index for learning rate calculating
            step_size_up (int): the step size of half cycle. Defaults to 20.
            min_rate (float): minimum learning rate in the cycle. Defaults to 3e-5.
            max_rate (float): maximum learning rate in the cycle. Defaults to 2e-4.

        Returns:
            float: learning rate value in current epoch training
        """
        # compute lr increase or decrease for each epoch
        lr_change_size = (max_rate - min_rate) / step_size_up

        # compute number of times adding/subtracting lr_change_size
        num = epoch % step_size_up

        # determine current period of lr increase/decrease
        para = int(epoch / step_size_up)

        # compute lr for current epoch
        if para % 2 == 0:
            lr = min_rate + num * lr_change_size

        else:
            lr = max_rate - num * lr_change_size

        return lr

    def reset_model(self):
        """initialize model with class parameter or updated parameter

        Returns:
            torch.Module: encoder, decoder, autoencoder and optimizer
        """

        encoder, decoder, join, optimizer = make_model_fn(
            self.device,
            self.learning_rate,
            self.en_original_step_size,
            self.de_original_step_size,
            self.pool_list,
            self.up_list,
            self.conv_size,
            self.scale,
            self.shear,
            self.rotation,
            self.rotate_clockwise,
            self.translation,
            self.Symmetric,
            self.mask_intensity,
            self.num_base,
            self.up_size,
            self.scale_limit,
            self.shear_limit,
            self.rotation_limit,
            self.trans_limit,
            self.adj_mask_para,
            self.crop_radius,
            self.sub_avg_coef,
            self.reduced_size,
            self.interpolate_mode,
            self.affine_mode,
            self.num_mask,
            self.fixed_mask,
            self.interpolate,
            self.revise_affine,
        )

        return encoder, decoder, join, optimizer

    def reset_loss_class(self):
        """function used for initializing loss class with initialized or updated parameters

        Returns:
            Class(Object): loss class
        """

        loss_fuc = AcumulatedLoss(
            self.device,
            reg_coef=self.reg_coef,
            scale_coef=self.scale_coef,
            shear_coef=self.shear_coef,
            norm_order=self.norm_order,
            scale_penalty=self.scale_penalty,
            shear_penalty=self.shear_penalty,
            mask_list=self.fixed_mask,
            weighted_mse=self.weighted_mse,
            reverse_mse=self.reverse_mse,
            weight_coef=self.weight_coef,
            interpolate=self.interpolate,
            batch_para=self.batch_para,
            cycle_consistent=self.cycle_consistent,
            dynamic_mask_region=self.dynamic_mask_region,
            soft_threshold=self.soft_threshold,
            hard_threshold=self.hard_threshold,
            con_div=self.con_div,
        )

        return loss_fuc

    def load_pretrained_weight(self, weight_path):
        """function used to load pretrained weight to neural network

        Args:
            weight_path (string): dictionary of pretrained weight

        Returns:
            torch.Module: pytorch model with pretrained weight loaded
        """

        # resets the model
        encoder, decoder, join, optimizer = self.reset_model()

        # load the pretrained weight
        if self.device == torch.device("cpu"):
            check_ccc = torch.load(weight_path, map_location=self.device)
        else:
            check_ccc = torch.load(weight_path)

        # load the pretrained weight to model
        join.load_state_dict(check_ccc["net"])
        encoder.load_state_dict(check_ccc["encoder"])
        decoder.load_state_dict(check_ccc["decoder"])
        optimizer.load_state_dict(check_ccc["optimizer"])

        # initial model in training class
        self.join = join
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer

    def show_pickup_dots(
        self,
        x_axis,
        y_axis,
        img_size=None,
    ):
        """function to show pick up dots in real space domain

        Args:
            x_axis (list): list of x coordinates of dots
            y_axis (list): list of y coordinates of dots
            img_size (_type_, optional): _description_. Defaults to None.
        """
        # initialize the image size if not given
        if img_size is None:
            x_size = y_size = int(np.sqrt(self.data_set.shape[0]))
        else:
            # set size of x,y coordinates
            x_size = img_size[0]
            y_size = img_size[1]
        # raise problem if not select 6 dots
        if len(x_axis) < 6 or len(y_axis) < 6:
            return "please select 6 points for visualization"
        # pick first 6 pairs of coordinates
        x_axis = x_axis[0:6]
        y_axis = y_axis[0:6]
        # set mean image of real space domain if not exists
        if self.mean_real_space_domain is None:
            self.mean_real_space_domain = np.mean(
                self.data_set.reshape(x_size, y_size, -1), axis=2
            )
        # plot the image and the position of pick up points
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.plot(x_axis, y_axis, "r.")
        plt.imshow(self.mean_real_space_domain)
        # reshape the points coordinates into 1-d vector
        index_ = []
        for i in range(6):
            index_.append(y_axis[i] * y_size + x_axis[i])
        # switch it into numpy array
        self.sample_series = np.array(index_)

    def show_transforming_sample(
        self,
        mask=None,
        clim=[0, 1],
        clim_d=[0, 1],
        file_name="",
        train_process="1",
        cmap="viridis",
    ):
        """function to show the visualization for pick up points

        Args:
            mask (tensor/numpy, optional): boolean mask in numpy or tensor format. Defaults to None.
            clim (list, optional): color range of visualization. Defaults to [0,1].
            clim_d (list, optional): color range of difference. Defaults to [0,1].
            file_name (str, optional): initial name of the file. Defaults to ''.
            train_process (str, optional): determine use which dataset to show. Defaults to '1'.
            cmap (str, optional): color map of imshow. Defaults to 'viridis'.
        """
        # use the pre select index of dataset for visualization, use dataset without rotation when train process '1'
        if train_process == "1":
            visual_data = self.data_set[self.sample_series]
            x = torch.tensor(visual_data, dtype=torch.float).to(self.device)
            y = None
        # use dataset with rotation when train process not '1'
        else:
            visual_data = [self.rotate_data[i] for i in self.sample_series]
            # load dataset into dataloader
            x, y = next(iter(DataLoader(visual_data, batch_size=6, shuffle=False)))
            x = x.to(self.device, dtype=torch.float)
            y = y.to(self.device, dtype=torch.float)

        # use model predicts the results, training type depends on interpolated mode
        if self.interpolate:
            (
                predicted_x,
                predicted_base,
                predicted_input,
                kout,
                theta_1,
                theta_2,
                theta_3,
                adj_mask,
                new_list,
                x_inp,
            ) = self.join(x, y)
        else:
            (
                predicted_x,
                predicted_base,
                predicted_input,
                kout,
                theta_1,
                theta_2,
                theta_3,
                adj_mask,
                new_list,
            ) = self.join(x, y)

        # initial mask value if not pre defined
        if mask is None:
            mask = 0
        # upgrid mask size if not equal to predict results
        elif mask.shape[-2:] != predicted_base.shape[-2:]:
            mask = upsample_single_mask(mask=mask, up_size=predicted_base.shape[-2:])
        else:
            mask = mask

        # visualize results
        fig, ax = plt.subplots(6, 5, figsize=(25, 30))
        for i in range(6):
            # add subtitle
            if i == 0:
                ax[i][0].title.set_text("raw input")
                ax[i][1].title.set_text("transformed base")
                ax[i][2].title.set_text("transformed input")
                ax[i][3].title.set_text("learned base")
                ax[i][4].title.set_text("difference")
            # remove the x,y tick labels for each image
            ax[i][0].set_xticklabels("")
            ax[i][0].set_yticklabels("")
            ax[i][1].set_xticklabels("")
            ax[i][1].set_yticklabels("")
            ax[i][2].set_xticklabels("")
            ax[i][2].set_yticklabels("")
            ax[i][3].set_xticklabels("")
            ax[i][3].set_yticklabels("")
            ax[i][4].set_xticklabels("")
            ax[i][4].set_yticklabels("")
            # determine the raw input depends on interpolate mode
            if self.interpolate:
                input_img = x_inp[i].squeeze().detach().cpu()
            else:
                input_img = x[i].squeeze().detach().cpu()
            im0 = ax[i][0].imshow(input_img, cmap=cmap, clim=clim)
            add_colorbar(im0, ax[i, 0])
            # plot show base with reverse affine transform
            reverse_base = predicted_input[i].squeeze().detach().cpu()
            reverse_base[~mask] = 0
            im1 = ax[i][1].imshow(reverse_base, cmap=cmap, clim=clim)
            add_colorbar(im1, ax[i, 1])
            # plot show input with affine transform
            transformed_input = predicted_x[i].squeeze().detach().cpu()
            transformed_input[~mask] = 0
            im2 = ax[i][2].imshow(transformed_input, cmap=cmap, clim=clim)
            add_colorbar(im2, ax[i, 2])
            # plot show generated base
            learned_base = predicted_base[i].squeeze().detach().cpu()
            learned_base[~mask] = 0
            im3 = ax[i][3].imshow(learned_base, cmap=cmap, clim=clim)
            add_colorbar(im3, ax[i, 3])
            # plot show MSE between generated base and input with affine transform
            im4 = ax[i][4].imshow(
                (transformed_input - learned_base) ** 2, cmap=cmap, clim=clim_d
            )
            add_colorbar(im4, ax[i, 4])
        # save figure
        plt.savefig(
            f"{self.folder_path}/{file_name}_show_affine_process_of_pickup_samples.svg"
        )

    def predict(
        self,
        sample_index=None,
        train_process="1",
        save_strain=False,
        save_rotation=False,
        save_translation=False,
        save_classification=False,
        save_base=False,
        file_name="",
        num_workers=0,
    ):
        """function to predict and save results

        Args:
            sample_index (np.array, optional): 1-D array of index of input dataset if exists. Defaults to None.
            train_process (str, optional): determine the training process of prediction. Defaults to '1'.
            save_strain (bool, optional): determine if strain weights saved. Defaults to False.
            save_rotation (bool, optional): determine if rotation weights saved. Defaults to False.
            save_translation (bool, optional): determine if translation weights saved. Defaults to False.
            save_classification (bool, optional): determine if classification weights saved. Defaults to False.
            save_base (bool, optional): determine if generated base weights saved. Defaults to False.
            file_name (float/int/str, optional): set the initial of file name. Defaults to ''.
            num_workers (int, optional): set number of workers in dataloader. Defaults to 0.
        """
        # create sample index for reproducing results
        if sample_index is None:
            # if sample index is None, include all index into sample index
            sample_index = np.arange(len(self.data_set))
        # determine which results should be reproduced
        if train_process == "1":
            # load dataset into dataloader
            data_iterator = DataLoader(
                self.data_set[sample_index],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        else:
            only_sample = [self.rotate_data[i] for i in sample_index]
            data_iterator = DataLoader(
                only_sample,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        # create infrastructure to load trained weights, include rotation, strain, translation and classification
        rotation = np.zeros([len(self.data_set[sample_index]), 2])
        scale_shear = np.zeros([len(self.data_set[sample_index]), 4])
        translation = np.zeros([len(self.data_set[sample_index]), 2])
        select_k = np.zeros([len(self.data_set[sample_index]), self.num_base])

        # predict weights with pretrained model
        for i, x_value in enumerate(
            tqdm(data_iterator, leave=True, total=len(data_iterator))
        ):
            with torch.no_grad():
                # determine x and y based on training process
                if train_process == "1":
                    x = x_value.to(self.device, dtype=torch.float)
                    y = None
                else:
                    x, y = x_value
                    x = x.to(self.device, dtype=torch.float)
                    y = y.to(self.device, dtype=torch.float)

                # determine the number of input based on interpolate mode, predict results.
                if self.interpolate:
                    (
                        predicted_x,
                        predicted_base,
                        predicted_input,
                        kout,
                        theta_1,
                        theta_2,
                        theta_3,
                        adj_mask,
                        new_list,
                        x_inp,
                    ) = self.join(x, y)
                # determine the number of input based on interpolate mode, predict results.
                else:
                    (
                        predicted_x,
                        predicted_base,
                        predicted_input,
                        kout,
                        theta_1,
                        theta_2,
                        theta_3,
                        adj_mask,
                        new_list,
                    ) = self.join(x, y)

                # save weights into infrastructure
                if x.shape[0] == self.batch_size:

                    scale_shear[i * self.batch_size : (i + 1) * self.batch_size] = (
                        theta_1[:, :, 0:2].cpu().detach().numpy().reshape(-1, 4)
                    )
                    rotation[i * self.batch_size : (i + 1) * self.batch_size] = (
                        theta_2[:, :, 0].cpu().detach().numpy()
                    )
                    translation[i * self.batch_size : (i + 1) * self.batch_size] = (
                        theta_3[:, :, 2].cpu().detach().numpy()
                    )
                    select_k[i * self.batch_size : (i + 1) * self.batch_size] = (
                        kout.cpu().detach().numpy().reshape(-1, self.num_base)
                    )
                # save weights into infrastructure
                else:
                    scale_shear[i * self.batch_size :] = (
                        theta_1[:, :, 0:2].cpu().detach().numpy().reshape(-1, 4)
                    )
                    rotation[i * self.batch_size :] = (
                        theta_2[:, :, 0].cpu().detach().numpy()
                    )
                    translation[i * self.batch_size :] = (
                        theta_3[:, :, 2].cpu().detach().numpy()
                    )
                    select_k[i * self.batch_size :] = (
                        kout.cpu().detach().numpy().reshape(-1, self.num_base)
                    )

        # save weights into public variables to the class
        self.generated_base = predicted_base[0].cpu().detach().numpy()
        self.strain_matrix = scale_shear
        self.rotation_matrix = rotation
        self.translation_matrix = translation
        self.classification_matrix = select_k

        # set file name according to insert
        if type(file_name) == float or type(file_name) == int:
            file_name = format(int(file_name * 100), "02d") + "Per"
        file_name += f"_{train_process}_train_process"

        # save strain if mode is on
        if save_strain:
            np.save(
                f"{self.folder_path}/{file_name}_scale_shear.npy", self.strain_matrix
            )
        # save rotation if mode is on
        if save_rotation:
            np.save(
                f"{self.folder_path}/{file_name}_rotation.npy", self.rotation_matrix
            )
        # save translation if mode is on
        if save_translation:
            np.save(
                f"{self.folder_path}/{file_name}_translation.npy",
                self.translation_matrix,
            )
        # save classification if mode is on
        if save_classification:
            np.save(
                f"{self.folder_path}/{file_name}_classification.npy",
                self.classification_matrix,
            )
        # save generated base if mode is on
        if save_base:
            np.save(
                f"{self.folder_path}/{file_name}_generated_base.npy",
                self.generated_base,
            )

    def save_predict(
        self,
        train_process="1",
        save_strain=False,
        save_rotation=False,
        save_translation=False,
        save_classification=False,
        save_base=False,
        file_name="",
    ):
        """function to save predict results

        Args:
            train_process (str, optional): determine the training process of prediction. Defaults to '1'.
            save_strain (bool, optional): determine if strain weights saved. Defaults to False.
            save_rotation (bool, optional): determine if rotation weights saved. Defaults to False.
            save_translation (bool, optional): determine if translation weights saved. Defaults to False.
            save_classification (bool, optional): determine if classification weights saved. Defaults to False.
            save_base (bool, optional): determine if generated base weights saved. Defaults to False.
            file_name (float/int/str, optional): set the initial of file name. Defaults to ''.
        """
        # set file name according to insert
        if type(file_name) == float or type(file_name) == int:
            file_name = format(int(file_name * 100), "02d") + "Per"
        file_name += f"_{train_process}_train_process"

        # save strain if mode is on
        if save_strain:
            np.save(
                f"{self.folder_path}/{file_name}_scale_shear.npy", self.strain_matrix
            )
        # save rotation if mode is on
        if save_rotation:
            np.save(
                f"{self.folder_path}/{file_name}_rotation.npy", self.rotation_matrix
            )
        # save translation if mode is on
        if save_translation:
            np.save(
                f"{self.folder_path}/{file_name}_translation.npy",
                self.translation_matrix,
            )
        # save classification if mode is on
        if save_classification:
            np.save(
                f"{self.folder_path}/{file_name}_classification.npy",
                self.classification_matrix,
            )
        # save generated base if mode is on
        if save_base:
            np.save(
                f"{self.folder_path}/{file_name}_generated_base.npy",
                self.generated_base,
            )

    def train_process(self):
        """function call the train process for model training"""
        # fix seed of the model
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        # create folder directory to save weight
        make_folder(self.folder_path)

        # if dynamic_mask_region is True, the interpolate should also be set to True
        if self.dynamic_mask_region:
            self.interpolate = True
        # initial check mask if not pre-defined
        if not self.check_mask:
            self.check_mask = self.fixed_mask
        # learning rate for training
        learning_rate = round(self.learning_rate, 6)

        # minimum learning rate
        min_rate = round(self.learning_rate, 6)
        # maximum learning rate
        max_rate = round(self.max_rate, 6)
        # coefficient of l norm regularization
        reg_coef = round(self.reg_coef, 9)
        # coefficient of scale regularization
        scale_coef = round(self.scale_coef, 2)
        # coefficient of shear regularization
        shear_coef = round(self.shear_coef, 2)

        # initialize coefficient to record lr decay condition
        patience = 0

        # initialize model
        encoder, decoder, join, optimizer = self.reset_model()

        # set lr scheduler if set_scheduler is True
        if self.set_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=min_rate,
                max_lr=max_rate,
                step_size_up=self.step_size_up,
                cycle_momentum=False,
            )
            # if set_scheduler is True, turn off lr_decay and lr_circle mode
            self.lr_decay = False
            self.lr_circle = False
        else:
            lr_scheduler = None

        # dynamic_mask_region is True, means in second training process, the dateset is [image, rotation]
        # dynamic_mask_region is False, means in first training process, the dateset is [image, None]
        if self.dynamic_mask_region:
            train_iterator = DataLoader(
                self.rotate_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )

            test_iterator = DataLoader(
                self.rotate_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

        else:
            train_iterator = DataLoader(
                self.data_set, batch_size=self.batch_size, shuffle=True, num_workers=0
            )

            test_iterator = DataLoader(
                self.data_set, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

        # set total epoch to train
        N_EPOCHS = self.epochs

        # initialize loss class
        loss_class = self.reset_loss_class()

        # initialize best loss to infinite
        best_train_loss = float("inf")

        for epoch in range(N_EPOCHS):
            # load pretrained weight result of previous epoch training
            if self.interpolate:
                # set the range of epoch for updating (potentially learning rate and mask region)
                if epoch > self.epoch_start_update and epoch <= self.epoch_end_update:
                    encoder, decoder, join, optimizer = self.reset_model()
                    if self.device == torch.device("cpu"):
                        check_ccc = torch.load(file_path, map_location=self.device)
                    else:
                        check_ccc = torch.load(file_path)

                    join.load_state_dict(check_ccc["net"])
                    encoder.load_state_dict(check_ccc["encoder"])
                    decoder.load_state_dict(check_ccc["decoder"])
                    optimizer.load_state_dict(check_ccc["optimizer"])

            # update learning rate if lr_decay is True
            if self.lr_decay:
                optimizer.param_groups[0]["lr"] = learning_rate
            # update learning rate if lr_circle is True
            elif self.lr_circle:
                optimizer.param_groups[0]["lr"] = self.lr_circular(
                    epoch,
                    step_size_up=self.step_size_up,
                    min_rate=min_rate,
                    max_rate=max_rate,
                )
            # update loss class if dynamic_mask_region is True
            if self.dynamic_mask_region:
                loss_class = self.reset_loss_class()

            # compute and return loss dictionary
            loss_dictionary = loss_class.__call__(
                join,
                train_iterator,
                optimizer,
            )
            # load loss value to save in weights' name
            train_loss = loss_dictionary["train_loss"]
            L2_loss = loss_dictionary["l2_loss"]
            Scale_Loss = loss_dictionary["scale_loss"]
            Shear_Loss = loss_dictionary["shear_loss"]

            # save mask list and generated base in each epoch
            if self.interpolate:
                name_of_file = (
                    self.folder_path
                    + f"/L1:{reg_coef:.10f}_scale:{scale_coef:.3f}_shear:{shear_coef:.3f}_lr:{learning_rate:.6f}_Epoch:{epoch:04d}_trainloss:{train_loss:.6f}_"
                )

                # save mask list and base to particular name
                Show_Process(
                    join,
                    test_iterator,
                    self.fixed_mask,
                    name_of_file,
                    self.device,
                    self.interpolate,
                )
                # update mask list according to generated base in particular epoch period
                if epoch >= self.epoch_start_update and epoch < self.epoch_end_update:
                    center_mask_list, rotate_center = inverse_base(
                        name_of_file, self.check_mask, radius=self.radius
                    )
                    self.fixed_mask = center_mask_list

            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
            print(".............................")
            # save weights, including encoder, decoder, autoencoder and optimizer.
            checkpoint = {
                "net": join.state_dict(),
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "mse_loss": train_loss,
            }
            # convert variable into string format to save in file name
            lr_ = format(optimizer.param_groups[0]["lr"], ".6f")
            scale_form = format(scale_coef, ".4f")
            shear_form = format(shear_coef, ".4f")
            cust_form = format(self.batch_para, "0d")
            file_path = (
                self.folder_path
                + "/Weight_lr:"
                + lr_
                + "_scale_cof:"
                + scale_form
                + "_shear_cof:"
                + shear_form
                + "_custom_para:"
                + cust_form
                + f"_epoch:{epoch:04d}_trainloss:{train_loss:.5f}_l1:{L2_loss:.5f}_scal:{Scale_Loss:.5f}_shr:{Shear_Loss:.5f}.pkl"
            )

            # determine if save every weight is necessary (if interpolate mode is True, save_every_weight should be True)
            if self.save_every_weights:
                torch.save(checkpoint, file_path)

                # update learning rate
                if self.lr_decay:
                    if epoch >= self.epoch_start_compare:
                        if best_train_loss > train_loss:
                            best_train_loss = train_loss
                            # initialize patience parameter
                            patience = 0

                            learning_rate = 1.2 * learning_rate

                        else:
                            patience += 1

                            if patience > 0:
                                learning_rate = learning_rate * 0.8

            else:
                # start update loss after epoch_start_compare
                if epoch >= self.epoch_start_compare:
                    if best_train_loss > train_loss:
                        best_train_loss = train_loss
                        # save model weights after epoch_start_save
                        if epoch >= self.epoch_start_save:
                            torch.save(checkpoint, file_path)
                        # update learning rate according to lr_decay
                        if self.lr_decay:
                            patience = 0
                            learning_rate = 1.2 * learning_rate

                    else:
                        if self.lr_decay:
                            patience += 1

                            if patience > 0:
                                learning_rate = learning_rate * 0.8
            # update learning rate according to lr_scheduler
            if lr_scheduler != None:
                lr_scheduler.step()
