import torch
import os
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from auto4dstem.nn.mixins.datamixins import DataMixin
from auto4dstem.nn.mixins.imagemixins import ImageMixin
from auto4dstem.nn.mixins.modelmixins import ModelMixin
from auto4dstem.random.seeds import set_seed
from ..transformations.image import add_rotation
from ..data.DataProcess import STEM4D_DataSet
from ..viz.util import (
    inverse_base,
    Show_Process,
    upsample_single_mask,
)
from ..viz.viz import add_colorbar
from .CC_ST_AE.CC_ST_AE import make_model_fn
from .Loss_Function import AccumulatedLoss
from dataclasses import dataclass, field
from m3util.util.IO import make_folder
from m3util.viz.text import labelfigs
from m3util.util.kwargs import filter_cls_params
import torch.nn as nn
import torch.optim as optim


@dataclass
class Train(
    DataMixin,
    ImageMixin,
    ModelMixin,
):
    """Class for managing the training process, including dataset loading, preprocessing, and loss initialization.

    This class integrates various mixins to handle data, image, and model properties, facilitating a comprehensive training setup.

    Attributes:
        join (Optional[nn.Module]): Module for joining different parts of the model. Defaults to None.
        encoder (Optional[nn.Module]): Encoder module of the model. Defaults to None.
        decoder (Optional[nn.Module]): Decoder module of the model. Defaults to None.
        optimizer (Optional[optim.Optimizer]): Optimizer for training the model. Defaults to None.

    Methods:
        __init__: Sets up the Train class with specified parameters.
        load_data: Handles loading and preprocessing of the dataset.
        initialize_loss: Sets up the loss function for training.
        train_model: Executes the training process with specified parameters.
        save_model: Persists the trained model weights to storage.
        update_mask: Modifies the dynamic mask list during training iterations.
        compute_loss: Calculates the loss value during training.
        adjust_learning_rate: Modifies the learning rate as needed during training.
        visualize_results: Generates visual representations of training outcomes.
        save_results: Archives the results obtained during training.
    """
    
    join: Optional[nn.Module] = None
    encoder: Optional[nn.Module] = None
    decoder: Optional[nn.Module] = None
    optimizer: Optional[optim.Optimizer] = None
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Post-initialization method to replace __init__().

        This method loads the dataset for initialization and sets up the initial model structure.
        """
        self.verbose = self.kwargs.get('verbose', False)
        
        self.load_data()

    def load_data(self):
        """
        Generates the dataset for training.

        This function loads pretrained rotation weights if provided, adjusts the rotation degree,
        and sets the seed for reproducibility.
        """

        # Load pretrained rotation weights if learned_rotation is a directory
        if isinstance(self.learned_rotation, str):
            self.learned_rotation = np.load(self.learned_rotation)

        # Adjust rotation degree if learned_rotation is provided
        if self.learned_rotation is not None:
            self.learned_rotation = add_rotation(
                self.learned_rotation, self.coarse_learned_angle_adjustment
            )

        # fix seed to reproduce results
        set_seed(seed=self.seed)

        dataset_params = filter_cls_params(STEM4D_DataSet, vars(self))
        if self.verbose:
            print(dataset_params)
        self.data_class = STEM4D_DataSet(**dataset_params)

        # return the stem dataset
        self.data_set = self.data_class.stem4d_data

        # pair each stem image with pretrained rotation
        if self.learned_rotation is not None:
            self.rotate_data = self.data_class.stem4d_rotation
            
        # computes the real space average image of the dataset
        self.compute_ave_real_space_image(**self.kwargs)


    # def _load_from_file(self, **kwargs):
    #     """
    #     Loads the dataset from a file.
    #     """

    #     index = kwargs.get("index", None)

    #     # load the dataset
    #     if self.data_dir.endswith(".h5") or self.data_dir.endswith(".mat"):
    #         print(self.data_dir)  # Printing the data directory for logging purposes
    #         with h5py.File(self.data_dir, "r") as f:  # Open the file in read mode
    #             stem4d_data = f["output4D"][:] if index is None else f["output4D"][index]  # Extract the data

    #     # Check if the data directory ends with '.npy' extension
    #     elif self.data_dir.endswith(".npy"):
    #         print(self.data_dir)
    #         stem4d_data = np.load(self.data_dir, mmap_mode='r')[index]  # Load just the index using NumPy
    #     else:
    #        raise ValueError("no correct format of input")

    #     return stem4d_data

    def raw_data(self, **kwargs):
        # stem4d_data = self.data_set / self.intensity_scaler
        # stem4d_data = stem4d_data.reshape(
        #     -1, stem4d_data.shape[-2], stem4d_data.shape[-1]
        # )
        
        data = self.data_class.raw_data

        index = kwargs.get("index", None)
        if index is not None:
            return data[index]
        else:
            return data

        
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

    def initialize_model(self):
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
            self.upsample_dimensions,
            self.scale_limit,
            self.shear_limit,
            self.rotation_limit,
            self.trans_limit,
            self.learnable_mask_intensity,
            self.reverse_affine_transform_crop_radius,
            self.COM_threshold_coef,
            self.embedding_size,
            self.upsampling_interpolation_mode,
            self.affine_interpolation_mode,
            self.dynamic_mask_to_loss_function,
            self.interpolate,
            self.reverse_affine,
        )

        return encoder, decoder, join, optimizer

    def reset_loss_class(self):
        """function used for initializing loss class with initialized or updated parameters

        Returns:
            Class(Object): loss class
        """

        loss_fuc = AccumulatedLoss(
            self.device,
            reg_coef=self.regularization_coef,
            scale_coef=self.scale_regularization_coef,
            shear_coef=self.shear_regularization_coef,
            norm_order=self.norm_order,
            scale_penalty=self.scale_penalty,
            shear_penalty=self.shear_penalty,
            mask_list=self.dynamic_mask_to_loss_function,
            weighted_mse=self.weighted_mse_flag,
            reverse_mse=self.mse_difference_sign_preference_flag,
            weight_coef=self.weighted_mse_coef,
            interpolate=self.interpolate,
            batch_para=self.large_batch_training_param,
            cycle_consistent=self.cycle_consistent_flag,
            dynamic_mask_region=self.adaptive_mask_loss_flag,
            soft_threshold=self.soft_loss_threshold,
            hard_threshold=self.hard_loss_threshold,
            noise_loss_scaling_factor=self.noise_loss_scaling_factor,
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
        encoder, decoder, join, optimizer = self.initialize_model()

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
        self, x_axis, y_axis, img_size=None, add_label=True, label_style="wb"
    ):
        """function to show pick up dots in real space domain

        Args:
            x_axis (list): list of x coordinates of dots
            y_axis (list): list of y coordinates of dots
            img_size (_type_, optional): _description_. Defaults to None.
            add_label (bool, optional): determine if add label to figure.
            label_style (str, optional): determine label style. Defaults to 'wb'
        """
        
        # TODO: fix for non-square images
        # initialize the image size if not given
        y_size, x_size = self.get_image_size(img_size)
            
        # raise problem if not select 6 dots
        if len(x_axis) != len(y_axis):
            raise ValueError("please insert valid xaxis and yaxis")
            
        # plot the image and the position of pick up points
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        axs.set_xticklabels([])
        axs.set_yticklabels([])
        axs.plot(x_axis, y_axis, "r.")
        axs.imshow(self.mean_real_space_domain)
        # add label to the image
        if add_label:
            labelfigs(
                axs,
                number=0,
                style=label_style,
                loc="tl",
                size=20,
                inset_fraction=(0.1, 0.1),
            )
        # reshape the points coordinates into 1-d vector
        index_ = []
        for i in range(len(x_axis)):
            index_.append(y_axis[i] * y_size + x_axis[i])
        # switch it into numpy array
        self.sample_series = np.array(index_)

    def get_image_size(self, **kwargs):
        """
        Determines the size of the image.

        This function calculates the size of the image based on the provided keyword arguments.
        If 'img_size' is not provided, it assumes a square image and calculates the size based on the dataset shape.

        Args:
            **kwargs: Arbitrary keyword arguments. Expected keys:
                - img_size (tuple, optional): A tuple containing the dimensions of the image (x_size, y_size).

        Returns:
            tuple: A tuple containing the dimensions of the image (y_size, x_size).
        """
        if kwargs.get('img_size') is None:
            y_size = int(np.sqrt(self.data_set.shape[0]))
            x_size = y_size
        else:
            # set size of x,y coordinates
            x_size = kwargs.get('img_size')[0]
            y_size = kwargs.get('img_size')[1]
        return y_size, x_size

    def compute_ave_real_space_image(self, **kwargs):
        """
        Computes the average real space image of the dataset.

        This function calculates the mean of the dataset in the real space domain.
        It first determines the size of the image using the provided keyword arguments.
        If the attribute 'mean_real_space_domain' does not exist, it computes the mean
        of the reshaped dataset and assigns it to 'mean_real_space_domain'.

        Args:
            **kwargs: Arbitrary keyword arguments. Expected keys:
                - img_size (tuple, optional): A tuple containing the dimensions of the image (x_size, y_size).

        Returns:
            None
        """
        y_size, x_size = self.get_image_size(**kwargs)
        if not hasattr(self, 'mean_real_space_domain'):
            self.mean_real_space_domain = np.mean(
                self.data_set.reshape(x_size, y_size, -1), axis=2
            )

    def show_transforming_sample(
        self,
        mask=None,
        clim=[0, 1],
        clim_d=[0, 1],
        file_name="",
        train_process="1",
        cmap="viridis",
        save_figure=True,
        add_label=True,
        label_style="wb",
    ):
        """function to show the visualization for pick up points

        Args:
            mask (tensor/numpy, optional): boolean mask in numpy or tensor format. Defaults to None.
            clim (list, optional): color range of visualization. Defaults to [0,1].
            clim_d (list, optional): color range of difference. Defaults to [0,1].
            file_name (str, optional): initial name of the file. Defaults to ''.
            train_process (str, optional): determine use which dataset to show. Defaults to '1'.
            cmap (str, optional): color map of imshow. Defaults to 'viridis'.
            save_figure (bool, optional): determine if save needed. Defaults to True.
            add_label (bool, optional): determine if add label to figure. Defaults to True.
            label_style (str, optional): determine label style. Defaults to 'wb'.
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
            x, y = next(
                iter(
                    DataLoader(
                        visual_data, batch_size=len(self.sample_series), shuffle=False
                    )
                )
            )
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
        # create h5 file to save noisy image
        hf = h5py.File(
            f"{self.folder_path}/transformed_sample_of_index_{self.sample_series}.h5",
            "w",
        )

        # visualize results
        fig, ax = plt.subplots(
            len(self.sample_series), 5, figsize=(25, 5 * len(self.sample_series))
        )
        for i in range(len(self.sample_series)):
            # remove the x,y tick labels for each image
            for j in range(5):
                ax[i][j].set_xticklabels("")
                ax[i][j].set_yticklabels("")

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
            # add generated results in h5 file
            hf.create_dataset(
                f"{self.sample_series[i]}",
                data=[input_img, learned_base, reverse_base, transformed_input],
            )
            add_colorbar(im4, ax[i, 4])
            # add subtitle
            if i == 0:
                ax[i][0].title.set_text("raw input")
                ax[i][1].title.set_text("transformed base")
                ax[i][2].title.set_text("transformed input")
                ax[i][3].title.set_text("learned base")
                ax[i][4].title.set_text("difference")
                # add label to the first row
                if add_label:
                    for j in range(5):
                        labelfigs(
                            ax[i][j],
                            number=j,
                            style=label_style,
                            loc="tl",
                            size=20,
                            inset_fraction=(0.1, 0.1),
                        )
        hf.close()
        # save figure
        if save_figure:
            plt.savefig(
                f"{self.folder_path}/{file_name}_show_affine_process_of_pickup_samples.svg"
            )

    def update_save_dict(self, **kwargs):
        save_dict = self.save_dict.copy()
        for key in kwargs:
            if key in self.save_dict:
                save_dict[key] = kwargs[key]
        return save_dict

    def predict(
        self,
        sample_index=None,
        train_process="1",
        file_name="",
        num_workers=0,
        **kwargs,
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
        if isinstance(file_name, float) or isinstance(file_name, int):
            file_name = format(int(file_name * 100), "02d") + "Per"
        file_name += f"_{train_process}_train_process"

        self.save_arrays(file_name, **kwargs)

    def save_predict(
        self,
        train_process="1",
        file_name="",
        **kwargs,
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
        if isinstance(file_name, float) or isinstance(file_name, int):
            file_name = format(int(file_name * 100), "02d") + "Per"
        file_name += f"_{train_process}_train_process"

        self.save_arrays(file_name, **kwargs)

    def save_arrays(self, file_name, **kwargs):
        update_save_dict = self.update_save_dict(**kwargs)

        save_dict = {
            "save_strain": {
                "save": update_save_dict["save_strain"],
                "name": "scale_shear",
                "array": self.strain_matrix,
            },
            "save_rotation": {
                "save": update_save_dict["save_rotation"],
                "name": "rotation",
                "array": self.rotation_matrix,
            },
            "save_translation": {
                "save": update_save_dict["save_translation"],
                "name": "translation",
                "array": self.translation_matrix,
            },
            "save_classification": {
                "save": update_save_dict["save_classification"],
                "name": "classification",
                "array": self.classification_matrix,
            },
            "save_base": {
                "save": update_save_dict["save_base"],
                "name": "generated_base",
                "array": self.generated_base,
            },
        }

        for key, value in save_dict.items():
            if value["save"]:
                np.save(
                    f"{self.folder_path}/{file_name}_{value['name']}.npy",
                    value["array"],
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
        if self.adaptive_mask_loss_flag:
            self.interpolate = True
        # initial check mask if not pre-defined
        if not self.initial_mask:
            self.initial_mask = self.dynamic_mask_to_loss_function
        # learning rate for training
        learning_rate = round(self.learning_rate, 6)

        # minimum learning rate
        min_rate = round(self.learning_rate, 6)
        # maximum learning rate
        max_rate = round(self.scheduler_max_learning_rate, 6)
        # coefficient of l norm regularization
        reg_coef = round(self.regularization_coef, 9)
        # coefficient of scale regularization
        scale_coef = round(self.scale_regularization_coef, 2)
        # coefficient of shear regularization
        shear_coef = round(self.shear_regularization_coef, 2)

        # initialize coefficient to record lr decay condition
        patience = 0

        # initialize model
        encoder, decoder, join, optimizer = self.initialize_model()

        # set lr scheduler if set_scheduler is True
        if self.learning_rate_scheduler_flag:
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=min_rate,
                max_lr=max_rate,
                step_size_up=self.epochs_per_learning_rate_half_cycle,
                cycle_momentum=False,
            )
            # if set_scheduler is True, turn off lr_decay and lr_circle mode
            self.learning_rate_decay_flag = False
            self.adaptive_learning_rate_circle_cycler_flag = False
        else:
            lr_scheduler = None

        # dynamic_mask_region is True, means in second training process, the dateset is [image, rotation]
        # dynamic_mask_region is False, means in first training process, the dateset is [image, None]
        if self.adaptive_mask_loss_flag:
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
                if epoch > self.epoch_start_mask_updates and epoch <= self.epoch_end_mask_updates:
                    encoder, decoder, join, optimizer = self.initialize_model()
                    if self.device == torch.device("cpu"):
                        check_ccc = torch.load(file_path, map_location=self.device)
                    else:
                        check_ccc = torch.load(file_path)

                    join.load_state_dict(check_ccc["net"])
                    encoder.load_state_dict(check_ccc["encoder"])
                    decoder.load_state_dict(check_ccc["decoder"])
                    optimizer.load_state_dict(check_ccc["optimizer"])

            # update learning rate if lr_decay is True
            if self.learning_rate_decay_flag:
                optimizer.param_groups[0]["lr"] = learning_rate
            # update learning rate if lr_circle is True
            elif self.adaptive_learning_rate_circle_cycler_flag:
                optimizer.param_groups[0]["lr"] = self.lr_circular(
                    epoch,
                    step_size_up=self.epochs_per_learning_rate_half_cycle,
                    min_rate=min_rate,
                    max_rate=max_rate,
                )
            # update loss class if dynamic_mask_region is True
            if self.adaptive_mask_loss_flag:
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
                    self.dynamic_mask_to_loss_function,
                    name_of_file,
                    self.device,
                    self.interpolate,
                )
                # update mask list according to generated base in particular epoch period
                if epoch >= self.epoch_start_mask_updates and epoch < self.epoch_end_mask_updates:
                    center_mask_list, rotate_center = inverse_base(
                        name_of_file,
                        self.initial_mask,
                        radius=self.diffraction_spot_mask_radius,
                    )
                    self.dynamic_mask_to_loss_function = center_mask_list

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
            cust_form = format(self.large_batch_training_param, "0d")
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
            if self.save_all_weights:
                torch.save(checkpoint, file_path)

                # update learning rate
                if self.learning_rate_decay_flag:
                    if epoch >= self.epoch_delay_saving:
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
                if epoch >= self.epoch_delay_saving:
                    if best_train_loss > train_loss:
                        best_train_loss = train_loss
                        
                        # TODO: We Might be able to remove this.
                        # save model weights after epoch_start_save
                        if epoch >= self.epoch_start_save:
                            torch.save(checkpoint, file_path)
                        # update learning rate according to lr_decay
                        if self.learning_rate_decay_flag:
                            patience = 0
                            learning_rate = 1.2 * learning_rate

                    else:
                        if self.learning_rate_decay_flag:
                            patience += 1

                            if patience > 0:
                                learning_rate = learning_rate * 0.8
            # update learning rate according to lr_scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()

