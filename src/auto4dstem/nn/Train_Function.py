import torch
import os
import random
from torch.utils.data import DataLoader
from typing import Optional
from ..Data.DataProcess import STEM4D_DataSet
from ..Viz.util import make_folder, inverse_base, Show_Process
import numpy as np
from .CC_ST_AE import make_model_fn
from .Loss_Function import AcumulatedLoss
from dataclasses import dataclass, field


@dataclass
class TrainClass:
    data_dir: str
    device: torch.device = torch.device("cpu")
    seed: int = 42
    crop: tuple = ((28, 228), (28, 228))
    transpose: tuple = (2,3,0,1)
    background_weight: float = 0.2
    learned_rotation: any = None  # Specify the data type as required
    background_intensity: bool = True
    standard_scale: Optional[float] = None
    up_threshold: float = 1000
    down_threshold: float = 0
    reg_coef: float = 1
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
    revise_affine: bool=True
    soft_threshold: float = 1.5
    hard_threshold: float = 3
    con_div: int = 15

    """class of the training process, including load and preprocess the dataset and initialize loss class.

    Args:
        data_dir (string): directory of the dataset
        device (torch.device): set the device to run the model. Defaults to torch.device('cpu')
        seed (int): set the seed to make the training reproducible. Defaults to 42.
        crop (tuple, optional): the range of index for image cropping. Defaults to ((28,228),(28,228)).
        background_weight (float, optional): set the intensity of background noise for simulated dataset. Defaults to 0.2.
        learned_rotation (numpy array, optional): The numpy array represents pretrained rotation value if exists. Defaults to None.
        background_intensity (bool): determine if the input dataset is simulated data or not. Defaults to True.
        standard_scale (float, optional): determine if the input dataset needs standard scale or not, the value can determine the scale in data processing. Defaults to None.
        up_threshold (float): determine the value of up threshold of dataset. Defaults to 1000.
        down_threshold (float): determine the value of down threshold of dataset. Default to 0.
        reg_coef (float): set the value of parameter multiplied by l norm. Defaults to 1.
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
        interpolate (bool, optional): turn upgrid version when inserting images into loss function. Defaults to True.
        revise_affine (bool): set to determine if need to add revise affine to image with affine transformation. Default to True.
        soft_threshold (float): set the value of threshold where using MAE replace MSE. Defaults to 1.5.
        hard_threshold (float): set the value of threshold where using hard threshold replace MAE. Defaults to 3.
        con_div (int): set the value of parameter divided by loss value. Defaults to 15.
    """

    def __post_init__(self):
        """replace __init__(), load dataset for initialization 
        """
        self.reset_dataset()

    def reset_dataset(self):
        """function for generating dataset"""
        
        # fix seed to reproduce results
        os.environ['PYTHONHASHSEED'] = str(self.seed)
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
            crop = self.crop,
            transpose = self.transpose,
            background_intensity=self.background_intensity,
            rotation=self.learned_rotation,
            standard_scale = self.standard_scale,
            up_threshold = self.up_threshold,
            down_threshold = self.down_threshold   
        )
        
        # return the stem dataset 
        self.data_set = self.data_class.stem4d_data

        # pair each stem image with pretrained rotation
        if self.learned_rotation is not None:
            self.rotate_data = self.data_class.stem4d_rotation

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
        """function used for initializing loss class

        Returns:
            Class(Object): loss class
        """

        loss_fuc = AcumulatedLoss(
            self.device,
            reg_coef=self.reg_coef,
            scale_coef=self.scale_coef,
            shear_coef=self.shear_coef,
            norm_order=self.reg_coef,
            scale_penalty=self.scale_penalty,
            shear_penalty=self.shear_penalty,
            mask_list=self.fixed_mask,
            batch_para=self.batch_para,
            weighted_mse=self.loss_type,
            weight_coef=self.weight_coef,
            upgrid_img=self.interpolate,
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
        check_ccc = torch.load(weight_path)

        # load the pretrained weight to model
        join.load_state_dict(check_ccc["net"])
        encoder.load_state_dict(check_ccc["encoder"])
        decoder.load_state_dict(check_ccc["decoder"])
        optimizer.load_state_dict(check_ccc["optimizer"])

        return join, encoder, decoder, optimizer

    def train_process(
        self,
        learning_rate=3e-5,
        max_rate=2e-4,
        reg_coef=1e-6,
        scale_coef=10,
        shear_coef=1,
        batch_para=1,
        step_size_up=20,
        set_scheduler=False,
        loss_type="custom",
        weight_coef=5,
        lr_decay=True,
        lr_circle=False,
        batch_size=4,
        epochs=20,
        epoch_start_compare=0,
        epoch_start_save=0,
        epoch_start_update = 0,
        epoch_end_update = 100,
        folder_path="",
        save_every_weights=True,
        dynamic_mask_region=True,
        cycle_consistent=True,
    ):
        """function call the train process for model training

        Args:
            learning_rate (float): initial learning rate set to training process. Defaults to 3e-5.
            max_rate (float): maximum learning rate in the training cycle. Defaults to 2e-4.
            reg_coef (float): coefficient of l norm regularization. Defaults to 1e-6.
            scale_coef (float): coefficient of scale regularization. Defaults to 10.
            shear_coef (float): coefficient of shear regularization. Defaults to 1.
            batch_para (int):  set the value of parameter multiplied by batch size. Defaults to 1.
            step_size_up (int): the step size of half cycle. Defaults to 20.
            set_scheduler (bool): determine whether using torch.optim.lr_scheduler.CyclicLR function generate learning rate. Defaults to False.
            loss_type (str):  set the type of the loss function ('custom' means weighted MSE, any else means MSE). Defaults to 'custom'.
            weight_coef (int):set the value of weight when using weighted MSE as loss function. Defaults to 2.
            lr_decay (bool): determine whether using learning rate decay after each epoch training. Defaults to True.
            lr_circle (bool): determine whether using lr_circular() function generate learning rate after each epoch. Defaults to False.
            batch_size (int): minibatch value. Defaults to 4.
            epochs (int): determine the number of training epochs. Defaults to 20.
            epoch_start_compare (int): index of epoch to record and save training loss. Defaults to 0.
            epoch_start_save (int): index of epoch to start save pretrained weights. Defaults to 0.
            folder_path (str): folder dictionary to save pretrained weights. Defaults to ''.
            save_every_weights (bool): determine whether to save every pretrained weights. Defaults to True.
            dynamic_mask_region (bool): determine whether use dynamic mask list when computing loss. Defaults to True.
            cycle_consistent (bool): determine whether computing loss cycle consistently. Defaults to True.
        """
        # fix seed of the model
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            
         # create folder directory to save weight
        make_folder(folder_path)
        
        # learning rate for training
        learning_rate = round(learning_rate, 6)
        
        # minimum learning rate
        min_rate = round(learning_rate, 6)
        # maximum learning rate
        max_rate = round(max_rate, 6)
        # coefficient of l norm regularization
        reg_coef = round(reg_coef, 9)
        # coefficient of scale regularization
        scale_coef = round(scale_coef, 2)
        # coefficient of shear regularization
        shear_coef = round(shear_coef, 2)

        self.reg_coef = reg_coef
        self.scale_coef = scale_coef
        self.shear_coef = shear_coef
        self.batch_para = batch_para
        self.loss_type = loss_type
        self.weight_coef = weight_coef
        
        # initialize coefficient to record lr decay condition
        patience = 0

        # initialize model 
        encoder, decoder, join, optimizer = self.reset_model()

        # set lr scheduler if set_scheduler is True
        if set_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=min_rate,
                max_lr=max_rate,
                step_size_up=step_size_up,
                cycle_momentum=False,
            )
        # if set_scheduler is True, turn off lr_decay and lr_circle mode
            lr_decay = False
            lr_circle = False
        else:
            lr_scheduler = None

        # dynamic_mask_region is True, means in second training process, the dateset is [image, rotation]
        # dynamic_mask_region is False, means in first training process, the dateset is [image, None]
        if dynamic_mask_region:
            train_iterator = DataLoader(
                self.rotate_data, batch_size=batch_size, shuffle=True, num_workers=0
            )

            test_iterator = DataLoader(
                self.rotate_data, batch_size=batch_size, shuffle=False, num_workers=0
            )

        else:
            train_iterator = DataLoader(
                self.data_set, batch_size=batch_size, shuffle=True, num_workers=0
            )

            test_iterator = DataLoader(
                self.data_set, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # set total epoch to train
        N_EPOCHS = epochs

        # initialize loss class
        loss_class = self.reset_loss_class()
        
        # initialize best loss to infinite
        best_train_loss = float("inf")

        for epoch in range(N_EPOCHS):
        # load pretrained weight result of previous epoch training 
            if self.interpolate:
                # set the range of epoch for updating (potentially learning rate and mask region)
                if epoch > epoch_start_update and epoch <= epoch_end_update:
                    encoder, decoder, join, optimizer = self.reset_model()

                    check_ccc = torch.load(file_path)

                    join.load_state_dict(check_ccc["net"])
                    encoder.load_state_dict(check_ccc["encoder"])
                    decoder.load_state_dict(check_ccc["decoder"])
                    optimizer.load_state_dict(check_ccc["optimizer"])
            
            # update learning rate if lr_decay is True 
            if lr_decay:
                optimizer.param_groups[0]["lr"] = learning_rate
            # update learning rate if lr_circle is True
            elif lr_circle:
                optimizer.param_groups[0]["lr"] = self.lr_circular(
                    epoch,
                    step_size_up=step_size_up,
                    min_rate=min_rate,
                    max_rate=max_rate,
                )
            # update loss class if dynamic_mask_region is True
            if dynamic_mask_region:
                loss_class = self.reset_loss_class()

            # compute and return loss dictionary
            loss_dictionary = loss_class.__call__(
                join,
                train_iterator,
                optimizer,
                batch_para=self.batch_para,
                cycle_consistent=cycle_consistent,
                upgrid_img=self.interpolate,
                dynamic_mask_region=dynamic_mask_region,
            )
            # load loss value to save in weights' name
            train_loss = loss_dictionary["train_loss"]
            L2_loss = loss_dictionary["l2_loss"]
            Scale_Loss = loss_dictionary["scale_loss"]
            Shear_Loss = loss_dictionary["shear_loss"]

            # save mask list and generated base in each epoch
            if self.interpolate:
                name_of_file = (
                    folder_path
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
                if epoch >= epoch_start_update and epoch < epoch_end_update:
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
            cust_form = format(batch_para, "0d")
            file_path = (
                folder_path
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
            if save_every_weights:
                torch.save(checkpoint, file_path)
            
            # update learning rate
                if lr_decay:
                    if epoch >= epoch_start_compare:
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
                if epoch >= epoch_start_compare:
                    if best_train_loss > train_loss:
                        best_train_loss = train_loss

                        if epoch >= epoch_start_save:
                            torch.save(checkpoint, file_path)

                        if lr_decay:
                            patience = 0
                            learning_rate = 1.2 * learning_rate

                    else:
                        if lr_decay:
                            patience += 1

                            if patience > 0:
                                learning_rate = learning_rate * 0.8
            # update learning rate according to lr_scheduler
            if lr_scheduler != None:
                lr_scheduler.step()
