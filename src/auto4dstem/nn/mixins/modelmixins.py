from dataclasses import dataclass, field

import torch
import numpy as np

@dataclass
class SaveWeightMixin:
    """Class for managing the saving of model weights during training.

    Attributes:
        epochs_delay_saving (int): Number of epochs to delay saving the model -- early epochs are not important to save. Defaults to 0.
        epoch_start_save (int): Index of the epoch to start saving pretrained weights. Defaults to 0.
        epoch_start_update (int): Index of the epoch to start updating the dynamic mask. Defaults to 0.
        epoch_end_update (int): Index of the epoch to stop updating the dynamic mask. Defaults to 100.
        folder_path (str): Directory path to save the pretrained weights. Defaults to "save_weight".
        save_every_weights (bool): Flag to determine whether to save weights at every epoch regardless if the loss is better or not. Defaults to True.
        save_dict (dict): Dictionary of flags to determine which transformation matrix to save. Defaults to {}.
    """

    epoch_delay_saving: int = 0

    # TODO: revisit - we might 
    epoch_start_save: int = 0


    epoch_start_mask_updates: int = 0
    epoch_end_mask_updates: int = 100
    folder_path: str = "save_weight"
    save_all_weights: bool = True

    save_dict: dict = field(default_factory=lambda: {
        "save_strain": False,
        "save_rotation": False,
        "save_translation": False,
        "save_classification": False,
        "save_base": False,
    })


@dataclass
class LearnableAffineTransformMixin:
    """class of the LearnableAffineTransformMixin process, including set the learnable affine transform parameters.

    Attributes:
        scale (bool): set to True if the model include scale affine transform
        shear (bool): set to True if the model include shear affine transform
        rotation (bool): set to True if the model include rotation affine transform
        rotate_clockwise (bool): set to True if the image is restricted to be rotated along one direction, making it unique[]
        translation (bool): set to True if the model include translation affine transform
        symmetric (bool): set to True if the shear affine transform is symmetric
        scale_threshold (float): set the threshold for scale. Defaults to 0.05.
        shear_threshold (float): set the threshold for shear. Defaults to 0.1.
        rotation_threshold (float): set the threshold for rotation. Defaults to 0.1.
        trans_threshold (float): set the threshold for translation. Defaults to 0.15.
        scale_regularizer (float): set the regularizer for scale. Defaults to 0.03.
        shear_regularizer (float): set the regularizer for shear. Defaults to 0.03.
        scale_regularization_coef (float): set the coefficient for scale regularization. Defaults to 10.
        shear_regularization_coef (float): set the coefficient for shear regularization. Defaults to 1.
    """

    scale: bool = True
    shear: bool = True
    rotation: bool = True
    rotate_clockwise: bool = True
    translation: bool = False
    symmetric: bool = True

    # bounds
    scale_threshold: float = 0.05
    shear_threshold: float = 0.1
    rotation_threshold: float = 0.1
    translation_threshold: float = 0.15

    # regularizer
    scale_regularizer: float = 0.03
    shear_regularizer: float = 0.03
    scale_regularization_coef: float = 10
    shear_regularization_coef: float = 1


@dataclass
class ModelHyperParameterMixin:
    """class of the ModelHyperParameterMixin process, including set the model hyper-parameters.

    Attributes:
        encoder_input_dimensions (list of integer): list of input image size to encoder. Defaults to [200,200].
        decoder_input_dimensions (list of integer, optional): list of image size to decoder before reconstruction. Defaults to [5,5].
        pool_list (list of int): the list of parameter for each 2D MaxPool layer. Defaults to [5,4,2].
        upsample_list (list of int): the list of parameter for each 2D Upsample layer. Defaults to [2,4,5].
        num_conv_filters (int): the number of filters number goes to each block. Defaults to 128.
        num_base (int): the number of base. This is the number of crystal structure to learn. Defaults to 1.
        upsample_dimensions (int): the size of image for upsampling for calculating MSE loss. Defaults to 800.
        embedding_size (int): the size of embedding for the K-top layer. Defaults to 20.
        adaptive_mask_loss_flag (bool): determine whether using adaptive mask loss. Defaults to True.
        cycle_consistent_flag (bool): Flag to train with just the cycle consistent loss. This is a benefit when training the dataset with significant amount of noise. Defaults to True.
    """

    input_image_dim: list = field(default_factory=lambda: [200, 200])
    decoder_input_dimensions: list = field(default_factory=lambda: [5, 5])
    pool_list: list = field(default_factory=lambda: [5, 4, 2])
    upsample_list: list = field(init=False)
    number_channels: int = 128
    num_base: int = 1
    upsample_dimensions: int = 800
    embedding_size: int = 20

    # transformation flags
    interpolate: bool = True
    reverse_affine: bool = True

    # dynamic mask region
    adaptive_mask_loss_flag: bool = True
    cycle_consistent_flag: bool = True
    
    def __post_init__(self):
        if hasattr(self, 'upsample_list') and self.upsample_list is not None:
            self.upsample_list = list(reversed(self.pool_list))


@dataclass
class TrainingHyperParameterMixin:
    """class of the TrainingHyperParameterMixin process, including set the hyper-parameters.

    Attributes:
        batch_size (int): set the batch size for training. Defaults to 4.
        epochs (int): set the number of epochs for training. Defaults to 20.
        learning_rate_scheduler_flag (bool): determine whether using torch.optim.lr_scheduler.CyclicLR function generate learning rate. Defaults to False.
        learning_rate (float): set the learning rate for ADAM optimization. Defaults to 3e-5.
        scheduler_max_learning_rate (float): set the maximum learning rate for the learning rate scheduler. If maximum rate is set learning_rate is the minimum rateDefaults to 2e-4.
        epochs_per_learning_rate_half_cycle (int): number of epochs for the learning rate scheduler. Defined as the number to go from minimum to maximum learning rate, or maximum to minimum learning rate. Defaults to 20.
        learning_rate_decay_flag (bool): determine whether using learning rate decay.  This is used to decay the learning rate during the training process. Defaults to True.
        adaptive_learning_rate_circle_cycler_flag (bool): determine whether using adaptive learning rate circle cycler. This is used to adjust the learning rate during the training process and is based on the current loss.. Defaults to False.
        soft_loss_threshold (float): set the value of threshold where using MAE replace MSE. Defaults to 1.5.
        hard_loss_threshold (float): set the value of threshold where using hard threshold replace MAE. Defaults to 3.
        noise_loss_scaling_factor (int): set the value of parameter divided by loss value this is based on the background noise. This is used to reduce the loss value when the background noise is high. Defaults to 15.
        large_batch_training_param (int): mini-batch adjustment parameter for training larger than memory batch sizes.Defaults to 1.
        weighted_mse_flag (bool): determine whether using weighted MSE in loss function. Defaults to True.
        weighted_mse_coef (int): set the value of weight when using weighted MSE as loss function. Defaults to 5.
        mse_difference_sign_preference_flag (bool): select positive of negative difference between the predicted and target images, this makes a difference because of the thresholding. Defaults to True.
    """

    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 3e-5

    # Learning rate scheduler, if maximum rate is set learning_rate is the minimum rate
    learning_rate_scheduler_flag: bool = False
    scheduler_max_learning_rate: float = 2e-4
    epochs_per_learning_rate_half_cycle: int = 20

    # Learning rate schedulers
    learning_rate_decay_flag: bool = True
    adaptive_learning_rate_circle_cycler_flag: bool = False

    # bounds for transitions between loss functions
    soft_loss_threshold: float = 1.5
    hard_loss_threshold: float = 3

    # loss scaling factor based on background noise
    noise_loss_scaling_factor: int = 15

    large_batch_training_param: int = 1

    weighted_mse_flag: bool = True
    weighted_mse_coef: int = 5
    mse_difference_sign_preference_flag: bool = True


@dataclass
class MaskMixin:
    """class of the MaskMixin process, including set the mask parameters.

    Attributes:
        diffraction_spot_mask_radius (int): sets the radius of the mask used for the diffraction spots, in units of pixels. Defaults to 45.
        learnable_mask (bool): set to True if the mask is learnable. Defaults to True.
        learnable_mask_intensity (float): set the intensity of the learnable mask. Defaults to 0.
        reverse_affine_transform_crop_radius (int): set the radius of the crop used for the reverse affine transform. This region should be larger than the diffraction spot mask radius to avoid clipping. Defaults to 60.
        COM_threshold_coef (float): set the threshold for the center of mass operation. Defaults to 1.5.
        dynamic_mask_to_loss_function (list of tensor, optional): The list of tensor with binary type. This is a dynamic mask applied during the calculation of the loss function. Defaults to None.
        initial_mask (list of tensor, optional): The list of tensor with binary type used for mask initialization. Generally this is set to the initial value of dynamic_mask_to_loss_function. Defaults to None.
    """

    diffraction_spot_mask_radius: int = 45
    reverse_affine_transform_crop_radius: int = 60
    learnable_mask: bool = True
    learnable_mask_intensity: float = 0
    COM_threshold_coef: float = 1.5
    dynamic_mask_to_loss_function: list[torch.Tensor] | None = None
    initial_mask: list[torch.Tensor] | None = None


@dataclass
class RegularizationMixin:
    """class of the RegularizationMixin process, including set the regularization parameters.

    Attributes:
        norm_order (float): set the value of parameter multiplied by l norm. Defaults to 1. This would be l1 norm if norm_order is 1, l2 norm if norm_order is 2.
        regularization_coef (float): set the value of parameter multiplied by regularization. Defaults to 1e-6.
    """

    norm_order: float = 1
    regularization_coef: float = 1e-6


@dataclass
class FineTuningPreTrainMixin:
    """class of the PreTrainMixin process, including set the pretrained rotation parameters.

    Attributes:
        learned_rotation (numpy array / string, optional): The numpy array/ directory of rotation weights represents pretrained rotation value if exists. Defaults to None.
        coarse_learned_angle_adjustment (int): The rotation degree added to learned_rotation if exists. Defaults to 0.
    """

    learned_rotation: any = None
    coarse_learned_angle_adjustment: int = 0

@dataclass
class ModelMixin(FineTuningPreTrainMixin,
    RegularizationMixin,
    MaskMixin,
    TrainingHyperParameterMixin,
    ModelHyperParameterMixin,
    LearnableAffineTransformMixin,
    SaveWeightMixin):
    """Class for managing the model properties during training."""