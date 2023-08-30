import torch
import random
from torch.utils.data import DataLoader
from ..Data.DataProcess import*
import numpy as np
from .CC_ST_AE import*
from .Loss_Function import*

class Train_Class:
    

    def __init__(self,
                 data_dir,
                 device=torch.device('cpu'),
                 seed=42,
                 crop = ((28,228),(28,228)),
                 background_weight = 0.2,
                 learned_rotation = None,
                 background_intensity = True,
                 reg_coef = 1,
                 radius = 45,
                 learning_rate = 3e-5,
                 en_original_step_size = [200,200],
                 de_original_step_size = [5,5],
                 pool_list= [5,4,2],
                 up_list = [2,4,5],
                 conv_size = 128,
                 scale = True,
                 shear = True,
                 rotation = True,
                 rotate_clockwise = True,
                 translation = False,
                 Symmetric = True,
                 mask_intensity = True,
                 num_base=1,
                 up_size=800,
                 scale_limit=0.05,
                 scale_penalty=0.03,
                 shear_limit=0.1,
                 shear_penalty=0.03,
                 rotation_limit=0.1,
                 trans_limit = 0.15,
                 adj_mask_para=0,
                 crop_radius = 60,
                 sub_avg_coef = 1.5,
                 reduced_size=20,
                 interpolate_mode = 'bicubic',
                 affine_mode = 'bicubic',
                 num_mask = 1,
                 fixed_mask = None,
                 check_mask = None,
                 interpolate = True,
                 soft_threshold = 1.5,
                 hard_threshold = 3,
                 con_div=15
                 ):
        """class of the training process, including load and preprocess the dataset and initialize loss class.

        Args:
            data_dir (string): directory of the dataset
            device (torch.device): set the device to run the model. Defaults to torch.device('cpu')
            seed (int): set the seed to make the training reproducible. Defaults to 42.
            crop (tuple, optional): the range of index for image cropping. Defaults to ((28,228),(28,228)).
            background_weight (float, optional): set the intensity of background noise for simulated dataset. Defaults to 0.2.
            learned_rotation (numpy array, optional): The numpy array represents pretrained rotation value if exists. Defaults to None.
            background_intensity (bool, optional): determine if the input dataset is simulated data or not. Defaults to True.
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
            soft_threshold (float): set the value of threshold where using MAE replace MSE. Defaults to 1.5.
            hard_threshold (float): set the value of threshold where using hard threshold replace MAE. Defaults to 3.
            con_div (int): set the value of parameter divided by loss value. Defaults to 15.
        """
        self.seed = seed
        self.device = device
        self.crop = crop
        self.data_dir = data_dir
        self.background_weight = background_weight
        self.background_intensity = background_intensity
        self.learned_rotation = learned_rotation
        self.reg_coef = reg_coef
        self.radius = radius
        self.reset_dataset()
        
        # Set the parameter for the model structure
        self.learning_rate = learning_rate
        self.en_original_step_size = en_original_step_size
        self.de_original_step_size = de_original_step_size
        self.pool_list= pool_list
        self.up_list = up_list
        self.conv_size = conv_size
        self.scale = scale
        self.shear = shear
        self.rotation = rotation
        self.rotate_clockwise = rotate_clockwise
        self.translation = translation
        self.Symmetric = Symmetric
        self.mask_intensity = mask_intensity
        self.num_base=num_base
        self.up_size=up_size
        self.scale_limit=scale_limit
        self.scale_penalty=scale_penalty
        self.shear_penalty=shear_penalty
        self.shear_limit=shear_limit
        self.rotation_limit=rotation_limit
        self.trans_limit =trans_limit
        self.adj_mask_para=adj_mask_para
        self.crop_radius = crop_radius
        self.sub_avg_coef = sub_avg_coef
        self.reduced_size=reduced_size
        self.interpolate_mode = interpolate_mode
        self.affine_mode = affine_mode
        self.num_mask = num_mask
        self.fixed_mask = fixed_mask
        self.interpolate = interpolate
        self.check_mask = check_mask
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.con_div = con_div
        
        
    def reset_dataset(self):
        """function for generating dataset
        """
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        self.data_class = STEM4D_DataSet(self.data_dir,self.background_weight,
                                         background_intensity=self.background_intensity,rotation = self.learned_rotation)
        self.data_set = self.data_class.stem4d_data
        
        
        
        if self.learned_rotation is not None:
            
            self.rotate_data = self.data_class.stem4d_rotation
            
    
    def lr_circular(self,
                    epoch,
                    step_size_up = 20,
                    min_rate = 3e-5,
                    max_rate = 2e-4,
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
        
        lr_change_size = (max_rate-min_rate)/step_size_up
        
        num = epoch%step_size_up
        para = int(epoch/step_size_up)
        
        if para%2==0:
            lr = min_rate + num*lr_change_size
            
        else:
            lr = max_rate - num*lr_change_size
        
        return lr
    
    

    
    def reset_model(self):
        
        encoder, decoder, join, optimizer = make_model_fn(self.device,
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
                                                        )
        
        return encoder, decoder, join, optimizer
    
    
    def reset_loss_class(self):
        """function used for initializing loss class

        Returns:
            Class(Object): loss class
        """
        
        loss_fuc = AcumulatedLoss(
                         self.device,
                         reg_coef = self.reg_coef,
                         scale_coef = self.scale_coef,
                         shear_coef = self.shear_coef,
                         norm_order = self.reg_coef,
                         scale_penalty=self.scale_penalty,
                         shear_penalty=self.shear_penalty,
                         mask_list = self.fixed_mask,
                         batch_para = self.batch_para,
                         loss_type = self.loss_type,
                         weight_coef = self.weight_coef,
                         upgrid_img = self.interpolate,
                         soft_threshold = self.soft_threshold,
                         hard_threshold = self.hard_threshold,
                         con_div=self.con_div
                         )
        
        return loss_fuc
    
    def load_pretrained_weight(self,
                               weight_path):
        """function used to load pretrained weight to neural network

        Args:
            weight_path (string): dictionary of pretrained weight

        Returns:
            torch.Module: pytorch model with pretrained weight loaded
        """
        
        encoder, decoder, join, optimizer = self.reset_model()

        check_ccc = torch.load(weight_path) 

        join.load_state_dict(check_ccc['net'])
        encoder.load_state_dict(check_ccc['encoder'])
        decoder.load_state_dict(check_ccc['decoder'])
        optimizer.load_state_dict(check_ccc['optimizer'])
        
        return join, encoder, decoder, optimizer
    
            
        
    def train_process(self,
                    learning_rate = 3e-5,
                    max_rate = 2e-4,
                    reg_coef = 1e-6,
                    scale_coef = 10,
                    shear_coef = 1,
                    batch_para = 1,
                    step_size_up = 20,
                    set_scheduler = False,
                    loss_type = 'custom',
                    weight_coef = 5,
                    lr_decay = True,
                    lr_circle = False,
                    batch_size = 4,
                    epochs=20,
                    epoch_start_compare = 0,
                    epoch_start_save = 0,
                    folder_path = '',
                    save_every_weights = True,
                    dynamic_mask_region=True,
                    cycle_consistent = True,
                    
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
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        make_folder(folder_path)


        learning_rate = int(learning_rate*1e6)/1e6
        min_rate = int(learning_rate*1e6)/1e6
        max_rate = int(max_rate*1e6)/1e6
        reg_coef = int(reg_coef*1e9)/1e9
        scale_coef = int(scale_coef*1e2)/1e2
        shear_coef = int(shear_coef*1e2)/1e2
        
        self.reg_coef = reg_coef
        self.scale_coef = scale_coef
        self.shear_coef = shear_coef
        self.batch_para = batch_para
        self.loss_type = loss_type
        self.weight_coef = weight_coef
        patience = 0
        
        encoder, decoder, join, optimizer = self.reset_model()
        
            
        if set_scheduler:
        
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_rate, max_lr=max_rate,
                                                  step_size_up=step_size_up,cycle_momentum=False)
            
            lr_decay = False
            lr_circle = False
        else: 
            
            lr_scheduler = None
        
        if dynamic_mask_region:
            
            train_iterator = DataLoader(self.rotate_data, batch_size=batch_size, shuffle=True, num_workers=0)
        
            test_iterator = DataLoader(self.rotate_data, batch_size=batch_size, shuffle=False, num_workers=0)
            
        else:
            train_iterator = DataLoader(self.data_set, batch_size=batch_size, shuffle=True, num_workers=0)

            test_iterator = DataLoader(self.data_set, batch_size=batch_size, shuffle=False, num_workers=0)

        N_EPOCHS = epochs

        loss_class = self.reset_loss_class()
        
        best_train_loss = float('inf')


        for epoch in range(N_EPOCHS):
    #    This loss function include the entropy loss with increasing coefficient value
    
            if self.interpolate:
                if epoch >0 and epoch<2:
                    
 #                   self.learning_rate = learning_rate

                    encoder, decoder, join, optimizer = self.reset_model()

                    check_ccc = torch.load(file_path) 

                    join.load_state_dict(check_ccc['net'])
                    encoder.load_state_dict(check_ccc['encoder'])
                    decoder.load_state_dict(check_ccc['decoder'])
                    optimizer.load_state_dict(check_ccc['optimizer'])
                    

            if lr_decay:

                optimizer.param_groups[0]['lr'] = learning_rate   

            elif lr_circle:

                optimizer.param_groups[0]['lr'] = self.lr_circular(epoch,
                                                                   step_size_up = step_size_up,
                                                                   min_rate = min_rate,
                                                                   max_rate = max_rate,
                                                                   )


            if dynamic_mask_region:
                
                loss_class = self.reset_loss_class()
                
                
                

            loss_dictionary = loss_class.__call__(join,
                                                  train_iterator,
                                                  optimizer,
                                                  batch_para=self.batch_para,
                                                  cycle_consistent = cycle_consistent,
                                                  upgrid_img = self.interpolate,
                                                  dynamic_mask_region = dynamic_mask_region)
                

            train_loss = loss_dictionary['train_loss']
            L2_loss = loss_dictionary['l2_loss']
            Scale_Loss = loss_dictionary['scale_loss']
            Shear_Loss = loss_dictionary['shear_loss']
            
            if self.interpolate:
                
                name_of_file = folder_path + f'/L1:{reg_coef:.10f}_scale:{scale_coef:.3f}_shear:{shear_coef:.3f}_lr:{learning_rate:.6f}_Epoch:{epoch:04d}_trainloss:{train_loss:.6f}_'
                Show_Process(join,
                             test_iterator,
                             self.fixed_mask,
                             name_of_file,
                             self.device,
                             self.interpolate)
                
                if epoch==0:
    
                    center_mask_list, rotate_center = \
                    inverse_base(name_of_file, self.check_mask, radius = self.radius)
                    self.fixed_mask = center_mask_list

                

    #        VAE_L /= len(train_iterator)
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')
    #        print(f'......... VAE Loss: {VAE_L:.4f}')
            print('.............................')
        
            checkpoint = {
                    "net": join.state_dict(),
                    "encoder":encoder.state_dict(),
                    "decoder":decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    'mse_loss': train_loss,
                }
            lr_ = format(optimizer.param_groups[0]['lr'],'.6f')
            scale_form = format(scale_coef,'.4f')
            shear_form = format(shear_coef,'.4f')
            cust_form = format(batch_para,'0d')
            file_path = folder_path+'/Weight_lr:'+lr_+'_scale_cof:'+scale_form+'_shear_cof:'+shear_form+'_custom_para:'+cust_form+\
                       f'_epoch:{epoch:04d}_trainloss:{train_loss:.5f}_l1:{L2_loss:.5f}_scal:{Scale_Loss:.5f}_shr:{Shear_Loss:.5f}.pkl'
            
            if save_every_weights:
                
                torch.save(checkpoint, file_path)
                
                if lr_decay:
                    if epoch>=epoch_start_compare:

                        if best_train_loss > train_loss:
                            best_train_loss = train_loss

                            patience = 0

                            learning_rate = 1.2 * learning_rate

                        else:

                            patience +=1

                            if patience >0:
                                learning_rate = learning_rate*0.8
     
            else:     
                    
                if epoch>=epoch_start_compare:

                    if best_train_loss > train_loss:
                        best_train_loss = train_loss
                        
                        if epoch>=epoch_start_save:
                            
                            torch.save(checkpoint, file_path)
                        
                        if lr_decay:
                            
                            patience = 0
                            learning_rate = 1.2 * learning_rate

                    else:
                        if lr_decay:
                            
                            patience +=1

                            if patience >0:
                                learning_rate = learning_rate*0.8

            if lr_scheduler!= None:
                lr_scheduler.step()