import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from ..Viz.util import upsample_mask


class AcumulatedLoss(nn.Module):
    def __init__(
        self,
        device=torch.device("cpu"),
        reg_coef=0,
        scale_coef=0,
        shear_coef=0,
        norm_order=1,
        scale_penalty=0.04,
        shear_penalty=0.03,
        mask_list=None,
        batch_para=1,
        loss_type="custom",
        weight_coef=2,
        upgrid_img=False,
        soft_threshold=1.5,
        hard_threshold=3,
        con_div=15,
    ):
        """Class of the loss function

        Args:
            device (torch.device): set the device to run the model. Defaults to torch.device('cpu').
            reg_coef (float): set the value of parameter multiplied by l norm. Defaults to 0.
            scale_coef (float): set the value of parameter multiplied by scale regularization. Defaults to 0.
            shear_coef (float): set the value of parameter multiplied by shear regularization. Defaults to 0.
            norm_order (float): set the type of norm to compute. Defaults to 1.
            scale_penalty (float): set the scale limitation where to start adding regularization. Defaults to 0.04.
            shear_penalty (float): set the shear limitation where to start adding regularization. Defaults to 0.03.
            mask_list (list of tensor, optional): The list of tensor with binary type. Defaults to None.
            batch_para (int): set the value of parameter multiplied by batch size. Defaults to 1.
            loss_type (str): set the type of the loss function ('custom' means weighted MSE, any else means MSE). Defaults to 'custom'.
            weight_coef (int): set the value of weight when using weighted MSE as loss function. Defaults to 2.
            upgrid_img (bool): turn upgrid version when inserting images into loss function. Defaults to False.
            soft_threshold (float): set the value of threshold where using MAE replace MSE. Defaults to 1.5.
            hard_threshold (float): set the value of threshold where using hard threshold replace MAE. Defaults to 3.
            con_div (int): set the value of parameter divided by loss value. Defaults to 15.
        """

        super(AcumulatedLoss, self).__init__()

        self.device = device
        self.reg_coef = reg_coef
        self.scale_coef = scale_coef
        self.shear_coef = shear_coef
        self.norm_order = norm_order
        self.scale_penalty = scale_penalty
        self.shear_penalty = shear_penalty
        self.mask_list = mask_list
        self.loss_type = loss_type
        self.weight_coef = weight_coef
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.con_div = con_div

    def __call__(
        self,
        model,
        data_iterator,
        optimizer,
        batch_para=1,
        cycle_consistent=True,
        upgrid_img=False,
        dynamic_mask_region=False,
    ):
        """function used to run the single epoch training and return the loss value

        Args:
            model (torch. Module): the pytorch neural network model
            data_iterator (torch.utils.data.Dataloader): Input data in Dataloader format
            optimizer (torch.optim): optimizer of the model
            batch_para (int): set the value of parameter multiplied by batch size. Defaults to 1.
            cycle_consistent (bool): Turn the cycle consistent mode when computing loss value. Defaults to True.
            upgrid_img (bool): turn upgrid version when inserting images into loss function. Defaults to False.
            dynamic_mask_region (bool): determine which function to call when computing loss value. Defaults to False.

        Returns:
            dictionary: dictionary with different type of loss value.
        """

        self.cycle_consistent = cycle_consistent
        self.upgrid_img = upgrid_img
        self.batch_para = batch_para
        self.dynamic_mask_region = dynamic_mask_region

        train_loss = 0
        L2_loss = 0
        Scale_Loss = 0
        Shear_Loss = 0

        model.train()

        optimizer.zero_grad()

        NUM_ACCUMULATION_STEPS = self.batch_para
        # NUM_ACCUMULATION_STEPS = div # Full gradient

        # for x in tqdm(train_iterator, leave=True, total=len(train_iterator)):
        for batch_idx, x_value in enumerate(tqdm(data_iterator)):
            if type(x_value) != list:
                x = x_value.to(self.device, dtype=torch.float)
                y = None
            else:
                x, y = x_value
                x = x.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.float)

            if self.upgrid_img:
                (
                    predicted_x,
                    predicted_base,
                    predicted_input,
                    kout,
                    theta_1,
                    theta_2,
                    adj_mask,
                    new_list,
                    x_inp,
                ) = model(x, y)

                self.mask_list = upsample_mask(
                    self.mask_list, x.shape[-1], x_inp.shape[-1]
                )

            else:
                (
                    predicted_x,
                    predicted_base,
                    predicted_input,
                    kout,
                    theta_1,
                    theta_2,
                    adj_mask,
                    new_list,
                ) = model(x, y)

            l2_loss = (
                self.reg_coef
                * torch.norm(predicted_base.squeeze(), p=self.norm_order)
                / x.shape[0]
            )

            scale_loss = self.scale_coef * (
                torch.mean(F.relu(abs(theta_1[:, 0, 0] - 1) - self.scale_penalty))
                + torch.mean(F.relu(abs(theta_1[:, 1, 1] - 1) - self.scale_penalty))
            )

            shear_loss = self.shear_coef * torch.mean(
                F.relu(abs(theta_1[:, 0, 1]) - self.shear_penalty)
            )

            initial_loss = l2_loss + scale_loss + shear_loss

            if self.dynamic_mask_region:
                loss = self.dynamic_mask_list(
                    x_inp,
                    predicted_x,
                    predicted_base,
                    predicted_input,
                    new_list,
                    initial_loss,
                    con_div=self.con_div,
                )

            else:
                loss = self.fix_mask_list(
                    x, predicted_x, predicted_base, predicted_input, initial_loss
                )

            train_loss += loss.detach().cpu().numpy()
            L2_loss += l2_loss.detach().cpu().numpy()
            Scale_Loss += scale_loss.detach().cpu().numpy()
            Shear_Loss += shear_loss.detach().cpu().numpy()

            # Backward pass
            loss = loss / NUM_ACCUMULATION_STEPS
            loss.backward()

            # Update the weights
            if ((batch_idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (
                batch_idx + 1 == len(data_iterator)
            ):
                optimizer.step()
                optimizer.zero_grad()

        train_loss = train_loss / len(data_iterator)
        L2_loss = L2_loss / len(data_iterator)
        Scale_Loss = Scale_Loss / len(data_iterator)
        Shear_Loss = Shear_Loss / len(data_iterator)

        loss_dictionary = {
            "train_loss": train_loss,
            "l2_loss": L2_loss,
            "scale_loss": Scale_Loss,
            "shear_loss": Shear_Loss,
        }

        return loss_dictionary

    def weighted_difference_loss(self, x, y, n=2, reverse=True):
        """Adds a weight to the MSE loss based in the difference is positive or negative.

        Args:
            x (torch.tensor): torch.tensor format image
            y (torch.tensor): torch.tensor format image
            n (int): value of the weight. Defaults to 2.
            reverse (bool): determine adding weight on positive loss or negative loss. Defaults to True.
        """

        # switches the order of the arguments when calculating the difference
        if reverse:
            diff = x - y
        else:
            diff = y - x

        # determines the positions to weight
        index_pos = torch.where(diff > 0)
        index_neg = torch.where(diff < 0)

        value = (
            torch.sum(diff[index_pos] ** 2) + n * torch.sum(diff[index_neg] ** 2)
        ) / torch.numel(x)

        return value

    def fix_mask_list(
        self, x, predicted_x, predicted_base, predicted_input, initial_loss
    ):
        """function for computing loss with fixed mask region

        Args:
            x (torch.tensor): torch.tensor format image
            predicted_x (torch.tensor): torch.tensor format image
            predicted_base (torch.tensor): torch.tensor format image
            predicted_input (torch.tensor): torch.tensor format image
            initial_loss (float): loss value before going to the function

        Returns:
            float: loss value
        """

        loss = initial_loss + 0

        for i, mask in enumerate(self.mask_list):
            if self.loss_type == "custom":
                loss += self.weighted_difference_loss(
                    predicted_x.squeeze()[:, mask],
                    predicted_base.squeeze()[:, mask],
                    n=self.weight_coef,
                ) + self.weighted_difference_loss(
                    x.squeeze()[:, mask],
                    predicted_input.squeeze()[:, mask],
                    n=self.weight_coef,
                )
            else:
                loss += F.mse_loss(
                    predicted_base.squeeze()[:, mask],
                    predicted_x.squeeze()[:, mask],
                    reduction="mean",
                ) + F.mse_loss(
                    predicted_input.squeeze()[:, mask],
                    x.squeeze()[:, mask],
                    reduction="mean",
                )

        if loss > self.soft_threshold:
            loss = initial_loss + 0

            for i, mask in enumerate(self.mask_list):
                loss += F.l1_loss(
                    predicted_base.squeeze()[:, mask],
                    predicted_x.squeeze()[:, mask],
                    reduction="mean",
                ) + F.l1_loss(
                    predicted_input.squeeze()[:, mask],
                    x.squeeze()[:, mask],
                    reduction="mean",
                )

            loss -= 1

        if loss > self.hard_threshold:
            loss = initial_loss + self.hard_threshold

        return loss

    def dynamic_mask_list(
        self,
        x,
        predicted_x,
        predicted_base,
        predicted_input,
        new_list,
        initial_loss,
        con_div=15,
    ):
        """function for computing loss with dynamic mask region

        Args:
            x (torch.tensor): torch.tensor format image
            predicted_x (torch.tensor): torch.tensor format image
            predicted_base (torch.tensor): torch.tensor format image
            predicted_input (torch.tensor): torch.tensor format image
            new_list (list of tensor): The list of tensor with binary type
            initial_loss (float): loss value before going to the function
            con_div (int): set the value of parameter divided by loss value. Defaults to 15.

        Returns:
            float: loss value
        """

        loss = len(self.mask_list) * initial_loss

        for i, mask in enumerate(self.mask_list):
            if self.cycle_consistent:
                loss += F.mse_loss(
                    predicted_base.squeeze()[:, mask],
                    predicted_x.squeeze()[:, mask],
                    reduction="mean",
                )
            # set the loss for the generated input and input
            sub_loss = 0
            for k in range(x.shape[0]):
                sub_loss += F.mse_loss(
                    predicted_input[k].squeeze()[new_list[i][k]],
                    x[k].squeeze()[new_list[i][k]],
                    reduction="mean",
                )

            loss += sub_loss / x.shape[0]

        loss = loss / (len(self.mask_list) * con_div)

        if loss > self.soft_threshold:
            loss = len(self.mask_list) * initial_loss

            for i, mask in enumerate(self.mask_list):
                if self.cycle_consistent:
                    loss += F.l1_loss(
                        predicted_base.squeeze()[:, mask],
                        predicted_x.squeeze()[:, mask],
                        reduction="mean",
                    )

                # set the loss for the generated input and input
                sub_loss = 0

                for k in range(x.shape[0]):
                    sub_loss += F.l1_loss(
                        predicted_input[k].squeeze()[new_list[i][k]],
                        x[k].squeeze()[new_list[i][k]],
                        reduction="mean",
                    )

                loss += sub_loss / x.shape[0]

            loss = loss / (len(self.mask_list) * con_div)

            loss -= 1

        if loss > self.hard_threshold:
            loss = initial_loss + self.hard_threshold

        return loss
