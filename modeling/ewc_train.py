"""
Continual learning with Elastic Weight Consolidation (EWC).

The original attention architecture is not changed.

Training order:

    Stage 1 (2010-2012)
            ↓
    calculate Fisher information
            ↓
    Stage 2 (2013-2014) + EWC
            ↓
    calculate Fisher information
            ↓
    Stage 3 (2015-2016) + EWC
            ↓
    calculate Fisher information
            ↓
    Stage 4 (2017-2018) + EWC

Run this file from the modeling directory:

    cd modeling
    python ewc_train.py
"""


# ---------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------

import os
import random
import timeit
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn # for neural netwoek layers
import torch.optim as optim # for optimizers

import torchvision.transforms as transforms # for augumentations

from torch.utils.data import DataLoader, ConcatDataset

# Reuse the repository's existing model.
from attention_model import Attn_Net

# Reuse the repository's existing datasets.
from dataloader import (
    MyJP2Dataset,
    NFDataset,
    FLDataset,
    Balancer,
)

# Reuse the repository's existing TSS and HSS calculation.
from evaluation import sklearn_Compatible_preds_and_targets


# ---------------------------------------------------------------------
# 2. COMMAND-LINE ARGUMENTS
# ---------------------------------------------------------------------
# Creates a command-line argument reader for the training program.
parser = argparse.ArgumentParser(
    description="Full-disk attention model with EWC continual learning"
)

# Number of epochs used for every continual-learning stage.
# python ewc_train.py --epochs=20
parser.add_argument(
    "--epochs",
    type=int,
    default=30,
    help="Number of epochs for each stage",
)

# Number of images passed through the model together.
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Training batch size",
)

# Input image size expected by the existing model.
parser.add_argument(
    "--im_size",
    type=int,
    default=256,
    help="Input image size",
)

# Keep the same attention option as train.py.
# 1 means attention model; another value means standard CNN.
parser.add_argument(
    "--attention",
    type=int,
    default=1,
    help="Use 1 for attention model",
)

# Initial learning rate for each stage.
parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="Initial learning rate",
)

# Preserve the original repository's weight decay.
"""
Weight decay gently pushes model weights toward zero during training.
This helps prevent the model from memorizing training images and reduces overfitting.
"""
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.5,
    help="Optimizer weight decay",
)

# Controls how strongly EWC protects previous knowledge.
#
# Large value:
#     protects previous stages more strongly,
#     but makes learning the new stage harder.
#
# Small value:
#     allows easier learning of the new stage,
#     but may cause more forgetting.
parser.add_argument(
    "--ewc_lambda",
    type=float,
    default=1000.0,
    help="Strength of the EWC penalty",
)

# We do not necessarily need to use every old training image to estimate
# Fisher information. This limits the number of batches used.
#
# Use 0 if you want to use every available training batch.
parser.add_argument(
    "--fisher_batches",
    type=int,
    default=50,
    help="Batches used to estimate Fisher; use 0 for all batches",
)

# how many separate CPU processes load and prepare images simultaneously.
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="Number of DataLoader workers",
)

# Random seed for reproducibility.
"""
Using the same seed helps produce the same:
- Data shuffling
- Random rotations
- Initial model weights
"""
parser.add_argument(
    "--seed",
    type=int,
    default=4,
    help="Random seed",
)
# Reads the command-line options and stores them in opt
"""
python ewc_train.py --epochs 20
Then:
opt.epochs  # 20
"""
opt = parser.parse_args()


# ---------------------------------------------------------------------
# 3. RANDOM SEED
# ---------------------------------------------------------------------

def seed_everything(seed):
    """
    Set the random seeds so repeated experiments are more reproducible.
    """
    # function that controls randomness.
    random.seed(seed)

    # PYTHONHASHSEED controls the order of hash-based Python operations.
    # os.environ["PYTHONHASHSEED"] = str(seed)

    # Set the NumPy random seed.
    np.random.seed(seed)

    # Set the PyTorch CPU random seed.
    torch.manual_seed(seed)

    # Set random seeds for all available GPUs.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) #Sets the seed for the current GPU.
        torch.cuda.manual_seed_all(seed) #Sets the seed for all other current GPU.

# Requests repeatable CUDA operations when possible.
    torch.backends.cudnn.deterministic = True

# Stops CUDA from automatically choosing potentially nondeterministic algorithms.
    torch.backends.cudnn.benchmark = False


seed_everything(opt.seed)


# ---------------------------------------------------------------------
# 4. DEVICE CONFIGURATION
# ---------------------------------------------------------------------

# Use a GPU when one is available. Otherwise, use the CPU.
use_cuda = torch.cuda.is_available() #Checks whether a CUDA GPU is available.
device = torch.device("cuda" if use_cuda else "cpu")

print("Device:", device)

if use_cuda:
    print("Number of available GPUs:", torch.cuda.device_count())
    torch.cuda.empty_cache() #Releases unused cached GPU memory.


# ---------------------------------------------------------------------
# 5. FILE LOCATIONS
# ---------------------------------------------------------------------

# The existing repository expects the magnetogram images here.
IMAGE_DIRECTORY = "/data/hmi_jpgs_512/"

# create_continual_stages.py should create these CSV files.
STAGE_DIRECTORY = Path(
    "../data_labeling/data_labels/continual_stages"
) 

# Training CSV for every chronological stage.
STAGE_TRAIN_FILES = {
    1: STAGE_DIRECTORY / "Stage1_train.csv",
    2: STAGE_DIRECTORY / "Stage2_train.csv",
    3: STAGE_DIRECTORY / "Stage3_train.csv",
    4: STAGE_DIRECTORY / "Stage4_train.csv",
}

# Untouched and naturally imbalanced test CSV for every stage.
STAGE_TEST_FILES = {
    1: STAGE_DIRECTORY / "Stage1_test.csv",
    2: STAGE_DIRECTORY / "Stage2_test.csv",
    3: STAGE_DIRECTORY / "Stage3_test.csv",
    4: STAGE_DIRECTORY / "Stage4_test.csv",
}

# Folder where continual-learning checkpoints will be saved.
OUTPUT_DIRECTORY = Path("trained_models/continual")

# Create the output directory when it does not exist.
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 6. CHECK THE STAGE CSV FILES
# ---------------------------------------------------------------------

def check_stage_files():
    """
    Verify that all required stage CSV files exist and have correct columns.

    This check happens before expensive training begins.
    """

    all_files = list(STAGE_TRAIN_FILES.values())
    all_files += list(STAGE_TEST_FILES.values())

# Checks each CSV one at a time.
    for csv_file in all_files:
        # Stop immediately if one expected file is missing.
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Required continual-learning CSV was not found: {csv_file}"
            )

# Loads the CSV into a Pandas table.
        dataframe = pd.read_csv(csv_file)

        # Every stage CSV must contain these two columns.
        required_columns = {"label", "goes_class"}

        if not required_columns.issubset(dataframe.columns):
            raise ValueError(
                f"{csv_file} must contain label and goes_class columns."
            )

# Finds the unique labels
        # Labels must be binary: 0 for NF and 1 for FL.
        available_labels = set(dataframe["goes_class"].unique())

        if not available_labels.issubset({0, 1}):
            raise ValueError(
                f"{csv_file} contains labels other than 0 and 1."
            )

        # TSS and HSS require both classes in the test data.
        if "test" in csv_file.name.lower(): #Checks whether the current file is a test CSV.
            # Ensures test data contains both non-flare and flare samples.
            if available_labels != {0, 1}:
                raise ValueError(
                    f"{csv_file} does not contain both flare and "
                    f"non-flare samples. TSS/HSS cannot be evaluated "
                    f"properly."
                )

        # Prints the summary for examples
        # Checked Stage1_train.csv: 10800 samples, NF=8700, FL=2100
        print(
            f"Checked {csv_file}: "
            f"{len(dataframe)} samples, "
            f"NF={(dataframe['goes_class'] == 0).sum()}, "
            f"FL={(dataframe['goes_class'] == 1).sum()}"
        )


# ---------------------------------------------------------------------
# 7. DATA LOADING
# ---------------------------------------------------------------------

def dataloading(stage_number):
    """
    Build the DataLoaders for one chronological stage.

    Three loaders are returned:

    train_loader:
        Contains non-flare images, original flare images, augmented flare
        images and repeated flare images produced by Balancer.

    test_loader:
        Contains the untouched stage test data. It is not augmented or
        balanced.

    fisher_loader:
        Contains the original, unaugmented stage training data.
        Fisher information should represent the actual old task rather
        than thousands of repeated copies.
    """

    # Get the CSV paths for the requested stage.
    csv_file_train = STAGE_TRAIN_FILES[stage_number]
    csv_file_test = STAGE_TEST_FILES[stage_number]

    # ---------------------------------------------------------------
    # Normal transformation
    # ---------------------------------------------------------------

    # This is applied to non-flare images, test images, Fisher images
    # and original flare images.
    transformations = transforms.Compose([
        transforms.Resize(opt.im_size),
        transforms.ToTensor(),
    ])

    # ---------------------------------------------------------------
    # Flare-class augmentations from the original train.py
    # ---------------------------------------------------------------

    # Randomly rotate flare images between -5 and +5 degrees.
    rotation = transforms.Compose([
        transforms.Resize(opt.im_size),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ToTensor(),
    ])

    # p=1.0 means a 100% probability of flipping the image horizontally.
    # Always flip flare images horizontally.
    hr_flip = transforms.Compose([
        transforms.Resize(opt.im_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
    ])

    # Always flip flare images vertically.
    vr_flip = transforms.Compose([
        transforms.Resize(opt.im_size),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
    ])

    # ---------------------------------------------------------------
    # Separate non-flare and flare training samples
    # ---------------------------------------------------------------

    # Load all original non-flare training images.
    ori_nf = NFDataset(
        csv_file=csv_file_train,
        root_dir=IMAGE_DIRECTORY,
        transform=transformations,
    )

    # Load the original flare images without augmentation.
    ori_fl = FLDataset(
        csv_file=csv_file_train,
        root_dir=IMAGE_DIRECTORY,
        transform=transformations,
    )

    # Load a horizontally flipped version of every flare image.
    hr_flip_fl = FLDataset(
        csv_file=csv_file_train,
        root_dir=IMAGE_DIRECTORY,
        transform=hr_flip,
    )

    # Load a vertically flipped version of every flare image.
    vr_flip_fl = FLDataset(
        csv_file=csv_file_train,
        root_dir=IMAGE_DIRECTORY,
        transform=vr_flip,
    )

    # Load a randomly rotated version of every flare image.
    rotation_fl = FLDataset(
        csv_file=csv_file_train,
        root_dir=IMAGE_DIRECTORY,
        transform=rotation,
    )

    # Stop if one of the training classes is completely empty.
    if len(ori_nf) == 0:
        raise ValueError(
            f"Stage {stage_number} training data contains no NF samples."
        )

    if len(ori_fl) == 0:
        raise ValueError(
            f"Stage {stage_number} training data contains no FL samples."
        )

    # ---------------------------------------------------------------
    # Dynamically select flare datasets -- because every stage has a different imbalance.
    # ---------------------------------------------------------------

    # The original train.py used either five or six flare copies,
    # depending on the fold.
    # Fold 2 used 5 flare appearances per original flare.
    # Folds 1, 3 and 4 used 6 flare appearances per original flare.
    # That rule cannot be reused here directly because
    # the new chronological stages have different class distributions.
    #
    # These are the same datasets used by the original train.py.
    # Repeated ori_fl and hr_flip_fl reproduce its original approach.
    possible_flare_datasets = [
        ori_fl,
        hr_flip_fl,
        vr_flip_fl,
        rotation_fl,
        hr_flip_fl,
        ori_fl,
    ]

    # Store the flare datasets that can be included without creating
    # more flare appearances than non-flare appearances.
    selected_flare_datasets = []

    # Count the flare appearances added to the training dataset.
    flare_appearances = 0

    for flare_dataset in possible_flare_datasets:
        # Number of samples after adding this flare dataset.
        proposed_count = flare_appearances + len(flare_dataset)

        # Add the dataset only when it does not exceed the NF count.
        if proposed_count <= len(ori_nf):
            selected_flare_datasets.append(flare_dataset)
            flare_appearances = proposed_count
        else:
            # Every later candidate has the same length, so stop here.
            break

    # The Balancer supplies the remaining flare appearances required
    # to reach approximately equal FL and NF training counts.
    additional_flare_samples = len(ori_nf) - flare_appearances

    # Begin the combined training set with original non-flare samples.
    training_parts = [ori_nf]

    # Add original and augmented flare datasets.
    training_parts.extend(selected_flare_datasets)

    # Use the corrected modulo Balancer only when more flare samples
    # are required.
    if additional_flare_samples > 0:
        bal_set = Balancer(
            csv_file_train,
            IMAGE_DIRECTORY,
            additional_flare_samples,
            transformations,
        )

        training_parts.append(bal_set)

    # Join the NF, FL, augmented FL and repeated FL datasets.
    train_set = ConcatDataset(training_parts)

    # ---------------------------------------------------------------
    # Untouched test dataset
    # ---------------------------------------------------------------

    # The test dataset is not augmented and is not balanced.
    test_set = MyJP2Dataset(
        csv_file=csv_file_test,
        root_dir=IMAGE_DIRECTORY,
        transform=transformations,
    )

    # ---------------------------------------------------------------
    # Fisher-information dataset
    # ---------------------------------------------------------------

    # Fisher information uses the original Stage training CSV.
    #
    # It does not use:
    # - test data,
    # - augmented copies,
    # - Balancer repetitions.
    fisher_set = MyJP2Dataset(
        csv_file=csv_file_train,
        root_dir=IMAGE_DIRECTORY,
        transform=transformations,
    )

    # ---------------------------------------------------------------
    # Create the DataLoaders
    # ---------------------------------------------------------------

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=use_cuda,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=use_cuda,
        shuffle=False,
    )

    fisher_loader = DataLoader(
        dataset=fisher_set,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=use_cuda,
        shuffle=True,
    )

    print(f"\nStage {stage_number} DataLoader:")
    print(f"Original NF training samples: {len(ori_nf)}")
    print(f"Original FL training samples: {len(ori_fl)}")
    print(f"FL datasets selected: {len(selected_flare_datasets)}")
    print(f"FL appearances before Balancer: {flare_appearances}")
    print(f"Additional Balancer samples: {additional_flare_samples}")
    print(f"Final combined training length: {len(train_set)}")
    print(f"Untouched test length: {len(test_set)}")

    return train_loader, test_loader, fisher_loader


# ---------------------------------------------------------------------
# 8. ACCESS THE ORIGINAL MODEL INSIDE DATAPARALLEL
# ---------------------------------------------------------------------
"""
Two GPUs: model → DataParallel → Attn_Net
                              ↓
                         model.module

One GPU:  model → Attn_Net
"""
def unwrap_model(model):
    """
    Return the original Attn_Net.

    With two GPUs:
        model is nn.DataParallel
        original model is model.module

    With one GPU or CPU:
        model is already the original Attn_Net
    """

# Checks whether the model is wrapped by DataParallel for multiple GPUs.
    if isinstance(model, nn.DataParallel):
        return model.module #If wrapped, .module returns the original Attn_Net inside it.

    return model


# ---------------------------------------------------------------------
# 9. ESTIMATE FISHER INFORMATION
# ---------------------------------------------------------------------

def estimate_fisher(model, fisher_loader): #Receives the trained model and original stage-training data.
    """
    Estimate diagonal Fisher information after training one stage.

    Fisher information estimates the importance of every trainable
    parameter for the stage that was just learned.

    Large Fisher value:
        The parameter was important for the old stage.

    Small Fisher value:
        The parameter was less important for the old stage.
    """

    # Access Attn_Net directly so parameter names remain the same whether
    # training uses one GPU or multiple GPUs.
    base_model = unwrap_model(model)

    # Create a zero-filled Fisher tensor for every trainable parameter.
    # fisheer storage
    fisher = {
        name: torch.zeros_like(parameter, device=device)
        for name, parameter in base_model.named_parameters()
        if parameter.requires_grad
    }


    """
    [
    (
        "conv_block1.0.weight",
        Parameter(tensor([[[[0.12, -0.08, ...]]]]))
    ),
    (
        "conv_block1.0.bias",
        Parameter(tensor([0.01, 0.02, ...]))
    ),
    (
        "classify.weight",
        Parameter(tensor([[0.15, -0.22, ...]]))
    )
]
    
    """
    # Evaluation mode prevents BatchNorm running statistics from changing.
    # While calculating Fisher, we only want gradients to measure parameter importance. We do not want to modify the model.
    model.eval() #freezes BatchNorm’s stored mean and variance.
    """
    model.eval() tells the model:
“We are not training normally now, so do not update BatchNorm’s stored statistics.”

During normal training, BatchNorm keeps updating its stored average and variance using every new batch:
    """
    # Count how many batches contributed to the Fisher estimate.
    processed_batches = 0

    # Loads batches from fisher_loader
    for batch_index, (images, targets) in enumerate(fisher_loader):
        # fisher_batches=0 means use the complete Fisher DataLoader ie the complete training dataset
        if opt.fisher_batches > 0:
            if batch_index >= opt.fisher_batches:
                break

        # Move the current batch to the GPU or CPU.
        images = images.to(device=device, non_blocking=use_cuda)
        targets = targets.to(device=device, non_blocking=use_cuda)

        # Clear gradients from the previous Fisher batch.
        model.zero_grad(set_to_none=True)

        # Attn_Net returns:
        # logits, attention map 1, attention map 2, attention map 3.    
        """
        scores → flare/non-flare prediction score s
            _      → ignored attention maps
        """
        scores, _, _, _ = model(images)

        # Calculate the normal old-stage classification loss.
        loss = nn.functional.cross_entropy(scores, targets)

        # Calculate gradients for every model parameter.
        loss.backward()

        # Add the squared gradient to the parameter's Fisher value.
        #
        # Squared gradient is used because:
        # - its value is always non-negative;
        # - a large gradient suggests the parameter strongly affects loss.
        for name, parameter in base_model.named_parameters(): 
            if parameter.grad is not None: #Ensures the parameter received a gradient.
                # Adds the squared gradient to that parameter’s accumulated Fisher value.
                fisher[name] += parameter.grad.detach().pow(2) #detach Gets the gradient without attaching it to another computation graph.

        processed_batches += 1

    # Prevent division by zero if the DataLoader was empty.
    if processed_batches == 0:
        raise RuntimeError(
            "No batches were available for Fisher estimation."
        )

    # It calculates each parameter’s average squared gradient across all processed batches, which is accumated in all batches
    # representing its importance to the old stage.
    for name in fisher:
        fisher[name] /= processed_batches

    print(
        f"Fisher information estimated using "
        f"{processed_batches} batches."
    )

    return fisher


# ---------------------------------------------------------------------
# 10. SAVE THE OLD PARAMETER VALUES
# ---------------------------------------------------------------------

def save_current_parameters(model):
    """
    Save a frozen copy of the model parameters after learning a stage.

    These parameter values become theta_old in the EWC equation.
    """

    base_model = unwrap_model(model)

    old_parameters = {
        name: parameter.detach().clone()
        for name, parameter in base_model.named_parameters()
        if parameter.requires_grad
    }

    return old_parameters


# ---------------------------------------------------------------------
# 11. CALCULATE THE EWC PENALTY
# ---------------------------------------------------------------------
"""
Defines a function that calculates how much the current model’s important parameters have changed from previous stages.
- model: the model currently being trained.
- ewc_history: saved old parameters and Fisher values from previous stages.
- Return value: one total penalty added to the current classification loss.

sample of ewc history:
ewc_history = [
    {
        "stage": 1,

        "fisher": {
            "conv_block1.0.weight": torch.tensor([0.8, 0.1]),
            "classify.weight": torch.tensor([0.6, 0.2]),
        },

        "old_parameters": {
            "conv_block1.0.weight": torch.tensor([0.5, -0.3]),
            "classify.weight": torch.tensor([0.9, 0.4]),
        },
    },

    {
        "stage": 2,

        "fisher": {
            "conv_block1.0.weight": torch.tensor([0.7, 0.2]),
            "classify.weight": torch.tensor([0.5, 0.3]),
        },

        "old_parameters": {
            "conv_block1.0.weight": torch.tensor([0.6, -0.2]),
            "classify.weight": torch.tensor([0.8, 0.5]),
        },
    },
]

"""
def calculate_ewc_penalty(model, ewc_history):
    """
    Calculate EWC protection for all previously learned stages.

    ewc_history contains one Fisher dictionary and one old-parameter
    dictionary for every completed stage.

    For each previous stage:

        penalty =
            Fisher × (current parameter - old parameter)^2

    Penalties from all parameters and old stages are added together.
    """

    # Access the original Attn_Net parameters.
    base_model = unwrap_model(model)

    # Start with a scalar value of zero on the selected device.
    total_penalty = torch.zeros((), device=device)

    # Loop through every previously learned stage.
    for old_task in ewc_history:
        old_fisher = old_task["fisher"]
        old_parameters = old_task["old_parameters"]

        # Compare every current parameter with its saved old value.
        for name, parameter in base_model.named_parameters():
            if name in old_fisher:
                # Measure how much this parameter has changed.
                parameter_change = (
                    parameter - old_parameters[name]
                )

                # Important parameters have large Fisher values.
                parameter_penalty = (
                    old_fisher[name]
                    * parameter_change.pow(2)
                ).sum()

                # Add this parameter's penalty to the complete penalty.
                # we will add penalty of all the parameter
                total_penalty += parameter_penalty

    return total_penalty


# ---------------------------------------------------------------------
# 12. TRAIN ONE CONTINUAL-LEARNING STAGE
# ---------------------------------------------------------------------

def train_one_stage(
    model, #current attention model.
    stage_number, #current stage, from 1 to 4.
    train_loader, #current stage’s balanced training data.
    optimizer,
    criterion,
    ewc_history, #Fisher values and saved parameters from older stages.
):
    """
    Train the model on one chronological stage.

    Stage 1:
        EWC history is empty, so only classification loss is used.

    Stages 2-4:
        Classification loss and EWC penalty are used together.
    """

    print("\n" + "=" * 70)
    print(f"TRAINING STAGE {stage_number}")
    print("=" * 70)

    # Loop through the requested number of epochs.
    for epoch in range(1, opt.epochs + 1):
        #
        # The learning rate is divided by two every three epochs.
        if epoch % 3 == 0:
            for optimizer_group in optimizer.param_groups:
                optimizer_group["lr"] /= 2

        # Start measuring the training time for this epoch.
        start_train = timeit.default_timer()

        # Enables normal training behaviour, including updating BatchNorm statistics.
        model.train()

        # Store losses accumulated during this epoch.
        classification_loss_sum = 0.0
        ewc_loss_sum = 0.0
        total_loss_sum = 0.0

        # Store predictions and correct labels for the existing TSS/HSS function.
        prediction_list = []
        target_list = []

        for batch_index, (images, targets) in enumerate(train_loader):
            # Move the images and labels to the selected device.
            images = images.to(
                device=device,
                non_blocking=use_cuda,
            )

            targets = targets.to(
                device=device,
                non_blocking=use_cuda, #Can make CPU-to-GPU transfer faster when CUDA is used.
            )

            # Remove gradients calculated for the previous batch.
            optimizer.zero_grad(set_to_none=True)

            # Forward pass through the unchanged attention model.
            scores, _, _, _ = model(images)

            # Normal classification loss for the current stage.
            classification_loss = criterion(scores, targets)

            # Stage 1 has no previous knowledge to protect.
            if len(ewc_history) == 0:
                weighted_ewc_loss = torch.zeros(
                    (),
                    device=device,
                )

            else:
                # Calculate the raw EWC penalty.
                raw_ewc_penalty = calculate_ewc_penalty(
                    model,
                    ewc_history,
                )

                # Lambda controls how strongly old knowledge is protected.
                #
                # Division by two follows the common EWC equation.
                weighted_ewc_loss = (
                    opt.ewc_lambda / 2.0
                ) * raw_ewc_penalty

            # Complete continual-learning objective.
            total_loss = (
                classification_loss
                + weighted_ewc_loss
            )

            # Calculate gradients for classification and EWC together.
            total_loss.backward()

            # Update the model parameters.
            optimizer.step()

            # Find the predicted class:
            # 0 means non-flare and 1 means flare.
            # Selects the class with the highest score for every image.
            """
            scores = [
    [2.1, 0.4],  # Image 1
    [0.3, 1.8]   # Image 2
]
            """

            # For each image, it finds the class with the highest score.
            # _ contains the maximum score values, but we do not need them, we just need labels for FL or NF
            # it look which position has max value ie position 0 or position 1
            _, predictions = torch.max(scores, dim=1) #dim=1 means look horizontally across the class scores of each image:

            # Move predictions to CPU before storing them.
            # This prevents the lists from consuming GPU memory.
            prediction_list.append(predictions.detach().cpu())
            target_list.append(targets.detach().cpu())

            # Add the current batch losses to the epoch totals.
            classification_loss_sum += classification_loss.item() #Adds the current batch’s classification loss to the epoch total.
            ewc_loss_sum += weighted_ewc_loss.item()
            total_loss_sum += total_loss.item()
            #.item() converts a one-value PyTorch tensor into a normal Python number.


        # Stop the training timer. Records the epoch’s finishing time.
        stop_train = timeit.default_timer()

        # Number of batches processed during this epoch.
        number_of_batches = len(train_loader)

        # Calculates the average classification loss per batch.
        average_classification_loss = (
            classification_loss_sum / number_of_batches
        )

        average_ewc_loss = (
            ewc_loss_sum / number_of_batches
        )``

        average_total_loss = (
            total_loss_sum / number_of_batches
        )

        # Join all batches into one large tensor.
        #
        # Wrapping the combined tensors inside lists keeps them compatible
        # with the repository's existing evaluation function.

        # Combines predictions from all batches into one tensor.
        combined_predictions = torch.cat(prediction_list)
        combined_targets = torch.cat(target_list)

        train_tss, train_hss = (
            sklearn_Compatible_preds_and_targets(
                [combined_predictions],
                [combined_targets],
            )
        )

        current_learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"Stage {stage_number} | "
            f"Epoch {epoch}/{opt.epochs} | "
            f"Classification loss: "
            f"{average_classification_loss:.4f} | "
            f"EWC loss: {average_ewc_loss:.4f} | "
            f"Total loss: {average_total_loss:.4f} | "
            f"TSS: {train_tss:.4f} | "
            f"HSS: {train_hss:.4f} | "
            f"LR: {current_learning_rate:.8f} | "
            f"Time: {stop_train - start_train:.2f}s"
        )


# ---------------------------------------------------------------------
# 13. EVALUATE ONE TEST STAGE
# ---------------------------------------------------------------------

"""
Disables gradient calculation inside the 
function because testing does not update model weights. This saves memory and computation.
"""
@torch.no_grad()
def evaluate_stage(model, stage_number, test_loader):
    """
    Evaluate the current model on one untouched stage test set.

    EWC is not included in test loss because test loss measures only
    predictive performance.
    """

    # Disable training behavior and gradient tracking.
    # BatchNorm uses its stored statistics instead of updating them.
    model.eval()

    # Use ordinary, unweighted cross-entropy for comparable test loss.
    test_criterion = nn.CrossEntropyLoss().to(device)

    test_loss_sum = 0.0
    prediction_list = []
    target_list = []

    for images, targets in test_loader:
        # Move data to the selected device.
        images = images.to(
            device=device,
            non_blocking=use_cuda,
        )

        targets = targets.to(
            device=device,
            non_blocking=use_cuda,
        )

        # Forward pass through the model.
        scores, _, _, _ = model(images)

        # Test loss measures how wrong or uncertain the model’s predictions are on unseen test data.
        # Calculate ordinary test classification loss.
        test_loss = test_criterion(scores, targets)

        # Select the class with the largest output score.
        _, predictions = torch.max(scores, dim=1)

        # Store predictions and labels on CPU.
        prediction_list.append(predictions.cpu())
        target_list.append(targets.cpu())

        test_loss_sum += test_loss.item()

    if len(test_loader) == 0:
        raise RuntimeError(
            f"Stage {stage_number} test DataLoader is empty."
        )

    average_test_loss = test_loss_sum / len(test_loader)

    # Combine all batches before calling the existing evaluator.
    combined_predictions = torch.cat(prediction_list)
    combined_targets = torch.cat(target_list)

    test_tss, test_hss = (
        sklearn_Compatible_preds_and_targets(
            [combined_predictions],
            [combined_targets],
        )
    )

    print(
        f"Test Stage {stage_number} --> "
        f"Loss: {average_test_loss:.4f}, "
        f"TSS: {test_tss:.4f}, "
        f"HSS: {test_hss:.4f}"
    )

    return average_test_loss, test_tss, test_hss


# ---------------------------------------------------------------------
# 14. SAVE A CONTINUAL-LEARNING CHECKPOINT
# ---------------------------------------------------------------------

def save_checkpoint(model, stage_number, ewc_history):
    """
    Save the model and EWC information after one completed stage.

    Saving Fisher and old parameters allows training to be resumed later.
    """

    base_model = unwrap_model(model)

    # Move EWC tensors to CPU before saving.
    # This makes the checkpoint easier to load on another machine.
    saved_ewc_history = []

    for old_task in ewc_history:
        saved_fisher = {
            name: value.detach().cpu()
            for name, value in old_task["fisher"].items()
        }

        saved_parameters = {
            name: value.detach().cpu()
            for name, value in old_task["old_parameters"].items()
        }

        saved_ewc_history.append({
            "stage": old_task["stage"],
            "fisher": saved_fisher,
            "old_parameters": saved_parameters,
        })

    checkpoint = {
        # Stage most recently completed.
        "completed_stage": stage_number,

        # Unchanged attention-model weights.
        "model_state_dict": base_model.state_dict(),

        # Fisher and parameter information from learned stages.
        "ewc_history": saved_ewc_history,

        # Store important experiment settings.
        "ewc_lambda": opt.ewc_lambda,
        "image_size": opt.im_size,
        "attention": opt.attention,
        "seed": opt.seed,
    }

    checkpoint_path = (
        OUTPUT_DIRECTORY
        / f"attention_ewc_stage{stage_number}.pth"
    )

    torch.save(checkpoint, checkpoint_path)

    print(f"Saved checkpoint: {checkpoint_path}")


# ---------------------------------------------------------------------
# 15. MAIN CONTINUAL-LEARNING PIPELINE
# ---------------------------------------------------------------------

def train():
    """
    Train one model sequentially from Stage 1 through Stage 4.
    """

    # Confirm that all stage CSV files are ready.
    check_stage_files()

    # Determine whether the attention architecture should be enabled.
    attention_enabled = opt.attention == 1

    # Create Attn_Net only once.
    #
    # Stage 1 begins with Kaiming initialization, exactly like train.py.
    net = Attn_Net(
        im_size=opt.im_size,
        num_classes=2,
        attention=attention_enabled,
        init="kaimingUniform",
    ).to(device)

    # Preserve support for the original repository's two-GPU setup.
    #
    # Unlike the original train.py, this also works with one GPU or CPU.
    if torch.cuda.device_count() >= 2:
        model = nn.DataParallel(
            net,
            device_ids=[0, 1],
        ).to(device)
    else:
        model = net

    # Use the same ordinary cross-entropy loss as train.py.
    #
    # The DataLoader performs class balancing, so class weights are not
    # added here.
    criterion = nn.CrossEntropyLoss().to(device)

    # EWC history starts empty because no stage has been learned yet.
    #
    # After Stage 1 it contains:
    #     Fisher 1 and parameters 1
    #
    # After Stage 2 it contains:
    #     Fisher 1, parameters 1, Fisher 2 and parameters 2
    ewc_history = []

    # Keep the test DataLoaders of all previously encountered stages.
    seen_test_loaders = {}

    # Store results for later analysis.
    evaluation_results = []

    # Process stages in chronological order.
    for stage_number in range(1, 5):
        print("\n" + "#" * 70)
        print(f"STARTING CONTINUAL-LEARNING STAGE {stage_number}")
        print("#" * 70)

        # Create this stage's training, testing and Fisher loaders.
        train_loader, test_loader, fisher_loader = dataloading(
            stage_number
        )

        # Remember this test loader for future forgetting evaluation.
        seen_test_loaders[stage_number] = test_loader

        # Recreate the optimizer for the new stage.
        #
        # The optimizer is restarted, but model weights are NOT reset.
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
        )

        # Train on the current stage.
        #
        # Stage 1 automatically uses no EWC penalty because
        # ewc_history is empty.
        train_one_stage(
            model=model,
            stage_number=stage_number,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            ewc_history=ewc_history,
        )

        # -------------------------------------------------------------
        # Evaluate on every stage learned so far
        # -------------------------------------------------------------

        print("\n" + "-" * 70)
        print(
            f"EVALUATION AFTER LEARNING STAGE {stage_number}"
        )
        print("-" * 70)

        for evaluated_stage, evaluated_loader in (
            seen_test_loaders.items()
        ):
            test_loss, test_tss, test_hss = evaluate_stage(
                model=model,
                stage_number=evaluated_stage,
                test_loader=evaluated_loader,
            )

            # Each row records:
            # - how far training had progressed;
            # - which old/current test stage was evaluated.
            evaluation_results.append({
                "trained_until_stage": stage_number,
                "evaluated_stage": evaluated_stage,
                "test_loss": test_loss,
                "TSS": test_tss,
                "HSS": test_hss,
            })

        # -------------------------------------------------------------
        # Consolidate knowledge from the stage just learned
        # -------------------------------------------------------------

        print(
            f"\nCalculating Fisher information for "
            f"Stage {stage_number}..."
        )

        # Estimate which parameters were important to this stage.
        current_fisher = estimate_fisher(
            model=model,
            fisher_loader=fisher_loader,
        )

        # Save the values of the parameters after learning this stage.
        current_old_parameters = save_current_parameters(model)

        # Add this stage's information to EWC history.
        ewc_history.append({
            "stage": stage_number,
            "fisher": current_fisher,
            "old_parameters": current_old_parameters,
        })

        # Save the model and EWC information.
        save_checkpoint(
            model=model,
            stage_number=stage_number,
            ewc_history=ewc_history,
        )

        if use_cuda:
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Save all evaluation results
    # -----------------------------------------------------------------

    results_dataframe = pd.DataFrame(evaluation_results)

    results_path = (
        OUTPUT_DIRECTORY
        / "continual_evaluation_results.csv"
    )

    results_dataframe.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING COMPLETED")
    print("=" * 70)

    print(f"Evaluation results saved to: {results_path}")

    # Display the final TSS evaluation matrix.
    #
    # Rows show how many stages had been learned.
    # Columns show which stage was tested.
    print("\nTSS matrix:")
    print(
        results_dataframe.pivot(
            index="trained_until_stage",
            columns="evaluated_stage",
            values="TSS",
        )
    )

    # Display the final HSS evaluation matrix.
    print("\nHSS matrix:")
    print(
        results_dataframe.pivot(
            index="trained_until_stage",
            columns="evaluated_stage",
            values="HSS",
        )
    )


# Run training only when this file is executed directly.
if __name__ == "__main__":
    train()