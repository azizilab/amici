import os
import pickle

import matplotlib.pyplot as plt
import ncem.models as models
from ncem.data import customLoader, get_data_custom
from ncem.interpretation import InterpreterInteraction
from ncem.train import TrainModelInteractions


def get_model_parameters(niche_size):
    """
    Get the default model parameters for the NCEM model.

    Parameters are taken from the NCEM interactions
    tutorial: https://github.com/theislab/ncem_tutorials/blob/main/tutorials/model_tutorial_interactions.ipynb.

    Args:
        niche_size: int, niche size

    Returns
    -------
        exp_params: dict, experiment parameters
        model_params: dict, model parameters
        train_params: dict, training parameters
    """
    exp_params = {
        "log_transforms": False,
        "radius": niche_size,
        "seed": 18,
    }
    model_params = {
        "optimizer": "adam",
        "learning_rate": 0.05,
        "n_eval_nodes_per_graph": 5,
        "l1_coef": 0.0,
        "l2_coef": 0.0,
        "use_domain": True,
        "use_interactions": True,
        "scale_node_size": False,
        "output_layer": "linear",
    }
    train_params = {
        "epochs": 400,
        "epochs_warmup": 0,
        "batch_size": 128,
        "max_steps_per_epoch": 20,
        "validation_batch_size": 64,
        "patience": 20,
        "lr_schedule_min_lr": 1e-10,
        "lr_schedule_factor": 0.5,
        "lr_schedule_patience": 50,
        "monitor_partition": "val",
        "monitor_metric": "loss",
        "shuffle_buffer_size": None,
        "early_stopping": True,
        "reduce_lr_plateau": True,
    }
    return exp_params, model_params, train_params


def load_ncem_from_weights(
    adata,
    labels_key,
    exp_params,
    model_path,
    model_args_path,
):
    """
    Load the NCEM model from the saved weights, arguments, and experiment parameters.

    Args:
        adata: AnnData object
        labels_key: str, key in adata.obs to use as labels
        exp_params: dict, experiment parameters
        model_path: str, path to the saved model weights
        model_args_path: str, path to the saved model arguments

    Returns
    -------
        interpreter: InterpreterInteraction object
    """
    # Define the interpreter for interactions
    interpreter = InterpreterInteraction()

    # Load model arguments from pickle file
    with open(model_args_path, "rb") as file:
        model_args = pickle.load(file)

    # Initialize the model using the saved model weights and arguments
    interpreter.model = models.ModelInteractions(**model_args)
    interpreter._model_kwargs = model_args
    interpreter.model_class = "interactions"
    interpreter.reinitialize_model(changed_model_kwargs=model_args)
    interpreter.model.training_model.load_weights(model_path)

    # Load the data into the interpreter
    interpreter.data = customLoader(
        adata=adata, cluster=labels_key, patient=None, library_id=None, radius=exp_params.get("radius")
    )
    get_data_custom(interpreter=interpreter)

    return interpreter


def train_ncem(
    adata,
    labels_key,
    exp_params,
    model_params,
    train_params,
    model_path,
    model_args_path,
):
    """
    Train the NCEM model.

    Args:
        adata: AnnData object
        labels_key: str, key in adata.obs to use as labels
        exp_params: dict, experiment parameters
        model_params: dict, model parameters
        train_params: dict, training parameters
        model_path: str, path to save the model weights
        model_args_path: str, path to save the model arguments

    Returns
    -------
        model_history: dict, model history
    """
    # Define the interactions trainer
    trainer = TrainModelInteractions()
    trainer.init_estim(log_transform=exp_params.get("log_transforms"))

    # Use the custom loader to load the anndata object
    trainer.estimator.data = customLoader(
        adata=adata, cluster=labels_key, patient=None, library_id=None, radius=exp_params.get("radius")
    )
    get_data_custom(interpreter=trainer.estimator)

    trainer.estimator.init_model(
        **model_params,
    )

    # Print the model summary
    print("Initialized Model Summary")
    print(trainer.estimator.model.training_model.summary())

    # Train the model
    trainer.estimator.train(
        **train_params,
    )
    trainer.estimator.simulation = (
        False  # TODO: Fix this later but temporary hack because get_data_custom does not set this param
    )

    # Save the model weights and the model arguments
    trainer.estimator.model.training_model.save_weights(model_path)
    with open(model_args_path, "wb") as file:
        pickle.dump(trainer.estimator.model.args, file)

    # Get the model history
    model_history = trainer.estimator.model.training_model.history.history
    return model_history, trainer


def plot_ncem_loss_curves(
    model_history,
    radius,
    save_dir,
):
    """
    Plot the training, validation, and reconstruction loss curves for the NCEM model.

    Args:
        model_history: dict, model history
        radius: int, radius
        save_dir: str, directory to save the loss curves

    Returns
    -------
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    train_loss = model_history["loss"]
    val_loss = model_history["val_loss"]
    recons_loss = model_history["gaussian_reconstruction_loss"]
    val_recons_loss = model_history["val_gaussian_reconstruction_loss"]

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.plot(recons_loss, label="Gaussian Reconstruction Loss")
    plt.plot(val_recons_loss, label="Validation Gaussian Reconstruction Loss")
    plt.title("NCEM Model Training, Validation, and Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_dir}/ncem_loss_curves_radius_{radius}.png")
    plt.close()
