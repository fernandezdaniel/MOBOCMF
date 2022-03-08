import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import torch
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf

sys.path.append(".")

from dgpmf.models.dgp import DGP_Base
from dgpmf.models.layers_init import init_layers
from dgpmf.likelihoods.likelihood import Gaussian
from utils.dataset import Test_Dataset, Training_Dataset
from utils.process_flags import manage_experiment_configuration
from utils.tensorflow_learning import fit_with_metrics, score

args = manage_experiment_configuration()

# Set eager execution
# tf.config.run_functions_eagerly(True)

train_indexes, test_indexes = train_test_split(
    np.arange(len(args.dataset)), test_size=0.1, random_state=2147483647
)

train_dataset = Training_Dataset(
    args.dataset.inputs[train_indexes], args.dataset.targets[train_indexes]
)
test_dataset = Test_Dataset(
    args.dataset.inputs[test_indexes],
    args.dataset.targets[test_indexes],
    train_dataset.inputs_mean,
    train_dataset.inputs_std,
)

# Get SVGP layers
layers = init_layers(
    train_dataset.inputs, train_dataset.output_dim, **vars(args)
)

train_loader = train_dataset.get_generator(batch_size=args.batch_size)
val_loader = test_dataset.get_generator(batch_size=args.batch_size)


# Instantiate Likelihood
ll = Gaussian()

# Create dgp object
dgp = DGP_Base(
    train_dataset.inputs,
    train_dataset.targets,
    ll,
    layers,
    len(train_indexes),
    num_samples=args.num_samples_train,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    dtype=args.dtype,
)

# Define optimizer and compile model
opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

# Perform training
train_hist, val_hist = fit_with_metrics(
    dgp,
    train_loader,
    opt,
    val_generator=val_loader,
    epochs=args.epochs,
)



dgp.num_samples = args.num_samples_test
test_metrics = score(dgp, val_loader)

print("TEST RESULTS: ")
print("\t - NELBO: {}".format(test_metrics["LOSS"])) # Print KL # Print one-GP
print("\t - NLL: {}".format(test_metrics["NLL"]))
print("\t - RMSE: {}".format(test_metrics["RMSE"]))
print("\t - CRPS: {}".format(test_metrics["CRPS"]))

df = pd.DataFrame.from_dict(train_hist)
df_val = pd.DataFrame.from_dict(val_hist)

fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 3)
ax3 = fig.add_subplot(2, 2, 2)
ax4 = fig.add_subplot(2, 2, 4)

loss = df[["LOSS"]].to_numpy().flatten()
ax3.plot(loss, label="Training loss")
ax3.legend()
ax3.set_title("Loss evolution")
ax4.plot(
    np.arange(loss.shape[0] // 2, loss.shape[0]),
    loss[loss.shape[0] // 2 :],
    label="Training loss",
)
ax4.legend()
ax4.set_title("Loss evolution in last half of epochs")


ax1.plot(df[["RMSE"]].to_numpy(), label="Training RMSE")
ax1.plot(df_val[["RMSE"]].to_numpy(), label="Validation RMSE")
ax1.legend()
ax1.set_title("RMSE evolution")
ax2.plot(df[["NLL"]].to_numpy(), label="Training NLL")
ax2.plot(df_val[["NLL"]].to_numpy(), label="Validation NLL")
ax2.legend()
ax2.set_title("NLL evolution")
filename = "dataset={}_gp_layers={}_epochs={}_genkern={}_n_inducings={}_batch_size={}_split={}{}".format(
    args.dataset_name,
    "-".join(str(i) for i in args.gp_layers),
    str(args.epochs),
    args.genkern,
    str(args.num_inducing_points),
    str(args.batch_size),
    str(args.split),
    args.name_flag,
)

plt.savefig("plots/" + filename + ".png")
# open file for writing
f = open("plots/" + filename + ".txt", "w")

# write file
f.write(str(test_metrics))

# close file
f.close()

# plt.show()
