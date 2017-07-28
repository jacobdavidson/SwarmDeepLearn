import numpy as np
import tensorflow as tf

import os
import pickle
import argparse

from utils import DataLoader
from model import Model

savefile = 'save/generated.csv'

def main():
    # Define the parser
    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=5,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=1000,
                        help='Predicted length of the trajectory')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=1,
                        help='Dataset to be tested on')

    # Read the arguments
    sample_args = parser.parse_args()

    # Load the saved arguments to the model from the config file
    with open(os.path.join('save', 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # Initialize with the saved args
    model = Model(saved_args, True)
    # Initialize TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize TensorFlow saver
    saver = tf.train.Saver()

    # Get the checkpoint state to load the model from
    ckpt = tf.train.get_checkpoint_state('save')
    print('loading model: ', ckpt.model_checkpoint_path)

    # Restore the model at the checpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Dataset to get data from
    dataset = [sample_args.test_dataset]

    # Initialize the dataloader object to
    # Get sequences of length obs_length
    data_loader = DataLoader(1, sample_args.obs_length+1, dataset, True)

    # Reset the data pointers of the data loader object
    data_loader.reset_batch_pointer()

    x, y = data_loader.next_batch()

    # The observed part of the trajectory
    obs_traj = x[0][:sample_args.obs_length]
    # Get the complete trajectory with both the observed and the predicted part from the model
    complete_traj = model.sample(sess, obs_traj, x[0], num=sample_args.pred_length)

    np.savetxt(savefile, complete_traj, delimiter=",")

    print("Data generation complete!")

if __name__ == '__main__':
    main()
