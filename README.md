## Five fish
The code is in two separate files, one for training and generating
predictions, and the other for running simulations.

### Training and generating predictions
five_fish_model.ipynb:

  Description:

  -At the beginning, we define global variables for network parameters.  These
  can be modified as desired.

  -Next, we define functions for downloading and preprocessing data.  These
  are all encompassed in the function download_and_preprocess_data().

  -plot_fit() is a helper function for visualizing the training's progress.

  -get_minibatch() is a helper function for producing a batch.

  -In the block directly below "Create and Run the Graph", we define the graph
  structure.  Note that this must be run before defining the later functions that
  take graph=graph as an argument.  The train_model() function is used to train
  the model.

  -The section "Generate predictions" defines functions for use in generating
  predictions based on the trained model.  The output of the main function in
  this section, generate_prediction(), is a numpy ndarray with 10 columns.  Each
  row is one time-step.  The first 5 columns are fish 1 through 5's x positions;
  the last 5 are their y positions.

  -"Now run the code" depicts a typical use case.

  Use:

  -Define all functions above

  -Create training, validation, and test sets by running download_and_preprocess_data()

  -Train the model using train_model()

  -Generate and save a trajectory by running
  ```
  savefile = './ModelOutputs/generated_trajectory.csv'
  seed = x_valid[0]  # an initial window to feed into the generative model
  pred = generate_prediction(seed, prediction_length=5000, restore_path=model_path, progress_counter=40)
  np.savetxt(savefile, pred, delimiter=',')
  ```

### Running simulations

five_fish_sim.ipynb

## One fish

Again, the code is in two separate files.  Use is much the same as with the five
fish model.  A minor difference is that in the one-fish model, the inputs given
are the fish's distance to all four walls.

The notebook for training and generating predictions is called "one_fish_model.ipynb",
and the notebook for visualizing simulations is called "fish_sim.ipynb"

## Next Steps

The further directions that I think are the most interesting are the following:

1). To change the architecture to be a social LSTM architecture. Our very limited attempt was to modify the tensorflow social LSTM code found here: https://github.com/xuerenlv/social-lstm-tf. This attempt is in our folder "unsocial-social-lstm".

2). The next change would be to change the loss function which currently is mean square error. We instead would like to predict a probability distribution and use log-likehood loss. Our attempt can be found here "unsuccessful-log_likelihood.ipynb". Scroll down to find the cell with the pertinent information.

We very quickly tried both of these approaches but we never got them to work as well as our previous models. We believe that this is because of bugs in our code and that these approaches still have lots of merit.
