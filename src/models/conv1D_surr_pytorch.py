import torch
import torch.nn as nn
import surrogate.surrogate as surr
import numpy as np

class conv1D_surr(nn.Module, surr):
    """
    A PyTorch Lightning module representing a 1D convolutional neural network
    for surrogate modeling of data.

    Args:
        spectrum_channel_nb (int): Number of channels in the output spectrum.
        x_features_nb (int): Number of features in the input data.

    Returns:
        torch.Tensor: A tensor representing the output of the convolutional neural network.
    """
    def __init__(self, spectrum_channel_nb: int, x_features_nb: int):
        super().__init__()

        # Define the layers of the neural network
        self.Dense1 = nn.Linear(x_features_nb, 128)
        self.Dense2 = nn.Linear(128, 64)
        self.Dense3 = nn.Linear(64, 32)
        self.Dense4 = nn.Linear(32, 16)

        # Reshape encoded input to a 3D tensor with shape (None, 1, 16)
        self.Reshape = nn.Reshape((16, 1))

        # Decode the encoded input using Conv1DTranspose
        # first use wide kernel size to learn the global structure and interraction between channels
        self.Conv1DT1 = nn.ConvTranspose1d(1, 32, kernel_size=32, stride=2, padding=16, output_padding=1, bias=False)
        self.Conv1DT2 = nn.ConvTranspose1d(32, 64, kernel_size=16, stride=2, padding=8, output_padding=1, bias=False)
        self.Conv1DT3 = nn.ConvTranspose1d(64, 128, kernel_size=8, stride=2, padding=4, output_padding=1, bias=False)
        # then use narrow kernel size to learn the local structure
        self.Conv1DT4 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False)
        self.Conv1DT5 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2, padding=0, output_padding=1, bias=False)
        # finally filter the output to get the desired output shape
        self.Conv1D = nn.Conv1d(32, spectrum_channel_nb, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        """
        Forward pass of the convolutional neural network.

        Args:
            x (torch.Tensor): Input data as a tensor of shape (batch_size, x_features_nb).

        Returns:
            torch.Tensor: A tensor representing the output of the convolutional neural network.
        """
        # Encode the input using dense layer
        x = nn.functional.relu(self.Dense1(x))
        x = nn.functional.relu(self.Dense2(x))
        x = nn.functional.relu(self.Dense3(x))
        x = nn.functional.relu(self.Dense4(x))

        # Reshape encoded input to a 3D tensor with shape (None, 1, 16)
        x = self.Reshape(x)

        # Decode the encoded input using Conv1DTranspose
        # first use wide kernel size to learn the global structure and interraction between channels
        x = nn.functional.relu(self.Conv1DT1(x))
        x = nn.functional.relu(self.Conv1DT2(x))
        x = nn.functional.relu(self.Conv1DT3(x))
        # then use narrow kernel size to learn the local structure
        x = nn.functional.relu(self.Conv1DT4(x))
        x = nn.functional.relu(self.Conv1DT5(x))
        # finally filter the output to get the desired output shape
        for i in range(4):
            x = nn.functional.relu(self.Conv1D(x))
        return x

        
    def train_surrogate(self, x_train : np.array, x_test: np.array, y_train: np.array, y_test: np.array,
                         epochs : int =100, batch_size : int =32, loss : str = 'mse', learn_rate : float =0.001, dropout_rate : float =0.3, verbose : int =1) :
        """Train the autoencoder on the given data"""
        
        # Define the model
        self.compile(optimizer=Adam(learning_rate=learn_rate), loss=loss)

        # create a TensorBoard callback
        tensorboard_callback = TensorBoard(log_dir='./tb_logs')

        history = self.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=verbose,
            callbacks=[tensorboard_callback]
        )

        self.history = pd.DataFrame(history.history)
        self.history['epoch'] = self.history.index.values