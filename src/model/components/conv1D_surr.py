import torch.nn as nn
import numpy as np
from torchinfo import summary

class conv1D_surr(nn.Module):
    """
    A PyTorch Lightning module representing a 1D convolutional neural network
    for surrogate modeling of data.

    Args:
        spectrum_channel_nb (int): Number of channels in the output spectrum.
        x_features_nb (int): Number of features in the input data.

    Returns:
        torch.Tensor: A tensor representing the output of the convolutional neural network.
    """
    def __init__(self,  x_input_size: int, spectrum_decomp_length:int, spectrum_channel_nb: int, dropout_rate:float):
        super().__init__()
        self.spectrum_channel_nb = spectrum_channel_nb
        self.x_input_size = x_input_size
        self.spectrum_decomp_length = spectrum_decomp_length
        self.dropout = nn.Dropout(p=dropout_rate)
        # Define the layers of the neural network
        self.Dense1 = nn.Linear(self.x_input_size, 128)
        self.Dense2 = nn.Linear(128, 64)
        self.Dense3 = nn.Linear(64, 32)
        self.Dense4 = nn.Linear(32, 16)


        # Decode the encoded input using Conv1DTranspose
        self.Conv1DT1 = nn.ConvTranspose1d(16, 32, kernel_size=32, stride=2 )
        self.Conv1DT2 = nn.ConvTranspose1d(32, 64, kernel_size=2, stride=2 )
        self.Conv1DT3 = nn.ConvTranspose1d(64, 128, kernel_size=2, stride=2)
        # then use narrow kernel size to learn the local structure
        self.Conv1DT4 = nn.ConvTranspose1d(128, 256, kernel_size=2, stride=2)
        self.Conv1DT5 = nn.ConvTranspose1d(256, 512, kernel_size=2, stride=2)
        # finally filter the output to get the desired output shape
        self.Conv1D1 = nn.Conv1d(512,256,kernel_size=2,stride=2)
        self.Conv1D2 = nn.Conv1d(256,128,kernel_size=2,stride=2)
        self.Conv1D3 = nn.Conv1d(128,64,kernel_size=2,stride=2)

        stride = 2
        k_size = 64-(self.spectrum_decomp_length-1)*stride
        self.Conv1D4 = nn.Conv1d(64,self.spectrum_channel_nb,kernel_size=k_size,stride=stride)

        summary(self, input_size=(self.x_input_size,))     

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
        x = self.dropout(x)
        x = nn.functional.relu(self.Dense3(x))
        x = nn.functional.relu(self.Dense4(x))

        # Reshape encoded input to a 3D tensor with shape (None, 1, 16)
        x = x.view(-1, 16, 1)

        # Decode the encoded input using Conv1DTranspose
        # first use wide kernel size to learn the global structure and interraction between channels
        x = nn.functional.relu(self.Conv1DT1(x))
        x = nn.functional.relu(self.Conv1DT2(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.Conv1DT3(x))
        # then use narrow kernel size to learn the local structure
        x = nn.functional.relu(self.Conv1DT4(x))
        x = nn.functional.relu(self.Conv1DT5(x))
        x = self.dropout(x)
        # finally filter the output to get the desired output shape
        x = nn.functional.relu(self.Conv1D1(x))
        x = nn.functional.relu(self.Conv1D2(x))
        x = nn.functional.relu(self.Conv1D3(x))
        x = nn.functional.relu(self.Conv1D4(x))
        x= x.permute(0, 2, 1)
        # for i in range(1):
        #     x = nn.functional.relu(self.Conv1D(x))
        return x

        
    # def train_surrogate(self, x_train : np.array, x_test: np.array, y_train: np.array, y_test: np.array,
    #                      epochs : int =100, batch_size : int =32, loss : str = 'mse', learn_rate : float =0.001, dropout_rate : float =0.3, verbose : int =1) :
    #     """Train the autoencoder on the given data"""
        
    #     # Define the model
    #     self.compile(optimizer=Adam(learning_rate=learn_rate), loss=loss)

    #     # create a TensorBoard callback
    #     tensorboard_callback = TensorBoard(log_dir='./tb_logs')

    #     history = self.fit(
    #         x_train,
    #         y_train,
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         validation_data=(x_test, y_test),
    #         verbose=verbose,
    #         callbacks=[tensorboard_callback]
    #     )

    #     self.history = pd.DataFrame(history.history)
    #     self.history['epoch'] = self.history.index.values

if __name__ == '__main__':
    # Test the model
    model = conv1D_surr(6, 23, 6)
    batch_size = 32
    summary(model, input_size = (batch_size, 7))