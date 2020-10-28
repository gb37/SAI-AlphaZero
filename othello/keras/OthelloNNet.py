import sys
sys.path.append('..')
from utils import *

import argparse
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.backend import *
from tensorflow.keras.losses import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#tf.debugging.set_log_device_placement(True)
#tf.executing_eagerly()
tf.compat.v1.disable_eager_execution()

class OthelloNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        # Input layer
        self.input_boards = Input(shape=(self.board_x, self.board_y))            # s: batch_size x board_x x board_y
        self.input_komi = Input(shape=(1,))                                      # batch_size x komi

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)    # batch_size  x board_x x board_y x 1
        
        # Hidden Layer 1 ( batch_size  x board_x x board_y x num_channels )
        h_conv1 = Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)
        h_conv1 = BatchNormalization(axis=3)(h_conv1)
        h_conv1 = Activation('relu')(h_conv1)
        
        # Hidden Layer 2 ( batch_size  x board_x x board_y x num_channels )
        h_conv2 = Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)
        h_conv2 = BatchNormalization(axis=3)(h_conv2)
        h_conv2 = Activation('relu')(h_conv2)
        
        # Hidden Layer 3 ( batch_size  x (board_x-2) x (board_y-2) x num_channels )
        h_conv3 = Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)
        h_conv3 = BatchNormalization(axis=3)(h_conv3)
        h_conv3 = Activation('relu')(h_conv3)
        
        # Hidden Layer 4 ( batch_size  x (board_x-4) x (board_y-4) x num_channels )
        h_conv4 = Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv3)
        h_conv4 = BatchNormalization(axis=3)(h_conv4)
        h_conv4 = Activation('relu')(h_conv4)
        
        # Flatten
        h_conv4_flat = Flatten()(h_conv4)
        
        # Dropout Layer 1 ( batch_size x 1024 )
        s_fc1 = Dense(1024, use_bias=False)(h_conv4_flat)
        s_fc1 = BatchNormalization(axis=1)(s_fc1)
        s_fc1 = Activation('relu')(s_fc1)
        s_fc1 = Dropout(args.dropout)(s_fc1)
        
        # Dropout Layer 2 ( batch_size x 1024 )
        s_fc2 = Dense(512, use_bias=False)(s_fc1)
        s_fc2 = BatchNormalization(axis=1)(s_fc2)
        s_fc2 = Activation('relu')(s_fc2)
        s_fc2 = Dropout(args.dropout)(s_fc2)
        
        # Policy - Output Layer ( batch_size x self.action_size )
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
        
        # Value - Output Layer ( batch_size x 1 )
        #self.v = Dense(1, activation='tanh', name='v')(s_fc2)
        #self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
         
        # SAI part
        # alpha ( batch_size x 1 )
        self.alpha = Dense(1, activation='linear', name='alpha')(s_fc2)
        
        # beta ( batch_size x 1 )
        self.beta = Dense(1, activation='exponential', name='beta')(s_fc2)
        
        # Add Komi to Alpha
        self.komi =  Lambda(lambda x: x, name='komi')(self.input_komi)
        self.alpha_komi = Add()( [self.alpha, self.komi] )

        # Sigma - Output Layer ( batch_size x 1 )
        #self.sigma = sigmoid( Multiply()( [self.beta, self.alpha_komi] ) )    # sigma = 1 / { 1 + exp[ -beta x (alpha + komi) ] }
        self.sigma = tanh( Multiply()( [self.beta, self.alpha_komi] ) )    # sigma = 1 / { 1 + exp[ -beta x (alpha + komi) ] }
        
        # Build model
        self.model = Model(inputs=[self.input_boards, self.input_komi], outputs=[self.pi, self.sigma, self.alpha])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error',None], optimizer=Adam(args.lr))
        
        # Plot model
        tf.keras.utils.plot_model(self.model, show_shapes=True, expand_nested=True)
        self.model.summary()
