#pyenv shell miniconda3-4.2.12/envs/3D_CNN_tf24

import os
import zipfile
import numpy as np
import copy
import tensorflow as tf
import cv2

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical, plot_model
#import keras
#from keras import layers
#from keras.utils import to_categorical
from tensorflow.keras import backend as K
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
from more_itertools import chunked
import itertools
import random
from scipy import ndimage
#from scipy.ndimage.interpolation import shift
import scipy.stats
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
import pickle

#import three_d_resnet_builder

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        
        if DA_flip:
            flip_axis = random.choice([0, 1, 2])
            if flip_axis != 2:
                volume = np.flip(volume, flip_axis)
        
         
        if DA_rotate:
            # idefine some rotation angles
            angles = [-2, -1, 0, 1, 2]
            # pick angles at random
            angle = random.choice(angles)
            # rotate volume
            volume = ndimage.rotate(volume, angle, reshape=False)
            volume[volume < 0] = 0
            volume[volume > 1] = 1
            
        
        #shift
        if DA_shift:
            shift_axis = random.choice([0, 1, 2])
            shift_size = int(len(volume[shift_axis]) * (np.random.rand()*0.5 - 0.25))
            #shift_size = int(len(volume[shift_axis]) * (np.random.rand()*0.2 - 0.1))
            shift_vec = [0, 0, 0]
            shift_vec[shift_axis] = shift_size
            #volume = ndimage.shift(volume, shift = shift_vec, cval = 0)
            volume = np.roll(volume, shift_size, axis = shift_axis)
            if shift_axis == 0:
                if shift_axis > 0:
                    volume[:shift_size,:, :] = 0
                else:
                    volume[shift_size:,:, :] = 0
            elif shift_axis == 1:
                if shift_size > 0:
                    volume[:, :shift_size, :] = 0
                else:
                    volume[:, shift_size:, :] = 0
            elif shift_axis == 2:
                if shift_size > 0:
                    volume[:, :, :shift_size] = 0
                else:
                    volume[:, :, shift_size:] = 0
            
        #shift intensity
        if DA_shift_intensity:
            volume = volume + (np.random.rand()*0.002 - 0.001)
        
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def set_seed(seed=0):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    if DA_valid:
        volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def get_model(width=128, height=128, depth=64, dropout_rate = 0.3):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding='same')(inputs)
    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=(1,1,4))(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=(1,1,4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=(1,1,2))(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)


    x = layers.Flatten()(x)
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(units=3, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def get_deep_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    #x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    #x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    #x = layers.Flatten()(x)
    #x = layers.Dropout(0.4)(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(units=3, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def get_C3D_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    #x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    #x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    #x = layers.Flatten()(x)
    #x = layers.Dropout(0.4)(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(units=3, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


def normalize(X):
    mean = np.mean(X, axis=(1,2,3), keepdims=True)
    sd = np.std(X, axis=(1,2,3), keepdims=True)
    return (X-mean)/(sd+1e-7)


class ZCAWhitening:
  def __init__(self, epsilon=1E-6):
    self.epsilon = epsilon
    self.mean = None
    self.zca = None

  def fit(self, x):
    self.mean = np.mean(x, axis=0)
    x_ = x - self.mean
    cov = np.dot(x_.T, x_) / x_.shape[0]
    E, D, _ = np.linalg.svd(cov)
    D = np.sqrt(D) + self.epsilon
    self.zca = np.dot(E, np.dot(np.diag(1.0 / D), E.T))
    return self

  def transform(self, x):
    x_ = x - self.mean
    return np.dot(x_, self.zca.T)



def loadtxt3(filename, skip = 0):
    data = []
    for l in open(filename).readlines():
        data.append(float(l))
    return data


def load_data(path, z_size = 800):
    data_list = []
    for i in range(1,4):
        print(i)
        data = []
        for x in range(60):
            x_data = []
            for y in range(60):
                if i == 2 and path == 'data/Control/C_':
                    file = path+str(i)+'/'+str(x).zfill(3)+'_'+str(y).zfill(3)+'_interpolation.csv'
                else:
                    file = path+str(i)+'/'+str(x).zfill(3)+'_'+str(y).zfill(3)+'.csv'
                x_data.append(loadtxt3(file)[1:z_size+1])
            data.append(x_data)
        data = np.array(data)
        data_list.append(data)
    return data_list

def prepare_subBlock_list(data_list, subBlock_size = 20):
    block_list = []
    for i in range(3):
        tmp_list = []
        for x in range(3):
            for y in range(3):
                tmp_list.append(data_list[i][x*subBlock_size:(x+1)*subBlock_size,y*subBlock_size:(y+1)*subBlock_size])
                
        block_list.append(tmp_list) 
    return block_list



def z_correction(intensity, z):
    att_db=50 #減衰定数[dB/m]
    #bottom_p=650 #堆積層表面位置 (z sample)
    #att_cor = Math.Pow(10, att_db / 6670 / 20 * (z - bottom_p))  #補正式　zは読み込んでいるzの値
    att_cor = 10 ** (att_db / 6670 / 20 * (z))  #補正式　zは読み込んでいるzの値
    intensity_cor = intensity * att_cor * 15
    return intensity_cor

def calc_correction(data_list):
    data_list = copy.copy(data_list)
    for z in range(data_list.shape[2]):
        data_list[:,:,z] = z_correction(data_list[:,:,z], z)
    return data_list

def save_Grad_CAM(original_data, cam_data, file_name):
    #Original image
    z_size = original_data.shape[2]
    array_img = np.concatenate([np.concatenate([original_data[:,:,j*50+i] for i in range(50)], axis = 0) for j in range(int(z_size/50))], axis = 1)
    v_max = np.max(array_img)
    array_img = Image.fromarray(np.sqrt(array_img/v_max)*255)
    array_img = array_img.convert('RGB') #if color == 'RGB', then the image is converted to color img.
    array_img.save(file_name+'original.png')

    #Grad_CAM image
    z_size = cam_data.shape[2]
    resized_cam_data = []
    for z in range(z_size):
        resized_cam = cv2.resize(cam_data[:,:,z], (20, 20), cv2.INTER_LINEAR)
        resized_cam_data+=[resized_cam for i in range(32)]
    resized_cam_data = np.array(resized_cam_data)
    z_size = resized_cam_data.shape[0]
    resized_array_img = np.concatenate([np.concatenate([resized_cam_data[j*50+i] for i in range(50)], axis = 0) for j in range(int(z_size/50))], axis = 1)
    jetcam = cv2.applyColorMap(np.uint8(255 * resized_array_img), cv2.COLORMAP_JET)
    cv2.imwrite(file_name+'GradCAM.png', jetcam)
    
    #overlayed image
    overlay_cam = (np.float32(jetcam)*0.3 + np.array(array_img)*0.7)
    cv2.imwrite(file_name+'overlay_GradCAM.png', overlay_cam)


def Grad_CAM_visualization(model, x_test, y_test, fold, output_file_header):
    """Grad-CAM function"""
    layer_name = 'conv3d_'+str(fold*6 + 5)
    grad_model = models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    for i in range(len(x_test)):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x_test[i:i+1])
            class_idx = np.argmax(predictions[0])
            loss = predictions[:, class_idx]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')

        guided_grads = gate_f * gate_r * grads
        weights = np.mean(guided_grads, axis=(0, 1, 2))
        cam = np.dot(output, weights)
        

        cam = np.maximum(cam, 0)      # Passing through ReLU
        cam /= np.max(cam)            # scale 0 to 1.0


        save_Grad_CAM(x_test[i], cam, 'fig/Vis_'+output_file_header+'fold'+str(fold)+'_test'+str(i)+'_GT'+str(np.argmax(y_test[i]))+'_Pred'+str(class_idx)+'_')



if __name__ == "__main__":

    total_fold = 5
    seed = 0
    data_correction = True
    gain_for_non_corrected = 1 #check for intensity gain for row data to compare correction effect
    z_init = 650
    z_range = 800
    #Data augmentation setting
    DA_flip = True
    DA_rotate = False
    DA_shift = True
    DA_shift_intensity = False
    DA_valid = False #Validation datasetのDAをするか. defaultはFalseだがやった方がいい時もある気がするがゆらぎが大きくなる可能性もあるか.
    #Model setting
    dropout_rate = 0.3 #default = 0.3

    #Training setting
    learning_rate = 0.0001
    best_val_acc_threshold = 0.7 #0にすると普通のプロセス. >0にするとvalidationのスコアが閾値を超えるまで学習を繰り返す 
    max_train_loop = 3

    #output file header
    output_file_header = 'v_DAflip'+str(DA_flip)+'_DAshift'+str(DA_shift)+'_DC'+str(data_correction)+'_'


    # Load training data index
    #初回のみ
    if not (os.path.exists("C_data.pickle") and os.path.exists("H_data.pickle") and os.path.exists("Ma_data.pickle") and os.path.exists("Mi_data.pickle")):
        C_data_list = load_data('data/Control/C_', z_size = 2000)
        H_data_list = load_data('data/Hydrobiaulvae/Hu_', z_size = 2000)
        Ma_data_list = load_data('data/Macoma/Ma_', z_size = 2000)
        Mi_data_list = load_data('data/Mix/Mix_', z_size = 2000)

        with open("C_data.pickle", mode="wb") as f:
            pickle.dump(C_data_list, f)
        with open("H_data.pickle", mode="wb") as f:
            pickle.dump(H_data_list, f)
        with open("Ma_data.pickle", mode="wb") as f:
            pickle.dump(Ma_data_list, f)
        with open("Mi_data.pickle", mode="wb") as f:
            pickle.dump(Mi_data_list, f)



    #pickleのロード
    with open("C_data.pickle", mode="rb") as f:
        C_data_list = pickle.load(f)
    with open("H_data.pickle", mode="rb") as f:
        H_data_list = pickle.load(f)
    with open("Ma_data.pickle", mode="rb") as f:
        Ma_data_list = pickle.load(f)
    with open("Mi_data.pickle", mode="rb") as f:
        Mi_data_list = pickle.load(f)


    print('C:', C_data_list[0].shape, 'min', np.min(C_data_list), 'max', np.max(C_data_list))
    print('H:', H_data_list[0].shape, 'min', np.min(H_data_list), 'max', np.max(H_data_list))
    print('Ma:', Ma_data_list[0].shape, 'min', np.min(Ma_data_list), 'max', np.max(Ma_data_list))
    print('Mi:', Mi_data_list[0].shape, 'min', np.min(Mi_data_list), 'max', np.max(Mi_data_list))

    max_value = np.max([np.max(C_data_list), np.max(H_data_list), np.max(Ma_data_list), np.max(Mi_data_list)])
    print('max_value', max_value)

    #data correction
    if data_correction:
        C_data_list = [calc_correction(C_data_list[i]) for i in range(3)]
        H_list = [calc_correction(H_data_list[i]) for i in range(3)]
        Ma_data_list = [calc_correction(Ma_data_list[i]) for i in range(3)]
        Mi_data_list = [calc_correction(Mi_data_list[i]) for i in range(3)]

    C_data_list = [gain_for_non_corrected*data for data in C_data_list]
    H_data_list = [gain_for_non_corrected*data for data in H_data_list]
    Ma_data_list = [gain_for_non_corrected*data for data in Ma_data_list]
    Mi_data_list = [gain_for_non_corrected*data for data in Mi_data_list]

    #blockの分割. 
    C_data_list = [data[:,:,z_init:z_init+z_range] for data in C_data_list]
    H_data_list = [data[:,:,z_init:z_init+z_range] for data in H_data_list]
    Ma_data_list = [data[:,:,z_init:z_init+z_range] for data in Ma_data_list]
    Mi_data_list = [data[:,:,z_init:z_init+z_range] for data in Mi_data_list]

    #preparation of sub blocks
    C_subBlock_list = prepare_subBlock_list(C_data_list)
    H_subBlock_list = prepare_subBlock_list(H_data_list)
    Ma_subBlock_list = prepare_subBlock_list(Ma_data_list)
    Mi_subBlock_list = prepare_subBlock_list(Mi_data_list)

    print('sub block shape:', C_subBlock_list[1][1].shape)


    x_trvlts = C_subBlock_list[0] +C_subBlock_list[1] + C_subBlock_list[2] + \
              H_subBlock_list[0] + H_subBlock_list[1] + H_subBlock_list[2]+ \
               Ma_subBlock_list[0] + Ma_subBlock_list[1] + Ma_subBlock_list[2]
    y_trvlts = [0]*9*3 + [1]*9*3 + [2]*9*3
    #x_test = Ma_subBlock_list[0]
    #y_test = [2]*9


    #x_test = np.concatenate([CrD_3D_data_list[CrD_3D_test_idx], NSC_3D_data_list[NSC_3D_test_idx], UC_3D_data_list[UC_3D_test_idx]], axis = 0)
    #y_test = np.concatenate([CrD_labels[CrD_3D_test_idx], NSC_labels[NSC_3D_test_idx], UC_labels[UC_3D_test_idx]], axis = 0)

    def my_to_categorical(data):
        cat = [[0,0,0] for i in range(len(data))]
        for i,v in enumerate(data):
            cat[i][v] = 1
        return np.array(cat)

    x_trvlts = np.array(x_trvlts).astype(np.float32)
    y_trvlts = my_to_categorical(y_trvlts)


    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    set_seed(seed)
    rand_perm = np.random.permutation(range(len(x_trvlts)))
    print('data', len(rand_perm),'rand_perm', rand_perm)


    y_test_pred_prob_all = []
    y_test_all = []
    cls_rep_all = []

    for fold in range(total_fold):
        splitted_rand_perm = np.array_split(rand_perm, total_fold)
        print(splitted_rand_perm)
        test_id_list = rand_perm[splitted_rand_perm[fold]]
        valid_id_list = rand_perm[splitted_rand_perm[fold - 1]]
        train_id_list = rand_perm[list(splitted_rand_perm[fold - 4])+list(splitted_rand_perm[fold - 3])+list(splitted_rand_perm[fold - 2])]
        print('train_id:', train_id_list)
        print('valid_id:', valid_id_list)
        print('test_id:', test_id_list)

        x_train = x_trvlts[train_id_list]
        y_train = y_trvlts[train_id_list]
        x_val = x_trvlts[valid_id_list]
        y_val = y_trvlts[valid_id_list]
        x_test = x_trvlts[test_id_list]
        y_test = y_trvlts[test_id_list]

        print('x_train', x_train.shape)

        #ZCAWhitening
        def calc_ZCAWhitening(X):
            x_zcaw = X.reshape(X.shape[0], -1)
            zcaw = ZCAWhitening().fit(x_zcaw)
            return zcaw.transform(x_zcaw).reshape(X.shape)

        #x_train = calc_ZCAWhitening(x_train.astype(np.uint8))
        #x_train = calc_ZCAWhitening(x_train.astype(np.float16))
        #x_val = calc_ZCAWhitening(x_val.astype(np.float16))
        #x_test = calc_ZCAWhitening(x_test.astype(np.float16))

        print(
            "Number of samples in train and validation are %d and %d."
            % (x_train.shape[0], x_val.shape[0])
        )

        #sys.exit()

        # Define data loaders.
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        batch_size = 2 #2 in original
        # Augment the on the fly during training.
        train_dataset = (
            train_loader.shuffle(len(x_train))
            .map(train_preprocessing)
            .batch(batch_size)
            .prefetch(2)
        )
        # Only rescale.
        validation_dataset = (
            validation_loader.shuffle(len(x_val))
            .map(validation_preprocessing)
            .batch(batch_size)
            .prefetch(2)
        )

        best_val_acc = -1
        train_loop = 0
        while best_val_acc < best_val_acc_threshold and train_loop < max_train_loop:
            train_loop+=1

            # Build model.
            #model = get_C3D_model(width=128, height=128, depth=64)  #get_deep_model(width=128, height=128, depth=64)
            model = get_model(width=20, height=20, depth=z_range, dropout_rate = dropout_rate) 
            plot_model(model, to_file='fig/'+output_file_header+'model.png', show_shapes = True, dpi=300)
            model.summary()


            # Compile model.
            initial_learning_rate = learning_rate #original 0.0001
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps=100, decay_rate=0.8, staircase=True
            )
            model.compile(
                loss = "categorical_crossentropy", #loss="binary_crossentropy",
                #optimizer = keras.optimizers.Adadelta(),
                optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                #optimizer=keras.optimizers.Adam(),
                metrics=["acc"],
            )

            # Define callbacks.
            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                "3d_image_classification.h5", save_best_only=True
            )
            early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", verbose=1, patience=40)
            #early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=20)

            # Train the model, doing validation at the end of each epoch
            epochs = 50
            result = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                shuffle=True,
                verbose=1,
                #class_weight = class_weight,
                callbacks=[checkpoint_cb, early_stopping_cb],
            )
            #print('history', result.history)
            best_val_acc = np.max(result.history['val_acc'])
            print('best val_acc', best_val_acc)

            
            
            fig, ax = plt.subplots(1, 2, figsize=(20, 3))
            ax = ax.ravel()

            for i, metric in enumerate(["acc", "loss"]):
                ax[i].plot(model.history.history[metric])
                ax[i].plot(model.history.history["val_" + metric])
                ax[i].set_title("Model {}".format(metric))
                ax[i].set_xlabel("epochs")
                ax[i].set_ylabel(metric)
                ax[i].legend(["train", "val"])
            plt.savefig('fig/'+output_file_header+'training_curve_'+str(fold)+'.png', dpi = 300)



            
        #Evaluation for test data
        model.load_weights('3d_image_classification.h5')
        y_test_pred_prob = model.predict(x_test)
        y_test_pred = np.argmax(y_test_pred_prob, axis=1)
        y_test_label = np.argmax(y_test, axis=1)
        print('y_test_GT:', y_test_label)
        print('y_test_pred:', y_test_pred_prob)
        print('Confusion matrix \n', confusion_matrix(y_test_label, y_test_pred))
        print('Classification report:\n', classification_report(y_test_label, y_test_pred))
        

        y_test_pred_prob_all += list(y_test_pred_prob)
        y_test_all += list(y_test_label)
        cls_rep_all.append(classification_report(y_test_label, y_test_pred))


        
        Grad_CAM_visualization(model, x_test, y_test, fold, output_file_header)

        #tf.keras.backend.clear_session()
        #del model

    print('===Cross-validation Result====')

    y_test_pred_all = np.argmax(y_test_pred_prob_all, axis = 1)
    print('Confusion matrix:\n', confusion_matrix(y_test_all, y_test_pred_all))
    print('Classification report:\n', classification_report(y_test_all, y_test_pred_all))

    for fold in range(total_fold):
        print('fold ', fold, '\n', cls_rep_all[fold])

    def plot_multiclass_roc_curve(y_test, y_pred, model_name=""):
        y_test = np.array(my_to_categorical(y_test))
        y_score = np.array(y_pred)
        n_classes = y_score.shape[-1]
        print('c_classes', n_classes)
        print('y_test', y_test)
        print('y_score', y_score)

        lw = 2
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        #ax.plot(fpr["micro"], tpr["micro"],
        #            label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["micro"]),
        #            color='deeppink', linestyle=':', linewidth=6)

        ax.plot(fpr["macro"], tpr["macro"],
                    label='macro-average ROC curve (area = {0:0.2f})'
                         ''.format(roc_auc["macro"]),
                    color='navy', linestyle=':', linewidth=6)

        colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
        class_list = ['Control', 'Hydrobia ulvae', 'Macoma']
        for i, color in zip(range(n_classes), colors):
            #k = [k for k, v in data["class_indices"].items() if v == i][0]
            ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class '+class_list[i]+' (area = {0:0.2f})'
                         ''.format(roc_auc[i]))

        ax.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=30, labelpad=20,)
        ax.set_ylabel('True Positive Rate',  fontsize=30, labelpad=20,)
        ax.xaxis.set_tick_params(direction="out", labelsize=20, width=3, pad=8)  # x軸の目盛りの調整
        ax.yaxis.set_tick_params(direction="out", labelsize=20, width=3, pad=8)  # y軸の目盛りの調整
        ax.set_title(f"ROC curves\n(model: {model_name})", fontsize=25, pad=10)
        ax.legend(loc="lower right",  fontsize=12)
        fig.tight_layout()
        return fig

    fig = plot_multiclass_roc_curve(y_test_all, y_test_pred_prob_all, '3D-CNN')
    fig.savefig('fig/'+output_file_header+'ROC.png', dpi = 300)






