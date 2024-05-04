

# Note: VVG stands for GBVV in Spanish, and noVVG for non-GBVV


# In[1]: Import libraries


import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import graphviz
import pydot
import pydotplus
import time


from matplotlib import pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
import tensorflow.nn as nn
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout, BatchNormalization, Lambda, Layer, PReLU, Conv1D, GRU, Concatenate, AveragePooling1D, MaxPool1D, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping



# In[2]: Restart Keras session

K.clear_session()



# In[3]: Load data

#Load the csv with 38 audio features from all 78 volunteers
data_input_path = 'C:/Users/Emma Reyner Fuentes/Desktop/'
name = data_input_path + '1sec_window_videolabel_librosa2.csv' 

df = pd.read_csv(name);
df.pop("Unnamed: 0");



# In[4]: Create the DataFrame


data_columns = ["Mean.MFCC0", "Mean.MFCC1", "Mean.MFCC2", "Mean.MFCC3", "Mean.MFCC4", "Mean.MFCC5", "Mean.MFCC6", "Mean.MFCC7", "Mean.MFCC8", "Mean.MFCC9", "Mean.MFCC10", "Mean.MFCC11", "Mean.MFCC12", "Mean.Energy", "Mean.ZCR", "Mean.Centroid", "Mean.Rolloff", "Mean.Flatness", "Mean.Pitch", "Std.MFCC0", "Std.MFCC1", "Std.MFCC2", "Std.MFCC3", "Std.MFCC4", "Std.MFCC5", "Std.MFCC6", "Std.MFCC7", "Std.MFCC8", "Std.MFCC9", "Std.MFCC10", "Std.MFCC11", "Std.MFCC12", "Std.Energy", "Std.ZCR", "Std.Centroid", "Std.Rolloff", "Std.Flatness", "Std.Pitch"]
domain_col = ['User']
cond_col = ['VVG/noVVG']

df['User'] = df['User'].astype('category').cat.codes + 1
df['VVG/noVVG'] = df['VVG/noVVG'].astype('category').cat.codes
df = pd.concat([df["User"], df["VVG/noVVG"], df[data_columns]], axis=1)


# In[5]: Make even the number of samples of GBVV and non-GBVV

# Randomly eliminate 1 out of every 4 samples of GBVV
    


filt = df['VVG/noVVG'] == 0 #Filter GBVVs
num_rows = filt.sum() #Number of GBVV samples

num_rows_elim = num_rows // 4 #Number of rows to be removed
ind_elim = np.random.choice(df[filt].index, size=num_rows_elim, replace=False) #Random indexes of rows to be removed
filt_inverse = ~df.index.isin(ind_elim) #Select rows not to be removed

df = df[filt_inverse]
df = df.reset_index() #Final DF to work with

#Check the number of samples for each group
count = len(df[(df['VVG/noVVG'] == 0)])
print("Number og GBVV rows':", count)
count2 = len(df[(df['VVG/noVVG'] == 1)])
print("Number of nonGBVV rows:", count2)

print(df.shape)

#Check the number of samples per user
df.head()

count_by_user = df.groupby('User').agg(Count=('User', 'size'), Value=('VVG/noVVG', 'first')).reset_index()
print(count_by_user)



# In[]:

# ## MODEL DEFINITIONS


# In[6]: Customized plot function


def personalized_plot_model(model):

    # Mostrar resumen
    model.summary()

    # Mostrar esquema del modelo
    return tf.keras.utils.plot_model(
      model,
      show_shapes=True,
      show_dtype=False,
      show_layer_names=True,
      rankdir='TB',
      expand_nested=True,
      dpi=96,
      layer_range=None,
      show_layer_activations=True,
      show_trainable=True) 
    

# In[7]: ENCODER


def create_encoder(num_feats, embedding_size=128): # If not defined, by default embedding_size = 32
    """
    From an audio feature dataframe, create an encoder to extract embeddings
    """    
    inputs = Input(shape=(num_feats)) 
    encoded = Dense(256, activation='relu')(inputs)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    embeddings = Dense(128, activation='relu')(encoded)
     
    model = Model(inputs=inputs, outputs=embeddings,  name='Encoder_classif')

    return model


# In[8]: SPEAKER CLASSIFIER


def speaker_classif(unique_domains, embedding_size=128):
    
    inputs = Input(shape=(embedding_size,))
    x_spk = Dense(128, activation='relu')(inputs)
    x_spk = Dense(64, activation='relu')(x_spk)
    outputs_spk = Dense(unique_domains, activation='relu')(x_spk)
    
    model = Model(inputs=inputs, outputs=outputs_spk,  name='Speaker_classif')

    return model


# In[9]: CONDITION CLASSIFIER


def condition_classif(unique_conditions, embedding_size=128):
    
    inputs = Input(shape=(embedding_size,))
    x_cond = Dense(64, activation='relu')(inputs)
    x_cond = Dense(32, activation='relu')(x_cond)
    x_cond = Dense(16, activation='relu')(x_cond)
    outputs_cond = Dense(unique_conditions, activation='relu')(x_cond)
    
    model = Model(inputs=inputs, outputs=outputs_cond, name='Condition_classif')
        
    return model


# In[10]: Variables for models definition

num_feats = df[data_columns].shape[1]
print(df[data_columns].shape[1])
unique_domains = np.unique(df[domain_col].values).shape[0] 
unique_conds = np.unique(df[cond_col].values).shape[0]
embedding_size = 128


# In[]: 
    
    # # # LEAVE ONE SUBJECT OUT (LOSO) SPLIT LOOP # # #


# In[11]: Variables to store the metrics

#Note: users from 1 to 39 (included) - GBVV; from 40 to 78 - nonGBVV

logo = LeaveOneGroupOut()
subjects = df["User"]

# Vectors for condition isolated LOSO split
accuracies_cond = np.array([])
f1_scores_cond = np.array([])
y_pred_total_cond = np.array([])
y_true_total_cond = np.array([])
conf_mtx_total_cond = np.array([])

# Vectors for spk isolated LOSO split
accuracies_spk = np.array([])
f1_scores_spk = np.array([])
y_pred_total_spk = np.array([])
y_true_total_spk = np.array([])
conf_mtx_total_spk = np.array([])

# Vectors for unlearnt spk LOSO split
accuracies_unlspk = np.array([])
f1_scores_unlspk = np.array([])
y_pred_total_unlspk = np.array([])
y_true_total_unlspk = np.array([])
conf_mtx_total_unlspk = np.array([])

# Vectors for adversarial LOSO split
accuracies_main = np.array([])
f1_scores_main = np.array([])
y_pred_total_main = np.array([])
y_true_total_main = np.array([])
conf_mtx_total_main = np.array([])

X = df[data_columns]
y = df[cond_col] #Not used in LOSO but we have to put it

# In[12]: Start of the LOSO loop

for train_index, test_index in logo.split(X, y, subjects):
    user = df["User"][test_index[0]]
    print("User ", df["User"][test_index[0]]) #Print to know which user are we at
    
    # Make the data split
    test_df = df.iloc[test_index]
    train_df = df.iloc[train_index]
    
    #Separate 30% of the training users for validation
    train_df, val_df = train_test_split(train_df, test_size=0.3, random_state=42)

    X_train = train_df[data_columns]
    X_val = val_df[data_columns]
    X_test = test_df[data_columns]

    y_cond_train = train_df[cond_col]
    y_cond_val = val_df[cond_col]
    y_cond_test = test_df[cond_col]

    y_dom_train = train_df[domain_col]
    y_dom_val = val_df[domain_col]
    y_dom_test = test_df[domain_col]

    #Info about the variables:
    print("train_df: ", train_df.shape)

    print("X_train: ", X_train.shape)
    print("X_val: ", X_val.shape)
    print("X_test: ", X_test.shape)

    print("y_train: ", y_cond_train.shape)
    print("y_val: ", y_cond_val.shape)
    print("y_test: ", y_cond_test.shape)

    print("d_train: ", y_dom_train.shape)
    print("d_val: ", y_dom_val.shape)
    print("d_test: ", y_dom_test.shape)
    
    # METRICS FOR MATRIX USER/VIDEO
    acc_main_vid = np.zeros((14, 2))
    acc_cond_vid = np.zeros((14, 2))
    for i in range(0, 14):
        acc_main_vid[i, 0] = i+1
        acc_cond_vid[i, 0] = i+1
        
    # Models' parameters
    starter_learning_rate = 0.000000001
    decay_steps = 10000
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps)
    sgd_optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate_fn, momentum=0.9)



    
    # In[13]: (within LOSO loop)   
# ---------------------------Training GBVV Condition Classifier (Isolated)----------------------------
    
    # Note: As GANNs have convergence problems due to the random initialization of weights (Ref: https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/), 
    #we will train first each of the SC (Speaker Classifier) and GBVVCC (GBVV Condition Classifier) separately, to "initialize" the weights properly, before the adversarial training fashion

    
    # Model Definition
    encoder_block = create_encoder(num_feats, embedding_size=128)
    condition_block = condition_classif(unique_conds, embedding_size=128)
    
    # Connections
    inputs = encoder_block.input
    outputs = condition_block(encoder_block(inputs))
    
    gbvv_classif_model = Model(inputs, outputs=outputs, name='gbvv_classif_model')
    
    # Model Compilation
    gbvv_classif_model.compile(optimizer=sgd_optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    
    # Model Plot
    personalized_plot_model(gbvv_classif_model)
    
    
    # Model Training
    
    print("User ", df["User"][test_index[0]])
    print("Training of the COND ISO model")
    
    y_true_cond = y_cond_test
    
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_dom_train, y_cond_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)
    
    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_dom_val, y_cond_val))
    val_dataset = val_dataset.batch(32)
    
    print("Fit gbvv_classif_model on training data")
    history_gbvv = gbvv_classif_model.fit(
    X_train,
    y_cond_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_cond_val),
    )
    
    #Evaluate in the excluded user
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = gbvv_classif_model.evaluate(X_test, y_cond_test)
    print("test loss, test acc:", results)
    
    # Generate predictions 
    predictions = gbvv_classif_model.predict(X_test)
    # Obtener el índice de la clase con la probabilidad más alta
    y_pred_indices = np.argmax(predictions, axis=1)
    # Crear un arreglo binario basado en la comparación
    y_pred_cond = np.where(y_pred_indices == 1, 1, 0)
    
    # Confusion matrix
    cm_cond = confusion_matrix(y_true_cond, y_pred_cond)
    
    # Accuracy y f1:
    acc_cond = accuracy_score(y_true_cond, y_pred_cond)
    f1_cond = f1_score(y_true_cond, y_pred_cond, pos_label = 0)
    
    
    # Plot history
    
    # summarize history for accuracy
    plt.plot(history_gbvv.history['accuracy'])
    plt.plot(history_gbvv.history['val_accuracy'])
    title = 'COND ISO Model Accuracy of User' + str(df["User"][test_index[0]])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history_gbvv.history['loss'])
    plt.plot(history_gbvv.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

    
    # In[14]: (within LOSO loop)  
# ---------------------------Training Speaker Classifier (Isolated)----------------------------
    
    
    # Model Definition
    encoder_block.trainable = True # Permitimos que el encoder aprenda inicialmente del spk 
    speaker_block = speaker_classif(unique_domains+1, embedding_size=128)
    
    # Connections
    inputs = encoder_block.input
    outputs = speaker_block(encoder_block(inputs))
    
    spk_classif_model = Model(inputs, outputs=outputs, name='speaker_classif_model')
    
    # Model Compilation
    spk_classif_model.compile(optimizer=sgd_optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    
    # Model Plot
    personalized_plot_model(spk_classif_model)
    
    
    # Model Training
    
    print("Fit spk_classif_model on training data")
    history_spk = spk_classif_model.fit(
        X_train,
        y_dom_train,
        batch_size=32,
        epochs=100, #!!! no es 20 como el de arriba si no 50
        validation_data=(X_val, y_dom_val),
    )
    
    y_true_spk = y_dom_val
    
    # Evaluate the model on the val data using `evaluate`
    #Note: we do not evaluate on test data because we cannot predict a speaker never seen
    print("Evaluate on val data")
    results = spk_classif_model.evaluate(X_val, y_dom_val)
    print("test loss, test acc:", results)
    
    # Generate predictions 
    predictions = spk_classif_model.predict(X_val)
    y_pred_spk = np.argmax(predictions, axis=1)

    # Accuracy:
    acc_spk = accuracy_score(y_true_spk, y_pred_spk)
    
    
    # Plot history
    
    # summarize history for accuracy
    plt.plot(history_spk.history['accuracy'])
    plt.plot(history_spk.history['val_accuracy'])
    title = "SPK Model accuracy sparing user " + str(df["User"][test_index[0]])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history_spk.history['loss'])
    plt.plot(history_spk.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
    
    # In[]: (within LOSO loop)  
    
# ---------------------------Training Models in Adversarial Fashion----------------------------

    # In[15]: Definitions
    
    # Parameters
    batch_size = 16
    num_epochs = 100 
    landa_param = 0.2
    
    starter_learning_rate = 0.000000001
    decay_steps = 10000
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps)
    
    
    # Metrics instantiation
    train_main_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    train_domain_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    val_main_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_domain_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    
    # Variables storage
    train_loss_domain_array = np.zeros((num_epochs,1))
    train_loss_main_array = np.zeros((num_epochs,1))
    train_acc_domain_array = np.zeros((num_epochs,1))
    train_acc_main_array = np.zeros((num_epochs,1))
    
    val_acc_domain_array = np.zeros((num_epochs,1))
    val_acc_main_array = np.zeros((num_epochs,1))
    
    
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_dom_train, y_cond_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_dom_val, y_cond_val))
    val_dataset = val_dataset.batch(batch_size)
    

    
    # DOMAIN model definition
    encoder_block.trainable = False # Encoder is frozen
    
    inputs_domain = encoder_block.input
    domain_model = Model(inputs_domain, outputs=speaker_block(encoder_block(inputs_domain)), name='domain_model')

    personalized_plot_model(domain_model)
    
    
    # MAIN model definition
    encoder_block.trainable = True # Encoder is learning
    speaker_block.trainable = False # Speaker classif is frozen
    
    inputs_main = encoder_block.input
    main_model = Model(inputs_main, outputs=[speaker_block((encoder_block(inputs_main))), \
                                                 condition_block(encoder_block(inputs_main))], \
                            name='main_model')
    
    
    personalized_plot_model(main_model)
    
    

    
    # In[16]: Optimizers and Loss functions  (within LOSO loop)  
    
    
    # Optimizers
    optimizer_domain = sgd_optimizer
    optimizer_main = sgd_optimizer
    
    optimizer_domain.legacy = True
    optimizer_main.legacy = True
    

    # Loss domain model
    loss_domain = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Custom loss main model. 
      #Ref1: https://github.com/inoryy/tensorflow2-deep-reinforcement-learning/issues/3
      #Ref2: https://stackoverflow.com/questions/58565394/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-c#:~:text=The%20representation%20of%20the%20targets%20are%20the%20only%20difference%2C%20the%20results%20should%20be%20the%20same%20since%20they%20are%20both%20calculating%20categorical%20crossentropy.
    class CustomEntropyLoss(tf.keras.losses.Loss): 
        def __init__(self):
            super().__init__()
        def call(self, y_true, y_pred):

            custom_loss1 = landa_param*tf.keras.losses.sparse_categorical_crossentropy(y_true[1], y_pred[1], from_logits=True) # GBVV vs NoGBVV
            custom_loss2 = (1-landa_param)*tf.math.divide(tf.keras.losses.categorical_crossentropy(nn.softmax(y_pred[0], axis=1), y_pred[0], from_logits=True),tf.math.log(tf.constant([2.]))) # Speaker
    
            return custom_loss1-custom_loss2
    
    loss_main = CustomEntropyLoss()
    
    


    
    # In[17]: Custom Adversarial Training Functions  (within LOSO loop)  
    
    
    # MAIN STEP - training
    @tf.function
    def train_main_step(x, y_dom, y_cond):
        with tf.GradientTape() as tape:
            logits_main = main_model(x, training=True)  
            loss_value_main = loss_main([y_dom, y_cond], logits_main) 
    
        main_grads = tape.gradient(loss_value_main, main_model.trainable_weights)
        optimizer_main.apply_gradients(zip(main_grads, main_model.trainable_weights))
        train_main_acc_metric.update_state(y_cond, logits_main[1])
    
        return loss_value_main
    


    # MAIN STEP - evaluation
    @tf.function
    def eval_main_step(x, y_cond):
        val_main_logits = main_model(x, training=False)
        val_main_acc_metric.update_state(y_cond, val_main_logits[1]) # Va a ser bueno en condition
        print('y_cond:')
        print(y_cond)
        print('val_main_logits:')
        print(val_main_logits)
    
    

    
    # DOMAIN STEP - training
    @tf.function
    def train_domain_step(x, y_dom):
    
        with tf.GradientTape() as tape:
    
            logits_domain = domain_model(x_batch_train, training=True) 
            loss_value_domain = loss_domain(y_dom, logits_domain)
    
        domain_grads = tape.gradient(loss_value_domain, domain_model.trainable_weights)
        optimizer_domain.apply_gradients(zip(domain_grads, domain_model.trainable_weights))
        train_domain_acc_metric.update_state(y_dom, logits_domain)
    
        return loss_value_domain
    
    
    
    # DOMAIN STEP - evaluation
    @tf.function
    def eval_domain_step(x, y_dom):
        val_domain_logits = domain_model(x, training=False)
        val_domain_acc_metric.update_state(y_dom, val_domain_logits) # Va a ser bueno en speaker
    
  
    
  
    # Testing the trained encoder + condition block
    @tf.function
    def final_test(X_test, y_cond):
        test_main_logits = main_model(X_test, training=False)
        val_main_acc_metric.update_state(y_cond, test_main_logits[1]) # Va a ser bueno en condition
        return y_cond, test_main_logits
    
    
    
    # In[18]: CUSTOM ADVERSARIAL TRAINING LOOP  (within LOSO loop)  
    
    
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_dom_train, y_cond_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_dom_val, y_cond_val))
    val_dataset = val_dataset.batch(batch_size)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print("\nStart of epoch %d" % (epoch,))
        print("User: ", user)
    
        ## 1 - DOMAIN STEP: Iterate over the batches of the dataset for DOMAIN Model
        encoder_block.trainable = False # Encoder is frozen
        speaker_block.trainable = True # Speaker is active
    
        for step, (x_batch_train, y_batch_dom_train, y_batch_cond_train) in enumerate(train_dataset):
            # Coger weights del Enc main
            if step == 0:
                domain_model.layers[1].set_weights(main_model.layers[1].get_weights())
    
            loss_value_domain = train_domain_step(x_batch_train, y_batch_dom_train) 
    
            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "   Training loss for domain model (for one batch) at step %d: %.4f"
                    % (step, float(loss_value_domain))
                )
                print("   Seen so far: %s samples" % ((step + 1) * batch_size))
    
        # Display metrics at the end of each domain epoch.
        train_domain_acc = train_domain_acc_metric.result()
        print("Training domain acc over epoch (Speaker Recognition): %.4f" % (float(train_domain_acc),))
        train_loss_domain_array[epoch] = float(loss_value_domain)
        train_acc_domain_array[epoch] = float(train_domain_acc)
    
        # Reset training metrics at the end of each domain epoch
        train_domain_acc_metric.reset_states()
    
        # Run a validation loop at the end of each DOMAIN step.
        for x_batch_val, y_batch_dom_val, y_batch_cond_val in val_dataset:
            eval_domain_step(x_batch_val, y_batch_dom_val)
    
        val_domain_acc = val_domain_acc_metric.result()
        val_domain_acc_metric.reset_states()
        print("Validation domain acc (Speaker Recognition): %.4f" % (float(val_domain_acc),))
        print("   Time taken: %.2fs" % (time.time() - start_time))
        val_acc_domain_array[epoch] = val_domain_acc
        
        
    
        ## 2- 2 MAIN STEPs: Iterate over the batches of the dataset for MAIN Model
    
        encoder_block.trainable = True # Encoder is active
        speaker_block.trainable = False # Speaker is frozen
        
        # 1st MAIN STEP
        for step, (x_batch_train, y_batch_dom_train, y_batch_cond_train) in enumerate(train_dataset):
            loss_value_main = train_main_step(x_batch_train, y_batch_dom_train, y_batch_cond_train) 
    
            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "   Training loss for main Model (for one batch) at step %d: %.4f"
                    % (step, float(loss_value_main))
                )
                print("   Seen so far: %s samples" % ((step + 1) * batch_size))
        # 2nd MAIN STEP
        for step, (x_batch_train, y_batch_dom_train, y_batch_cond_train) in enumerate(train_dataset):
            loss_value_main = train_main_step(x_batch_train, y_batch_dom_train, y_batch_cond_train) 
    
            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "   Training loss for main Model (for one batch) at step %d: %.4f"
                    % (step, float(loss_value_main))
                )
                print("   Seen so far: %s samples" % ((step + 1) * batch_size))
    
        # Display metrics at the end of the 2 main steps.
        train_main_acc = train_main_acc_metric.result()
        print("Training main acc over epoch (GBVV Condition): %.4f" % (float(train_main_acc),))
        train_loss_main_array[epoch] = float(loss_value_main)
        train_acc_main_array[epoch] = float(train_main_acc)
    
        # Reset training metrics
        train_main_acc_metric.reset_states()
    
        # Run a validation loop 
        for x_batch_val, y_batch_dom_val, y_batch_cond_val in val_dataset:
            eval_main_step(x_batch_val, y_batch_cond_val)
    
        val_main_acc = val_main_acc_metric.result()
        val_main_acc_metric.reset_states()
        print("Validation main acc (GBVV Condition): %.4f" % (float(val_main_acc),))
        print("   Time taken: %.2fs" % (time.time() - start_time))
        val_acc_main_array[epoch] = val_main_acc
    
    # # END OF THE ADVERSARIAL LOOP # #
    
    #After finishing all 3-stepped epochs, we evaluate in the excluded user:
    print("Evaluate on test data")
    y_true_main, logits_y_pred = final_test(X_test, y_cond_test)
    
    # Sigmoid function to get probabilities
    y_pred_probabilities = 1 / (1 + np.exp(-logits_y_pred[1]))
    
    # Getting the predicted classes
    y_pred_indices = np.argmax(y_pred_probabilities, axis=1)
    y_pred_main = np.where(y_pred_indices == 1, 1, 0)
    
    # Confusion matrix
    cm_main = confusion_matrix(y_true_main, y_pred_main)
    
    # Accuracy and f1:
    acc_main = accuracy_score(y_true_main, y_pred_main)
    f1_main = f1_score(y_true_main, y_pred_main, pos_label = 0)
    
    
    # Plot history
    
    plt.figure()
    plt.plot(range(num_epochs), train_loss_domain_array[0:100], 'dodgerblue', label='Training loss domain model')
    plt.title('Training loss domain con lambda = '+str(landa_param))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    #plt.savefig(data_input_path+'imgs/training_loss_domain_lambda'+str(landa_param)+'.png')
    
    plt.figure()
    plt.plot(range(num_epochs), train_loss_main_array[0:100], 'darkorange', label='Training loss mainersarial model')
    plt.title('Training loss main con lambda = '+str(landa_param))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    #plt.savefig(data_input_path+'imgs/training_loss_main_lambda'+str(landa_param)+'.png')
    
    plt.figure()
    plt.plot(range(num_epochs), train_acc_domain_array[0:100], 'turquoise', label='Train Accuracy domain model')
    plt.plot(range(num_epochs), train_acc_main_array[0:100], 'gold', label='Train Accuracy mainersarial model')
    plt.title('Train Accuracy con lambda = '+str(landa_param))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #plt.savefig(data_input_path+'imgs/train_accuracy_lambda'+str(landa_param)+'.png')
    
    
    plt.figure()
    plt.plot(range(num_epochs), val_acc_domain_array[0:100], 'turquoise', label='Validation Accuracy domain model')
    plt.plot(range(num_epochs), val_acc_main_array[0:100], 'gold', label='Validation Accuracy mainersarial model')
    plt.title('Validation Accuracy con lambda = '+str(landa_param))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #plt.savefig(data_input_path+'imgs/validation_accuracy_lambda'+str(landa_param)+'.png')
    
        
    # In[19]: UNLEARNT SPEAKER MODEL  (within LOSO loop)  
    
    # This model uses the encoder that unlearnt the speaker traits during adversarial training, and pairs it with a new speaker
    # classifier. Then, we train it to see how much the encoder actually unlearnt from speaker traits.
    
    
    # Definition
    encoder_block.trainable = False # Encoder is frozen
    speaker_block_new = speaker_classif(unique_domains+1, embedding_size=128)
    
    # Connections
    inputs = encoder_block.input
    outputs = speaker_block_new(encoder_block(inputs))
    
    spk_classif_model_new = Model(inputs, outputs=outputs, name='speaker_classif_model_new')
    
    # Compile
    spk_classif_model_new.compile(optimizer=sgd_optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    
    # Model Plot
    personalized_plot_model(spk_classif_model_new)
    
    
    # Model Training
    
    print("Fit NEW spk_classif_model on training data")
    history_spk_new = spk_classif_model_new.fit(
        X_train,
        y_dom_train,
        batch_size=32,
        epochs=100, 
        validation_data=(X_val, y_dom_val),
    )
    
    y_true_unlspk = y_dom_val
    
    # Evaluate the model on the val data
    print("Evaluate on val data")
    results = spk_classif_model_new.evaluate(X_val, y_dom_val)
    print("test loss, test acc:", results)
    
    # Generate predictions 
    predictions = spk_classif_model_new.predict(X_val)
    y_pred_unlspk = np.argmax(predictions, axis=1)    
    
    # Accuracy
    acc_unlspk = accuracy_score(y_true_unlspk, y_pred_unlspk)
    
    
    # Plot history
    
    # summarize history for accuracy
    plt.plot(history_spk_new.history['accuracy'])
    plt.plot(history_spk_new.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history_spk_new.history['loss'])
    plt.plot(history_spk_new.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[20]: RESULTS, concatenating all results and saving to CSV  (within LOSO loop)  
    

    #--------Condition Isolated----------
    # Concatenate
    accuracies_cond = np.append(accuracies_cond, acc_cond)
    f1_scores_cond = np.append(f1_scores_cond, f1_cond)
    y_pred_total_cond = np.append(y_pred_total_cond, y_pred_cond)
    y_true_total_cond = np.append(y_true_total_cond, y_true_cond)
    conf_mtx_total_cond = np.append(conf_mtx_total_cond, cm_cond)
        
    # Save to csv
    dftocsv = pd.DataFrame()
    dftocsv['Accuracy'] = acc_cond
    dftocsv['F1 score'] = f1_cond
    dftocsv['y pred'] = y_pred_cond
    dftocsv['y true'] = y_true_cond
        
    name = str(df["User"][test_index[0]]) + "_cond_iso.csv"
    #dftocsv.to_csv(name)
    
    #--------Speaker Isolated----------
    # Concatenate
    accuracies_spk = np.append(accuracies_spk, acc_spk)
    y_pred_total_spk = np.append(y_pred_total_spk, y_pred_spk)
    y_true_total_spk = np.append(y_true_total_spk, y_true_spk)
        
    # Save to csv
    dftocsv = pd.DataFrame()
    dftocsv['Accuracy'] = acc_spk
    dftocsv['y pred'] = y_pred_spk
    dftocsv['y true'] = y_true_spk
        
    name = str(df["User"][test_index[0]]) + "_spk_iso.csv"
    #dftocsv.to_csv(name)
    
    
    #--------Adversarial----------
    # Concatenate
    accuracies_main = np.append(accuracies_main, acc_main)
    f1_scores_main = np.append(f1_scores_main, f1_main)
    y_pred_total_main = np.append(y_pred_total_main, y_pred_main)
    y_true_total_main = np.append(y_true_total_main, y_true_main)
    conf_mtx_total_main = np.append(conf_mtx_total_main, cm_main)

    print("The accuracies per user for the adversarial model are: \n", accuracies_main)

    # Save to csv
    dftocsv = pd.DataFrame()
    dftocsv['Accuracy'] = acc_main
    dftocsv['F1 score'] = f1_main
    dftocsv['y pred'] = y_pred_main
    dftocsv['y true'] = y_true_main
        
    name = str(df["User"][test_index[0]]) + "_main.csv"
    #dftocsv.to_csv(name)
    
    #--------Unlearnt Speaker----------
    # Concatenate
    accuracies_unlspk = np.append(accuracies_unlspk, acc_unlspk)
    y_pred_total_unlspk = np.append(y_pred_total_unlspk, y_pred_unlspk)
    y_true_total_unlspk = np.append(y_true_total_unlspk, y_true_unlspk)
        
    # Save to csv
    dftocsv = pd.DataFrame()
    dftocsv['Accuracy'] = acc_unlspk
    dftocsv['y pred'] = y_pred_unlspk
    dftocsv['y true'] = y_true_unlspk
        
    # name = str(df["User"][test_index[0]]) + "_spk_new.csv"
    # dftocsv.to_csv(name)

# # # # END OF THE LOSO LOOP # # # #

# In[21]: Presenting the global metrics for all 78 users


# -------Global metrics for Condition Isolated--------
mean_acc = np.mean(accuracies_cond)
std_acc = np.std(accuracies_cond)
mean_f1 = f1_score(y_true_total_cond, y_pred_total_cond, pos_label = 0)

print("The mean accuracy for the COND ISO model for 78 users is: ", mean_acc, "; and the std is: ", std_acc)
print("The f1-score for the COND ISO model for 78 users is: ", mean_f1)

# Confusion matrix
cm = confusion_matrix(y_true_total_cond, y_pred_total_cond)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predictions')
plt.ylabel('True Labels')
plt.title('Confusion Matrix COND ISO 78 users')

plt.show()

# -------Global metrics for Speaker Isolated--------
mean_acc = np.mean(accuracies_spk)
std_acc = np.std(accuracies_spk)

print("The mean accuracy for the SPK ISO model for 78 users is: ", mean_acc, "; and the std is: ", std_acc)


# -------Global metrics for Adversarial Model--------

mean_acc = np.mean(accuracies_main)
std_acc = np.std(accuracies_main)

mean_f1 = f1_score(y_true_total_main, y_pred_total_main)

print("The mean accuracy for the main MODEL for 78 users is: ", mean_acc, "; and the std is: ", std_acc)
print("The mean f1-score for the main MODEL for 78 users is: ", mean_f1)

# Confusion matrix
cm = confusion_matrix(y_true_total_main, y_pred_total_main)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predictions')
plt.ylabel('True Labels')
plt.title('Confusion Matrix main MODEL')

plt.show()


# -------Global metrics for Unlearnt Speaker--------
mean_acc = np.mean(accuracies_unlspk)
std_acc = np.std(accuracies_unlspk)

print("The mean accuracy for the NEW SPK model for 78 users is: ", mean_acc, "; and the std is: ", std_acc)

