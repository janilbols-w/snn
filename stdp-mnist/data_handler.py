import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
# %matplotlib inline

class DataHandler:
    def __init__(self, randseed = 20181121, MNIST_data_path = '/home/janilbols/Code/data/mnist/', image_size = 28, no_of_different_labels = 10):        
        self.randseed = randseed
        self.data_path = MNIST_data_path
        self.image_size = image_size # width and length
        self.no_of_different_labels = no_of_different_labels #  i.e. 0, 1, 2, 3, ..., 9
        self.image_pixels = image_size * image_size
        
        self.train_data = np.array([])
        self.train_label = np.array([])
        self.train_label_one_hot = np.array([])
        self.test_data = np.array([])
        self.test_label = np.array([])
        self.test_label_one_hot = np.array([])
        self.shuffle_train_order = np.array([])
        self.shuffle_test_order = np.array([])
        
    def load_data(self, flag_From_Pickle = True, flag_normalize = False, flag_Debug = True):
        ''' 
            Load MNIST data from MNIST_data_path
            
            Flags:
            - flag_From_Pickle: try to load data from pickle file, else from csv
            - flag_normalize: load data from csv, with normalization
            - flag_Debug: print some debug message
            
            Logic:            
            - If failed, return false
              Else return true
        '''
        flag_okay = False
        
        #-- Load from pickle -------------------------------------------------------
        if flag_From_Pickle:
            if not(os.path.isfile("%s" % self.data_path + "pickled_mnist.pkl")):
                print("File not found:%s" % self.data_path + "pickled_mnist.pkl")
                print("solution: Try open with csv file")
            else:
                if flag_Debug:
                    print("data loading from pickle...")
                with open("%s" % self.data_path + "pickled_mnist.pkl", "r") as fh:
                    data = pickle.load(fh)
                self.train_data = data[0]
                self.test_data = data[1]
                self.train_label = data[2]
                self.test_label = data[3]
                self.train_label_one_hot = data[4]
                self.test_label_one_hot = data[5]
                flag_okay = True
                return flag_okay

        #-- Load from csv file -----------------------------------------------------   
        if not(os.path.isfile('%s' % self.data_path + "mnist_train.csv")):
            print("File not found:%s" % self.data_path + "mnist_train.csv")
            flag_okay = False
            return flag_okay
        if not(os.path.isfile('%s' % self.data_path + "mnist_train.csv")):
            print("File not found:%s" % self.data_path + "mnist_train.csv")
            if_okay = False
            return flag_okay
        
        if flag_Debug:
            print("Data Loading: Train")
        train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
        if flag_Debug:
            print("Data Loading: Test")
        test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 
        if flag_Debug:
            print("Data Normalizing: transfer into [0,1]")
        fac = 255  *0.99 + 0.01
        self.train_data = np.asfarray(train_data[:, 1:]) / fac # train_imgs
        self.test_data = np.asfarray(test_data[:, 1:]) / fac # test_imgs
        self.train_label = np.asfarray(train_data[:, :1]) # train_labels 
        self.test_label = np.asfarray(test_data[:, :1]) # test_labels
        
        lr = np.arange(no_of_different_labels)
        # transform labels into one hot representation
        train_labels_one_hot = (lr==train_labels).astype(np.float)
        test_labels_one_hot = (lr==test_labels).astype(np.float)
        # we don't want zeroes and ones in the labels neither:
        
        if flag_normalize:
            train_labels_one_hot[train_labels_one_hot==0] = 0.01
            train_labels_one_hot[train_labels_one_hot==1] = 0.99
            test_labels_one_hot[test_labels_one_hot==0] = 0.01
            test_labels_one_hot[test_labels_one_hot==1] = 0.99

        self.train_label_one_hot = train_labels_one_hot
        self.test_label_one_hot = test_labels_one_hot
        
        flag_okay = True
        return flag_okay
    
    def shuffle_data(self):
        self.shuffle_train_order = np.arange(len(self.train_data))
        self.shuffle_test_order = np.arange(len(self.test_data))
        np.random.seed(self.randseed)
        np.random.shuffle(self.shuffle_train_order)
        np.random.seed(self.randseed)
        np.random.shuffle(self.shuffle_test_order)
        
        self.train_data = self.train_data[self.shuffle_train_order] 
        self.train_label = self.train_label[self.shuffle_train_order]
        self.test_data = self.test_data[self.shuffle_test_order]
        self.test_label = self.test_label[self.shuffle_test_order]
        self.train_label_one_hot = self.train_label_one_hot[self.shuffle_train_order]
        self.test_label_one_hot = self.test_label_one_hot[self.shuffle_test_order]
        return True
        
    def get_data_nparray(self):
        return self.train_data, self.train_label, self.test_data, self.test_label, self.train_label_one_hot, self.test_label_one_hot
    
    
    
    def displayID(self, data_type ='train', data_id=0):
        if data_type == 'train':
            if len(self.train_data)<=data_id:
                print("Data-ID not found in Train-set!")
                return False
            img = self.train_data[data_id]
            label = self.train_label[data_id]
            one_hot = self.train_label_one_hot
        elif data_type == 'test':
            if len(self.test_data)<=data_id:
                print("Data-ID not found in Test-set!")
                return False
            img = self.test_data[data_id]
            label = self.test_label[data_id]
            one_hot = self.test_label_one_hot
        else:
            print("Unknown Data Type!")
            return False
        img = img.reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.show()
        print(label)
        print(one_hot)
        
        return True

# =========================================================================== #
# End of class DataHandler
# =========================================================================== #    

def plotNeuronsSpikes(spike_table):
    n_neuron = np.shape(spike_table)[0]
    n_spikes = np.shape(spike_table[0])[0]
    labs = np.ones([n_neuron,n_spikes])
    for i in range(n_neuron): 
        labs[i] *= i
    plt.figure(figsize=(16,4))
    plt.plot(spike_table,labs, 'ro')
    plt.show()

def generatePoissonSpikes(t_stop = 100, n_neurons = 10, 
                          delta_steps = 1.0, period = 2.0, norm_sigma = 0.05, flag_Debug = False):
    # init
    poisson_spike_time = np.array([])
    spike_time = np.random.normal(0, period * (1 + norm_sigma), n_neurons)
    if flag_Debug:
        print "origin:\n", spike_time
    min_spike_time = min(spike_time)
    while min_spike_time < 0:
        spike_step = np.random.poisson(lam = delta_steps, size=(n_neurons))
        if flag_Debug:
            print(spike_step)
        id_temp = np.where(spike_time<0)
        spike_time[id_temp] += spike_step[id_temp] * np.random.normal(period, period * norm_sigma ,len(id_temp))
        min_spike_time = min(spike_time)
    poisson_spike_time = np.append(poisson_spike_time,spike_time,axis=0)
    if flag_Debug:
        print "init:\n", spike_time
    # simulate till t_stop
    while min_spike_time < t_stop:
    #make sure all the neurons spikes until stop time
        spike_step = np.random.poisson(lam = delta_steps, size=(n_neurons))
        spike_step[spike_step<1] = 1
        if flag_Debug:
            print(spike_step)
        id_temp = np.where(spike_time>=t_stop)
        spike_step[id_temp] = 0
        spike_time += spike_step * np.random.normal(period, period * norm_sigma ,len(id_temp))
        if flag_Debug:
            print(spike_time)
        min_spike_time = min(spike_time)
        poisson_spike_time = np.append(poisson_spike_time,spike_time,axis=0)
    poisson_spike_time = poisson_spike_time.reshape(-1,n_neurons)
    poisson_spike_time = np.transpose(poisson_spike_time)
    return poisson_spike_time


def generateDataMaskedSpikes(data, sample_period, firing_period, thresh = 0.2, flag_Debug = True):
    '''
        data format:
            np.array(n_sample,n_element)
    '''
    n_samples, n_neurons = np.shape(data)[:2]
    if flag_Debug:
        print "n_samples = ", n_samples
        print "n_neurons = ",n_neurons
    t_stop = n_samples * sample_period
    init_spike_table = generatePoissonSpikes(t_stop = t_stop, n_neurons = n_neurons, period = firing_period)
    init_spike_table[init_spike_table>=t_stop] = 0
    # init empty masked_spike_table
    masked_spike_table = np.zeros(np.shape(init_spike_table))
    for i_neuron in np.arange(n_neurons):
        if flag_Debug:
            print "=============================================="
            print "----------------- i_neuron ------------------- ", i_neuron
            print "=============================================="
        # convert spike time to sample id
        id_sample = [ int(spike_time/sample_period) for spike_time in init_spike_table[i_neuron]] 
        if flag_Debug:
            print(id_sample)
        # thresh the spike into binary {0,1}
        binary_mask = (data[id_sample,i_neuron] > thresh) * 1
        if flag_Debug:
            print(data[id_sample,i_neuron])
            print(binary_mask)
        #id_4_masked = np.where(init_spike_table[i,id_sample])
        masked_spike_table[i_neuron] = binary_mask * init_spike_table[i_neuron]
    return init_spike_table, masked_spike_table

def build_spike_sequences(spike_table):
    def spike_time_gen(i):
        """Spike time generator. `i` should be an array of indices."""
        return [Sequence(spike_table[j]) for j in i]
    return spike_time_gen
