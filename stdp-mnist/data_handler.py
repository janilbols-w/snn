import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from pyNN.parameters import Sequence
import neo
from quantities import ms


# %matplotlib inline
class LargeScaleList:

    def __init__(self, n_neurons = 784):
        self.data = np.zeros((n_neurons,))
        self.capacity = n_neurons
        self.size = 0

    def add(self, x):
        x_len = len(x)
        if x_len + self.size >= self.capacity:
            self.capacity *= 4
            newdata = np.zeros((self.capacity,))
            newdata[:self.size] = self.data[:self.size]
            self.data = newdata
        self.data[self.size:self.size+x_len] = x
        self.size += x_len

    def finalize(self,shape=5):
        data = self.data[:self.size]
        data = np.reshape(data, (-1,shape))
        return np.transpose(data)


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
        
        self.sample_period = -1
        self.firing_period = -1
        self.batch_size = -1
        self.t_stop = -1
        
        self.train_in_spikes = np.array([])
        self.train_out_spikes = np.array([])
        self.test_in_spikes = np.array([])
        self.test_out_spikes = np.array([])
        
    def load_data(self, flag_From_Pickle = True, flag_normalize = False, flag_Debug = True, flag_save_pickle = True):
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
        no_of_different_labels = 10 # MNIST output dimension
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
        train_data = np.loadtxt(self.data_path + "mnist_train.csv", 
                        delimiter=",")
        if flag_Debug:
            print("Data Loading: Test")
        test_data = np.loadtxt(self.data_path + "mnist_test.csv", 
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
        train_labels_one_hot = (lr==self.train_label).astype(np.float)
        test_labels_one_hot = (lr==self.test_label).astype(np.float)
        # we don't want zeroes and ones in the labels neither:
        
        if flag_normalize:
            train_labels_one_hot[train_labels_one_hot==0] = 0.01
            train_labels_one_hot[train_labels_one_hot==1] = 0.99
            test_labels_one_hot[test_labels_one_hot==0] = 0.01
            test_labels_one_hot[test_labels_one_hot==1] = 0.99

        self.train_label_one_hot = train_labels_one_hot
        self.test_label_one_hot = test_labels_one_hot
        
        if flag_save_pickle:
            with open("%s" % self.data_path + "pickled_mnist.pkl", "w") as fh:
                data = (self.train_data, 
                        self.test_data, 
                        self.train_label,
                        self.test_label,
                        self.train_label_one_hot,
                        self.test_label_one_hot)
                pickle.dump(data, fh)
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
    
    
    
    def display_ID(self, data_type, data_id=0):
        '''
            data_type should be 'train' or 'test'
        '''
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
    # end of display_ID()
    
    def generate_spikes(self, data_type, sample_period, firing_period, thresh, flag_rebuild = False):
        '''
            if flag_rebuild==False:
                try(return the data that built)
            return in_spikes, out_spikes
        '''
        if data_type == 'train':
            if not flag_rebuild:
                if len(self.train_in_spikes)>0 and len(self.train_out_spikes)>0:
                    return self.train_in_spikes, self.train_out_spikes
            in_set = self.train_data
            out_set = self.train_label_one_hot
        elif data_type == 'test':
            if not flag_rebuild:
                if len(self.train_in_spikes)>0 and len(self.train_out_spikes)>0:
                    return self.train_in_spikes, self.train_out_spikes
            in_set = self.test_data
            out_set = self.test_label_one_hot
        else:
            print("unknown data type: should be `train` or `test`")
            return  np.array([]), np.array([])
        
        _, in_spikes = generateDataMaskedSpikes(data=in_set,
                                                 sample_period=sample_period,
                                                 firing_period=firing_period,
                                                 thresh=thresh)
        _, out_spikes = generateDataMaskedSpikes(data=out_set,
                                                 sample_period=sample_period,
                                                 firing_period=firing_period,
                                                 thresh=thresh)
        self.sample_period = sample_period
        self.firing_period =  firing_period
        
        if data_type == 'train':
            self.train_in_spikes = in_spikes
            self.train_out_spikes = out_spikes
        if data_type == 'test':
            self.test_in_spikes = in_spikes
            self.test_out_spikes = out_spikes
        return in_spikes, out_spikes
    # end of generate_spikes()
    
    def get_batch_spikes(self, data_type, i_batch, batch_size=-1, sample_period=-1):
        '''
            data_type should be 'train' or 'test'
        '''
        if batch_size < 0:
            if self.batch_size>0:
                batch_size = self.batch_size
            else:
                print("* batch_size sould be greater than 0, or you should initial it at first!")
                return np.array([]), np.array([])
        
        if sample_period < 0:
            if self.sample_period>0:
                sample_period = self.sample_period
            else:
                print("* sample_period sould be greater than 0, or your spike sequence is not generated!")
                return np.array([]), np.array([])
        
        if data_type == 'train':
            if len(self.train_data)<=i_batch*batch_size:
                print("Out of range, batch not found in Train-set!")
                return np.array([]), np.array([])
            len_in = np.shape(self.train_in_spikes)[0]
            in_spikes = self.train_in_spikes
            len_out = np.shape(self.train_out_spikes)[0]
            out_spikes = self.train_out_spikes
        elif data_type == 'test':
            if len(self.test_data)<=i_batch*batch_size:
                print("Out of range, batch not found in Test-set!")
                return np.array([]), np.array([])
            len_in = np.shape(self.test_in_spikes)[0]
            in_spikes = self.test_in_spikes
            len_out = np.shape(self.test_out_spikes)[0]
            out_spikes = self.test_out_spikes
        else:
            print("Unknown Data Type!")
            return np.array([]), np.array([])
        time_start = i_batch*batch_size*sample_period
        in_spikes = in_spikes-time_start
        in_spikes = in_spikes.reshape(len_in,-1)
        out_spikes = out_spikes-time_start
        out_spikes = out_spikes.reshape(len_out,-1)
        
        in_spikes[in_spikes<=0] = np.max(in_spikes) + batch_size*sample_period
        out_spikes[out_spikes<=0] = np.max(out_spikes) + batch_size*sample_period
        return np.sort(in_spikes), np.sort(out_spikes)
        #return in_spikes, out_spikes
        #return in_spikes[in_spikes>0], out_spikes[out_spikes>0]
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
    def get_batch_size(self):
        return self.batch_size
    
    def set_sample_period(self, sample_period):
        self.sample_period = sample_period
        
    def get_sample_period(self):
        return self.sample_period
    
    def set_firing_period(self, firing_period):
        self.firing_period = firing_period
        
    def get_firing_period(self):
        return self.firing_period
    
    def get_sample_size(self, data_type):
        if data_type == 'train':
            return np.shape(self.train_data)[0]
        elif data_type == 'test':
            return np.shape(self.test_data)[0]
        else:
            print("unknown data_type! please use 'train' or 'test'")
            return -1
    # end of get_batch_spikes
        
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
    # iniindex 10000 is out of bounds fot
    poisson_sp_handler = LargeScaleList(n_neurons=n_neurons)
    #poisson_spike_time = np.array([])
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
    poisson_sp_handler.add(spike_time)    
    # poisson_spike_time = np.append(poisson_spike_time,spike_time,axis=0)
    if flag_Debug:
        print "init:\n", spike_time
    # simulate till t_stop
    PERCENTAGE_4_MONITOR = 0 #used for monitoring process
    MONITOR_STEP = 5 #used for monitoring process
    while min_spike_time < t_stop:
    #make sure all the neurons spikes until stop time
        if (min_spike_time/t_stop*100 - PERCENTAGE_4_MONITOR)>= MONITOR_STEP:
            print "generate poisson spike till %.1f in %.1f : %.3f"%(min_spike_time, t_stop, min_spike_time/t_stop*100),"%"
            PERCENTAGE_4_MONITOR = int(min_spike_time/t_stop*100)
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
        poisson_sp_handler.add(spike_time)         
    return poisson_sp_handler.finalize(n_neurons) #poisson_spike_time


def generateDataMaskedSpikes(data, sample_period, firing_period, thresh = 0.2, flag_Debug = False):
    '''
        data format:
            np.array(n_sample,n_element)
    '''
    print "func generateDataMaskedSpikes: data shape - ", np.shape(data)
    n_samples, n_neurons = np.shape(data)[:2]
    if flag_Debug:
        print "n_samples = ", n_samples
        print "n_neurons = ",n_neurons
    t_stop = n_samples * sample_period
    print "n_samples = ", n_samples
    print "n_neurons = ", n_neurons
    print "t_stop = ", t_stop
    print "sample_period = ", sample_period
    print "firing_period = ", firing_period
    
    print "generate random poisson spikes..."
    init_spike_table = generatePoissonSpikes(t_stop = t_stop, n_neurons = n_neurons, period = firing_period)
    init_spike_table[init_spike_table>=t_stop] = 0
    
    print "applying data-mask on ..."
    # init empty masked_spike_table
    masked_spike_table = np.zeros(np.shape(init_spike_table))
    
    PERCENTAGE_4_MONITOR = 0 #used for monitoring process
    MONITOR_STEP = 5 #used for monitoring process
    for i_neuron in np.arange(n_neurons):
        if flag_Debug:
            print "=============================================="
            print "----------------- i_neuron ------------------- ", i_neuron
            print "=============================================="
        # convert spike time to sample id
        if (i_neuron*1.0/n_neurons*100 - PERCENTAGE_4_MONITOR)>= MONITOR_STEP:
            print "masking spikes for neuron-",i_neuron, " in ", n_neurons, "%.3f"%(i_neuron*1.0/n_neurons*100), "%"
            PERCENTAGE_4_MONITOR = int(i_neuron*1.0/n_neurons*100)
        id_sample = [ int(spike_time/sample_period) for spike_time in init_spike_table[i_neuron]] 
        if flag_Debug:
            print(id_sample)
        # thresh the spike into binary {0,1}
        binary_mask = (data[id_sample,i_neuron] > thresh) * 1
        if flag_Debug:
            print(data[id_sample,i_neuron])
            print(binary_mask)

        masked_spike_table[i_neuron] = binary_mask * init_spike_table[i_neuron]
    
    # TODO
    # make all the time-zero spikes have no effect for the simulation
    masked_spike_table[masked_spike_table<=0] =  t_stop + 10*sample_period
    print np.shape(masked_spike_table)
    return init_spike_table, np.sort(masked_spike_table)

def build_spike_sequences(spike_time_table):
    def spike_time_gen(i):
        """Spike time generator. `i` should be an array of indices."""
        #print "spike_time_gen i:\n",i
        return [Sequence(spike_time_table[j]) for j in i]
    return spike_time_gen


class WeightRecorder(object):
    """
    ---->  Copied from official tutorial 'simple_STDP.py'  <----
    Recording of weights is not yet built in to PyNN, so therefore we need
    to construct a callback object, which reads the current weights from
    the projection at regular intervals.
    """
    def __init__(self, sampling_interval, projection):
        self.interval = sampling_interval
        self.projection = projection
        self._weights = []

    def __call__(self, t):
        self._weights.append(self.projection.get('weight', format='list', with_address=False))
        return t + self.interval

    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms,
                                  name="weight")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._weights[0])))
        return signal