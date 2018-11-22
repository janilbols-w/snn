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