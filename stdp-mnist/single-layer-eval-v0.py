# -*- coding: utf-8 -*-
from math import exp
import neo
from quantities import ms
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.utility.plotting import DataTable
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
from data_handler import *

import time
print("v-0")
TIMER_GLOBAL_START = time.time()

op_simulator = "neuron" # 'brian' #
op_debug = False
op_dendritic_delay_fraction = False
op_fit_curve = False
op_plot_figure = True #False #
op_timer = True

sim_timestep = 0.01      # (ms) timestep for simulator
firing_period = 0.1    # (ms) interval between spikes
n_neurons = 784         # number of input layer synapses (MNIST image dimension)
n_out = 10              # number of output layer synapses 
batch_size = 10
sample_period = 1.0 # (ms) for each img sample
sample_num = 0 # to be updated after loading data
batch_time = batch_size * sample_period # each simulate  simulation duration
batch_num = 0 # to be updated after loading data
spike_thresh = 0.3 # when image pixel value greater than spike_thresh
delay = sim_timestep * 2              # (ms) synaptic time delay, default=3.0
min_delay = sim_timestep
max_delay = sim_timestep * 5
cell_parameters = {
    "tau_m": 10.0,       # (ms)
    "v_thresh": -50.0,   # (mV)
    "v_reset": -60.0,    # (mV)
    "v_rest": -60.0,     # (mV)
    "cm": 1.0,           # (nF)
    "tau_refrac": firing_period / 2,  # (ms) long refractory period to prevent bursting
}


sim = import_module("pyNN."+op_simulator)

if op_debug: 
    init_logging(None, debug=True)

sim.setup(sim_timestep=sim_timestep, min_delay="auto")
#sim.setup(sim_timestep=sim_timestep, min_delay=min_delay, max_delay=max_delay)


# Load MNIST data
print "loading MNIST data..."
TIMER_START = time.time()
dh = DataHandler(randseed=19023895)
dh.load_data()
dh.shuffle_data()
train_data, train_label, test_data, test_label, train_one_hot, test_one_hot = dh.get_data_nparray()
#sample_num = np.shape(train_data)[0]
sample_num = np.shape(test_data)[0]
batch_num = int(sample_num/batch_size)
TIMER_END = time.time()
print "loaded!"
print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)
# generate spike table
print "generating spike table..."
TIMER_START = time.time()
init_test_in, test_in = generateDataMaskedSpikes(data=test_data,
                                                 sample_period=sample_period,
                                                 firing_period=firing_period,
                                                 thresh=spike_thresh)
init_test_out, test_out = generateDataMaskedSpikes(data=test_one_hot,
                                                 sample_period=sample_period,
                                                 firing_period=firing_period,
                                                 thresh=spike_thresh)
#spike_tabel = generatePoissonSpikes(t_stop = t_stop, n_neurons = n_neurons, period = firing_period)
in_spike_sequence_generator = build_spike_sequences(test_in)
out_spike_sequence_generator = build_spike_sequences(test_out)
TIMER_END = time.time()
print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)

# presynaptic population
print "simulator: " + op_simulator
print "buildup neural network..."
TIMER_START = time.time()
p_in_drive = sim.Population(n_neurons, sim.SpikeSourceArray(spike_times=in_spike_sequence_generator),
                    label="in_drive")
p_in = sim.Population(n_neurons, sim.IF_cond_exp(**cell_parameters),
                    initial_values={"v": cell_parameters["v_reset"]}, label="in")
p_out = sim.Population(n_out, sim.IF_cond_exp(**cell_parameters),
                    initial_values={"v": cell_parameters["v_reset"]}, label="out")

p_out_drive = sim.Population(n_out, sim.SpikeSourceArray(spike_times=out_spike_sequence_generator),
                    label="out_drive")

stdp_model = sim.STDPMechanism(
                #timing_dependence=sim.SpikePairRule(tau_plus=15.0, tau_minus=15.0,
                timing_dependence=sim.SpikePairRule(tau_plus=firing_period, tau_minus=firing_period,
                                                    A_plus=0.01, A_minus=0.012),
                weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.0000001),
                weight=0.00000005,
                delay=delay,
                dendritic_delay_fraction=float(op_dendritic_delay_fraction))

# Define Connectionsneuron
connections_in_drive = sim.Projection(p_in_drive, p_in, sim.OneToOneConnector(), 
                               sim.StaticSynapse(weight=10.0, delay=delay))

connections_mid = sim.Projection(p_in, p_out, sim.AllToAllConnector(), stdp_model)

connections_out_drive = sim.Projection(p_out_drive, p_out, sim.OneToOneConnector(), 
                               sim.StaticSynapse(weight=10.0, delay=delay))
#connections_out_drive = sim.Projection(p_out, p_out_drive, sim.OneToOneConnector(), sim.StaticSynapse(weight=10.0, delay=delay))
TIMER_END = time.time()

print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)

p_in_drive.record('spikes')
p_in.record( ['spikes'])
p_out.record(['spikes'])
p_out_drive.record('spikes')

weight_recorder = WeightRecorder(sampling_interval=1.0, projection=connections_mid)

# Start simulating!

def report_time(t):
    i = int(t/batch_time)
    return t + batch_time

print("start simulating! ...")
print("\tTotal simulation time: %d"%(sample_period * sample_num))
AVG_BATCH_TIME_USAGE = 0

#batch_num = 2 # reset for debugging!!!!! REMOVE WHEN RUN FULL TEST
for i in range(batch_num):
    TIMER_START = time.time()
    sim.run(batch_time, callbacks = [report_time, weight_recorder])
    TIMER_END = time.time()
    AVG_BATCH_TIME_USAGE = (AVG_BATCH_TIME_USAGE*i/(i+1)+(TIMER_END-TIMER_START)/(i+1))
    EXPECT_TIME_LEFT = AVG_BATCH_TIME_USAGE * (batch_num-i-1)
    print("\t Batch - %d in %d - \t%.2f" % (i, batch_num, i*1.0/batch_num*100) + "%")
    print("\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)+",\t remain time - %.5f"%EXPECT_TIME_LEFT)
print("Average Batch Time Usage: %5f"%AVG_BATCH_TIME_USAGE)
# Collect data
p_in_drive_data = p_in_drive.get_data().segments[0]
p_in_data = p_in.get_data().segments[0]
p_out_data = p_out.get_data().segments[0]
p_out_drive_data = p_out_drive.get_data().segments[0]

weights = weight_recorder.get_weights()
final_weights = np.array(weights[-1])
# === Clean up Simulator ========================================================
sim.end()

# === Plotting ==================================================================
#plotNeuronsSpikes(test_img_spike_table)
filename = normalized_filename("Results", "my_stdp", "pkl", op_simulator)
if op_plot_figure:
    TIMER_START = time.time()
    from pyNN.utility.plotting import Figure, Panel, DataTable
    figure_filename = filename.replace("pkl", "png")
    print "plotting ", figure_filename
    Figure(
        # p_in_drive spikes
        Panel(p_in_drive_data.spiketrains,
              yticks=True, markersize=0.2, xlim=(batch_time*(batch_num-1), batch_time*batch_num)),
        # p_in spikes
        Panel(p_in_data.spiketrains,
              yticks=True, markersize=0.2, xlim=(batch_time*(batch_num-1), batch_time)),
        # p_out spikes
        Panel(p_out_data.spiketrains,
              yticks=True, markersize=0.2, xlim=(batch_time*(batch_num-1), batch_time)),
        # p_out_drive spikes
        Panel(p_out_drive_data.spiketrains,
              yticks=True, markersize=0.2, xlim=(batch_time*(batch_num-1), batch_time)),
        # weights
        Panel(weights, xticks=True, yticks=True, xlabel="Time (ms)",
              legend=False, xlim=(batch_time*(batch_num-1), batch_time)),
        title="MY - STDP",
        annotations="Simulated with %s" % op_simulator.upper(),
        size = (16,8)
    ).save(figure_filename)
    TIMER_END = time.time()
    print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)




TIMER_GLOBAL_END = time.time()
print "Simulation Ends! - Total Time Usage: %.5f"%(TIMER_GLOBAL_END-TIMER_GLOBAL_START)



npz_filename = normalized_filename("Results", "single_layer_v0_weights", "npz", op_simulator)
np.savez(npz_filename, final_weights=final_weights)
