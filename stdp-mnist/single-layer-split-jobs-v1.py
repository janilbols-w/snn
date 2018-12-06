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

TIMER_GLOBAL_START = time.time()

op_simulator = "neuron"
op_debug = False
op_dendritic_delay_fraction = False
op_fit_curve = False
op_plot_figure = True # False #
op_timer = True

sim_timestep = 0.5      # (ms) timestep for simulator
firing_period = 10.0    # (ms) interval between spikes
n_neurons = 784         # number of input layer synapses (MNIST image dimension)
n_out = 10              # number of output layer synapses 
batch_size = 80
sample_period = 10 # (ms) for each img sample
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

#sim = import_module("pyNN."+op_simulator)

if op_debug: 
    init_logging(None, debug=True)



# Load MNIST data
print "loading MNIST data..."
TIMER_START = time.time()
dh = DataHandler(randseed=19023895)
dh.load_data()
dh.shuffle_data()
sample_num = dh.get_sample_size('test')
batch_num = int(sample_num/batch_size)
TIMER_END = time.time()
print "loaded!"
print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)
# generate spike table
print "generating spike table..."
TIMER_START = time.time()
_, _ = dh.generate_spikes(data_type='test',
                                            sample_period=sample_period,
                                            firing_period=firing_period,
                                            thresh=spike_thresh)
TIMER_END = time.time()
print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)


# === Start simulating! ======================================================
def report_time(t):
    # Do Nothing
    return t + batch_time
6
print("start simulating! ...")
print("\tTotal simulation time: %d"%(sample_period * sample_num))
AVG_BATCH_TIME_USAGE = 0
i_batch = 0
flag_init = True


# =============================================================================
# ===                           in loop                                     ===
#==============================================================================
while(i_batch<batch_num):
    TIMER_START = time.time()
    print("i_batch: %d in %d"% (i_batch, batch_num))
    # update spike sequence
    in_spikes, out_spikes = dh.get_batch_spikes(data_type='test', i_batch=i_batch, 
                                                batch_size=batch_size, sample_period=sample_period)
    in_spike_sequence_generator = build_spike_sequences(in_spikes)
    out_spike_sequence_generator = build_spike_sequences(out_spikes)
    
    print "\t spike_sequence shape:", np.shape(in_spikes), np.shape(out_spikes)
    # reset simulator
    print("\t - reseting simulator")
    sim = import_module("pyNN."+op_simulator)
    sim.setup(sim_timestep=0.3, min_delay="auto")
    p_in_drive = sim.Population(n_neurons, sim.SpikeSourceArray(spike_times=in_spike_sequence_generator),
                        label="in_drive")
    p_in = sim.Population(n_neurons, sim.IF_cond_exp(**cell_parameters),
                        initial_values={"v": cell_parameters["v_reset"]}, label="in")
    p_out = sim.Population(n_out, sim.IF_cond_exp(**cell_parameters),
                        initial_values={"v": cell_parameters["v_reset"]}, label="out")

    p_out_drive = sim.Population(n_out, sim.SpikeSourceArray(spike_times=out_spike_sequence_generator),
                        label="out_drive")
    stdp_model = sim.STDPMechanism(
                timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                                    A_plus=0.01, A_minus=0.012),
                weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.0000001),
                weight=  0.00000005, # 0.05, #
                delay=delay,
                dendritic_delay_fraction=float(op_dendritic_delay_fraction))

    # Define Connections
    connections_in_drive = sim.Projection(p_in_drive, p_in, sim.OneToOneConnector(), 
                                   sim.StaticSynapse(weight=10.0, delay=delay))

    connections_mid = sim.Projection(p_in, p_out, sim.AllToAllConnector(), stdp_model)

    connections_out_drive = sim.Projection(p_out_drive, p_out, sim.OneToOneConnector(), 
                                   sim.StaticSynapse(weight=10.0, delay=delay))
    if not flag_init:
        connections_mid.set(weight=final_weights)

    p_in_drive.record('spikes')
    p_in.record( ['spikes'])
    p_out.record(['spikes'])
    p_out_drive.record('spikes')
    weight_recorder = WeightRecorder(sampling_interval=1.0, projection=connections_mid)

    
    # Run simulator
    print("\t - start simulator")
    sim.run(batch_time, callbacks = [report_time, weight_recorder])
        
    # === Collect data ==============================================================
    p_in_drive_data = p_in_drive.get_data().segments[0]
    p_in_data = p_in.get_data().segments[0]
    p_out_data = p_out.get_data().segments[0]
    p_out_drive_data = p_out_drive.get_data().segments[0]

    weights = weight_recorder.get_weights()
    final_weights = np.array(weights[-1])
    # === Clean up Simulator ========================================================
    sim.end()
    print("\t - simulation ends")
    
    # if plotting
    
    # === Plotting ==================================================================
    #plotNeuronsSpikes(test_img_spike_table)
    filename = normalized_filename("Results", "my_stdp", "pkl", op_simulator)
    if op_plot_figure and i_batch%500==499 and False:
        from pyNN.utility.plotting import Figure, Panel, DataTable
        figure_filename = filename.replace("pkl", "png")
        print "\t - plotting ", figure_filename
        Figure(
            # p_in_drive spikes
            Panel(p_in_drive_data.spiketrains,
                  yticks=True, markersize=0.2, xlim=(0, batch_time)),
            # p_in spikes
            Panel(p_in_data.spiketrains,
                  yticks=True, markersize=0.2, xlim=(0, batch_time)),
            # p_out spikes
            Panel(p_out_data.spiketrains,
                  yticks=True, markersize=0.2, xlim=(0, batch_time)),
            # p_out_drive spikes
            Panel(p_out_drive_data.spiketrains,
                  yticks=True, markersize=0.2, xlim=(0, batch_time)),
            # weights
            Panel(weights, xticks=True, yticks=True, xlabel="Time (ms)",
                  legend=False, xlim=(0, batch_time)),
            title="MY - STDP",
            #annotations="Simulated with %s" % op_simulator.upper(),
            size = (16,8)
        ).save(figure_filename)
    
    
    progress = i_batch*1.0/batch_num*100
    if int(progress)%10== 0:
        npz_filename = normalized_filename("Results", "my_stdp_"+"%.2f"%progress, "npz", op_simulator)
        np.savez(npz_filename, final_weights=final_weights)
        
        
    
    # timer
    TIMER_END = time.time()
    AVG_BATCH_TIME_USAGE = (AVG_BATCH_TIME_USAGE*i_batch/(i_batch+1)+(TIMER_END-TIMER_START)/(i_batch+1))
    EXPECT_TIME_LEFT = AVG_BATCH_TIME_USAGE * (batch_num-i_batch-1)
    print("\t Batch - %d in %d - \t%.2f" % (i_batch, batch_num, progress) + "%")
    print("\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)+",\t remain time - %.5f"%EXPECT_TIME_LEFT)
    
    
    # release memory
    del p_in_drive, p_in, p_out, p_out_drive, in_spikes, out_spikes, in_spike_sequence_generator, out_spike_sequence_generator
    del connections_in_drive, connections_mid, connections_out_drive, sim
    del filename
    # update loop parameters
    flag_init = False
    i_batch += 1

# =============================================================================
# ===                           end loop                                    ===
#==============================================================================

TIMER_GLOBAL_END = time.time()
print "Simulation Ends! - Total Time Usage: %.5f"%(TIMER_GLOBAL_END-TIMER_GLOBAL_START)

filename = normalized_filename("Results", "my_stdp", "npz", op_simulator)
np.savez(npz_filename, final_weights=final_weights)
