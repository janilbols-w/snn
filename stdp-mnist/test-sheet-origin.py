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
op_plot_figure = True
op_timer = True






firing_period = 10.0    # (ms) interval between spikes
n_neurons = 784                  # number of synapses / number of presynaptic neurons
bach_size = 10
sample_period = 100 # (ms) for each img sample
simulate_time = bach_size * sample_period # regard as
spike_thresh = 0.3 # when image pixel value greater than spike_thresh
delay = 0.01              # (ms) synaptic time delay, default=3.0
min_delay = 0.01
max_delay = 0.1
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

sim.setup(timestep=0.01, min_delay=min_delay, max_delay=max_delay)


# Load MNIST data
print "loading MNIST data..."
TIMER_START = time.time()
dh = DataHandler(randseed=19023895)
dh.load_data()
dh.shuffle_data()
train_data, train_label, test_data, test_label, train_one_hot, test_one_hot = dh.get_data_nparray()
TIMER_END = time.time()
print "loaded!"
print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)
# generate spike table
print "generating spike table..."
TIMER_START = time.time()
init_test_img_spike_table, test_img_spike_table = generateDataMaskedSpikes(data=test_data,
                                                                             sample_period=sample_period,
                                                                             firing_period=firing_period,
                                                                             thresh=spike_thresh)
#spike_tabel = generatePoissonSpikes(t_stop = t_stop, n_neurons = n_neurons, period = firing_period)
img_spike_sequence_generator = build_spike_sequences(test_img_spike_table)
TIMER_END = time.time()
print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)

# presynaptic population
print "buildup neural network..."
TIMER_START = time.time()
p_in = sim.Population(n_neurons, sim.SpikeSourceArray(spike_times=img_spike_sequence_generator),
                    label="presynaptic")
p_mid_1 = sim.Population(n_neurons, sim.IF_cond_exp(**cell_parameters),
                    initial_values={"v": cell_parameters["v_reset"]}, label="postsynaptic_1")

stdp_model = sim.STDPMechanism(
                timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                                    A_plus=0.01, A_minus=0.012),
                weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.0000001),
                weight=0.00000005,
                delay=delay,
                dendritic_delay_fraction=float(op_dendritic_delay_fraction))
connections_1 = sim.Projection(p_in, p_mid_1, sim.OneToOneConnector(), 
                               sim.StaticSynapse(weight=10.0, delay=delay))
TIMER_END = time.time()
print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)

p_in.record('spikes')
p_mid_1.record(['spikes', 'v'])

print("start simulating! ...")
TIMER_START = time.time()
sim.run(simulate_time)


presynaptic_data = p_in.get_data().segments[0]
postsynaptic_data_1 = p_mid_1.get_data().segments[0]
TIMER_END = time.time()
print("Post-synaptic 1 spike times: %s" % postsynaptic_data_1.spiketrains[0])
print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)

#plotNeuronsSpikes(test_img_spike_table)
filename = normalized_filename("Results", "my_stdp", "pkl", op_simulator)
if op_plot_figure:
    TIMER_START = time.time()
    from pyNN.utility.plotting import Figure, Panel, DataTable
    figure_filename = filename.replace("pkl", "png")
    print "plotting ", figure_filename
    Figure(
        # raster plot of the presynaptic neuron spike times
        Panel(presynaptic_data.spiketrains,
              yticks=True, markersize=0.2, xlim=(0, simulate_time)),
        # membrane potential of the postsynaptic neuron 1
        Panel(postsynaptic_data_1.filter(name='v')[0],
              ylabel="Membrane-1 potential (mV)",
              data_labels=[p_mid_1.label], yticks=True, xlim=(0, simulate_time)),
        title="MY - STDP",
        annotations="Simulated with %s" % op_simulator.upper(),
        size = (16,8)
    ).save(figure_filename)
    TIMER_END = time.time()
    print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)
    
    
# === Clean up and quit ========================================================
sim.end()


TIMER_GLOBAL_END = time.time()
print "Simulation Ends! - Total Time Usage: %.5f"%(TIMER_GLOBAL_END-TIMER_GLOBAL_START)