if True:
    i_batch = np.random.randint(batch_num)
    print "i_batch = %d"%i_batch
    TIMER_START = time.time()
    from pyNN.utility.plotting import Figure, Panel, DataTable
    figure_filename = normalized_filename("Results", "my_stdp", "png", op_simulator)
    print "plotting ", figure_filename
    Figure(
        # p_in_drive spikes
        Panel(p_in_drive_data.spiketrains,
              yticks=True, markersize=0.2, xlim=(batch_time*i_batch, batch_time*(i_batch+1))),
        # p_in spikes
        Panel(p_in_data.spiketrains,
              yticks=True, markersize=0.2, xlim=(batch_time*i_batch, batch_time*(i_batch+1))),
        # p_out spikes
        Panel(p_out_data.spiketrains,
              yticks=True, markersize=0.2, xlim=(batch_time*i_batch, batch_time*(i_batch+1))),
        # p_out_drive spikes
        Panel(p_out_drive_data.spiketrains,
              yticks=True, markersize=0.2, xlim=(batch_time*i_batch, batch_time*(i_batch+1))),
        # weights
        Panel(weights, xticks=True, yticks=True, xlabel="Time (ms)",
              legend=False, xlim=(batch_time*i_batch, batch_time*(i_batch+1))),
        title="MY - STDP",
        size = (16,8)
    )
    TIMER_END = time.time()
    print "\t - Time Usage: %.5f"%(TIMER_END-TIMER_START)