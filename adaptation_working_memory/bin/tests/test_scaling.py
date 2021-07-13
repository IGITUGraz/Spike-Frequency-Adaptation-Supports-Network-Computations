import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy.random as rd
from lsnn.spiking_models import ALIF
from lsnn.guillaume_toolbox.matplotlib_extension import raster_plot

def test_dt_scaling():
    n_in = 5
    n_rec = 10
    n_input_spikes = 10
    p0 = 0.2 # connectivity


    T = 200 # in ms
    tau_delays = 5 # in ms
    tau_ref = 5 # in ms
    tau_a = 100 # in ms

    input_spike_trains = [(spike_ind, spike_t) for spike_ind, spike_t in
                          zip(rd.choice(n_in, n_input_spikes), rd.rand(n_input_spikes) * T)]

    w_in_var = rd.rand(n_in,n_rec) / np.sqrt(n_in) * (rd.rand(n_in,n_rec) < p0)
    w_rec_var = rd.rand(n_rec,n_rec) / np.sqrt(n_in) * (rd.rand(n_rec,n_rec) < p0)

    w_in_tau_delay = rd.choice(tau_delays,size=(n_in,n_rec)).reshape(n_in,n_rec) * tau_delays
    w_rec_tau_delay = rd.choice(tau_delays,size=(n_rec,n_rec)).reshape(n_rec,n_rec) * tau_delays

    dt_list = [1.,1.,0.25,0.1,0.025,0.01]
    out_list = []
    in_list = []
    b_list = []
    out_spike_times_list = []

    for dt in dt_list:

        w_in_delay = np.int_(w_in_tau_delay / dt)
        w_rec_delay = np.int_(w_rec_tau_delay / dt)
        n_refractory = int(tau_ref / dt)

        n_delay = max(np.max(w_in_delay),np.max(w_rec_delay))+1
        print('------')
        print('dt',dt)
        print('n_refractory',n_refractory)
        print('n_delay',n_delay)

        alif = ALIF(n_in=n_in,n_rec=n_rec,dt=dt,n_refractory=n_refractory,n_delay=n_delay,tau_adaptation=tau_a)

        n_time = int(T / dt)
        inputs_np = np.zeros(shape=(1,n_time,n_in))

        for i,t in input_spike_trains:
            n_t = int(t/dt)
            inputs_np[0,n_t,i] += 1/dt

        inputs = tf.constant(inputs_np,dtype=tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(alif,inputs,dtype=tf.float32)
        z,b = outputs

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        sess.run(tf.assign(alif.w_in_var,tf.cast(w_in_var,dtype=tf.float32)))
        sess.run(tf.assign(alif.w_rec_var,tf.cast(w_rec_var,dtype=tf.float32)))
        sess.run(tf.assign(alif.w_in_delay,tf.cast(w_in_delay,dtype=tf.int32)))
        sess.run(tf.assign(alif.w_rec_delay,tf.cast(w_rec_delay,dtype=tf.int32)))

        z,b = sess.run([z,b])
        z = z[0]
        b = b[0]

        output_spike_times = np.where(z)
        output_spike_times = [(i,t * dt) for i,t in zip(*list(output_spike_times))]

        in_list.append(inputs_np[0])
        out_list.append(z)
        b_list.append(b)
        out_spike_times_list.append(output_spike_times)

    fig,ax_list = plt.subplots(len(dt_list),3,figsize=(10,8))
    [raster_plot(ax,z) for ax,z in zip(ax_list[:,0],in_list)]
    [raster_plot(ax,z) for ax,z in zip(ax_list[:,1],out_list)]
    [ax.plot(b) for ax,b in zip(ax_list[:,2],b_list)]

    ax_list[0,0].set_title('input spikes')
    ax_list[0,1].set_title('recurrent spikes')
    ax_list[0,2].set_title('b_j(t)')

    for dt,ax in zip(dt_list,ax_list[:,2]):
        b_increment = (1 - np.exp(-dt/tau_a)) * 1 / dt
        ax.axhline(y=b_increment,linestyle='dashed',color='black')

    for dt, ax in zip(dt_list, ax_list[:, 0]):
        ax.set_ylabel('dt: {}'.format(dt))

    plt.tight_layout()
    plt.show()

def test_V0_scaling(fix_weight_init=True):
    n_in = 5
    n_rec = 10
    n_input_spikes = 10
    p0 = 0.2 # connectivity
    dt= 1.

    T = 200 # in ms
    n_delay = 5 # in ms
    n_ref = 5 # in ms
    tau_a = 100 # in ms

    input_spike_trains = [(spike_ind, spike_t) for spike_ind, spike_t in
                          zip(rd.choice(n_in, n_input_spikes), rd.rand(n_input_spikes) * T)]

    w_in_var = rd.rand(n_in,n_rec) / np.sqrt(n_in) * (rd.rand(n_in,n_rec) < p0)
    w_rec_var = rd.rand(n_rec,n_rec) / np.sqrt(n_in) * (rd.rand(n_rec,n_rec) < p0)
    w_in_delay = rd.choice(n_delay,size=(n_in,n_rec)).reshape(n_in,n_rec)
    w_rec_delay = rd.choice(n_delay,size=(n_rec,n_rec)).reshape(n_rec,n_rec)

    V0_list = [1.,1000.,1000000]
    out_list = []
    in_list = []
    b_list = []
    v_list = []
    out_spike_times_list = []

    for V0 in V0_list:

        print('------')
        print('V0',V0)

        alif = ALIF(n_in=n_in,n_rec=n_rec,V0=V0,n_refractory=n_ref,n_delay=n_delay,tau_adaptation=tau_a,thr= V0 * 0.01)

        n_time = int(T / dt)
        inputs_np = np.zeros(shape=(1,n_time,n_in))

        for i,t in input_spike_trains:
            n_t = int(t/dt)
            inputs_np[0,n_t,i] += 1/dt

        inputs = tf.constant(inputs_np,dtype=tf.float32)

        outputs, _ = tf.nn.dynamic_rnn(alif,inputs,dtype=tf.float32)
        z,b = outputs

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if fix_weight_init:
            sess.run(tf.assign(alif.w_in_var,tf.cast(w_in_var * V0,dtype=tf.float32)))
            sess.run(tf.assign(alif.w_rec_var,tf.cast(w_rec_var * V0,dtype=tf.float32)))
            sess.run(tf.assign(alif.w_in_delay,tf.cast(w_in_delay,dtype=tf.int32)))
            sess.run(tf.assign(alif.w_rec_delay,tf.cast(w_rec_delay,dtype=tf.int32)))

        z,b = sess.run([z,b])
        z = z[0]
        #v = v[0]
        b = b[0]

        output_spike_times = np.where(z)
        output_spike_times = [(i,t * dt) for i,t in zip(*list(output_spike_times))]

        in_list.append(inputs_np[0])
        out_list.append(z)
        b_list.append(b)
        #v_list.append(v)
        out_spike_times_list.append(output_spike_times)

    fig,ax_list = plt.subplots(len(V0_list),3,figsize=(10,8))
    [raster_plot(ax,z) for ax,z in zip(ax_list[:,0],in_list)]
    [raster_plot(ax,z) for ax,z in zip(ax_list[:,1],out_list)]
    [ax.plot(b) for ax,b in zip(ax_list[:,2],b_list)]
    #[ax.plot(v) for ax,v in zip(ax_list[:,3],v_list)]

    ax_list[0,0].set_title('input spikes')
    init_type_str = 'same' if fix_weight_init else 'random scaled'
    ax_list[0,1].set_title('recurrent spikes (' + init_type_str + ' weight init)')
    ax_list[0,2].set_title('b_j(t)')

    for V0,ax in zip(V0_list,ax_list[:,2]):
        b_increment = (1 - np.exp(-dt/tau_a)) * 1 / dt
        ax.axhline(y=b_increment,linestyle='dashed',color='black')

    for V0, ax in zip(V0_list, ax_list[:, 0]):
        ax.set_ylabel('V0: {}'.format(V0))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_V0_scaling(fix_weight_init=True)
    test_V0_scaling(fix_weight_init=False)
    test_dt_scaling()