# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:52:18 2016 by emin
"""
import math
import os
import sys
import argparse
import theano
import theano.tensor as T
import numpy as np
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import lasagne.objectives
import lasagne.init
from utils_lb import build_generators, build_model, build_loss, build_performance
import scipy.io as sio
import matplotlib.pyplot as plt
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recurrent Memory Experiment (Basic Condition)')
    parser.add_argument('--task', type=int, default=0, help='Task code')
    parser.add_argument('--model', type=int, default=0, help='Model code')
    parser.add_argument('--lambda_val', type=float, default=0.98, help='lambda (initialization for diagonal terms)')
    parser.add_argument('--sigma_val', type=float, default=0.0, help='sigma (initialization for off-diagonal terms)')
    args = parser.parse_args()

    diag_val    = args.lambda_val
    offdiag_val = args.sigma_val
    m_ind       = args.model
    t_ind       = args.task

    # Models and model-specific parameters
    model_list = [{"model_id":'LeInitRecurrent',"diag_val":diag_val,"offdiag_val":offdiag_val},
                  {"model_id":'LeInitRecurrentWithFastWeights',"diag_val":diag_val,"offdiag_val":offdiag_val,"gamma":0.0007},
                  {"model_id":'OrthoInitRecurrent',"init_val":diag_val},
                  {"model_id":'ResidualRecurrent',"leak_inp":1.0, "leak_hid":1.0},
                  {"model_id":'GRURecurrent',"diag_val":diag_val,"offdiag_val":offdiag_val},
                  {"model_id":'LeInitRecurrentWithLayerNorm', "diag_val": diag_val, "offdiag_val": offdiag_val},
                  ]

    # Tasks and task-specific parameters
    task_list = [{"task_id":'Harvey2012', "n_out":1, "sigtc":15.0, "stim_rate":1.0, "n_loc":1, "out_nonlin":lasagne.nonlinearities.sigmoid},
                 {"task_id":'Julia2015v1', "n_out":1, "sigtc":15.0, "stim_rate":1.0, "n_loc":1, "out_nonlin": lasagne.nonlinearities.sigmoid},
                 {"task_id":'Julia2015v2', "n_out":1, "sigtc":15.0, "stim_rate":1.0, "n_loc":1, "out_nonlin": lasagne.nonlinearities.sigmoid}
                # {"task_id":'COMP', "n_out":1, "n_loc":1, "out_nonlin": lasagne.nonlinearities.sigmoid}
                 ]
    '''
    # Task and model parameters
    ExptDict = {"model": model_list[m_ind],
                "task": task_list[t_ind],
                "tr_cond": 'all_gains',
                "test_cond": 'all_gains',
                "n_hid": 500,
                "n_in": 50,
                "batch_size": 50,
                "stim_dur": 25,
                "delay_dur": 100,
                "resp_dur": 25,
                "kappa": 2.0,
                "spon_rate": 0.1,
                "tr_max_iter": 25001,
                "test_max_iter": 2501}
    '''
   # Task and model parameters
    ExptDict = {"model": model_list[m_ind],
                "task": task_list[t_ind],
                "tr_cond": 'all_gains',
                "test_cond": 'all_gains',
                "n_hid": 200,
                "n_in": 30,
                "batch_size": 60,
                "stim_dur": 10,
                "delay_dur": 100,
                "resp_dur": 10,
                "kappa": 2.0,
                "spon_rate": 0.1,
                "tr_max_iter": 5000,
                "test_max_iter": 40}
    # Build task generators
    generator, test_generator, generator2, test_generator2 = build_generators(ExptDict)

    # Define the input and expected output variable
    input_var, target_var = T.tensor3s('input', 'target')

    # Build the model
    l_out, l_rec = build_model(input_var, ExptDict)

    # The generated output variable and the loss function
    if ExptDict["task"]["task_id"] in ['DE1','DE2','GDE2','VDE1','SINE','Julia2015v1','Julia2015v2']:
        pred_var = lasagne.layers.get_output(l_out)
    elif ExptDict["task"]["task_id"] in ['CD1','CD2','Harvey2012','Harvey2012Dynamic','Harvey2016','COMP']:
        pred_var = T.clip(lasagne.layers.get_output(l_out), 1e-6, 1.0 - 1e-6)

    # Build loss
    rec_act    = lasagne.layers.get_output(l_rec)
    l2_penalty = T.mean(lasagne.objectives.squared_error(rec_act[:,:,:], 0.0)) * 1e-4
    loss       = build_loss(pred_var, target_var, ExptDict) + l2_penalty

    # Create the update expressions
    params       = lasagne.layers.get_all_params(l_out, trainable=True)
    #updates      = lasagne.updates.adam(loss, params, learning_rate=0.0005)
    updates      = lasagne.updates.adam(loss, params, learning_rate=0.0005)

    # Compile the function for a training step, as well as the prediction function
    # and a utility function to get the inner details of the RNN
    train_fn     = theano.function([input_var, target_var], loss, updates=updates,
                                   allow_input_downcast=True)
    pred_fn      = theano.function([input_var], pred_var, allow_input_downcast=True)
    rec_layer_fn = theano.function([input_var], lasagne.layers.get_output(l_rec,
                                   get_details=True), allow_input_downcast=True)

    # TRAINING
    net_snap, opt_snap, s_vec, opt_vec, net_vec, infloss_vec = [], [], [], [], [], []
    for i, (example_input, example_output, s, opt) in generator:
        score = train_fn(example_input, example_output)
        net   = pred_fn(example_input)
        s_vec.append(s)
        opt_vec.append(opt)
        net_vec.append(np.squeeze(net))
       # if i % 500 == 0:
        if i % 100 == 0:
            opt_vec = np.asarray(opt_vec)
            net_vec = np.asarray(net_vec)
            s_vec   = np.asarray(s_vec)
            infloss = build_performance(s_vec,opt_vec,net_vec,ExptDict)
            infloss_vec.append(infloss)
           # print 'Batch #%d; X-ent: %.6f; Inf. loss: %.6f'.format(i, score, infloss)
            print ('Batch {:n}; X-ent: {:f}; Inf. loss: {:f}'.format(i, score, infloss))
            net_snap.append(np.squeeze(net))
            opt_snap.append(opt)
            s_vec   = []
            opt_vec = []
            net_vec = []
        
        if i % 60 == 0:
            sh = np.squeeze(net)
            plt.subplot(211)
            plt.imshow(sh)
            plt.subplot(212)
            plt.imshow(opt)
            plt.show()
    opt_snap = np.asarray(opt_snap)
    net_snap = np.asarray(net_snap)
        
    # TESTING
    s_vec, opt_vec, net_vec, ex_hid_vec, ex_inp_vec = [], [], [], [], []
    for i, (example_input, example_output, s, opt) in test_generator:
        net            = pred_fn(example_input)
        example_hidden = rec_layer_fn(example_input)
        s_vec.append(s)
        opt_vec.append(opt)
        net_vec.append(np.squeeze( net ))
#        if i % 500 == 0:
        if i % 5 == 0:
            ex_hid_vec.append(example_hidden)
            ex_inp_vec.append(example_input)
        infloss_test = build_performance(s_vec,opt_vec,net_vec,ExptDict)    
    sio.savemat('200-Julia1_sigma%f_lambda%f_model%i_task%i_everything - 1.mat'%(offdiag_val, diag_val, m_ind, t_ind),
                {'all_params_list':lasagne.layers.get_all_param_values(l_out, trainable=True),
                 'inpResps':ex_inp_vec,
                 'hidResps':ex_hid_vec,
                 'infloss_test':infloss_test,
                 'opt_snap':opt_snap,
                 'net_snap':net_snap,
                 'infloss_vec':np.asarray(infloss_vec)})    
    # TRAINING 2
    net_snap, sh_snap, s_vec, opt_vec, net_vec, infloss_vec = [], [], [], [], [], []
    for i, (example_input, example_output, s, opt) in generator2:
        score = train_fn(example_input, example_output)
        net   = pred_fn(example_input)
        s_vec.append(s)
        opt_vec.append(opt)
        net_vec.append(np.squeeze(net))
       # if i % 500 == 0:
        if i % 100 == 0:
            opt_vec = np.asarray(opt_vec)
            net_vec = np.asarray(net_vec)
            s_vec   = np.asarray(s_vec)
            infloss = build_performance(s_vec,opt_vec,net_vec,ExptDict)
            infloss_vec.append(infloss)
           # print 'Batch #%d; X-ent: %.6f; Inf. loss: %.6f'.format(i, score, infloss)
            print ('Batch {:n}; X-ent: {:f}; Inf. loss: {:f}'.format(i, score, infloss))
            net_snap.append(np.squeeze(net))
            opt_snap.append(opt)
            s_vec   = []
            opt_vec = []
            net_vec = []
        if i % 100 == 0:
            sh = np.squeeze(net)
            plt.subplot(211)
            plt.imshow(sh)
            plt.subplot(212)
            plt.imshow(opt)
            plt.show()
    opt_snap = np.asarray(opt_snap)
    net_snap = np.asarray(net_snap)

    # TESTING 2
    s_vec, opt_vec, net_vec, ex_hid_vec, ex_inp_vec = [], [], [], [], []
    for i, (example_input, example_output, s, opt) in test_generator2:
        net            = pred_fn(example_input)
        example_hidden = rec_layer_fn(example_input)
        s_vec.append(s)
        opt_vec.append(opt)
        net_vec.append(np.squeeze( net ))
#        if i % 500 == 0:
        if i % 5 == 0:
            ex_hid_vec.append(example_hidden)
            ex_inp_vec.append(example_input)

    opt_vec = np.asarray(opt_vec)
    net_vec = np.asarray(net_vec)
    s_vec   = np.asarray(s_vec)
    infloss_test = build_performance(s_vec,opt_vec,net_vec,ExptDict)
    print ('Test data; Inf. loss: {:f}'.format(infloss_test))

    # Input and hidden layer activities
    ex_hid_vec = np.asarray(ex_hid_vec)
    ex_hid_vec = np.reshape(ex_hid_vec,(-1, generator.total_dur, ExptDict["n_hid"]))

    ex_inp_vec = np.asarray(ex_inp_vec)
    ex_inp_vec = np.reshape(ex_inp_vec,(-1, generator.total_dur, ExptDict["task"]["n_loc"] * generator.n_in))

    # SAVE TRAINED MODEL
    sio.savemat('200-Julia1_sigma%f_lambda%f_model%i_task%i_everything - 2.mat'%(offdiag_val, diag_val, m_ind, t_ind),
                {'all_params_list':lasagne.layers.get_all_param_values(l_out, trainable=True),
                 'inpResps':ex_inp_vec,
                 'hidResps':ex_hid_vec,
                 'opt_snap':opt_snap,
                 'net_snap':net_snap,
                 'infloss_test':infloss_test,
                 'infloss_vec':np.asarray(infloss_vec)})
