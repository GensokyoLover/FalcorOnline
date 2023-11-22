import sys
import os
import falcor
from FalcorVariableRenderer import FalcorVariableRenderer

from utils import *
from generators.pixel_generator import PixelGenerator
from generators.positional_pixel_generator import PositionalPixelGenerator
from generators.moe_positional_pixel_generator import MoE
from generators.moe_positional_pixel_generator import MoE_Dispatch
from generators.moe_positional_pixel_generator import MoE_Dispatch_Encoder
from losses import *
from train.samplers.mcmc_sampler import MarkovChainSampler

import random
import numpy as np
import configargparse
import time
import torch
import torch.optim as optim
from utils import *
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
from multiprocessing import Pool
import pynvml
pynvml.nvmlInit()
alpha = 0.5 # mix
load_weight = 1
def show_gpu_infomation(id):
    handle = pynvml.nvmlDeviceGetHandleByIndex(id) # 指定显卡号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.used/1024**2

def initialize_markov_chain(patch, scene_path,config_path, tonemap):
    global renderer
    renderer = FalcorVariableRenderer(tonemap_type=tonemap,deviceType = falcor.DeviceType.Vulkan,deviceID = 3)
    renderer.load_scene(scene_path)
    renderer.load_config(config_path)
    global patch_range, patch_size

    patch_size = patch



def run_markov_chain(mc_sampler, resolution):
    patch_range = [resolution[0] - patch_size, resolution[1] - patch_size]
    custom_values = dict()
    # Create a vector that represents the proposed state
    mc_state = []
    variableDict = renderer.variables
    for key in variableDict:
        paraSize = variableDict[key]
        parameters = []
        for j in range(paraSize):
            parameters.append(mc_sampler.get_sample())
            mc_state.append(parameters[j])
        custom_values[key] = np.array(parameters)
    # Get patch position from Markov Chain
    patch_position = [int(mc_sampler.get_sample() * patch_range[0]), int(mc_sampler.get_sample() * patch_range[1])]
    mc_state.append(patch_position[0] / patch_range[0] if patch_range[0] > 0 else 0)
    mc_state.append(patch_position[1] / patch_range[1] if patch_range[1] > 0 else 0)

    # Render scene configuration
    renderer.resolution = resolution
    renderer.startPosition = falcor.uint2(patch_position[0],patch_position[1])
    renderer.width = patch_size
    renderer.height = patch_size

    buffers, gt, custom_values = renderer.get_custom_render(custom_values, need_buffers=True, need_image=True)
    inputs = stack_inputs(buffers, renderer.variables, [*custom_values.values()])
    #print(id(renderer))
    return inputs, gt, mc_sampler.last_step, mc_state

variableList = []
nameToList = {}
custom_values = dict()
tim = 0
def calculate_test_loss(path,number):
    
    global variableList,nameToList,custom_values,tim
    if len(variableList) == 0:
        for j in range(len(renderer.variables_ids)):
            var_id = renderer.variables_ids[j]
            if not (var_id in nameToList):
                nameToList[var_id] = []
            f = open(path + '/' + var_id + '.txt', 'r')
            for i in range(number):
                data = f.readline()
                data = data[0:-1]
                numbers = data.split()
                for q in range(len(numbers)):
                    numbers[q] = float(numbers[q])
                nameToList[var_id].append(numbers)
    inputs = []
    gts = []
    for i in range(number):
        custom_values = dict()
        data = np.load(path + '/' + str(i) + 'sample.npz')
        input_buffer = []
        for j in range(6):
           input_buffer.append(data['arr_' + str(j)])
        gt = data['arr_6']
        for j in range(len(renderer.variables_ids)):
            var_id = renderer.variables_ids[j]
            custom_values[var_id] = nameToList[var_id][i]
        input = stack_inputs(input_buffer, renderer.variables, [*custom_values.values()])
        input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        inputs.append(input)
        gts.append(gt)
    inputs = torch.cat(inputs, dim=0)
    gts = torch.cat(gts, dim=0)
    lenth = inputs.shape[0]
    lpLossSum = 0
    l1LossSum = 0
    smapeLossSum = 0
    if conf.debug:
        print("beforce test calculate gpu cost:")
        print(show_gpu_infomation(1))
    for i in range(lenth):
        print(show_gpu_infomation(1))
        # show_gpu_infomation(1)
        optimizer.zero_grad()
        outputs = model(inputs[i].to(gpu))
        if conf.debug:
            print("after model train: ")
            print(show_gpu_infomation(1))
        lpLoss,maLp = LP(outputs, gts[i].to(gpu))
        lpLoss = lpLoss.item()
       
        
        l1Loss,maL1 = L1(outputs, gts[i].to(gpu))
        l1Loss = l1Loss.item()
        
        smapeLoss,maS = SMAPE(outputs, gts[i].to(gpu))
        smapeLoss = smapeLoss.item()
        if conf.debug:
            print("TestLPLoss : " + str(lpLoss))
            print("TestL1Loss : " + str(l1Loss))
            print("TestSmapeLoss : " + str(smapeLoss))
        
        lpLossSum += lpLoss
        l1LossSum += l1Loss
        smapeLossSum += smapeLoss
    lpLossMean = lpLossSum / lenth
    l1LossMean = l1LossSum / lenth
    smapeLossMean = smapeLossSum / lenth
    writer.add_scalar("TestLpLoss", lpLossMean,tim)
    writer.add_scalar("TestL1Loss", l1LossMean,tim)
    writer.add_scalar("TestSmapeLoss", smapeLossMean,tim)
    tim = tim + 1

def save_sample(inputs, gt, type):
    samples.append((inputs, gt))
    sample_weights.append(0)
    sample_type.append(type)
    return len(samples) - 1


uniform_multiple_variable = 2
uniform_total_samples = 0
mc_total_samples = 0





def generate_samples(pool, device):
    print("start generate")
    global mc_total_samples
    mc_sample_ids = []

    before = pool._pool[:]

    # Create resolutions array for processes
    resolutions = [resolution] * len(mc_samplers)

    # Render inputs and gts chosen by the MCs in parallel
    if conf.debug:
        print(len(resolutions))
    mc_results = pool.starmap_async(run_markov_chain, zip(mc_samplers, resolutions))
        
    # Wait for parallel results
    while not mc_results.ready():
        processing_time = time.time() - start_time
        # If one of the processes crashed repeat sample generation
        if any(proc.exitcode for proc in before) or processing_time > conf.timeout:
            print('Subprocess crashed, retrying...')
            # Terminate thread pool
            pool.close()
            pool.join()
            enoki.cuda_malloc_trim()
            return []
    else:
        mc_results = mc_results.get()
        if conf.debug:
            print(len(mc_results))
        for k in range(conf.num_patches):

            proposed_inputs = torch.from_numpy(mc_results[k][0]).unsqueeze(0)
            proposed_gt = torch.from_numpy(mc_results[k][1]).unsqueeze(0)
            proposed_type = mc_results[k][2]
            proposed_state = mc_results[k][3].copy()

            # Initialize MC with the first state
            if mc_samplers[k].current_state is None:
                optimizer.zero_grad(set_to_none=True)
                proposed_result = model(proposed_inputs.to(device))
                proposed_loss,max_loss = criterion(proposed_result, proposed_gt.to(device))
                if conf.arch == "pmoe" or conf.arch =="moe" or conf.arch=="pmoeE":
                    print("Moe Loss:")
                    print(proposed_loss)
                    print(model.balanced_loss)
                    proposed_loss = proposed_loss + model.balanced_loss * load_weight
                proposed_loss.backward()
                proposed_grad = compute_adam_grad_norm_reset(optimizer)
                proposed_metric = proposed_grad * metric_calculate(proposed_result,proposed_gt.to(device))
                
                mc_samplers[k].current_sample_id = save_sample(proposed_inputs, proposed_gt, proposed_type)
                mc_samplers[k].current_state = proposed_state.copy()
                mc_samplers[k].accept()
                mc_sample_ids.append(mc_samplers[k].current_sample_id)
                mc_states.append(mc_samplers[k].current_state)
                mc_samplers[k].mutate()
                writer.add_scalar("MC Metric",proposed_metric,mc_total_samples)
                writer.add_scalar("MC Loss",proposed_loss,mc_total_samples)
                mc_total_samples = mc_total_samples + 1
                continue

            if not conf.uniform: #
                optimizer.zero_grad(set_to_none=True)
                proposed_result = model(proposed_inputs.to(device))
                proposed_loss,max_loss = criterion(proposed_result, proposed_gt.to(device))
                if conf.arch == "pmoe" or conf.arch =="moe" or conf.arch=="pmoeE":
                    print("Moe Loss:")
                    print(proposed_loss)
                    print(model.balanced_loss)
                    proposed_loss = proposed_loss + model.balanced_loss * load_weight
                proposed_loss.backward()
                proposed_grad = compute_adam_grad_norm_reset(optimizer)
                proposed_metric = proposed_grad * metric_calculate(proposed_result,proposed_gt.to(device))

                # Compute current state metric
                current_inputs, current_gt = samples[mc_samplers[k].current_sample_id]
                current_result = model(current_inputs.to(device))
                current_loss,max_loss = criterion(current_result, current_gt.to(device))
                if conf.arch == "pmoe" or conf.arch =="moe" or conf.arch=="pmoeE":
                    print("Moe Loss:")
                    print(current_loss)
                    print(model.balanced_loss)
                    current_loss = current_loss + model.balanced_loss * load_weight
                current_loss.backward()
                current_grad = compute_adam_grad_norm_reset(optimizer)
                current_metric = current_grad * metric_calculate(current_result,current_gt.to(device))

            # Our aggressive policy
            acceptance_policy = lambda proposed_metric, currrent_metric: proposed_metric >= current_metric

            if conf.uniform or acceptance_policy(proposed_metric, current_metric):

                mc_samplers[k].accept()

                # Save current state
                # 重用策略并没有把之前弱了的样本丢掉或者进一步削弱，只是重新计算了loss，无视了grad的作用
                mc_samplers[k].current_sample_id = save_sample(proposed_inputs, proposed_gt, proposed_type)
                mc_samplers[k].current_state = proposed_state
                mc_total_samples = mc_total_samples + 1

            else:
                mc_samplers[k].reject()

                # Save proposed state for training
                proposed_sample_id = save_sample(proposed_inputs, proposed_gt, proposed_type)
                mc_sample_ids.append(proposed_sample_id)

                mc_states.append(proposed_state)
                writer.add_scalar("MC Metric",current_metric,mc_total_samples)
                writer.add_scalar("MC Loss",current_loss,mc_total_samples)
                mc_total_samples = mc_total_samples + 1

            mc_samplers[k].mutate()

            # Save current state for training
            mc_sample_ids.append(mc_samplers[k].current_sample_id)

            mc_states.append(mc_samplers[k].current_state)
    if conf.debug:
        print("generate finished")
    return mc_sample_ids

def set_random_seed():
     # Set random seeds
    np.random.seed(conf.seed)
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)

def metric_calculate(predict, gt):
    meanValue =  metricFunction(predict,gt)
    if conf.addMax:
        return meanValue + metricFunction.max
    return meanValue
    

# Multiprocessing needs to be on main thread
if __name__ == '__main__':
    
    multiprocessing.set_start_method('spawn', force=True)
    LP = LpipsLoss()
    L1 = DssimL1Loss()
    SMAPE = SMAPELoss()
    print("finished")
    conf = configargparse.ArgumentParser()

    # Samples
    conf.add('--training_samples', type=int, required=True, help='Number of training samples')
    conf.add('--validation_samples', type=int, required=True, help='Number of validation samples')

    # Directories
    conf.add('--scene_path', required=True, help='Path to the scene to be trained on')
    conf.add('--config_path', required=True, help='Path to the scene to be trained on')
    conf.add('--models_dir', default='./models', help='Path to models directory')
    conf.add('--log_dir', default='./runs/', help='Path to tensorboard logging directory')
    conf.add('--validation_dataset_dir', help='Path to training dataset directory')

    # Sample reuse parameters
    conf.add('--bootstrap_samples', type=int, default=100, help='Number of samples before reuse is activated')
    conf.add('--memory_samples', type=int, default=1e5, help='Number of maximum samples kept in memory')
    conf.add('--max_samples', type=int, default=-1, help='Maximum number of samples to be generated')
    conf.add('--reuse_bias', type=float, default=3.0, help='Bias towards reusing when saved and new losses are equal')
    conf.add('--ema_alpha', type=float, default=0.95, help='Coefficient that controls how fast EMA discounts older values')

    # Patch based rendering parameters
    conf.add('--num_patches', type=int, default=16, help='Number of Markov Chains')
    conf.add('--num_threads', type=int, default=4, help='Number of parallel threads')
    conf.add('--timeout', type=int, default=400, help='Timeout for parallel rendering')
    conf.add('--patch_size', type=int, default=32, help='Size of patch to be rendered')

    # Generators (default: Positional Pixel Generator)
    conf.add('--arch', default='ppixel', choices=['pixel', 'ppixel', 'pmoe', 'pmoeE', 'moe'])
    conf.add('--loss', default='dssim_l1', choices=['dssim_l1'])
    conf.add('--tonemap', default='log1p', choices=['log1p'])
    conf.add('--hidden_features', type=int, default=512, help='Number of hidden features for the generator')
    conf.add('--hidden_layers', type=int, default=8, help='Number of hidden layers for the generator')
    conf.add('--encoder_hidden_layers', type=int, default=2, help='Number of hidden layers for the generator')
    conf.add('--decoder_hidden_layers', type=int, default=4, help='Number of hidden layers for the generator')
    conf.add('--learning_rate', type=float, default=1e-4, help='Learning rate to use for training')

    # Ablations
    conf.add('--uniform', action='store_true', help='No MCMC')

    # Misc
    conf.add('--batch_size', type=int, default=1, help='Batch size')
    conf.add('--epochs', type=int, default=1000, help='Number of training epochs')
    conf.add('--max_res', type=int, default=600, help='Maximum resolution during training')
    conf.add('--res_step', type=int, default=4, help='Resolution increase step during training')
    conf.add('--check_freq', type=int, default=1, help='Save model every check_freq epochs')
    conf.add('--tensorboard', action='store_true', help='Whether to use tensorboard for visualization')
    conf.add('--save_states', action='store_true', help='Whether to save the MC states for debugging')
    conf.add('--save_checkpoint', action='store_true', help='Whether to save the model checkpoint')
    conf.add('--seed', type=int, default=0, help='Seed for random numbers generator')
    conf.add('--mutation_size_small', type=float, default=1.0 / 25.0, help='MCMC small mutation size')
    conf.add('--mutation_size_large', type=float, default=1.0 / 20.0, help='MCMC large mutation size')
    conf.add('--device', type=int, default=0, help='Device to use for Pytorch training')
    conf.add('--summary_freq', type=int, default=50, help='Frequency at which to display results')
    conf.add('--resume', type=str, default='', help='Resume checkpoint path from which to start training')
    # sht added 
    conf.add('--test_path', type=str, default='', help='test data sets path')
    conf.add('--test_sample', type=int, default=100, help='the number of test data sample')
    conf.add('--no_reused', action='store_true', help='whether to unenable reused module')
    conf.add('--debug', action='store_true', help='whether to unenable reused module')
    conf.add('--metric',default='L1',choices=['L1','RelativeError'])
    conf.add('--addMax',action='store_true',help='whether to add max * alpha to metric')
    conf.add('--expertNum',type=int, default=8, help='how many son you need')
    conf.add('--gateK',type=int, default=2, help='how many gate you choice')
    conf = conf.parse_args()

    # Set random seeds
    set_random_seed()
    
    # CreateMetricFunc
    if conf.metric == 'L1':
        metricFunction = L1Metric()
    elif conf.metric == 'RelativeError':
        metricFunction = RelativeErrorMetric()
    # Create directories
    models_dir = conf.models_dir
    log_dir = conf.log_dir

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # Create renderer
    renderer = FalcorVariableRenderer(tonemap_type=conf.tonemap,deviceType = falcor.DeviceType.Vulkan,deviceID = 3)

    # Load scene
    renderer.load_scene(conf.scene_path)
    renderer.load_config(conf.config_path)

    resolution = [renderer.width,renderer.height]
    
    # Validation set
    if conf.validation_samples > 0:
        validation_dataset = ConfigurableDataset(conf.validation_dataset_dir, renderer.variables_ids, data_samples=conf.validation_samples)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, num_workers=0, batch_size=conf.batch_size)


    mc_samplers = []
    ## sht added 
    mc_uniform_samplers = []
    for i in range(conf.num_patches*uniform_multiple_variable):
        mc_uniform_samplers.append(MarkovChainSampler(large_step_prob=1.0, mutation_size_large=conf.mutation_size_large, mutation_size_small=conf.mutation_size_small, dimensions=renderer.totalPara + 2))
    
    
    large_step_prob = 0.3

    if conf.uniform:
        large_step_prob = 1.0

    # Initialize Markov Chain samplers
    for i in range(conf.num_patches):
        mc_samplers.append(MarkovChainSampler(large_step_prob=large_step_prob, mutation_size_large=conf.mutation_size_large, mutation_size_small=conf.mutation_size_small, dimensions=renderer.totalPara + 2))
   
    training_samples = conf.training_samples

    # Losses and optimizer
    if conf.loss == 'dssim_l1':
        criterion = DssimL1Loss()
    if conf.loss == 'dssim_smape':
        criterion = DssimSMAPELoss()

    # Initialize the network
    if conf.arch == 'pixel':
        print('Using Pixel generator with ADAM and learning rate ' + str(conf.learning_rate))
        model = PixelGenerator(buffers_features=13, variables_features=renderer.totalPara, hidden_features=conf.hidden_features, hidden_layers=conf.hidden_layers)
    elif conf.arch == 'ppixel':
        print('Using Positional Pixel generator with ADAM and learning rate ' + str(conf.learning_rate))
        model = PositionalPixelGenerator(buffers_features=13, variables_features=renderer.totalPara, hidden_features=conf.hidden_features, hidden_layers=conf.hidden_layers)
    elif conf.arch == "moe":
        print("Using MOE Positional Pixel generator with ADAM and learning rate" + str(conf.learning_rate))
        #model = MoE(conf.expertNum,buffers_features=13,variables_features=renderer.totalPara,hidden_features=conf.hidden_features,hidden_layers=conf.hidden_layers,k = conf.gateK)
        model = MoE(conf.expertNum,buffers_features=13,variables_features=renderer.totalPara,hidden_features=conf.hidden_features,hidden_layers=conf.hidden_layers,k = conf.gateK)
    elif conf.arch == 'pmoe':
        print("Using MOE Positional Pixel generator with ADAM and learning rate" + str(conf.learning_rate))
        #model = MoE(conf.expertNum,buffers_features=13,variables_features=renderer.totalPara,hidden_features=conf.hidden_features,hidden_layers=conf.hidden_layers,k = conf.gateK)
        model = MoE_Dispatch(conf.expertNum,buffers_features=13,variables_features=renderer.totalPara,hidden_features=conf.hidden_features,encoder_hidden_layers=conf.encoder_hidden_layers,decoder_hidden_layers=conf.decoder_hidden_layers,k = conf.gateK)
    elif conf.arch == 'pmoeE':
        print("Using MOE Positional Pixel generator with ADAM and learning rate" + str(conf.learning_rate))
        #model = MoE(conf.expertNum,buffers_features=13,variables_features=renderer.totalPara,hidden_features=conf.hidden_features,hidden_layers=conf.hidden_layers,k = conf.gateK)
        model = MoE_Dispatch_Encoder(conf.expertNum,buffers_features=13,variables_features=renderer.totalPara,hidden_features=conf.hidden_features,encoder_hidden_layers=conf.encoder_hidden_layers,decoder_hidden_layers=conf.decoder_hidden_layers,k = conf.gateK) 
    print(renderer.totalPara)
    gpu = torch.device("cuda:" + str(conf.device))
    model.to(gpu)
    LP.to(gpu)
    L1.to(gpu)
    SMAPE.to(gpu)
    if conf.debug:
        print("gpu cost ")
        print(show_gpu_infomation(1))
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)

    if conf.tensorboard:
        writer = SummaryWriter(log_dir + "_" + conf.arch + "_" + str(conf.expertNum)\
            + "_" +str(conf.hidden_features) + "_"  + str(conf.hidden_layers)\
            + "_" + str(conf.gateK) + time.strftime('%Y%m%d-%H%M%S'))
    

    # Define model directory
    model_dir = conf.arch + '_'

    # Ablations tags
    if conf.uniform:
        model_dir += 'no_mcmc_'

    # Tonemapping tags
    if conf.tonemap == 'n2n':
        model_dir += 'n2n_'
    else:
        model_dir += 'log1p_'

    # Resolution increase tags
    if conf.res_step > 0:
        model_dir += 'step_' + str(conf.res_step) + '_' + str(conf.max_res) + '_'

    # Loss tags
    model_dir += conf.loss + '_'

    # Learning rate tags
    model_dir += str(conf.learning_rate) + '_'

    model_dir += time.strftime('%Y%m%d-%H%M%S')

    # Create model directory
    create_dir(os.path.join(models_dir, model_dir))
    model_path = os.path.join(models_dir, model_dir)

    epoch_train_losses = []
    epoch_validation_losses = []

    pool = Pool(initializer=initialize_markov_chain, initargs=[conf.patch_size, conf.scene_path, conf.tonemap], processes=conf.num_threads, maxtasksperchild=100)
    print('Found ' + str(renderer.totalPara) + ' variable scene parameters.')
    print('Started training..')

    # Initialize sample reuse parameters
    samples = []
    sample_weights = []
    sample_type = []

    # Total samples generated and samples kept in memory currently
    total_samples = 0
    memory_samples = 0

    # Exponential moving average of the two tracked losses
    ema_loss_saved = 0.0
    ema_loss_new = 0.0

    # Use checkpoint to resume training
    if conf.resume != '':
        print('Resuming from checkpoint..')
        load_dict = torch.load(conf.resume)
        optimizer.load_state_dict(load_dict["optimizer"])
        model.load_state_dict(load_dict["model"])
        resolution = load_dict["resolution"]
        load_dict = None
    total = 0;
    
    
    if conf.debug:
        print("before training gpu cost")
        print(show_gpu_infomation(1))
    
    for epoch in range(conf.epochs):# 每次迭代epoch 清空mcstates和runing_loss 渐进像素
        #calculate_test_loss(conf.test_path, conf.test_sample)
        running_loss = 0.0
        mc_states = []
        i = 0
        load_weight = load_weight * 0.98
        # Increase resolution every epoch
        resolution = [min(conf.max_res, res + conf.res_step) for res in resolution]

        while i < training_samples:
            # Inputs and gts that will be used for training
            inputs = []
            gt = []

            # Indices of (inputs, gt) in saved samples
            current_sample_indices = []

            start_time = time.time()
            is_reused = False
            # Always generate the first few samples
            if total_samples < conf.bootstrap_samples:
                current_sample_indices = generate_samples(pool, gpu)
                #generate_uniform_samples(pool,gpu)
                # If the sample generation crashed
                if len(current_sample_indices) == 0:
                    # Restart thread pool
                    pool = Pool(initializer=initialize_markov_chain, initargs=[conf.patch_size, conf.scene_path, conf.tonemap], processes=conf.num_threads, maxtasksperchild=100)
                    continue

                is_reused = False

                inputs = []
                gt = []

                # Get the generated sample from indices
                for batch, id in enumerate(current_sample_indices):
                    inputs_batch, gt_batch = samples[id]
                    inputs.append(inputs_batch)
                    gt.append(gt_batch)

                inputs = torch.cat(inputs, dim=0)
                gt = torch.cat(gt, dim=0)
            else:
                # Initialize new and saved loss
                if total_samples == conf.bootstrap_samples:
                    ema_loss_new = loss.item()
                    ema_loss_saved = ema_loss_new

                # Compute reuse probability
                reuse_prob = compute_reuse_prob(ema_loss_saved - ema_loss_new, conf.reuse_bias)
                # Plot the reuse probability
                if conf.no_reused == True:
                    reuse_prob = 0
                if conf.tensorboard:
                    writer.add_scalar('Reuse probability', reuse_prob, epoch * conf.training_samples + i)

                # Case reuse or case exceeded sample generation budget
                if random.random() < reuse_prob or 0 < conf.max_samples <= total_samples:
                    current_sample_indices = random.choices(range(len(samples)), weights=sample_weights, k=conf.num_patches)
                    for batch in range(conf.num_patches):
                        inputs_batch, gt_batch = samples[current_sample_indices[batch]]
                        inputs.append(inputs_batch)
                        gt.append(gt_batch)

                    inputs = torch.cat(inputs, dim=0)
                    gt = torch.cat(gt, dim=0)

                    is_reused = True
                # Case generate
                else:
                    current_sample_indices = generate_samples(pool, gpu)
                    # generate_uniform_samples(pool,gpu)
                    # If the sample generation crashed
                    if len(current_sample_indices) == 0:
                        # Restart thread pool
                        pool = Pool(initializer=initialize_markov_chain, initargs=[conf.patch_size, conf.scene_path, conf.tonemap], processes=conf.num_threads)
                        continue
                    is_reused = False

                    inputs = []
                    gt = []

                    # Get the generated sample from indices
                    for batch, id in enumerate(current_sample_indices):
                        inputs_batch, gt_batch = samples[id]
                        inputs.append(inputs_batch)
                        gt.append(gt_batch)

                    inputs = torch.cat(inputs, dim=0)
                    gt = torch.cat(gt, dim=0)
                    memory_samples += conf.num_patches

            memory_samples = len(samples)
            if is_reused == False:
                total_samples += 1
            else:
                sampler_time = time.time()
                print("Sampler Time:  " + str(sampler_time-start_time))
            
            # Training step
            print("start model training")
            optimizer.zero_grad()
            outputs = model(inputs.to(gpu))
            loss,max_loss = criterion(outputs, gt.to(gpu))
            if conf.arch == "pmoe" or conf.arch =="moe" or conf.arch=="pmoeE":
                print("Moe Loss:")
                print(loss)
                print(model.balanced_loss)
                loss = loss + model.balanced_loss * load_weight
            metricFunction(outputs,gt.to(gpu))
            loss.backward()
            optimizer.step()
            # Update weights based on loss
            loss_map = criterion.loss_map
            loss_map_max = criterion.loss_map
            metric_map = metricFunction.loss_map
            metric_map_max = metricFunction.loss_map
            # Average to batch dimension
            loss_map = loss_map.mean(1).mean(1).mean(1)
            loss_map_max = loss_map_max.max(1).values.max(1).values.max(1).values
        
            metric_map = metric_map.mean(1).mean(1).mean(1)
            metric_map_max = metric_map_max.max(1).values.max(1).values.max(1).values
            if conf.addMax:
                metric_map = metric_map + metric_map_max * metricFunction.alpha
            if conf.debug:
                print(current_sample_indices)
            for batch in range(inputs.shape[0]):
                sample_weights[current_sample_indices[batch]] = metric_map[batch].item()

                # Use large step samples to update EMAs
                if sample_type[current_sample_indices[batch]] == 'large':

                    if is_reused:
                        ema_loss_saved = (1 - conf.ema_alpha) * metric_map[batch].item() + conf.ema_alpha * ema_loss_saved
                    else:
                        ema_loss_new = (1 - conf.ema_alpha) * metric_map[batch].item() + conf.ema_alpha * ema_loss_new
            generate_lenth = conf.num_patches
            if conf.debug:
                print(current_sample_indices)
                print(memory_samples)
            if memory_samples > conf.memory_samples:
                g=memory_samples - conf.memory_samples
                while(g> 0):
                    samples.pop(0)
                    sample_weights.pop(0)
                    g = g - 1

                # Update indices
                for k in range(generate_lenth):
                    mc_samplers[k].current_sample_id -= generate_lenth
                    if mc_samplers[k].current_sample_id < 0:
                        mc_samplers[k].current_sample_id = random.choices(range(len(samples)), weights=sample_weights, k=1)[0]
                        print(mc_samplers[k].current_sample_id)
                        mc_samplers[k].current_state = None

                # Update counts
                memory_samples = len(samples)
            running_loss += loss.item()

            if conf.tensorboard and i % conf.summary_freq == 0:
                writer.add_scalars('Losses', {'saved': ema_loss_saved, 'new': ema_loss_new}, epoch * conf.training_samples + i)
                writer.add_scalar('Stored Samples Size', memory_samples, epoch * conf.training_samples + i)
                writer.add_scalar('Total Samples Size', total_samples, epoch * conf.training_samples + i)
            
            i += 1
            

            end_time = time.time()

            print('Time: ' + str(end_time - start_time))
            print('Training -- %d/%d' % (i, training_samples))
            print("at epoch " + str(epoch) + " trainCount = " + str(i) +  " sampleSize = " +str(len(samples)))

        # Print Markov Chain statistics
        large_acceptance_rate = 0
        small_acceptance_rate = 0

        for k in range(conf.num_patches):
            large_acceptance_rate += mc_samplers[k].get_large_acceptance_rate()
            small_acceptance_rate += mc_samplers[k].get_small_acceptance_rate()

            mc_samplers[k].reset_statistics()

        large_acceptance_rate /= conf.num_patches
        small_acceptance_rate /= conf.num_patches

        print('Large Acceptance Rate: %.2f' % large_acceptance_rate)
        print('Small Acceptance Rate: %.2f' % small_acceptance_rate)

        epoch_train_loss = running_loss / training_samples
        epoch_train_losses.append(epoch_train_loss)

        if conf.tensorboard:
            writer.add_scalar('Epoch loss', epoch_train_loss, epoch)

        print('[%d] Train loss: %.5f' % (epoch + 1, epoch_train_loss))

        if conf.validation_samples > 0:
            with torch.no_grad():
                running_loss = 0.0
                for i, data in enumerate(validation_loader, 0):
                    inputs, gt = data

                    outputs = model(inputs.to(gpu)).cpu()
                    loss = criterion(outputs.to(gpu), gt.to(gpu))

                    running_loss += loss.item()

            epoch_validation_loss = running_loss / conf.validation_samples
            epoch_validation_losses.append(epoch_validation_loss)

            print('[%d] Validation loss: %.5f' % (epoch + 1, epoch_validation_loss))

        if epoch % conf.check_freq == 0:
            print('[%d] Saving network..' % (epoch + 1))
            torch.save(model.state_dict(), model_path + '/model_' + str(epoch / conf.check_freq) + '.pth')
            write_mc_acceptance_rates(large_acceptance_rate, small_acceptance_rate, model_path + '/mc_acceptance_rates.csv')

            if conf.save_states:
                # Write MC states for visualization -- debugging purposes
                write_mc_states(mc_states, model_path + '/mc_states_' + str(epoch / conf.check_freq) + '.csv')

                mc_states = []

            if conf.save_checkpoint:
                # Write model and samplers checkpoint
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mcs': mc_samplers,
                    'resolution': resolution
                }
                torch.save(save_dict, model_path + '/resume_checkpoint_' + str(epoch / conf.check_freq) + '.pth')

                # Keep only the last checkpoint
                if os.path.exists(model_path + '/resume_checkpoint_' + str((epoch - 1) / conf.check_freq) + '.pth'):
                    os.remove(model_path + '/resume_checkpoint_' + str((epoch - 1) / conf.check_freq) + '.pth')

    print('Finished training')

    model_path = model_path + '_' + str(epoch_train_losses[-1]) + '.pth'

    torch.save(model.state_dict(), model_path)
