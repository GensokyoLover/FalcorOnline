import falcor
import torch
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import time
import json
from FalcorVariableRenderer import FalcorVariableRenderer
from mcmc_sampler import MarkovChainSampler
def main():
    falcor.Logger.verbosity = falcor.Logger.Level.Info
    config_path = r'/mnt/sht/FalcorOnline/scripts/python/variable.json'
    #scene_path = r'E:\EmeraldSquare_v4_1\EmeraldSquare_Day.pyscene'
    scene_path = 'test_scenes/cornell_box.pyscene'

    renderer = FalcorVariableRenderer()
    renderer.load_scene(scene_path)
    renderer.load_config(config_path)
    sampler = MarkovChainSampler(large_step_prob=1.0, mutation_size_large=0.05,
                       mutation_size_small=0.04, dimensions=5)
    start = time.time()
    for i in range(10):
        variableDict = renderer.variables
        custom_values = {}
        for key in variableDict:
            paraSize = variableDict[key]
            parameters = []
            for j in range(paraSize):
                parameters.append(sampler.get_sample())
            custom_values[key] = np.array(parameters)
        sampler.accept()
        sampler.mutate()
        buffer,gt,custom = renderer.get_custom_render(custom_values)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        end = time.time() - start
        print(end)
if __name__ == "__main__":
    main()
