import random

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
    config_path = r'E:\falcor7\falcor\scripts\python\variable.json'
    scene_path = 'test_scenes/cornell_box.pyscene'


    renderer = FalcorVariableRenderer()
    renderer.load_scene(scene_path)
    renderer.load_config(config_path)
    sampler = MarkovChainSampler(large_step_prob=1.0, mutation_size_large=0.05,
                       mutation_size_small=0.04, dimensions=17)
    start = time.time()

    for i in range(100):
        resolution =falcor.uint2(300+ i * 10 , 300 + i * 10)
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
        #renderer.startPosition = renderer.resolution/2
        buffer,gt,custom = renderer.get_custom_render(custom_values)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        cv2.imwrite("fun" + str(i) + ".exr",gt)
        end = time.time() - start
        print(end)
if __name__ == "__main__":
    main()
