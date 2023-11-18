import falcor
import torch
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import time
import json
import math
import tonemap
def sphereSample_uniform(u, v):
    phi = v * 2.0 * math.pi;
    cosTheta = 1.0 - u * 2;
    sinTheta = math.sqrt(1.0 - cosTheta * cosTheta);
    return [math.cos(phi) * sinTheta, math.sin(phi) * sinTheta, cosTheta]

def setup_renderpass(testbed):
    render_graph = testbed.create_render_graph("PathTracer")
    render_graph.create_pass("PathTracer", "PathTracer", {'samplesPerPixel': 1})
    render_graph.create_pass("GBufferRT", "GBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    render_graph.create_pass("AccumulatePass", "AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    render_graph.add_edge("GBufferRT.vbuffer", "PathTracer.vbuffer")
    render_graph.add_edge("GBufferRT.posW", "AccumulatePass.posW")
    render_graph.add_edge("GBufferRT.emissive", "AccumulatePass.emissive")
    render_graph.add_edge("GBufferRT.normW", "AccumulatePass.normW")
    render_graph.add_edge("GBufferRT.viewW", "AccumulatePass.viewW")
    render_graph.add_edge("GBufferRT.diffuseOpacity", "AccumulatePass.diffuseOpacity")
    render_graph.add_edge("GBufferRT.specRough", "AccumulatePass.specRough")
    render_graph.add_edge("PathTracer.color", "AccumulatePass.input")
    render_graph.mark_output("AccumulatePass.output")
    testbed.render_graph = render_graph
class FalcorVariableRenderer:
    def __init__(self, deviceType = falcor.DeviceType.D3D12, deviceID = 0):
        self.device = falcor.Device(type=deviceType, gpu=deviceID)
        self.width = 32
        self.height = 32
        self.renderer = falcor.Testbed(width=self.width, height=self.height, position = falcor.uint2(600,600),ow =falcor.uint2(1920,1080) ,create_window=False, device=self.device)
        self.renderer.setSpp(2000)
        self.scene = None
        self.integrator = None
        self.params = None
        self.variables_ids = None
        self.sensors = None
        self.emitters = None
        self.shapes = None
        self.shapegroups = None
        self.bsdfs = None
        self.initial_values = None
        self.variables = {}
        self.min_bounds = {}
        self.len_bounds = {}
    def load_config(self,variableFile):
        print(variableFile)
        with open(variableFile, encoding='utf-8', errors='ignore') as json_data:
            self.vb = json.load(json_data, strict=False)
        for i,value in self.vb.items():
            self.variables[i] = len(value[0])
            self.min_bounds[i] = np.array(value[0])
            self.len_bounds[i] = np.array(value[1]) - self.min_bounds[i]

    def load_scene(self, scene_filename):

        setup_renderpass(self.renderer)

        # Load the actual scene
        self.renderer.load_scene(scene_filename)

        self.scene = self.renderer.scene
        self.aabb = self.renderer.getSceneBound()
        print(self.aabb)
        self.sensor = self.scene.camera
        self.sensor.nearPlane = 0.00001
        self.sensor.farPlane = 10000000
        '''
        # Get the scene parameters
        params = traverse(scene)

        # Scene randomizable objects
        emitters = scene.emitters()
        shapes = scene.shapes()
        bsdfs = scene.bsdfs()
        sensors = scene.sensors()
        shapegroups = scene.shapegroups()

        # Variables parameters
        variables = []
        variables_ids = []
        initial_values = []

        # Retrieve chosen variable objects
        for i in range(len(shapes)):
            if 'var' in shapes[i].id():
                variables_ids.append(shapes[i].id())
                variables.append(shapes[i])

                try:
                    initial_values.append(get_values(params, shapes[i].id() + '.vertex_positions_buf'))
                except:
                    pass

        for i in range(len(shapegroups)):
            if 'var' in shapegroups[i].id():
                variables_ids.append(shapegroups[i].id())
                variables.append(shapegroups[i])
                # Initial values not necessary for shapegroups

        for i in range(len(emitters)):
            if 'var' in emitters[i].id():
                variables_ids.append(emitters[i].id())
                variables.append(emitters[i])
                # Initial values not necessary for emitters

        for i in range(len(bsdfs)):
            if 'var' in bsdfs[i].id():
                variables_ids.append(bsdfs[i].id())
                variables.append(bsdfs[i])
                # Initial values not necessary for bsdfs

        for i in range(len(sensors)):
            if 'var' in sensors[i].id():
                variables_ids.append(sensors[i].id())
                variables.append(sensors[i])
                # Initial values not necessary for sensors

        variable_params = []

        # Save the params that are variable and keep only those in the parameter map
        for i in range(len(variables)):
            if variables[i] in emitters:
                param_id = variables[i].id() + '.radiance.value'
                variable_params.append(param_id)

            elif variables[i] in shapes:
                # Handle shape group instances
                if variables[i].is_instance():
                    param_id = variables[i].id() + '.to_world'
                    variable_params.append(param_id)
                # Handle single shapes
                else:
                    param_id = variables[i].id() + '.vertex_positions_buf'
                    variable_params.append(param_id)

            elif variables[i] in bsdfs:
                if variables[i].is_glossy():
                    param_id = variables[i].id() + '.specular_reflectance.value'
                    variable_params.append(param_id)

                    param_id = variables[i].id() + '.diffuse_reflectance.value'
                    variable_params.append(param_id)

                    param_id = variables[i].id() + '.alpha.value'
                    variable_params.append(param_id)

                    param_id = variables[i].id() + '.alpha'
                    variable_params.append(param_id)

        params.keep(variable_params)

        self.scene = scene
        self.integrator = scene.integrator()
        self.params = params
        self.variables_ids = variables_ids
        self.variables = variables
        self.sensors = sensors
        self.emitters = emitters
        self.shapes = shapes
        self.shapegroups = shapegroups
        self.bsdfs = bsdfs
        self.initial_values = initial_values
        '''


    def setup_scene(self, custom_values):
        print(custom_values)
        for key in custom_values:
            finalvalue = self.min_bounds[key] + custom_values[key] * self.len_bounds[key]
            print(self.min_bounds[key])
            print(self.len_bounds[key])
            print(custom_values[key])
            print(finalvalue)
            if key == "sensor":
                pos = falcor.float3(finalvalue[0],finalvalue[1],finalvalue[2])
                direction = sphereSample_uniform(finalvalue[3],finalvalue[4])
                direction = falcor.float3(direction[0],direction[1],direction[2])
                self.sensor.position = pos
                target = pos + direction
                self.sensor.target = target
                self.sensor.up = falcor.float3(0, 1, 0)
                print(self.sensor)

            '''
            # Emitters
            if self.variables[i] in self.emitters:
                assert self.variables[i].num_parameters() in [1,
                                                              3], "Emitters need 1 or 3 parameters, defined in the xml as num_parameters"

                if self.variables[i].is_environment():
                    # TODO: WIP implementation of rotating environment map, currently it is rotated in a fixed range
                    self.variables[i].set_world_transform(AnimatedTransform(
                        ScalarTransform4f.rotate(ScalarVector3f(0.0, 1.0, 0.0),
                                                 100 + custom_values[self.variables[i].id()][0] * 40)))
                else:
                    param_id = self.variables[i].id() + '.radiance.value'

                    # Change the intensity of emission
                    if self.variables[i].num_parameters() == 1:
                        self.params[param_id] = self.variables[i].min_bounds() + cVector3f(
                            custom_values[self.variables[i].id()][0], custom_values[self.variables[i].id()][0],
                            custom_values[self.variables[i].id()][0]) * self.variables[i].range_bounds()

                    # Change X, Y, Z of emission individually
                    if self.variables[i].num_parameters() == 3:
                        self.params[param_id] = self.variables[i].min_bounds() + cVector3f(
                            custom_values[self.variables[i].id()][0], custom_values[self.variables[i].id()][1],
                            custom_values[self.variables[i].id()][2]) * self.variables[i].range_bounds()

            # Sensors
            elif self.variables[i] in self.sensors:
                assert self.variables[i].num_parameters() in [
                    5], "Sensors need 5 parameters, defined in the xml as num_parameters"

                bbox_range = self.scene.bbox().extents()

                pos = self.variables[i].min_bounds() + cVector3f(custom_values[self.variables[i].id()][0],
                                                                 custom_values[self.variables[i].id()][1],
                                                                 custom_values[self.variables[i].id()][2]) * \
                      self.variables[i].range_bounds()

                origin = np.array([pos[0], pos[1], pos[2]])

                if self.variables[i].num_parameters() == 5:
                    # Target ranges based on bbox
                    # target = np.array([self.scene.bbox().min[0] + bbox_range.x / 3 + (custom_values[self.variables[i].id()][3] * bbox_range.x / 3),
                    #                   self.scene.bbox().min[1] + (bbox_range.y / 3),
                    #                   self.scene.bbox().min[2] + bbox_range.z / 3 + (custom_values[self.variables[i].id()][4] * bbox_range.z / 3)])

                    normal_dir = hemisphereSample_uniform(custom_values[self.variables[i].id()][3],
                                                          custom_values[self.variables[i].id()][4])
                    target = np.array([normal_dir[0] + pos[0], normal_dir[1] + pos[1], normal_dir[2] + pos[2]])
                set_sensor(self.variables[i], origin, target)

            # Shapes
            elif self.variables[i] in self.shapes:
                assert self.variables[i].num_parameters() in [1, 2,
                                                              3], "Shapes need 1, 2 or 3 parameters, defined in the xml as num_parameters"

                param_index = 0

                custom_vector = [0, 0, 0]

                # If x has a range use it as a parameter
                if self.variables[i].range_bounds().x[0] > 0:
                    custom_vector[0] = custom_values[self.variables[i].id()][param_index]
                    param_index += 1

                # If y has a range use it as a parameter
                if self.variables[i].range_bounds().y[0] > 0:
                    custom_vector[1] = custom_values[self.variables[i].id()][param_index]
                    param_index += 1

                # If z has a range use it as a parameter
                if self.variables[i].range_bounds().z[0] > 0:
                    custom_vector[2] = custom_values[self.variables[i].id()][param_index]
                    param_index += 1

                translate = self.variables[i].min_bounds() + cVector3f(custom_vector[0], custom_vector[1],
                                                                       custom_vector[2]) * self.variables[
                                i].range_bounds()

                # Handle shape group instances
                if self.variables[i].is_instance():
                    param_id = self.variables[i].id() + '.to_world'

                    # The parameter left is for rotation
                    if self.variables[i].num_parameters() > param_index:
                        rotation_axis = ScalarVector3f(self.variables[i].rotation_axis()[0][0],
                                                       self.variables[i].rotation_axis()[1][0],
                                                       self.variables[i].rotation_axis()[2][0])
                        angle = (self.variables[i].min_angle() + custom_values[self.variables[i].id()][param_index] *
                                 self.variables[i].range_angle())[0]

                        self.variables[i].set_to_world(ScalarTransform4f.translate(
                            ScalarVector3f(translate.x[0], translate.y[0], translate.z[0])) * ScalarTransform4f.rotate(
                            rotation_axis, angle))
                    # Only translate
                    else:
                        self.variables[i].set_to_world(
                            ScalarTransform4f.translate(ScalarVector3f(translate.x[0], translate.y[0], translate.z[0])))

                    self.params.set_dirty(param_id)
                # Handle single shapes
                else:
                    param_id = self.variables[i].id() + '.vertex_positions_buf'

                    apply_translation_from(self.params, self.initial_values[i], translate, param_id)

            # Shapegroups
            elif self.variables[i] in self.shapegroups:
                assert self.variables[i].num_parameters() in [
                    1], "Shape groups need 1 defined in the xml as num_parameters"

                self.variables[i].set_alternative(int(min(0.99, custom_values[self.variables[i].id()][0]) * (
                        self.variables[i].num_alternatives() + 1)))
                custom_values[self.variables[i].id()][0] = int(min(0.99, custom_values[self.variables[i].id()][0]) * (
                        self.variables[i].num_alternatives() + 1)) / (self.variables[i].num_alternatives() + 1)

            # BSDFs
            elif self.variables[i] in self.bsdfs:
                min_bounds = self.variables[i].min_bounds()
                range_bounds = self.variables[i].range_bounds()

                # Glossy BSDFs
                if self.variables[i].is_glossy():
                    assert self.variables[i].num_parameters() in [1, 3, 4], "Glossy BSDFs can have 1, 3 or 4 parameters"

                    # Set variable reflectance
                    if self.variables[i].num_parameters() in [3, 4]:
                        specular_reflectance = cVector3f(min_bounds[0], min_bounds[1], min_bounds[2]) + cVector3f(
                            custom_values[self.variables[i].id()][0], custom_values[self.variables[i].id()][1],
                            custom_values[self.variables[i].id()][2]) * cVector3f(range_bounds[0], range_bounds[1],
                                                                                  range_bounds[2])

                        param_id = self.variables[i].id() + '.specular_reflectance.value'

                        self.params[param_id] = specular_reflectance

                    # Set variable roughness
                    if self.variables[i].num_parameters() in [1, 4]:
                        alpha = [min_bounds[3] + custom_values[self.variables[i].id()][
                            self.variables[i].num_parameters() - 1] * range_bounds[3]][0]

                        # TODO: trying different keys because rough conductor and rough plastic use different keys for alpha

                        # Rough conductor
                        try:
                            param_id = self.variables[i].id() + '.alpha.value'

                            self.params[param_id] = alpha
                        except:
                            pass

                        # Rough plastic
                        try:
                            param_id = self.variables[i].id() + '.alpha'

                            self.params[param_id] = alpha[0]
                        except:
                            pass

                # Diffuse BSDFs
                else:
                    assert self.variables[i].num_parameters() in [1, 3,
                                                                  4], "Diffuse BSDFs can have 1, 3 or 4 parameters"

                    reflectance = cVector3f(min_bounds[0], min_bounds[1], min_bounds[2]) + cVector3f(
                        custom_values[self.variables[i].id()][0], custom_values[self.variables[i].id()][1],
                        custom_values[self.variables[i].id()][2]) * cVector3f(range_bounds[0], range_bounds[1],
                                                                              range_bounds[2])

                    # Set variable reflectance
                    if self.variables[i].num_parameters() in [3, 4]:
                        self.variables[i].set_modifier(reflectance)

                    # Set texture index
                    if self.variables[i].num_parameters() in [1, 4]:
                        self.variables[i].set_alternative(
                            int(custom_values[self.variables[i].id()][3] * self.variables[i].num_alternatives()))
        '''

        return custom_values

    def get_custom_render(self, custom_values, need_image=True, need_buffers=True):

        # Set up the scene for the given custom  values and check intersection
        custom_values = self.setup_scene(custom_values)

        # Call the scene's integrator to render the loaded scene

        self.renderer.run()
        buffers = []
        gt = []
        abuffer = self.renderer.getEmissive(falcor.uint3(self.height, self.width, 19))
        abuffer = abuffer.reshape((self.height,self.width,19))
        gt = abuffer[:,:,0:3]
        buffer = abuffer[:,:,3:19]

        return buffers, gt, custom_values
