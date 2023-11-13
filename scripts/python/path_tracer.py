import falcor
import torch
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy

def setup_renderpass(testbed):
    render_graph = testbed.create_render_graph("PathTracer")
    render_graph.create_pass("PathTracer", "PathTracer", {'samplesPerPixel': 1})
    render_graph.create_pass("VBufferRT", "VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    render_graph.create_pass("AccumulatePass", "AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    render_graph.add_edge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    render_graph.add_edge("PathTracer.color", "AccumulatePass.input")
    render_graph.mark_output("AccumulatePass.output")
    testbed.render_graph = render_graph

def main():
    falcor.Logger.verbosity = falcor.Logger.Level.Info

    scene_path = 'test_scenes/cornell_box.pyscene'

    # Create device and setup renderer.
    device = falcor.Device(type=falcor.DeviceType.D3D12, gpu=0, enable_debug_layer=True)
    testbed = falcor.Testbed(width=1920, height=1080, create_window=True, device=device)
    setup_renderpass(testbed)

    # Load scene.
    testbed.load_scene(scene_path)
    testbed.frame()


    testbed.run()
    cc = testbed.getEmissive(falcor.uint3(1080, 1920, 16))
    emissive = cc[:,:,0:3].cpu().numpy()

    emissive = cv2.cvtColor(emissive,cv2.COLOR_BGR2RGB)
    cv2.imwrite("fungraphics.exr",emissive)


if __name__ == "__main__":
    main()
