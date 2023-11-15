import falcor
import torch
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy

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


    testbed.run()
    abuffer = testbed.getEmissive(falcor.uint3(1061, 1920, 19))
    print(abuffer.shape)

    abuffer = abuffer.reshape((1061, 1920, 19))

    gt = abuffer[:,:,0:3]
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    cv2.imwrite("fungraphics.exr", gt)

    emissve = abuffer[:, :, 3:6]
    emissve = cv2.cvtColor(emissve, cv2.COLOR_BGR2RGB)
    cv2.imwrite("emissve.exr", emissve)

    normal = abuffer[:, :, 6:9]
    normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
    cv2.imwrite("normal.exr", normal)

    pos = abuffer[:, :, 9:12]
    pos = cv2.cvtColor(pos, cv2.COLOR_BGR2RGB)
    cv2.imwrite("pos.exr", pos)

    wi = abuffer[:, :, 12:15]
    wi = cv2.cvtColor(wi, cv2.COLOR_BGR2RGB)
    cv2.imwrite("wi.exr", wi)

    albedoalpha = abuffer[:, :, 15:19]
    albedoalpha = cv2.cvtColor(albedoalpha, cv2.COLOR_BGR2RGB)
    cv2.imwrite("albedoalpha.exr", albedoalpha)
if __name__ == "__main__":
    main()
