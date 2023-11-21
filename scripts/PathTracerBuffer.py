from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("PathTracer")
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1})
    g.addPass(PathTracer, "PathTracer")
    GBufferRT = createPass("GBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(GBufferRT, "GBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer")
    g.add_edge("GBufferRT.posW", "AccumulatePass.posW")
    g.add_edge("GBufferRT.emissive", "AccumulatePass.emissive")
    g.add_edge("GBufferRT.normW", "AccumulatePass.normW")
    g.add_edge("GBufferRT.viewW", "AccumulatePass.viewW")
    g.add_edge("GBufferRT.diffuseOpacity", "AccumulatePass.diffuseOpacity")
    g.add_edge("GBufferRT.specRough", "AccumulatePass.specRough")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None
