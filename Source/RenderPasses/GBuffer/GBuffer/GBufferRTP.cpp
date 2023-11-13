/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Falcor.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "GBufferRTP.h"

namespace
{
const std::string kProgramRaytraceFile = "RenderPasses/GBuffer/GBuffer/GBufferRTP.rt.slang";
const std::string kProgramComputeFile = "RenderPasses/GBuffer/GBuffer/GBufferRTP.cs.slang";

// Scripting options.
const char kUseTraceRayInline[] = "useTraceRayInline";
const char kUseDOF[] = "useDOF";

// Ray tracing settings that affect the traversal stack size. Set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 4;
const uint32_t kMaxRecursionDepth = 1;

// Scripting options
const std::string kLODMode = "texLOD";

// Additional output channels.
const std::string kVBufferName = "vbuffer";
const ChannelList kGBufferExtraChannels = {
    // clang-format off
    { kVBufferName,                 "gVBuffer",                     "Visibility buffer",                                       true /* optional */, ResourceFormat::Unknown /* set at runtime */ },
    { "depth",                      "gDepth",                       "Depth buffer (NDC)",                                      true /* optional */, ResourceFormat::R32Float     },
    { "linearZ",                    "gLinearZ",                     "Linear Z and slope",                                      true /* optional */, ResourceFormat::RG32Float    },
    { "mvecW",                      "gMotionVectorW",               "Motion vector in world space",                            true /* optional */, ResourceFormat::RGBA16Float  },
    { "normWRoughnessMaterialID",   "gNormalWRoughnessMaterialID",  "Guide normal in world space, roughness, and material ID", true /* optional */, ResourceFormat::RGB10A2Unorm },
    { "guideNormalW",               "gGuideNormalW",                "Guide normal in world space",                             true /* optional */, ResourceFormat::RGBA32Float  },
    { "diffuseOpacity",             "gDiffOpacity",                 "Diffuse reflection albedo and opacity",                   true /* optional */, ResourceFormat::RGBA32Float  },
    { "specRough",                  "gSpecRough",                   "Specular reflectance and roughness",                      true /* optional */, ResourceFormat::RGBA32Float  },
    { "emissive",                   "gEmissive",                    "Emissive color",                                          true /* optional */, ResourceFormat::RGBA32Float  },
    { "viewW",                      "gViewW",                       "View direction in world space",                           true /* optional */, ResourceFormat::RGBA32Float  }, // TODO: Switch to packed 2x16-bit snorm format.
    { "time",                       "gTime",                        "Per-pixel execution time",                                true /* optional */, ResourceFormat::R32Uint      },
    { "disocclusion",               "gDisocclusion",                "Disocclusion mask",                                       true /* optional */, ResourceFormat::R32Float     },
    { "mask",                       "gMask",                        "Mask",                                                    true /* optional */, ResourceFormat::R32Float     },
    // clang-format on
};
} // namespace

GBufferRTP::GBufferRTP(ref<Device> pDevice, const Properties& props) : GBuffer(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
        FALCOR_THROW("GBufferRTP requires Shader Model 6.5 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        FALCOR_THROW("GBufferRTP requires Raytracing Tier 1.1 support.");

    parseProperties(props);

    // Create random engine
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_DEFAULT);
}

RenderPassReflection GBufferRTP::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    // Add all outputs as UAVs. These are all optional.
    addRenderPassOutputs(reflector, kGBufferChannels, ResourceBindFlags::UnorderedAccess, sz);
    addRenderPassOutputs(reflector, kGBufferExtraChannels, ResourceBindFlags::UnorderedAccess, sz);
    reflector.getField(kVBufferName)->format(mVBufferFormat);

    return reflector;
}

void GBufferRTP::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    GBuffer::execute(pRenderContext, renderData);

    // Update frame dimension based on render pass output.
    // In this pass all outputs are optional, so we must first find one that exists.
    ref<Texture> pOutput;
    auto findOutput = [&](const std::string& name)
    {
        auto pTex = renderData.getTexture(name);
        if (pTex && !pOutput)
            pOutput = pTex;
    };
    for (const auto& channel : kGBufferChannels)
        findOutput(channel.name);
    for (const auto& channel : kGBufferExtraChannels)
        findOutput(channel.name);

    if (!pOutput)
    {
        logWarning("GBufferRTP::execute() - Render pass has no connected outputs. Is this intended?");
        return;
    }
    FALCOR_ASSERT(pOutput);
    updateFrameDim(uint2(pOutput->getWidth(), pOutput->getHeight()));

    // If there is no scene, clear the output and return.
    if (mpScene == nullptr)
    {
        clearRenderPassChannels(pRenderContext, kGBufferChannels, renderData);
        clearRenderPassChannels(pRenderContext, kGBufferExtraChannels, renderData);
        return;
    }

    // Check for scene changes.
    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::SDFGridConfigChanged))
    {
        recreatePrograms();
    }

    // Configure depth-of-field.
    // When DOF is enabled, two PRNG dimensions are used. Pass this info to subsequent passes via the dictionary.
    mComputeDOF = mUseDOF && mpScene->getCamera()->getApertureRadius() > 0.f;
    if (mUseDOF)
    {
        renderData.getDictionary()[Falcor::kRenderPassPRNGDimension] = mComputeDOF ? 2u : 0u;
    }

    if (mLODMode == TexLODMode::RayDiffs)
    {
        // TODO: Remove this warning when the TexLOD code has been fixed.
        // logWarning("GBufferRTP::execute() - Ray differentials are not tested for instance transforms that flip the coordinate system
        // handedness. The results may be incorrect.");
    }

    mUseTraceRayInline ? executeCompute(pRenderContext, renderData) : executeRaytrace(pRenderContext, renderData);

    mFrameCount++;
}

void GBufferRTP::renderUI(Gui::Widgets& widget)
{
    // Render the base class UI first.
    GBuffer::renderUI(widget);

    // Ray tracing specific options.
    if (widget.dropdown("LOD Mode", mLODMode))
    {
        mOptionsChanged = true;
    }

    if (widget.checkbox("Use TraceRayInline", mUseTraceRayInline))
    {
        mOptionsChanged = true;
    }

    if (widget.checkbox("Use depth-of-field", mUseDOF))
    {
        mOptionsChanged = true;
    }
    widget.tooltip(
        "This option enables stochastic depth-of-field when the camera's aperture radius is nonzero. "
        "Disable it to force the use of a pinhole camera.",
        true
    );
}

Properties GBufferRTP::getProperties() const
{
    Properties props = GBuffer::getProperties();
    props[kLODMode] = mLODMode;
    props[kUseTraceRayInline] = mUseTraceRayInline;
    props[kUseDOF] = mUseDOF;
    return props;
}

void GBufferRTP::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    GBuffer::setScene(pRenderContext, pScene);

    recreatePrograms();
}

void GBufferRTP::parseProperties(const Properties& props)
{
    GBuffer::parseProperties(props);

    for (const auto& [key, value] : props)
    {
        if (key == kLODMode)
            mLODMode = value;
        else if (key == kUseTraceRayInline)
            mUseTraceRayInline = value;
        else if (key == kUseDOF)
            mUseDOF = value;
        // TODO: Check for unparsed fields, including those parsed in base classes.
    }
}

void GBufferRTP::recreatePrograms()
{
    mRaytrace.pProgram = nullptr;
    mRaytrace.pVars = nullptr;
    mpComputePass = nullptr;
}

void GBufferRTP::executeRaytrace(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mRaytrace.pProgram || !mRaytrace.pVars)
    {
        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(getShaderDefines(renderData));

        // Create ray tracing program.
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramRaytraceFile);
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        ref<RtBindingTable> sbt = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("miss"));
        sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));

        // Add hit group with intersection shader for displaced meshes.
        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("displacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
            );
        }

        // Add hit group with intersection shader for curves (represented as linear swept spheres).
        if (mpScene->hasGeometryType(Scene::GeometryType::Curve))
        {
            sbt->setHitGroup(
                0, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("curveClosestHit", "", "curveIntersection")
            );
        }

        // Add hit group with intersection shader for SDF grids.
        if (mpScene->hasGeometryType(Scene::GeometryType::SDFGrid))
        {
            sbt->setHitGroup(
                0, mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid), desc.addHitGroup("sdfGridClosestHit", "", "sdfGridIntersection")
            );
        }

        // Add hit groups for for other procedural primitives here.

        mRaytrace.pProgram = Program::create(mpDevice, desc, defines);
        mRaytrace.pVars = RtProgramVars::create(mpDevice, mRaytrace.pProgram, sbt);

        // Bind static resources.
        ShaderVar var = mRaytrace.pVars->getRootVar();
        RenderContext* pRenderContext = mpDevice->getRenderContext();

        const size_t elemCount = (size_t)mFrameDim.x * mFrameDim.y * 16;
        const size_t byteSize = elemCount * sizeof(float);
        FALCOR_CHECK(byteSize <= std::numeric_limits<uint32_t>::max(), "Buffer is too large.");

        if (mpBuffer == nullptr || mpBuffer->getElementCount() < elemCount)
        {
            // Create data buffer and CUDA shared buffer for async PyTorch access.
            // Pytorch can access the data in the shared buffer while we generate new data into the data buffer.
            // It is fine to recreate the buffers here without syncing as the caller is responsible for synchronization.
            logInfo("Reallocating buffers to size {} bytes", byteSize);
            mpBuffer = mpDevice->createStructuredBuffer(
                sizeof(float),
                elemCount * 16,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr,
                false
            );
        }
        var["bufferUav"] = mpBuffer;
        mpSampleGenerator->bindShaderData(var);
    }

    mRaytrace.pProgram->addDefines(getShaderDefines(renderData));

    ShaderVar var = mRaytrace.pVars->getRootVar();
    bindShaderData(var, renderData);

    // Dispatch the rays.
    mpScene->raytrace(pRenderContext, mRaytrace.pProgram.get(), mRaytrace.pVars, uint3(mFrameDim, 1));
}

void GBufferRTP::executeCompute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Create compute pass.
    if (!mpComputePass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramComputeFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(getShaderDefines(renderData));

        mpComputePass = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpComputePass->getRootVar();
        mpScene->setRaytracingShaderData(pRenderContext, var);
        mpSampleGenerator->bindShaderData(var);
    }

    mpComputePass->getProgram()->addDefines(getShaderDefines(renderData));

    ShaderVar var = mpComputePass->getRootVar();
    bindShaderData(var, renderData);

    mpComputePass->execute(pRenderContext, uint3(mFrameDim, 1));
}

DefineList GBufferRTP::getShaderDefines(const RenderData& renderData) const
{
    DefineList defines;
    defines.add("COMPUTE_DEPTH_OF_FIELD", mComputeDOF ? "1" : "0");
    defines.add("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");
    defines.add("LOD_MODE", std::to_string((uint32_t)mLODMode));
    defines.add("ADJUST_SHADING_NORMALS", mAdjustShadingNormals ? "1" : "0");

    // Setup ray flags.
    RayFlags rayFlags = RayFlags::None;
    if (mForceCullMode && mCullMode == RasterizerState::CullMode::Front)
        rayFlags = RayFlags::CullFrontFacingTriangles;
    else if (mForceCullMode && mCullMode == RasterizerState::CullMode::Back)
        rayFlags = RayFlags::CullBackFacingTriangles;
    defines.add("RAY_FLAGS", std::to_string((uint32_t)rayFlags));

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    defines.add(getValidResourceDefines(kGBufferChannels, renderData));
    defines.add(getValidResourceDefines(kGBufferExtraChannels, renderData));
    return defines;
}

void GBufferRTP::bindShaderData(const ShaderVar& var, const RenderData& renderData)
{
    FALCOR_ASSERT(mpScene && mpScene->getCamera());
    var["gGBufferRTP"]["frameDim"] = mFrameDim;
    var["gGBufferRTP"]["invFrameDim"] = mInvFrameDim;
    var["gGBufferRTP"]["frameCount"] = mFrameCount;
    var["gGBufferRTP"]["screenSpacePixelSpreadAngle"] = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(mFrameDim.y);

    // Bind output channels as UAV buffers.
    auto bind = [&](const ChannelDesc& channel)
    {
        ref<Texture> pTex = getOutput(renderData, channel.name);
        var[channel.texname] = pTex;
    };
    for (const auto& channel : kGBufferChannels)
        bind(channel);
    for (const auto& channel : kGBufferExtraChannels)
        bind(channel);
}

bool GBufferRTP::getGbufferPytorch(const uint3 dim, const uint32_t offset, GBufferRTP::PyTorchTensor data)
{
#if FALCOR_HAS_CUDA
    // We create a tensor and return to PyTorch. Falcor retains ownership of the memory.
    // The Pytorch side is free to access the tensor up until the next call to this function.
    // The caller is responsible for synchronizing the access or copying the data into its own memory.

    RenderContext* pRenderContext = mpDevice->getRenderContext();

    const size_t elemCount = (size_t)mFrameDim.x * mFrameDim.y * 16;
    const size_t byteSize = elemCount * sizeof(float);
    FALCOR_CHECK(byteSize <= std::numeric_limits<uint32_t>::max(), "Buffer is too large.");

    if (mpBuffer == nullptr || mpBuffer->getElementCount() < elemCount)
    {
        // Create data buffer and CUDA shared buffer for async PyTorch access.
        // Pytorch can access the data in the shared buffer while we generate new data into the data buffer.
        // It is fine to recreate the buffers here without syncing as the caller is responsible for synchronization.
        logInfo("Reallocating buffers to size {} bytes", byteSize);
        mpBuffer = mpDevice->createStructuredBuffer(
            sizeof(float),
            elemCount * 16,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mSharedGbuffer = createInteropBuffer(mpDevice, byteSize);
    }

    auto var = mpWritePass->getRootVar();
    var["bufferUav"] = mpBuffer;
    var["CB"]["dim"] = dim;
    var["CB"]["offset"] = offset;

    logInfo("Generating data on {}x{}x{} grid", dim.x, dim.y, dim.z);
    mpWritePass->execute(pRenderContext, dim);

    // Copy data to shared CUDA buffer.
    pRenderContext->copyResource(mSharedWriteBuffer.buffer.get(), mpBuffer.get());

    // Wait for copy to finish.
    pRenderContext->waitForFalcor();

    // Construct PyTorch tensor from CUDA buffer.
    const size_t shape[3] = {dim.x, dim.y, dim.z};
    const pybind11::dlpack::dtype dtype = pybind11::dtype<float>();
    int32_t deviceType = pybind11::device::cuda::value;
    int32_t deviceId = 0; // TODO: Consistent enumeration of GPU device IDs.

    TestPyTorchPass::PyTorchTensor tensor = TestPyTorchPass::PyTorchTensor(
        (void*)mSharedWriteBuffer.devicePtr, 3, shape, pybind11::handle() /* owner */, nullptr /* strides */, dtype, deviceType, deviceId
    );
    return tensor;
#else
    FALCOR_THROW("CUDA is not available.");
#endif
}
