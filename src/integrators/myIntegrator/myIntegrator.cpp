/**
 * 
 * 图形学大作业
 */
#include<iostream>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/gatherproc.h>
#include <mitsuba/render/renderqueue.h>
#include <mitsuba/render/renderproc.h>

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);

class myIntegrator : public MonteCarloIntegrator {
public:
    myIntegrator(const Properties &props)
        : MonteCarloIntegrator(props) { 
        /* 初始光子半径 (0 表示通过场景大小和分辨率推断) */
        m_initialRadius = props.getFloat("initialRadius", 0);
        /* Alpha值 */
        m_alpha = props.getFloat("alpha", .7);
        /* 光子数量 */
        m_photonCount = props.getInteger("photonCount", 25000000);
        /* 并行过程的粒度（0表示随机选择） */
        m_granularity = props.getInteger("granularity", 0);
        /* 光子映射的最大深度 */
        m_maxDepth = props.getInteger("maxDepth", -1);
        m_maxDepthRay = props.getInteger("maxDepthRay", 1);
        /* 启用 russian roulette 的深度 */
        m_rrDepth = props.getInteger("rrDepth", 3);
        /* 是否自动结束 */
        m_autoCancelGathering = props.getBoolean("autoCancelGathering", true);
        /* 光子映射趟数 */
        m_maxPasses = props.getInteger("maxPasses", -1);

        m_mutex = new Mutex();
        if (m_maxDepth <= 1 && m_maxDepth != -1)
            Log(EError, "Maximum depth must either be set to \"-1\" or \"2\" or higher!");
        if (m_maxPasses <= 0 && m_maxPasses != -1)
            Log(EError, "Maximum number of Passes must either be set to \"-1\" or \"1\" or higher!");
    
    }

    /// Unserialize from a binary data stream
    myIntegrator(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID) {

        if (m_initialRadius == 0) {
            /* 推测一个合适的光子半径
              (scene width / horizontal or vertical pixel count) * 5 */
            Float rad = scene->getBSphere().radius;
            Vector2i filmSize = scene->getSensor()->getFilm()->getSize();

            m_initialRadius = std::min(rad / filmSize.x, rad / filmSize.y) * 5;
            
        }
        Log(EInfo,"%lf",m_initialRadius);
        //此处是添加的代码
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sampler> indepSampler = static_cast<Sampler *> (PluginManager::getInstance()->
            createObject(MTS_CLASS(Sampler), Properties("independent")));

        /* 为每一个核心创建一个取样实例 */
        std::vector<SerializableObject *> samplers(sched->getCoreCount());
        for (size_t i=0; i<sched->getCoreCount(); ++i) {
            ref<Sampler> clonedSampler = indepSampler->clone();
            clonedSampler->incRef();
            samplers[i] = clonedSampler.get();
        }

        int indepSamplerResID = sched->registerMultiResource(samplers);

        m_totalEmissions = 0;
        m_totalPhotons = 0;
        //调用光子映射过程
        photonMapPass(0, queue, job, sceneResID, sensorResID, indepSamplerResID);

        SamplingIntegrator::render(scene, queue, job, sceneResID, sensorResID, samplerResID);
        return true;
    }

	void photonMapPass(int it, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID) {
        Log(EInfo, "Performing a photon mapping pass %i (" SIZE_T_FMT " photons so far)",
                it, m_totalPhotons);
        ref<Scheduler> sched = Scheduler::getInstance();

        /* 创建PhotonMap进程实例 */
        ref<GatherPhotonProcess> proc = new GatherPhotonProcess(
            GatherPhotonProcess::EAllSurfacePhotons, m_photonCount,
            m_granularity, m_maxDepth == -1 ? -1 : (m_maxDepth-1), m_rrDepth, true,
            m_autoCancelGathering, job);

        /* 连接资源 */
        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);

        sched->schedule(proc);
        sched->wait(proc);

        /* 生成PhotonMap */
        ref<PhotonMap> photonMap = proc->getPhotonMap();
        photonMap->build();
        Log(EDebug, "Photon map full. Shot " SIZE_T_FMT " particles, excess photons due to parallelism: "
            SIZE_T_FMT, proc->getShotParticles(), proc->getExcessPhotons());

        Log(EInfo, "Gathering ..");
        m_totalEmissions += proc->getShotParticles();
        m_totalPhotons += photonMap->size();
        
        m_photonMap = photonMap;

        /* 刷新工作队列 */
        queue->signalRefresh(job);
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* 一些别名和临时变量 */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        /* 找到光线的第一个交汇点 */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;
        //初始深度为1
        int fakedepth = 0;  //虚假深度，即镜面反射不会增加深度
        std::vector<Float> rayPdf;  //用于储存光路每个交点的Pdf值
        std::vector<Float> rayInvPdf;  //用于储存反向光路每个点的Pdf值
        std::vector<Spectrum> vecInvEval;  //用于储存反向光路每个点的Eval值
        std::vector<Spectrum> vecThroughput;  //用于储存光路每个点的throughput值
        std::vector<Vector> vecWi;     //用于储存光路每个点的入射方向
        while (rRec.depth <= m_maxDepthRay || m_maxDepthRay < 0) {
            if (!its.isValid() && rRec.depth==1) {
                /* 如果没有交会,返回环境光照 */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }
            //如果碰到光源就获得光源光亮
            if (rRec.depth==1 && its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || scattered))
                Li += throughput * its.Le(-ray.d);

            /* Include radiance from a subsurface scattering model if requested */
            if (rRec.depth==1 && its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

           

            const BSDF *bsdf = its.getBSDF(ray);

            /* Possibly include emitted radiance if requested */
            
            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */

            /* Estimate the direct illumination if this is requested */
            DirectSamplingRecord dRec(its);

            // if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                // (bsdf->getType() & BSDF::ESmooth)) {
                // Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                // if (!value.isZero()) {
                    // const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    // BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    // const Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    // if (!bsdfVal.isZero() && (!m_strictNormals
                            // || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                           using BSDF sampling */
                        // Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            // ? bsdf->pdf(bRec) : 0;

                        /* Weight using the power heuristic */
                        // Float weight = miWeight(dRec.pdf, bsdfPdf);
                        // Li += throughput * value * bsdfVal * weight;
                    // }
                // }
            // }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */
            Spectrum flux;
            Float M = (Float) m_photonMap->estimateRadianceRawNew(
                its, m_initialRadius, flux, (m_maxDepthRay == -1 ? INT_MAX : m_maxDepth), rRec.depth,
                 m_rrDepth, rayPdf, rayInvPdf,vecInvEval, throughput, vecWi); //需要传入当前vector
                // its, m_initialRadius, flux, (m_maxDepthRay == -1 ? INT_MAX : m_maxDepth)); 
            Li += throughput * flux/((Float) m_totalEmissions*m_initialRadius*m_initialRadius * M_PI); 
            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            if (bsdfWeight.isZero())
                break;
            if(bRec.sampledType == BSDF::EDeltaReflection)
            {
                fakedepth++;
                if(fakedepth<10)
                {
                    bool hitEmitter = false;
                    Spectrum value;
                    const Vector wo = its.toWorld(bRec.wo);
                    Float woDotGeoN = dot(its.geoFrame.n, wo);
                    ray = Ray(its.p, wo, ray.time);
                    throughput *= bsdfWeight;
                    eta *= bRec.eta;
                    scene->rayIntersect(ray, its);
                    /*if(scene->rayIntersect(ray, its)) {
                               // Intersected something - check if it was a luminaire 
                        if (its.isEmitter()) {
                            value = its.Le(-ray.d);
                            dRec.setQuery(ray, its);
                            hitEmitter = true;
                            }
                    } else {
                        // Intersected nothing -- perhaps there is an environment map? 
                        const Emitter *env = scene->getEnvironmentEmitter();
                        if (env) {
                            if (m_hideEmitters && !scattered)
                               break;

                            value = env->evalEnvironment(ray);
                            if (!env->fillDirectSamplingRecord(dRec, ray))
                                break;
                            hitEmitter = true;
                        } else {break;}
                    }
                    if ( hitEmitter &&
                    (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {  //只有在深度为1才执行
                       // Compute the prob. of generating that direction using the
                         // implemented direct illumination sampling technique 
                        const Float lumPdf = 0;
                        //Li += throughput * value * miWeight(bsdfPdf, lumPdf);
                    }*/
                continue;
                }
            }
            //添加
            //
            //
             if ((rRec.depth >= m_maxDepthRay && m_maxDepthRay > 0)
                || (m_strictNormals && dot(ray.d, its.geoFrame.n)
                    * Frame::cosTheta(its.wi) >= 0)) {

                /* Only continue if:
                   1. The current path length is below the specifed maximum
                   2. If 'strictNormals'=true, when the geometric and shading
                      normals classify the incident direction to the same side */
                break;
            }
            //
            Float tmpInvPdf; 
            Spectrum tmpInvEval;
            BSDFSamplingRecord tmpbRec = bRec;
            Vector tmpWi = bRec.wi;
            tmpbRec.wi = bRec.wo;
            tmpbRec.wo = bRec.wi;
            tmpInvPdf = bsdf->pdf(tmpbRec, tmpbRec.sampledType ==BSDF:: EDeltaReflection?EDiscrete:ESolidAngle);
            tmpInvEval = bsdf->eval(tmpbRec, tmpbRec.sampledType ==BSDF:: EDeltaReflection?EDiscrete:ESolidAngle);

            scattered |= bRec.sampledType != BSDF::ENull;

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
            if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                break;

            throughput *= bsdfWeight;
            eta *= bRec.eta;

            bool hitEmitter = false;
            Spectrum value;
            /* Keep track of the throughput and relative
               refractive index along the path */

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);
            if(scene->rayIntersect(ray, its)) {
                /* Intersected something - check if it was a luminaire */
                if (its.isEmitter()) {
                     value = its.Le(-ray.d);
                     dRec.setQuery(ray, its);
                     hitEmitter = true;
                 }
            } else {
                /* Intersected nothing -- perhaps there is an environment map? */
                const Emitter *env = scene->getEnvironmentEmitter();

                if (env) {
                    if (m_hideEmitters && !scattered)
                        break;

                    value = env->evalEnvironment(ray);
                    if (!env->fillDirectSamplingRecord(dRec, ray))
                        break;
                    hitEmitter = true;
                } else {
                    break;
                }
            }

            // Log(EInfo,"PDF %lf SAMPDF %lf",bsdf->pdf(bRec,bRec.sampledType ==BSDF:: EDeltaReflection?EDiscrete:ESolidAngle), bsdfPdf);
            // Log(EInfo,"WGT%s SAMWGT%s",bsdf->eval(bRec,bRec.sampledType ==  BSDF::EDeltaReflection? EDiscrete:ESolidAngle).toString().c_str(), (bsdfWeight* bsdfPdf).toString().c_str());

            
            /* If a luminaire was hit, estimate the local illumination and
               weight using the power heuristic */
            if ( hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {  //只有在深度为1才执行
                /* Compute the prob. of generating that direction using the
                   implemented direct illumination sampling technique */
                const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                    scene->pdfEmitterDirect(dRec) : 0;

                //Li += throughput * value * miWeight(bsdfPdf, lumPdf);
            }
                

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
               BSDF sample or if indirect illumination was not requested */
            if (!its.isValid()) break;
                
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_rrDepth) {  
                /* Russian roulette: try to keep path weights equal to one,
                   while accounting for the solid angle compression at refractive
                   index boundaries. Stop with at least some probability to avoid
                   getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                // Log(EInfo,"Q:%lf",q);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
                rayPdf.push_back(bsdfPdf*q);
            }
            else rayPdf.push_back(bsdfPdf);
            
                
                rayInvPdf.push_back(tmpInvPdf);
                vecThroughput.push_back(throughput);
                vecWi.push_back(tmpWi);
                vecInvEval.push_back(tmpInvEval);

            // Log(EInfo,"%lf %f",bsdfPdf,bsdfPdf);
        }

        /* Store statistics */
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        return Li;
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "MIPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
    private:
    // std::vector<PPMWorkUnit *> m_workUnits;
    ref<PhotonMap> m_photonMap;
    Float m_initialRadius, m_alpha;
    int m_photonCount, m_granularity;
    int m_maxDepth, m_rrDepth, m_maxDepthRay;
    size_t m_totalEmissions, m_totalPhotons;
    int m_blockSize;
    bool m_running;
    bool m_autoCancelGathering;
    ref<Mutex> m_mutex;
    int m_maxPasses;
};

MTS_IMPLEMENT_CLASS_S(myIntegrator, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(myIntegrator, "myIntegrator-BDPM");
MTS_NAMESPACE_END
