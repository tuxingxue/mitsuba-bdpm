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
                /* Initial photon query radius (0 = infer based on scene size and sensor resolution) */
        m_initialRadius = props.getFloat("initialRadius", 0);
        /* Alpha parameter from the paper (influences the speed, at which the photon radius is reduced) */
        m_alpha = props.getFloat("alpha", .7);
        /* Number of photons to shoot in each iteration */
        m_photonCount = props.getInteger("photonCount", 25000000);    //已修改,原始值为250000
        /* Granularity of the work units used in parallelizing the
           particle tracing task (default: choose automatically). */
        m_granularity = props.getInteger("granularity", 0);
        /* Longest visualized path length (<tt>-1</tt>=infinite). When a positive value is
           specified, it must be greater or equal to <tt>2</tt>, which corresponds to single-bounce
           (direct-only) illumination */
        m_maxDepth = props.getInteger("maxDepth", -1);          //已修改, 原始值为-
        m_maxDepthRay = props.getInteger("maxDepthRay", 1);
        /* Depth to start using russian roulette */
        m_rrDepth = props.getInteger("rrDepth", 3);
        /* Indicates if the gathering steps should be canceled if not enough photons are generated. */
        m_autoCancelGathering = props.getBoolean("autoCancelGathering", true);
        /* Maximum number of passes to render. -1 renders until the process is stopped. */
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
        // MonteCarloIntegrator::preprocess(scene, queue, job, sceneResID, sensorResID, samplerResID);

        if (m_initialRadius == 0) {
            /* Guess an initial radius if not provided
              (scene width / horizontal or vertical pixel count) * 5 */
            Float rad = scene->getBSphere().radius;
            Vector2i filmSize = scene->getSensor()->getFilm()->getSize();

            m_initialRadius = std::min(rad / filmSize.x, rad / filmSize.y) * 5;
        }

        //此处是添加的代码
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sampler> indepSampler = static_cast<Sampler *> (PluginManager::getInstance()->
            createObject(MTS_CLASS(Sampler), Properties("independent")));

        /* Create a sampler instance for every core */
        std::vector<SerializableObject *> samplers(sched->getCoreCount());
        for (size_t i=0; i<sched->getCoreCount(); ++i) {
            ref<Sampler> clonedSampler = indepSampler->clone();
            clonedSampler->incRef();
            samplers[i] = clonedSampler.get();
        }

        int indepSamplerResID = sched->registerMultiResource(samplers);

        m_totalEmissions = 0;
        m_totalPhotons = 0;
        photonMapPass(0, queue, job, sceneResID, sensorResID, indepSamplerResID);
        //以上是添加的代码

        SamplingIntegrator::render(scene, queue, job, sceneResID, sensorResID, samplerResID);
        return true;
    }

	void photonMapPass(int it, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID) {
        Log(EInfo, "Performing a photon mapping pass %i (" SIZE_T_FMT " photons so far)",
                it, m_totalPhotons);
        ref<Scheduler> sched = Scheduler::getInstance();

        /* Generate the global photon map */
        ref<GatherPhotonProcess> proc = new GatherPhotonProcess(
            GatherPhotonProcess::EAllSurfacePhotons, m_photonCount,
            m_granularity, m_maxDepth == -1 ? -1 : (m_maxDepth-1), m_rrDepth, true,
            m_autoCancelGathering, job);

        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);

        sched->schedule(proc);
        sched->wait(proc);

        ref<PhotonMap> photonMap = proc->getPhotonMap();
        photonMap->build();
        Log(EDebug, "Photon map full. Shot " SIZE_T_FMT " particles, excess photons due to parallelism: "
            SIZE_T_FMT, proc->getShotParticles(), proc->getExcessPhotons());

        Log(EInfo, "Gathering ..");
        m_totalEmissions += proc->getShotParticles();
        m_totalPhotons += photonMap->size();
        
        m_photonMap = photonMap;// 此条语句应修改为merge功能

        // film->clear();
        // #if defined(MTS_OPENMP)
            // #pragma omp parallel for schedule(dynamic)
        // #endif
        // for (int wuIdx = 0; wuIdx < (int) m_workUnits.size(); ++wuIdx) {
            // PPMWorkUnit *wu = m_workUnits[wuIdx];
            // Spectrum flux, contrib;

            // wu->block->clear();
            // for (size_t i=0; i<wu->gatherPoints.size(); ++i) {
                // GatherPoint &g = wu->gatherPoints[i];

                // if (g.radius == 0) {
                    // /* Generate a black sample -- necessary for proper
                    //    sample weight computation at surface boundaries */
                    // wu->block->put(g.sample, g.emission, 1);
                    // continue;
                // }

                // Float M = (Float) photonMap->estimateRadianceRaw( 
                    // g.its, g.radius, flux, m_maxDepth == -1 ? INT_MAX : (m_maxDepth-g.depth));
                // Float N = g.N;

                // if (N+M == 0) {
                    // g.flux = contrib = Spectrum(0.0f);
                // } else {
                    // Float ratio = (N + m_alpha * M) / (N + M);
                    // g.flux = (g.flux + flux) * ratio;
                    // g.radius = g.radius * std::sqrt(ratio);
                    // g.N = N + m_alpha * M;
                // }
                // contrib = g.flux / ((Float) m_totalEmissions * g.radius*g.radius * M_PI)
                    // + g.emission;
                // wu->block->put(g.sample, contrib * g.weight, 1);
            // }
            // LockGuard guard(m_mutex);
            // film->put(wu->block);
        // }
        queue->signalRefresh(job);
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;
        //depth orgini =1
        while (rRec.depth <= m_maxDepthRay || m_maxDepthRay < 0) {
            if (!its.isValid() && rRec.depth==1) {
                /* If no intersection could be found, potentially return
                   radiance from a environment luminaire if it exists */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }
			/* 新增代码: 计算PhotonMap的evaluate */
            Spectrum flux;
            Float M = (Float) m_photonMap->estimateRadianceRaw(
                its, m_initialRadius, flux, (m_maxDepthRay == -1 ? INT_MAX : m_maxDepth), rRec.depth); 
                // its, m_initialRadius, flux, (m_maxDepthRay == -1 ? INT_MAX : m_maxDepth)); 
                //计算方法需要修改, 跟photon的depth有关.
            Li += throughput * flux/((Float) m_totalEmissions*m_initialRadius*m_initialRadius * M_PI); //需要修改
            // Log(EInfo, Li.toString().c_str());

            const BSDF *bsdf = its.getBSDF(ray);

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || scattered))
                Li += throughput * its.Le(-ray.d);

            /* Include radiance from a subsurface scattering model if requested */
            if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

            if ((rRec.depth >= m_maxDepthRay && m_maxDepthRay > 0)
                || (m_strictNormals && dot(ray.d, its.geoFrame.n)
                    * Frame::cosTheta(its.wi) >= 0)) {

                /* Only continue if:
                   1. The current path length is below the specifed maximum
                   2. If 'strictNormals'=true, when the geometric and shading
                      normals classify the incident direction to the same side */
                break;
            }

            
            
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

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            if (bsdfWeight.isZero())
                break;

            scattered |= bRec.sampledType != BSDF::ENull;

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
            if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                break;

            bool hitEmitter = false;
            Spectrum value;

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);
            if (scene->rayIntersect(ray, its) && rRec.depth==1) {
                /* Intersected something - check if it was a luminaire */
                if (its.isEmitter()) {
                    value = its.Le(-ray.d);
                    dRec.setQuery(ray, its);
                    hitEmitter = true;
                }
            } else {
                /* Intersected nothing -- perhaps there is an environment map? */
                const Emitter *env = scene->getEnvironmentEmitter();

                if (env && rRec.depth==1) {
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

            /* Keep track of the throughput and relative
               refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;

            /* If a luminaire was hit, estimate the local illumination and
               weight using the power heuristic */
            // if (hitEmitter &&
                // (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                   implemented direct illumination sampling technique */
                // const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                    // scene->pdfEmitterDirect(dRec) : 0;
                // Li += throughput * value * miWeight(bsdfPdf, lumPdf);
            // }
                

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
               BSDF sample or if indirect illumination was not requested */
            if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                   while accounting for the solid angle compression at refractive
                   index boundaries. Stop with at least some probability to avoid
                   getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }
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
