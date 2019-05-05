/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/photonmap.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/phase.h>
#include <fstream>
#include <iostream>

MTS_NAMESPACE_BEGIN

PhotonMap::PhotonMap(size_t photonCount)
        : m_kdtree(0, PhotonTree::ESlidingMidpoint), m_scale(1.0f) {
    m_kdtree.reserve(photonCount);
    Assert(Photon::m_precompTableReady);
}

PhotonMap::PhotonMap(Stream *stream, InstanceManager *manager)
    : SerializableObject(stream, manager),
      m_kdtree(0, PhotonTree::ESlidingMidpoint) {
    Assert(Photon::m_precompTableReady);
    m_scale = (Float) stream->readFloat();
    m_kdtree.resize(stream->readSize());
    m_kdtree.setDepth(stream->readSize());
    m_kdtree.setAABB(AABB(stream));
    for (size_t i=0; i<m_kdtree.size(); ++i)
        m_kdtree[i] = Photon(stream);
}

void PhotonMap::serialize(Stream *stream, InstanceManager *manager) const {
    Log(EDebug, "Serializing a photon map (%s)",
        memString(m_kdtree.size() * sizeof(Photon)).c_str());
    stream->writeFloat(m_scale);
    stream->writeSize(m_kdtree.size());
    stream->writeSize(m_kdtree.getDepth());
    m_kdtree.getAABB().serialize(stream);
    for (size_t i=0; i<m_kdtree.size(); ++i)
        m_kdtree[i].serialize(stream);
}

PhotonMap::~PhotonMap() {
}

std::string PhotonMap::toString() const {
    std::ostringstream oss;
    oss << "PhotonMap[" << endl
        << "  size = " << m_kdtree.size() << "," << endl
        << "  capacity = " << m_kdtree.capacity() << "," << endl
        << "  aabb = " << m_kdtree.getAABB().toString() << "," << endl
        << "  depth = " << m_kdtree.getDepth() << "," << endl
        << "  scale = " << m_scale << endl
        << "]";
    return oss.str();
}
void PhotonMap::dumpOBJ(const std::string &filename) {
    std::ofstream os(filename.c_str());
    os << "o Photons" << endl;
    for (size_t i=0; i<m_kdtree.size(); ++i) {
        const Point &p = m_kdtree[i].getPosition();
        os << "v " << p.x << " " << p.y << " " << p.z << endl;
    }

    /// Need to generate some fake geometry so that blender will import the points
    for (size_t i=3; i<=m_kdtree.size(); i++)
        os << "f " << i << " " << i-1 << " " << i-2 << endl;
    os.close();
}

Spectrum PhotonMap::estimateIrradiance(
        const Point &p, const Normal &n,
        Float searchRadius, int maxDepth,
        size_t maxPhotons) const {
    SearchResult *results = static_cast<SearchResult *>(
        alloca((maxPhotons+1) * sizeof(SearchResult)));
    Float squaredRadius = searchRadius*searchRadius;
    size_t resultCount = nnSearch(p, squaredRadius, maxPhotons, results);
    Float invSquaredRadius = 1.0f / squaredRadius;

    /* Sum over all contributions */
    Spectrum result(0.0f);
    for (size_t i=0; i<resultCount; i++) {
        const SearchResult &searchResult = results[i];
        const Photon &photon = m_kdtree[searchResult.index];
        if (photon.getDepth() > maxDepth)
            continue;

        Vector wi = -photon.getDirection();
        Vector photonNormal = photon.getNormal();
        Float wiDotGeoN = dot(photonNormal, wi),
              wiDotShN  = dot(n, wi);

        /* Only use photons from the top side of the surface */
        if (dot(wi, n) > 0 && dot(photonNormal, n) > 1e-1f && wiDotGeoN > 1e-2f) {
            /* Account for non-symmetry due to shading normals */
            Spectrum power = photon.getPower() * std::abs(wiDotShN / wiDotGeoN);

            /* Weight the samples using Simpson's kernel */
            Float sqrTerm = 1.0f - searchResult.distSquared*invSquaredRadius;

            result += power * (sqrTerm*sqrTerm);
        }
    }

    /* Based on the assumption that the surface is locally flat,
       the estimate is divided by the area of a disc corresponding to
       the projected spherical search region */
    return result * (m_scale * 3 * INV_PI * invSquaredRadius);
}

Spectrum PhotonMap::estimateRadiance(const Intersection &its,
        Float searchRadius, size_t maxPhotons) const {
    SearchResult *results = static_cast<SearchResult *>(
        alloca((maxPhotons+1) * sizeof(SearchResult)));
    Float squaredRadius = searchRadius*searchRadius;
    size_t resultCount = nnSearch(its.p, squaredRadius, maxPhotons, results);
    Float invSquaredRadius = 1.0f / squaredRadius;

    /* Sum over all contributions */
    Spectrum result(0.0f);
    const BSDF *bsdf = its.getBSDF();
    for (size_t i=0; i<resultCount; i++) {
        const SearchResult &searchResult = results[i];
        const Photon &photon = m_kdtree[searchResult.index];
        Float sqrTerm = 1.0f - searchResult.distSquared*invSquaredRadius;

        Vector wi = its.toLocal(-photon.getDirection());

        BSDFSamplingRecord bRec(its, wi, its.wi, EImportance);
        result += photon.getPower() * bsdf->eval(bRec) * (sqrTerm*sqrTerm);
    }

    /* Based on the assumption that the surface is locally flat,
       the estimate is divided by the area of a disc corresponding to
       the projected spherical search region */
    return result * (m_scale * 3 * INV_PI * invSquaredRadius);
}

struct RawRadianceQuery {
    RawRadianceQuery(const Intersection &its, int maxDepth, int rayDepth)
      : its(its), maxDepth(maxDepth),rayDepth(rayDepth),result(0.0f) {
        bsdf = its.getBSDF();
    }

    inline void operator()(const Photon &photon) {
        Normal photonNormal(photon.getNormal());
        Vector wi = -photon.getDirection();
        Float wiDotGeoN = absDot(photonNormal, wi);

        if (photon.getDepth() > maxDepth
            || dot(photonNormal, its.shFrame.n) < 1e-1f
            || wiDotGeoN < 1e-2f)
            return;

        BSDFSamplingRecord bRec(its, its.toLocal(wi), its.wi, EImportance);

        // std::cerr<<photon.toString().c_str()<<endl;

        Spectrum value = photon.getPower() * bsdf->eval(bRec);
        if (value.isZero())
            return;

        /* Account for non-symmetry due to shading normals */
        value *= std::abs(Frame::cosTheta(bRec.wi) /
            (wiDotGeoN * Frame::cosTheta(bRec.wo)));

        if(rayDepth == -1)
            result += value;
        else 
            result += value / (rayDepth + photon.getDepth()-1);   //已修改
    }

    const Intersection &its;
    const BSDF *bsdf;
    int maxDepth;
    int rayDepth;
    Spectrum result;
};

struct RawRadianceQueryNew {
    RawRadianceQueryNew(const Intersection &its, int maxDepth, int rayDepth,int rrDepth,
        const std::vector<Float> & rayPdf,const std::vector<Float> & rayInvPdf,
        const std::vector<Spectrum> & vecInvEval,
        Spectrum throughput, const std::vector<Vector> & vecWi)
      : its(its), maxDepth(maxDepth),rayDepth(rayDepth),rrDepth(rrDepth),rayPdf(rayPdf),rayInvPdf(rayInvPdf),
      vecInvEval(vecInvEval),
      throughput(throughput),vecWi(vecWi),result(0.0f) {
        bsdf = its.getBSDF();
    }

    inline void operator()(const Photon &photon) {
        Normal photonNormal(photon.getNormal());
        Vector wi = -photon.getDirection();
        Float wiDotGeoN = absDot(photonNormal, wi);

        if (photon.getDepth() > maxDepth
            || dot(photonNormal, its.shFrame.n) < 1e-1f
            || wiDotGeoN < 1e-2f)
            return;

        BSDFSamplingRecord bRec(its, its.toLocal(wi), its.wi, EImportance);

        // std::cerr<<photon.toString().c_str()<<endl;

        Spectrum value = photon.getPower() * bsdf->eval(bRec);
        if (value.isZero())
            return;

        /* Account for non-symmetry due to shading normals */
        value *= std::abs(Frame::cosTheta(bRec.wi) /
            (wiDotGeoN * Frame::cosTheta(bRec.wo)));

        /* if(rayDepth == -1) //最简单的权重选择,也可以取得不错的效果
            result += value;
        else 
            result += value / (rayDepth + photon.getDepth()-1);  */
        
        //计算权重：
        Class * m_theClass = NULL;
        //光-》照
        int pd = photon.data.vecPdf.size() + 1;
        std::vector<Float> vecP;
        vecP.push_back(1.0);
        for(int i = 1; i<pd; i++)
            vecP.push_back(vecP[i-1] * photon.data.vecPdf[i-1]);
        Spectrum tp = photon.data.throughput;
        BSDFSamplingRecord tmpbRec(its,its.toLocal(wi), its.wi);
        Spectrum bEval = bsdf->eval(tmpbRec);//, tmpbRec.sampledType ==BSDF:: EDeltaReflection?EDiscrete:ESolidAngle);
        Float bPdf = bsdf->pdf(tmpbRec);//, tmpbRec.sampledType ==BSDF:: EDeltaReflection?EDiscrete:ESolidAngle);
        if(bPdf ==0 ) //Log(EError, "bPdf1=0");
        return;
        tp *= bEval / bPdf;
        Float q = 1.0;
        if(pd >= rrDepth){
            q = std::min(tp.max(), (Float) 0.95f);
        }
        tp /= q;
        vecP.push_back(vecP[pd-1] * bPdf * q);

        for(int i = rayDepth -2; i>=0; i--){
            tp*= vecInvEval[i] / rayInvPdf[i];
            Float q = 1.0;
            if(pd + rayDepth -1 - i >= rrDepth){
                q = std::min(tp.max(), (Float) 0.95f);
            }
            tp /= q;
            vecP.push_back(vecP[pd + rayDepth -2 -i] * rayInvPdf[i] * q);
        }

        // Log(EInfo,"1RAY%s;;%lf;;%s",bEval.toString().c_str(),bPdf,tp.toString().c_str());
        //Log(EInfo,"through %lf",throughput.toString().c_str());
        //照-》光
        std::vector<Float> vecC;
        vecC.push_back(1.0);
        for(int i = 1; i<rayDepth; i++)
            vecC.push_back(vecC[i-1] * rayPdf[i-1]);
        // Log(EInfo,"2RAY%s;;%lf;;%s",bEval.toString().c_str(),bPdf,tp.toString().c_str());
        tp = throughput;
        // Log(EInfo,"3RAY%s;;%lf;;%s",bEval.toString().c_str(),bPdf,tp.toString().c_str());
        BSDFSamplingRecord tmpbRec2(its,its.wi,its.toLocal(wi));
        bEval = bsdf->eval(tmpbRec2);//, tmpbRec2.sampledType ==BSDF:: EDeltaReflection?EDiscrete:ESolidAngle);
        bPdf = bsdf->pdf(tmpbRec2);//, tmpbRec2.sampledType ==BSDF:: EDeltaReflection?EDiscrete:ESolidAngle);
        if(bPdf ==0 ) //Log(EError, "bPdf2=0");
        return;
        // Log(EInfo,"4RAY%s;;%lf;;%s",bEval.toString().c_str(),bPdf,tp.toString().c_str());
        tp *= bEval / bPdf;
        q = 1.0;
        if(rayDepth>= rrDepth){
            q = std::min(tp.max(), (Float) 0.95f);
        }
        tp /= q;
        vecC.push_back(vecC[rayDepth-1] * bPdf *q);
        // Log(EInfo,"5RAY%s;;%lf;;%s",bEval.toString().c_str(),bPdf,tp.toString().c_str());

        for(int i = pd -2; i>=0; i--)
        {
            tp*= photon.data.vecInvEval[i] / photon.data.vecInvPdf[i];
            Float q = 1.0;
            if(rayDepth + pd -1 - i >= rrDepth){
                q = std::min(tp.max(), (Float) 0.95f);
            }
            tp /= q;
            vecC.push_back(vecC[pd + rayDepth -2 -i] * photon.data.vecInvPdf[i] * q);
        }
        // Log(EInfo,"6RAY%s;;%lf;;%s",bEval.toString().c_str(),bPdf,tp.toString().c_str());
        //Log(EInfo,"PHO%s;;%lf;;%lf",bEval.toString().c_str(),bPdf,tp);

        //分母；
        Float sum = 0.0;
        for(int i = 0; i<= pd+rayDepth -2; i++)
        {
            sum += vecP[i] * vecC[pd + rayDepth -2 - i];
        }
        Float prob = vecP[pd-1]* vecC[rayDepth -1] / sum;
    
        result += value * prob;
    }

    const Intersection &its;
    const BSDF *bsdf;
    int maxDepth;
    int rayDepth;
    int rrDepth;
    Spectrum result;
    std::vector<Float> rayPdf;
    std::vector<Float> rayInvPdf;
    std::vector<Spectrum> vecInvEval;
    Spectrum throughput;
    std::vector<Vector> vecWi;
};

size_t PhotonMap::estimateRadianceRaw(const Intersection &its,
        Float searchRadius, Spectrum &result, int maxDepth, int rayDepth) const {
    RawRadianceQuery query(its, maxDepth, rayDepth);
    size_t count = m_kdtree.executeQuery(its.p, searchRadius, query);
    result = query.result;
    return count;
}

size_t PhotonMap::estimateRadianceRawNew(const Intersection &its,
        Float searchRadius, Spectrum &result, int maxDepth, int rayDepth,int rrDepth,
        const std::vector<Float> & rayPdf,const std::vector<Float> & rayInvPdf,
        const std::vector<Spectrum> & vecInvEval,
        Spectrum throughput, const std::vector<Vector> & vecWi) const
{
    RawRadianceQueryNew query(its, maxDepth, rayDepth,rrDepth, rayPdf, rayInvPdf, vecInvEval,
        throughput, vecWi);
    size_t count = m_kdtree.executeQuery(its.p, searchRadius, query);
    result = query.result;
    return count;
}


MTS_IMPLEMENT_CLASS_S(PhotonMap, false, SerializableObject)
MTS_NAMESPACE_END
