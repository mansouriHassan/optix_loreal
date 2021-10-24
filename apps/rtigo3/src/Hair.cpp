
//
// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <random>

#include "inc/SceneGraph.h"
#include "inc/Hair.h"
#include "vector_types.h"

#include <optix.h>

#include "inc/Application.h"
#include "inc/CheckMacros.h"

#include <inc/MyAssert.h>



//#include "ProgramGroups.h"
//#include "Util.h"

namespace sg {

    Curves::Curves(const unsigned int id)
        : Node(id)
    {
        m_density = 1.f;
        m_disparity = 1.f;
    }
    
    Curves::Curves(const unsigned int id, float density, float disparity)
        : Node(id)
    {
        if (density < 1.f)
            std::cout << "Density must be a >1.f value, setting m_density to 1.f default value !!!" << std::endl;
        if (disparity > 1.f || disparity < 0.f)
            std::cout << "Disparity must be in [0.f; 1.f] value, clamping m_disparity to [0.f; 1.f] !!!" << std::endl;
        m_density = (density < 1.f)  ? 1.f : density;
        m_disparity = clamp(disparity, 0.f, 1.f);
    }

    NodeType Curves::getType() const
    {
        return NT_CURVES;
    }
    inline std::ostream& operator<<(std::ostream& o, float3 v)
    {
        o << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return o;
    }

    void Curves::setAttributes(std::vector<VertexAttributes> const& attributes)
    {
        m_attributes.resize(attributes.size());
        memcpy(m_attributes.data(), attributes.data(), sizeof(VertexAttributes) * attributes.size());
    }

    std::vector<VertexAttributes> const& Curves::getAttributes() const
    {
        return m_attributes;
    }

    void Curves::setIndices(std::vector<unsigned int> const& indices)
    {
        m_indices.resize(indices.size());
        memcpy(m_indices.data(), indices.data(), sizeof(unsigned int) * indices.size());
    }

    std::vector<unsigned int> const& Curves::getIndices() const
    {
        return m_indices;
    }

    void Curves::setThickness(std::vector<float> const& indices)
    {
        m_indices.resize(indices.size());
        memcpy(m_indices.data(), indices.data(), sizeof(float) * indices.size());
    }

    std::vector<float> const& Curves::getThickness() const { return m_thickness; };


    void Curves::createHairFromFile(const std::string& fileName)
    {
        std::ifstream input(fileName.c_str(), std::ios::binary);
        if (!input.is_open()) {
            std::cout << "Unable to open " + fileName + "." << std::endl;
        }

        input.read(reinterpret_cast<char*>(&m_header), sizeof(FileHeader));
        if (strncmp(m_header.magic, "HAIR", 4) != 0)
            std::cout << "Hair-file error: Invalid file format." + fileName << std::endl;
        m_header.fileInfo[87] = 0;
        m_header.defaultThickness = 0.009f;

        // Segments array(unsigned short)
        // The segements array contains the number of linear segments per strand;
        // thus there are segments + 1 control-points/vertices per strand.
        auto strandSegments = std::vector<unsigned short>(numberOfStrands());
        if (hasSegments())
        {
            if (!input.read(reinterpret_cast<char*>(strandSegments.data()), numberOfStrands() * sizeof(unsigned short)))
                std::cout << "Hair-file error: Cannot read segments." << std::endl;
        }
        else
        {
            std::fill(strandSegments.begin(), strandSegments.end(), defaultNumberOfSegments());
        }

        int density_int_part = unsigned int(m_density)-1;
        float density_dec_part = m_density - (float)density_int_part - 1 ;
         
        std::vector<unsigned short> strandSegments_copy = strandSegments;

        // copy the strandSegments density_int_part times 
        for (unsigned int i = 0; i < density_int_part; i++)
        {
            strandSegments.insert(strandSegments.end(), strandSegments_copy.begin(), strandSegments_copy.end());
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis_uni(0, 1);//uniform distribution between 0 and 1

        std::vector<unsigned int> duplicated_indices;

        for (unsigned int i = 0; i < numberOfStrands(); i++) {
            float rand = dis_uni(gen);
            if (rand <= density_dec_part)
            {
                strandSegments.push_back(strandSegments_copy[i]);
                duplicated_indices.push_back(i);
            }

        }

        // Compute strands vector<unsigned int>. Each element is the index to the
        // first point of the first segment of the strand. The last entry is the
        // index "one beyond the last vertex".
        m_strands = std::vector<int>(strandSegments.size() + 1);
        auto strand = m_strands.begin();
        *strand++ = 0;
        for (auto segments : strandSegments)
        {
            *strand = *(strand - 1) + 1 + segments;
            strand++;
        }

        // Points array(float)
        if (!hasPoints()) std::cout << "Hair-file error: File contains no points." << std::endl;
        m_points = std::vector<float3>(numberOfPoints());
        input.read(reinterpret_cast<char*>(m_points.data()), numberOfPoints() * sizeof(float3));

        // Complete vertex attributes
        VertexAttributes attribute;
        float3 point;
        for (unsigned int i = 0; i < numberOfPoints(); i++) {
            point.x = m_points[i].y;
            point.y = m_points[i].z;
            point.z = m_points[i].x; 
            m_points[i] = point;
            attribute.vertex = point;
 
            m_attributes.push_back(attribute);
        }

        std::normal_distribution<float> dis_norm(1.f, m_disparity/20.f);
        float rand_ratio; 

        for (unsigned int k = 0; k < density_int_part; k++)
        {
            for (unsigned int i = 0; i < numberOfStrands(); i++) {
                
                rand_ratio = dis_norm(gen);
                for (unsigned int j = m_strands[i]; j < m_strands[i+1]; j++) {
                    //std::cout << j << " ;" << std::endl;
                    point = m_points[j] * rand_ratio;
                    attribute.vertex = point;

                    m_points.push_back(point);
                    m_attributes.push_back(attribute);
                }
            }
        }
        
        for (auto i : duplicated_indices) {
            rand_ratio = dis_norm(gen);
            for (unsigned int j = m_strands[i]; j < m_strands[i+1]; j++) {
                point = m_points[j] * rand_ratio;
                attribute.vertex = point;

                m_points.push_back(point);
                m_attributes.push_back(attribute);
            }
        }

        m_header.numStrands = (uint32_t)strandSegments.size();
        m_header.numPoints = (uint32_t)m_attributes.size();

        if (!input) std::cout << "Hair-file error: Cannot read points." << std::endl;

        // Thickness array(float)
        m_thickness = std::vector<float>(numberOfPoints());
        if (hasThickness())
        {
            if (!input.read(reinterpret_cast<char*>(m_thickness.data()), numberOfPoints() * sizeof(float))) std::cout << "Hair-file error: Cannot read thickness." << std::endl;
        }
        else
        {
            std::fill(m_thickness.begin(), m_thickness.end(), defaultThickness());
        }

        if (hasAlpha()) std::cout << "Not implemented: Alpha data." << std::endl;
        if (hasColor()) std::cout << "Not implemented: Color data." << std::endl;

        //
        // Compute the axis-aligned bounding box for this hair geometry.
        //
        // expand the aabb by the maximum hair radius
        float max_width = defaultThickness();
        if (hasThickness())
        {
            max_width = *std::max_element(m_thickness.begin(), m_thickness.end());
        }
        std::cout << *this << std::endl;
    }


    void Curves::createHairFromFile(const std::string& fileName, const bool side)
    {
        std::ifstream input(fileName.c_str(), std::ios::binary);
        if (!input.is_open()) {
            std::cout << "Unable to open " + fileName + "." << std::endl;
        }

        input.read(reinterpret_cast<char*>(&m_header), sizeof(FileHeader));
        if (strncmp(m_header.magic, "HAIR", 4) != 0)
            std::cout << "Hair-file error: Invalid file format." + fileName << std::endl;
        m_header.fileInfo[87] = 0;
        m_header.defaultThickness = 0.009f;

        // Segments array(unsigned short)
        // The segements array contains the number of linear segments per strand;
        // thus there are segments + 1 control-points/vertices per strand.
        auto strandSegments = std::vector<unsigned short>(numberOfStrands());
        if (hasSegments())
        {
            if (!input.read(reinterpret_cast<char*>(strandSegments.data()), numberOfStrands() * sizeof(unsigned short)))
                std::cout << "Hair-file error: Cannot read segments." << std::endl;
        }
        else
        {
            std::fill(strandSegments.begin(), strandSegments.end(), defaultNumberOfSegments());
        }

        int density_int_part = unsigned int(m_density) - 1;
        float density_dec_part = m_density - (float)density_int_part - 1;

        std::vector<unsigned short> strandSegments_copy = strandSegments;

        // copy the strandSegments density_int_part times 
        for (unsigned int i = 0; i < density_int_part; i++)
        {
            strandSegments.insert(strandSegments.end(), strandSegments_copy.begin(), strandSegments_copy.end());
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis_uni(0, 1);//uniform distribution between 0 and 1

        std::vector<unsigned int> duplicated_indices;

        for (unsigned int i = 0; i < numberOfStrands(); i++) {
            float rand = dis_uni(gen);
            if (rand <= density_dec_part)
            {
                strandSegments.push_back(strandSegments_copy[i]);
                duplicated_indices.push_back(i);
            }

        }

        // Compute strands vector<unsigned int>. Each element is the index to the
        // first point of the first segment of the strand. The last entry is the
        // index "one beyond the last vertex".
        m_strands = std::vector<int>(strandSegments.size() + 1);
        auto strand = m_strands.begin();
        *strand++ = 0;
        for (auto segments : strandSegments)
        {
            *strand = *(strand - 1) + 1 + segments;
            strand++;
        }

        // Points array(float)
        if (!hasPoints()) std::cout << "Hair-file error: File contains no points." << std::endl;
        m_points = std::vector<float3>(numberOfPoints());
        input.read(reinterpret_cast<char*>(m_points.data()), numberOfPoints() * sizeof(float3));

        // Complete vertex attributes
        VertexAttributes attribute;
        float3 point;
        if (side==true) {
            for (unsigned int i = 0; i < numberOfPoints(); i++) {
                point.x = fabsf(m_points[i].y);
                point.y = m_points[i].z;
                point.z = m_points[i].x;
                m_points[i] = point;
                attribute.vertex = point;

                m_attributes.push_back(attribute);
            }
        }
        else {
            for (unsigned int i = 0; i < numberOfPoints(); i++) {
                point.x = -fabsf(m_points[i].y);
                point.y = m_points[i].z;
                point.z = m_points[i].x;
                m_points[i] = point;
                attribute.vertex = point;

                m_attributes.push_back(attribute);
            }
        }
        std::normal_distribution<float> dis_norm(1.f, m_disparity / 20.f);
        float rand_ratio;

        for (unsigned int k = 0; k < density_int_part; k++)
        {
            for (unsigned int i = 0; i < numberOfStrands(); i++) {

                rand_ratio = dis_norm(gen);
                for (unsigned int j = m_strands[i]; j < m_strands[i + 1]; j++) {
                    //std::cout << j << " ;" << std::endl;
                    point = m_points[j] * rand_ratio;
                    attribute.vertex = point;

                    m_points.push_back(point);
                    m_attributes.push_back(attribute);
                }
            }
        }

        for (auto i : duplicated_indices) {
            rand_ratio = dis_norm(gen);
            for (unsigned int j = m_strands[i]; j < m_strands[i + 1]; j++) {
                point = m_points[j] * rand_ratio;
                attribute.vertex = point;

                m_points.push_back(point);
                m_attributes.push_back(attribute);
            }
        }

        m_header.numStrands = (uint32_t)strandSegments.size();
        m_header.numPoints = (uint32_t)m_attributes.size();

        if (!input) std::cout << "Hair-file error: Cannot read points." << std::endl;

        // Thickness array(float)
        m_thickness = std::vector<float>(numberOfPoints());
        if (hasThickness())
        {
            if (!input.read(reinterpret_cast<char*>(m_thickness.data()), numberOfPoints() * sizeof(float))) std::cout << "Hair-file error: Cannot read thickness." << std::endl;
        }
        else
        {
            std::fill(m_thickness.begin(), m_thickness.end(), defaultThickness());
        }

        if (hasAlpha()) std::cout << "Not implemented: Alpha data." << std::endl;
        if (hasColor()) std::cout << "Not implemented: Color data." << std::endl;

        //
        // Compute the axis-aligned bounding box for this hair geometry.
        //
        // expand the aabb by the maximum hair radius
        float max_width = defaultThickness();
        if (hasThickness())
        {
            max_width = *std::max_element(m_thickness.begin(), m_thickness.end());
        }
        std::cout << *this << std::endl;
    }

    Curves::~Curves() {}

    std::string Curves::programName() const
    {
        switch (m_splineMode) {
        case LINEAR_BSPLINE:
            return "hitLinearCurve";
        case QUADRATIC_BSPLINE:
            return "hitQuadraticCurve";
        case CUBIC_BSPLINE:
            return "hitCubicCurve";
        default:
            std::cout << "Invalid b-spline mode !" << std::endl;
        }

        return "";
    }

    std::string Curves::programSuffix() const
    {
        switch (m_shadeMode) {
        case SEGMENT_U:
            return "SegmentU";
        case STRAND_U:
            return "StrandU";
        case STRAND_IDX:
            return "StrandIndex";
        default:
            std::cout << "Invalid hair-shading mode" << std::endl;
        }

        return "";
    }

    uint32_t Curves::numberOfStrands() const
    {
        return m_header.numStrands;
    }

    uint32_t Curves::numberOfPoints() const
    {
        return m_header.numPoints;
    }

    uint32_t Curves::defaultNumberOfSegments() const
    {
        return m_header.defaultNumSegments;
    }

    float Curves::defaultThickness() const
    {
        return m_header.defaultThickness;
    }

    float Curves::defaultAlpha() const
    {
        return m_header.defaultAlpha;
    }

    float3 Curves::defaultColor() const
    {
        return make_float3(m_header.defaultColor.x, m_header.defaultColor.y, m_header.defaultColor.z);
    }

    std::string Curves::fileInfo() const
    {
        return std::string(m_header.fileInfo);
    }

    bool Curves::hasSegments() const
    {
        return (m_header.flags & (0x1 << 0)) > 0;
    }

    bool Curves::hasPoints() const
    {
        return (m_header.flags & (0x1 << 1)) > 0;
    }

    bool Curves::hasThickness() const
    {
        return (m_header.flags & (0x1 << 2)) > 0;
    }

    bool Curves::hasAlpha() const
    {
        return (m_header.flags & (0x1 << 3)) > 0;
    }

    bool Curves::hasColor() const
    {
        return (m_header.flags & (0x1 << 4)) > 0;
    }

    std::vector<float3> Curves::points() const
    {
        return m_points;
    }

    std::vector<float> Curves::widths() const
    {
        return m_thickness;
    }

    int Curves::numberOfSegments() const
    {
        return numberOfPoints() - numberOfStrands() * curveDegree();
    }

    // Compute a list of all segment indices making up the curves array.
    //
    // The structure of the list is as follows:
    // * For each strand all segments are listed in order from root to tip.
    // * Segment indices are identical to the index of the first control-point
    //   of a segment.
    // * The number of segments per strand is dependent on the curve degree; e.g.
    //   a cubic segment requires four control points, thus a cubic strand with n
    //   control points will have (n - 3) segments.
    //
    std::vector<int unsigned> Curves::segments() const
    {
        std::vector<unsigned int> segments;
        // loop to one before end, as last strand value is the "past last valid vertex"
        // index
        for (auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand)
        {
            const int start = *(strand);                      // first vertex in first segment
            const int end = *(strand + 1) - curveDegree();  // second vertex of last segment
            for (int i = start; i < end; ++i)
            {
                segments.push_back(i);
            }
        }

        return segments;
    }

    std::vector<float2> Curves::strandU() const
    {
        std::vector<float2> strand_u;
        for (auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand)
        {
            const int   start = *(strand);
            const int   end = *(strand + 1) - curveDegree();
            const int   segments = end - start;  // number of strand's segments
            const float scale = 1.0f / segments;
            for (int i = 0; i < segments; ++i)
            {
                strand_u.push_back(make_float2(i * scale, scale));
            }
        }

        return strand_u;
    }

    std::vector<int> Curves::strandIndices() const
    {
        std::vector<int> strandIndices;
        int              strandIndex = 0;
        for (auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand)
        {
            const int start = *(strand);
            const int end = *(strand + 1) - curveDegree();
            for (auto segment = start; segment != end; ++segment)
            {
                strandIndices.push_back(strandIndex);
            }
            ++strandIndex;
        }

        return strandIndices;
    }

    std::vector<float3> Curves::strandRand() const
    {
        std::vector<float3> strandIndices;
        int              strandIndex = 0;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis_uni(0, 1);//uniform distribution between 0 and 1
        std::normal_distribution<float> dis_norm(0, 1);
        for (auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand)
        {
            float n_1 = dis_uni(gen);
            float n_2 = dis_norm(gen);
            float n_3 = dis_norm(gen);

            const int start = *(strand);
            const int end = *(strand + 1) - curveDegree();
            for (auto segment = start; segment != end; ++segment)
            {
                strandIndices.push_back(make_float3(n_1,n_2,n_3));
            }
            ++strandIndex;
        }

        return strandIndices;
    }

    std::vector<uint2> Curves::strandInfo() const
    {
        std::vector<uint2> strandInfo;
        unsigned int       firstPrimitiveIndex = 0;
        for (auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand)
        {
            uint2 info;
            info.x = firstPrimitiveIndex;                        // strand's start index
            info.y = *(strand + 1) - *(strand)-curveDegree();  // number of segments in strand
            firstPrimitiveIndex += info.y;                       // increment with number of primitives/segments in strand
            strandInfo.push_back(info);
        }
        return strandInfo;
    }

    void Curves::setRadiusMode(Radius radiusMode)
    {
        if (m_radiusMode != radiusMode)
        {
            m_radiusMode = radiusMode;
            if (CONSTANT_R == m_radiusMode)
            {
                // assign all radii the root radius
                const float r = m_thickness[0];
                for (auto ir = m_thickness.begin(); ir != m_thickness.end(); ++ir)
                    *ir = r;
            }
            else if (TAPERED_R == m_radiusMode)
            {
                const float r = m_thickness[0];
                for (auto strand = m_strands.begin(); strand != m_strands.end() - 1; ++strand)
                {
                    const int rootVertex = *(strand);
                    const int vertices = *(strand + 1) - rootVertex;  // vertices in strand
                    for (int i = 0; i < vertices; ++i)
                    {
                        m_thickness[rootVertex + i] = r * (vertices - 1 - i) / static_cast<float>(vertices - 1);
                    }
                }
            }
        }
    }

    std::string toString(bool b)
    {
        std::string result;
        if (b)
            result = "true";
        else
            result = "false";
        return result;
    }

    std::ostream& operator<<(std::ostream& o, Curves::SplineMode splineMode)
    {
        switch (splineMode)
        {
        case Curves::LINEAR_BSPLINE:
            o << "LINEAR_BSPLINE";
            break;
        case Curves::QUADRATIC_BSPLINE:
            o << "QUADRATIC_BSPLINE";
            break;
        case Curves::CUBIC_BSPLINE:
            o << "CUBIC_BSPLINE";
            break;
        default:
            std::cout << "Invalid spline mode." << std::endl;
        }

        return o;
    }

    std::ostream& operator<<(std::ostream& o, const Curves& curves)
    {
        o << "Hair: " << std::endl;
        o << "Number of strands:          " << curves.numberOfStrands() << std::endl;
        o << "Number of points:           " << curves.numberOfPoints() << std::endl;
        o << "Spline mode:                " << curves.m_splineMode << std::endl;
        o << "Contains segments:          " << toString(curves.hasSegments()) << std::endl;
        o << "Contains points:            " << toString(curves.hasPoints()) << std::endl;
        o << "Contains alpha:             " << toString(curves.hasAlpha()) << std::endl;
        o << "Contains color:             " << toString(curves.hasColor()) << std::endl;
        o << "Default number of segments: " << curves.defaultNumberOfSegments() << std::endl;
        o << "Default thickness:          " << curves.defaultThickness() << std::endl;
        o << "Default alpha:              " << curves.defaultAlpha() << std::endl;
        float3 color = curves.defaultColor();
        o << "Default color:              (" << color.x << ", " << color.y << ", " << color.z << ")" << std::endl;
        std::string fileInfo =curves.fileInfo();
        o << "File info:                  ";
        if (fileInfo.empty())
            o << "n/a" << std::endl;
        else
            o << fileInfo << std::endl;

        o << "Strands: [" << curves.m_strands[0] << "..." << curves.m_strands[curves.m_strands.size() - 1] << "]" << std::endl;
        o << "Points: [" << curves.m_points[0] << "..." << curves.m_points[curves.m_points.size() - 1] << "]" << std::endl;
        o << "Thickness: [" << curves.m_thickness[0] << "..." << curves.m_thickness[curves.m_thickness.size() - 1] << "]" << std::endl;
        o << "Segments: " << curves.segments().size() << std::endl;
        return o;
    }

}