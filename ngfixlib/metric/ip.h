#pragma once

#include "space.h"
#include <immintrin.h>
#include <stdexcept>
#include <cstddef>

namespace ngfixlib {

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

class IPSpace_float : public Space<float>
{
public:
    float (IPSpace_float::*fp)(const float*, const float*);

    virtual float dist_func(const float* vec0, const float* vec1) {
        return (this->*fp)(vec0, vec1);
    }

    // ============================================================
    // Scalar fallback
    // d(x,y) = 1 - x dot y
    // ============================================================
    float ipDistanceScalar(const float *pVect1,
                           const float *pVect2) {
        float sum = 0.0f;

        for (size_t i = 0; i < dim; ++i) {
            sum += pVect1[i] * pVect2[i];
        }

        return 1.0f - sum;
    }


    // ============================================================
    // AVX2 implementation
    //
    // AVX2 processes 8 float values per 256-bit vector.
    // Does NOT require FMA: use mul + add explicitly.
    // ============================================================
#if defined(__AVX2__)

    float ipDistanceAVX2(const float *pVect1,
                         const float *pVect2) {

        size_t i = 0;

        // Four independent accumulators improve instruction-level
        // parallelism during the dot product.
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        // Process 32 floats per iteration.
        for (; i + 31 < dim; i += 32) {

            __m256 a0 = _mm256_loadu_ps(pVect1 + i);
            __m256 b0 = _mm256_loadu_ps(pVect2 + i);

            __m256 a1 = _mm256_loadu_ps(pVect1 + i + 8);
            __m256 b1 = _mm256_loadu_ps(pVect2 + i + 8);

            __m256 a2 = _mm256_loadu_ps(pVect1 + i + 16);
            __m256 b2 = _mm256_loadu_ps(pVect2 + i + 16);

            __m256 a3 = _mm256_loadu_ps(pVect1 + i + 24);
            __m256 b3 = _mm256_loadu_ps(pVect2 + i + 24);

            sum0 = _mm256_add_ps(
                sum0,
                _mm256_mul_ps(a0, b0)
            );

            sum1 = _mm256_add_ps(
                sum1,
                _mm256_mul_ps(a1, b1)
            );

            sum2 = _mm256_add_ps(
                sum2,
                _mm256_mul_ps(a2, b2)
            );

            sum3 = _mm256_add_ps(
                sum3,
                _mm256_mul_ps(a3, b3)
            );
        }

        // Combine four accumulators.
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        // Process remaining blocks of 8 floats.
        for (; i + 7 < dim; i += 8) {

            __m256 a = _mm256_loadu_ps(pVect1 + i);
            __m256 b = _mm256_loadu_ps(pVect2 + i);

            sum0 = _mm256_add_ps(
                sum0,
                _mm256_mul_ps(a, b)
            );
        }

        // Horizontal reduction of 8 floats.
        float tmp[8];

        _mm256_storeu_ps(tmp, sum0);

        float sum =
            tmp[0] + tmp[1] +
            tmp[2] + tmp[3] +
            tmp[4] + tmp[5] +
            tmp[6] + tmp[7];

        // Scalar tail for dimensions not divisible by 8.
        for (; i < dim; ++i) {
            sum += pVect1[i] * pVect2[i];
        }

        return 1.0f - sum;
    }

#endif  // __AVX2__


    // ============================================================
    // Original AVX-512 implementation
    //
    // IMPORTANT:
    // Entire function bodies are hidden from the compiler when
    // AVX-512 is unavailable. Otherwise GCC can report
    // "target specific option mismatch".
    // ============================================================
#if defined(__AVX512F__) && defined(__FMA__)

    float ipDistanceSIMD16ExtAVX512(const float *pVect1v,
                                    const float *pVect2v) {

        const float *pVect1 = pVect1v;
        const float *pVect2 = pVect2v;

        size_t qty = dim;
        size_t qty16 = qty / 16;

        const float *pEnd1 = pVect1 + 16 * qty16;

        __m512 sum512 = _mm512_setzero_ps();

        size_t loop = qty16 / 4;

        while (loop--) {

            __m512 v1 = _mm512_loadu_ps(pVect1);
            __m512 v2 = _mm512_loadu_ps(pVect2);

            pVect1 += 16;
            pVect2 += 16;

            __m512 v3 = _mm512_loadu_ps(pVect1);
            __m512 v4 = _mm512_loadu_ps(pVect2);

            pVect1 += 16;
            pVect2 += 16;

            __m512 v5 = _mm512_loadu_ps(pVect1);
            __m512 v6 = _mm512_loadu_ps(pVect2);

            pVect1 += 16;
            pVect2 += 16;

            __m512 v7 = _mm512_loadu_ps(pVect1);
            __m512 v8 = _mm512_loadu_ps(pVect2);

            pVect1 += 16;
            pVect2 += 16;

            sum512 = _mm512_fmadd_ps(v1, v2, sum512);
            sum512 = _mm512_fmadd_ps(v3, v4, sum512);
            sum512 = _mm512_fmadd_ps(v5, v6, sum512);
            sum512 = _mm512_fmadd_ps(v7, v8, sum512);
        }

        while (pVect1 < pEnd1) {

            __m512 v1 = _mm512_loadu_ps(pVect1);
            __m512 v2 = _mm512_loadu_ps(pVect2);

            pVect1 += 16;
            pVect2 += 16;

            sum512 =
                _mm512_fmadd_ps(v1, v2, sum512);
        }

        float sum = _mm512_reduce_add_ps(sum512);

        return 1.0f - sum;
    }


    float ipDistanceSIMD4ExtAVX512(const float *pVect1v,
                                   const float *pVect2v) {

        const float *pVect1 = pVect1v;
        const float *pVect2 = pVect2v;

        size_t qty = dim;

        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        const float *pEnd1 =
            pVect1 + 16 * qty16;

        const float *pEnd2 =
            pVect1 + 4 * qty4;

        __m512 sum512 =
            _mm512_setzero_ps();

        size_t loop =
            qty16 / 4;

        while (loop--) {

            __m512 v1 =
                _mm512_loadu_ps(pVect1);

            __m512 v2 =
                _mm512_loadu_ps(pVect2);

            pVect1 += 16;
            pVect2 += 16;

            __m512 v3 =
                _mm512_loadu_ps(pVect1);

            __m512 v4 =
                _mm512_loadu_ps(pVect2);

            pVect1 += 16;
            pVect2 += 16;

            __m512 v5 =
                _mm512_loadu_ps(pVect1);

            __m512 v6 =
                _mm512_loadu_ps(pVect2);

            pVect1 += 16;
            pVect2 += 16;

            __m512 v7 =
                _mm512_loadu_ps(pVect1);

            __m512 v8 =
                _mm512_loadu_ps(pVect2);

            pVect1 += 16;
            pVect2 += 16;

            sum512 =
                _mm512_fmadd_ps(v1, v2, sum512);

            sum512 =
                _mm512_fmadd_ps(v3, v4, sum512);

            sum512 =
                _mm512_fmadd_ps(v5, v6, sum512);

            sum512 =
                _mm512_fmadd_ps(v7, v8, sum512);
        }

        while (pVect1 < pEnd1) {

            __m512 v1 =
                _mm512_loadu_ps(pVect1);

            __m512 v2 =
                _mm512_loadu_ps(pVect2);

            pVect1 += 16;
            pVect2 += 16;

            sum512 =
                _mm512_fmadd_ps(v1, v2, sum512);
        }

        float sum =
            _mm512_reduce_add_ps(sum512);

        __m128 sum128 =
            _mm_setzero_ps();

        while (pVect1 < pEnd2) {

            __m128 v1 =
                _mm_loadu_ps(pVect1);

            __m128 v2 =
                _mm_loadu_ps(pVect2);

            pVect1 += 4;
            pVect2 += 4;

            sum128 =
                _mm_fmadd_ps(v1, v2, sum128);
        }

        sum128 =
            _mm_hadd_ps(sum128, sum128);

        sum128 =
            _mm_hadd_ps(sum128, sum128);

        sum +=
            _mm_cvtss_f32(sum128);

        // scalar tail
        while (pVect1 < pVect1v + qty) {
            sum += (*pVect1) * (*pVect2);

            ++pVect1;
            ++pVect2;
        }

        return 1.0f - sum;
    }

#endif  // __AVX512F__ && __FMA__


    // ============================================================
    // Automatically select the fastest implementation supported
    // by the machine used to compile NGFix.
    // ============================================================
    IPSpace_float(size_t _dimension_)
        : Space<float>(_dimension_) {

#if defined(__AVX512F__) && defined(__FMA__)

        // Keep original NGFix AVX-512 path.
        if (dim % 16 == 0) {
            fp =
                &IPSpace_float::
                ipDistanceSIMD16ExtAVX512;
        }
        else if (dim % 4 == 0) {
            fp =
                &IPSpace_float::
                ipDistanceSIMD4ExtAVX512;
        }
        else {
            fp =
                &IPSpace_float::
                ipDistanceScalar;
        }

#elif defined(__AVX2__)

        // Your current machine will enter here.
        fp =
            &IPSpace_float::
            ipDistanceAVX2;

#else

        fp =
            &IPSpace_float::
            ipDistanceScalar;

#endif
    }
};

} // namespace ngfixlib