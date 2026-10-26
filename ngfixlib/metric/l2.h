#pragma once

#include "space.h"
#include <immintrin.h>
#include <cstddef>

namespace ngfixlib {

class L2Space_float : public Space<float>
{
public:

    virtual float dist_func(const float* vec0,
                            const float* vec1) {

        float result = 0.0f;
        size_t i = 0;

        // ========================================================
        // AVX-512
        // ========================================================
#if defined(__AVX512F__)

        size_t mul16 = dim - (dim % 16);

        __m512 sum = _mm512_setzero_ps();

        for (; i < mul16; i += 16) {

            __m512 x = _mm512_loadu_ps(&vec0[i]);
            __m512 y = _mm512_loadu_ps(&vec1[i]);

            __m512 diff = _mm512_sub_ps(x, y);

#if defined(__FMA__)
            sum = _mm512_fmadd_ps(diff, diff, sum);
#else
            sum = _mm512_add_ps(
                sum,
                _mm512_mul_ps(diff, diff)
            );
#endif
        }

        result = _mm512_reduce_add_ps(sum);


        // ========================================================
        // AVX2
        // Your current server will compile/use this branch
        // ========================================================
#elif defined(__AVX2__)

        size_t mul8 = dim - (dim % 8);

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        // Process 32 floats each iteration
        for (; i + 31 < mul8; i += 32) {

            __m256 x0 = _mm256_loadu_ps(&vec0[i]);
            __m256 y0 = _mm256_loadu_ps(&vec1[i]);

            __m256 x1 = _mm256_loadu_ps(&vec0[i + 8]);
            __m256 y1 = _mm256_loadu_ps(&vec1[i + 8]);

            __m256 x2 = _mm256_loadu_ps(&vec0[i + 16]);
            __m256 y2 = _mm256_loadu_ps(&vec1[i + 16]);

            __m256 x3 = _mm256_loadu_ps(&vec0[i + 24]);
            __m256 y3 = _mm256_loadu_ps(&vec1[i + 24]);

            __m256 d0 = _mm256_sub_ps(x0, y0);
            __m256 d1 = _mm256_sub_ps(x1, y1);
            __m256 d2 = _mm256_sub_ps(x2, y2);
            __m256 d3 = _mm256_sub_ps(x3, y3);

            sum0 = _mm256_add_ps(
                sum0,
                _mm256_mul_ps(d0, d0)
            );

            sum1 = _mm256_add_ps(
                sum1,
                _mm256_mul_ps(d1, d1)
            );

            sum2 = _mm256_add_ps(
                sum2,
                _mm256_mul_ps(d2, d2)
            );

            sum3 = _mm256_add_ps(
                sum3,
                _mm256_mul_ps(d3, d3)
            );
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        // Remaining blocks of 8 floats
        for (; i + 7 < mul8; i += 8) {

            __m256 x = _mm256_loadu_ps(&vec0[i]);
            __m256 y = _mm256_loadu_ps(&vec1[i]);

            __m256 diff = _mm256_sub_ps(x, y);

            sum0 = _mm256_add_ps(
                sum0,
                _mm256_mul_ps(diff, diff)
            );
        }

        // Horizontal reduction
        float tmp[8];

        _mm256_storeu_ps(tmp, sum0);

        result =
            tmp[0] +
            tmp[1] +
            tmp[2] +
            tmp[3] +
            tmp[4] +
            tmp[5] +
            tmp[6] +
            tmp[7];


        // ========================================================
        // No SIMD
        // ========================================================
#else

        i = 0;

#endif


        // ========================================================
        // Scalar tail
        // ========================================================
        for (; i < dim; ++i) {

            float diff = vec0[i] - vec1[i];

            result += diff * diff;
        }

        return result;
    }


    L2Space_float(size_t _dimension_)
        : Space<float>(_dimension_) {}
};

}  // namespace ngfixlib