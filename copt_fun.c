#include "copt_fun.h"
#include "stdio.h"
#include "threads.h"
#include "stdlib.h"
#include <immintrin.h>
#include <stdint.h>
#include <pthread.h>

/******************************************************************************
 * Usage: copt OP N LOOP
 *
 * copt measures the execution time impact of source level optimizations
 * in C. copt runs and times an unoptimized and optimized version of a given
 * operation and input size.
 *
 * Arguments description:
 * OP is the operation to run for this invocation of copt. There are four
 * possible operations, each of which takes exactly one argument N:
 *   0: initialize a pair of square integer matrices. N is the size of the
 *      matrices.
 *   1: initialize an integer array. N is the length of the array
 *   2: compute factorial with a recursive routine. N is the number for
 *      which the routine computes the factorial
 *   3: multiply two square integer matrices. N is the size of the matrix
 *
 * LOOP is the number of times to run the given operation with the given
 * argument. Timing starts before the first operation begins and ends when
 * the last operation has completed.
 *
 * OP, N, and LOOP all must be integers <= INT_MAX
 ******************************************************************************/

/******************************************************************************
 * Name:        Jacob Hebert
 * Collaboration: Some help from GPT and Stack Overflow
 ******************************************************************************/

int check(int x, int y)
{
  return x < y;
}

void set(int *mat, int i, int num)
{
  mat[i] = num;
}

void matrix_initialize_unopt(struct fn_args *args)
{
  int i, j, n;
  int *mat1, *mat2;

  n = args->n;
  mat1 = args->mem1;
  mat2 = args->mem2;

  for (i = 0; check(i, n); i++)
  {
    for (j = 0; check(j, n); j++)
    {
      set(mat1, i * n + j, i);
      set(mat2, i * n + j, i + 1);
    }
  }
}

void matrix_initialize_opt(struct fn_args *args)
{
  // TODO: optimized implementation goes here

  // 1. Function inlining, removing call to check in i loop; No speedup
  // 2. function inlining for check, in j loop: ~1.2x speedup
  // 3. function inlining for set (int* int, int) for both matrices: 2.5x speedup total
  // 4. Common sub-expresion elimination in j loop (i*n) 3.2x speedup total
  // 5. Common sub-expresion elimination in j loop (i+1) - SLOWER
  // 6. Loop unrolling J by a factor of 2: 3.3x
  // 7. Loop unrolling j by a factor of 5: 4x
  // 8. Common sub-expresion elimination in j unroll - i_offset + j => 5x speedup
  // 9. Map offset and sum vars to registers
  // 10. Map mat 1 and mat 2 to register : 5.7x
  // 11 map loop control to register
  // 12. Changed i_offset from i*n to i+=n: 4.7x to 5.7x
  // 13. Moved to AVX2 SIMD instructions and vectors -> 15.3x speedup

  register int i, j, n;
  register int *mat1, *mat2;
  register int i_offset = 0;

  n = args->n;
  mat1 = args->mem1;
  mat2 = args->mem2;

  // Prepare vectors for initialization
  for (i = 0; i < n; i++)
  {
    __m256i vec_i = _mm256_set1_epi32(i);         // Vector with all elements set to 'i'
    __m256i vec_i_sum = _mm256_set1_epi32(i + 1); // Vector with all elements set to 'i + 1'

    for (j = 0; j < n; j += 8)
    { // Process 8 elements per iteration
      // Store the vectors into mat1 and mat2
      _mm256_storeu_si256((__m256i *)&mat1[i_offset + j], vec_i);
      _mm256_storeu_si256((__m256i *)&mat2[i_offset + j], vec_i_sum);
    }
    i_offset += n;
  }
}
/*
int i, j, n;
register int *mat1, *mat2;
register int i_offset = 0, i_sum, j_offset, j_sum;

n = args->n;
mat1 = args->mem1;
mat2 = args->mem2;

for (i = 0; i < n; i++)
{
  // i_offset = i*n;
  i_sum = i + 1;
  for (j = 0; j < n; j += 5)
  {
    j_offset = i_offset + j;
    mat1[j_offset] = i;
    mat1[j_offset + 1] = i;
    mat1[j_offset + 2] = i;
    mat1[j_offset + 3] = i;
    mat1[j_offset + 4] = i;
    mat2[i_offset + j] = i_sum;
    mat2[j_offset + 1] = i_sum;
    mat2[j_offset + 2] = i_sum;
    mat2[j_offset + 3] = i_sum;
    mat2[j_offset + 4] = i_sum;
  }
  i_offset += n;
}
}
*/
void array_initialize_unopt(struct fn_args *args)
{
  int i, mod, n, *arr;

  n = args->n;
  arr = args->mem1;
  for (i = 0; i < n; i++)
  {
    mod = X % Y;
    arr[i] = i * mod * Z;
  }
}
struct thread_data
{
  int32_t *arr, modZ;
  int startIdx, endIdx;
};

void *thread_func(void *arg)
{
  struct thread_data *data = (struct thread_data *)arg;
  int32_t *arr = data->arr;
  int startIdx = data->startIdx;
  int endIdx = data->endIdx;
  __m256i vec_modZ = _mm256_set1_epi32(data->modZ);

  for (int i = startIdx; i < endIdx; i += 8)
  {
    __m256i vec_i = _mm256_setr_epi32(i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7);
    __m256i vec_result = _mm256_mullo_epi32(vec_i, vec_modZ);
    _mm256_storeu_si256((__m256i *)(arr + i), vec_result);
  }

  return NULL;
}

#define NUM_THREADS 2

void array_initialize_opt(struct fn_args *args)
{
  // 1. removed X % Y from for loop: (1.2x)
  // 2. removed mod * Z calculation from the for loop: (1.0) slower :(
  // 3. Unrolled loop to 2 places (1.2x)
  // 4. Moved int variables to the registers (2.5x)
  // 5. Unrolled loop to 8 places (2.5), no change
  // I STILL NEED 2x MORE :(
  // 6. Moved modz to 1 line (2.5x), no change
  // 7. Moved i * modz calculation to imult, then offset by multiplying modz (2.5 :( )
  // 8. Moved to 6 place loop unroll (4.9x, half the time :/ ) - 3033/2050 - just 1.5x to go!
  // 9. Since X,Y,Z are constants which result in 64, removed the mod calculation (4.9) - This wasn't in the loop, so it wasn't significant
  // 10. unrolled loop to 12 places (2.8) - moving to args : 1 300,000 20,000 to better reflect test2 in makefile
  // 11. SIMD Instructions!!!
  // 12. Multithreading!! - using make test_arr_init scared me because it didnt have a large enough array to remove overhead, but using make test gives a different story - 5.3x speedup- BEATING 02!!!!
  // 13. Playing around with NUM_THREADS allowed me to determine that 2 threads is optimal in this specific usecase- 16.3x speedup!
  pthread_t threads[NUM_THREADS];
  struct thread_data thread_args[NUM_THREADS];
  int n = args->n;
  int32_t *arr = args->mem1;
  const int32_t modZ = (X % Y) * Z;

  int chunkSize = n / NUM_THREADS;
  for (int i = 0; i < NUM_THREADS; i++)
  {
    thread_args[i].arr = arr;
    thread_args[i].startIdx = i * chunkSize;
    thread_args[i].endIdx = (i + 1) * chunkSize;
    thread_args[i].modZ = modZ;
    if (i == NUM_THREADS - 1)
      thread_args[i].endIdx = n; // Ensure the last thread covers the remainder

    pthread_create(&threads[i], NULL, thread_func, (void *)&thread_args[i]);
  }

  for (int i = 0; i < NUM_THREADS; i++)
  {
    pthread_join(threads[i], NULL);
  }
}

unsigned long long factorial_unopt_helper(unsigned long long n)
{
  if (n == 0ull)
    return 1ull;
  return n * factorial_unopt_helper(n - 1);
}

void factorial_unopt(struct fn_args *args)
{
  args->fac = factorial_unopt_helper((unsigned long long)args->n);
  // printf("Unopt: %llu\n", args->fac);
}

// Leaving this here for the funnies
unsigned long long factorialTable[] =
    {1,                       // 0
     1,                       // 1
     2,                       // 2
     6,                       // 3
     24,                      // 4
     120,                     // 5
     720,                     // 6
     5040,                    // 7
     40320,                   // 8
     362880,                  // 9
     3628800,                 // 10
     39916800,                // 11
     479001600,               // 12
     6227020800ULL,           // 13
     87178291200ULL,          // 14
     1307674368000ULL,        // 15
     20922789888000ULL,       // 16
     355687428096000ULL,      // 17
     6402373705728000ULL,     // 18
     121645100408832000ULL,   // 19
     2432902008176640000ULL}; // 20

unsigned long long factorial_opt_helper(unsigned long long n)
{
  // 1. Not recursive, allows for easier optimization, 2.4x
  // 2. Put the whole thing in a switch statement, if you dont calculate it you dont need to worry about overhead :) (13.1x)
  // 3. Using a global array and using n as an indexer for it makes this significantly faster! (well, not as much for me :/ ) (14.6x)

  // this is gonna be ugly but I think it'd be funny
  if (n > 20 || n < 0)
  {
    return 0;
  }
  return factorialTable[n];

  // original switch statement code

  // 20! is almost 2^64, so it's the max given ull datatype, so if n > 20, it overflows!
  if (n == 0 || n == 1)
  {
    return 1;
  }
  if (n > 20)
  {
    return 0;
  }
  switch (n)
  {
  case 2:
    return 2;
  case 3:
    return 6;
  case 4:
    return 24;
  case 5:
    return 120;
  case 6:
    return 720;
  case 7:
    return 5040;
  case 8:
    return 40320;
  case 9:
    return 362880;
  case 10:
    return 3628800;
  case 11:
    return 39916800;
  case 12:
    return 479001600;
  case 13:
    return 6227020800ULL;
  case 14:
    return 87178291200ULL;
  case 15:
    return 1307674368000ULL;
  case 16:
    return 20922789888000ULL;
  case 17:
    return 355687428096000ULL;
  case 18:
    return 6402373705728000ULL;
  case 19:
    return 121645100408832000ULL;
  case 20:
    return 2432902008176640000ULL;
  default:
  }
}

void factorial_opt(struct fn_args *args)
{
  args->fac = factorial_opt_helper((unsigned long long)args->n);
}

void matrix_multiply_unopt(struct fn_args *args)
{
  int i, j, k, n;
  int *mat1, *mat2, *res;

  n = args->n;
  mat1 = args->mem1;
  mat2 = args->mem2;
  res = args->mem3;

  for (i = 0; i < n; i++)
  {
    for (j = 0; j < n; j++)
    {
      res[i * n + j] = 0;
      for (k = 0; k < n; k++)
      {
        res[i * n + j] += mat1[i * n + k] * mat2[k * n + j];
      }
    }
  }
}

// Transposes input matrix in order to make the below algorithm even faster
void transpose_matrix(int *src, int *dst, int n)
{
  for (register int i = 0; i < n; ++i)
  {
    for (register int j = 0; j < n; ++j)
    {
      dst[j * n + i] = src[i * n + j];
    }
  }
}

#define TILE_SIZE 32
#define THREAD_COUNT 8
typedef struct
{
  int startRow;
  int endRow;
  int n;
  int *mat1;
  int *transposed_mat2;
  int *res;
} ThreadData;

void mat_thread_func(void *arg)
{
  ThreadData *data = (ThreadData *)arg;
  int startRow = data->startRow;
  int endRow = data->endRow;
  register int n = data->n;
  int *mat1 = data->mat1;
  int *transposed_mat2 = data->transposed_mat2;
  int *res = data->res;

  for (int i = startRow; i < endRow; i += TILE_SIZE)
  {
    for (int j = 0; j < n; j += TILE_SIZE)
    {
      for (int k = 0; k < n; k += TILE_SIZE)
      {
        for (int ii = i; ii < i + TILE_SIZE && ii < endRow; ++ii)
        {
          for (register int jj = j; jj < j + TILE_SIZE; jj += 8)
          {
            __m256i temp = _mm256_loadu_si256((__m256i *)&res[ii * n + jj]);
            for (register int kk = k; kk < k + TILE_SIZE; ++kk)
            {
              __m256i mat1_vec = _mm256_set1_epi32(mat1[ii * n + kk]);
              __m256i mat2_vec = _mm256_loadu_si256((__m256i *)&transposed_mat2[jj * n + kk]);
              temp = _mm256_add_epi32(temp, _mm256_mullo_epi32(mat1_vec, mat2_vec));
            }
            _mm256_storeu_si256((__m256i *)&res[ii * n + jj], temp);
          }
        }
      }
    }
  }
}

void matrix_multiply_opt(struct fn_args *args)
{
  // 1. Added n offset variable to reduce redundant calculations (1.2x speedup)
  // 2. Moved vars to registers (1.5x speedup)
  // 3. Added joffset variable to reduce redundant calculations (1.8x speedup) Need to get below 3033 ms, currently at 7783 ms
  // 4. Tiled matrix - 2.4x at 32
  // 5. Moved to AVX2 256-bit vectorization - 4.3x
  // 6. Transposed mat2 to make the multiplication horizontal by horizontal - 4.3...?
  // 7. Moved some loop control variables to registers - 4.5
  // 8. Multithreading - variable performance- 4.5x

  /* Pre tiled code

  register int i, j, k, n, noffset, joffset;
  register int *mat1, *mat2, *res;

  n = args->n;
  mat1 = args->mem1;
  mat2 = args->mem2;
  res = args->mem3;

  for (i = 0; i < n; i++)
  {
    noffset = i * n;
    for (j = 0; j < n; j++)
    {

      joffset = noffset + j;

      res[joffset] = 0;

      for (k = 0; k < n; k++)
      {
        res[joffset] += mat1[noffset + k] * mat2[k * n + j];
      }
    }
  }
  */

  /* Tiled mult
   register int i, j, k, ii, jj, kk, n;
   register int *mat1, *mat2, *res;

   n = args->n;
   mat1 = args->mem1;
   mat2 = args->mem2;
   res = args->mem3;

   for (i = 0; i < n; i += TILE_SIZE)
   {
     for (j = 0; j < n; j += TILE_SIZE)
     {
       for (k = 0; k < n; k += TILE_SIZE)
       {
         for (ii = i; ii < i + TILE_SIZE; ++ii)
         {
           for (jj = j; jj < j + TILE_SIZE; ++jj)
           {
             register int temp = res[ii * n + jj];
             for (kk = k; kk < k + TILE_SIZE; ++kk)
             {
               temp += mat1[ii * n + kk] * mat2[kk * n + jj];
             }
             res[ii * n + jj] = temp;
           }
         }
       }
     }
   }
   */

  /* AVX2 mult
  int n = args->n;
  int *mat1 = args->mem1;
  int *mat2 = args->mem2;
  int *res = args->mem3;

  int temp_sum[TILE_SIZE][TILE_SIZE] = {0};

  for (int i = 0; i < n; i += TILE_SIZE)
  {
    for (int j = 0; j < n; j += TILE_SIZE)
    {
      for (int k = 0; k < n; k += TILE_SIZE)
      {
        for (int ii = i; ii < i + TILE_SIZE; ++ii)
        {
          for (int jj = j; jj < j + TILE_SIZE; jj += 8)
          {
            __m256i temp = _mm256_loadu_si256((__m256i *)&res[ii * n + jj]);
            for (int kk = k; kk < k + TILE_SIZE; ++kk)
            {
              __m256i mat1_vec = _mm256_set1_epi32(mat1[ii * n + kk]);
              __m256i mat2_vec = _mm256_loadu_si256((__m256i *)&mat2[kk * n + jj]);
              temp = _mm256_add_epi32(temp, _mm256_mullo_epi32(mat1_vec, mat2_vec));
            }
            _mm256_storeu_si256((__m256i *)&res[ii * n + jj], temp);
          }
        }
      }
    }
  }
  */

  /*loop control to registers
  register int n = args->n;
  register int *mat1 = args->mem1;
  int *transposed_mat2 = (int *)malloc(n * n * sizeof(int)); // Allocate memory for the transposed matrix
  int *res = args->mem3;

  // Transpose mat2
  transpose_matrix(args->mem2, transposed_mat2, n);

  for (int i = 0; i < n; i += TILE_SIZE)
  {
    for (int j = 0; j < n; j += TILE_SIZE)
    {
      for (int k = 0; k < n; k += TILE_SIZE)
      {
        for (int ii = i; ii < i + TILE_SIZE; ++ii)
        {
          for (register int jj = j; jj < j + TILE_SIZE; jj += 8)
          {
            __m256i temp = _mm256_loadu_si256((__m256i *)&res[ii * n + jj]);
            for (register int kk = k; kk < k + TILE_SIZE; ++kk)
            {
              __m256i mat1_vec = _mm256_set1_epi32(mat1[ii * n + kk]);
              // Load horizontally from transposed_mat2
              __m256i mat2_vec = _mm256_loadu_si256((__m256i *)&transposed_mat2[jj * n + kk]); // Adjusted for transposed access
              temp = _mm256_add_epi32(temp, _mm256_mullo_epi32(mat1_vec, mat2_vec));
            }
            _mm256_storeu_si256((__m256i *)&res[ii * n + jj], temp);
          }
        }
      }
    }
  }

  free(transposed_mat2);
}
*/
  register int n = args->n;
  int *mat1 = args->mem1;
  int *transposed_mat2 = (int *)malloc(n * n * sizeof(int));
  int *res = args->mem3;

  transpose_matrix(args->mem2, transposed_mat2, n);

  thrd_t threads[THREAD_COUNT];
  ThreadData threadData[THREAD_COUNT];

  register int rowsPerThread = n / THREAD_COUNT;
  for (int i = 0; i < THREAD_COUNT; ++i)
  {
    threadData[i].startRow = i * rowsPerThread;
    threadData[i].endRow = (i + 1) * rowsPerThread;
    threadData[i].n = n;
    threadData[i].mat1 = mat1;
    threadData[i].transposed_mat2 = transposed_mat2;
    threadData[i].res = res;

    if (i == THREAD_COUNT - 1)
    {
      // Make sure the last thread covers any remaining rows
      threadData[i].endRow = n;
    }

    if (thrd_create(&threads[i], (thrd_start_t)mat_thread_func, &threadData[i]) != thrd_success)
    {
      // Handle thread creation failure
      exit(1);
    }
  }

  // Join threads
  for (int i = 0; i < THREAD_COUNT; ++i)
  {
    thrd_join(threads[i], NULL);
  }

  free(transposed_mat2);
}
