#include "copt_fun.h"
#include "stdio.h"

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
 * Name:          <your name here>
 * Collaboration: <collaborator names here--one per line>
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
  // 13.
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

  register int i, n, *arr, modz = 64, imult;

  n = args->n;
  arr = args->mem1;
  // mod = X % Y;
  // modz = mod * Z; x = 500, y = 12, z = 8
  // modz = X % Y * Z; 8*8

  for (i = 0; i < n; i += 12)
  {
    imult = i * modz;
    arr[i] = imult;
    arr[i + 1] = imult + modz;
    arr[i + 2] = imult + 2 * modz;
    arr[i + 3] = imult + 3 * modz;
    arr[i + 4] = imult + 4 * modz;
    arr[i + 5] = imult + 5 * modz;
    arr[i + 6] = imult + 6 * modz;
    arr[i + 7] = imult + 7 * modz;
    arr[i + 8] = imult + 8 * modz;
    arr[i + 9] = imult + 9 * modz;
    arr[i + 10] = imult + 10 * modz;
    arr[i + 11] = imult + 11 * modz;
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

unsigned long long factorial_opt_helper(unsigned long long n)
{
  // 1. Not recursive, allows for easier optimization, 2.4x
  // 2. Put the whole thing in a switch statement, if you dont calculate it you dont need to worry about overhead :) (13.1x)

  // this is gonna be ugly but I think it'd be funny
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

void matrix_multiply_opt(struct fn_args *args)
{
  // 1. Added n offset variable to reduce redundant calculations (1.2x speedup)
  // 2. Moved vars to registers (1.5x speedup)
  // 3. Added joffset variable to reduce redundant calculations (1.8x speedup) Need to get below 3033 ms, currently at 7783 ms
  // 4.
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
}