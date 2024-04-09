#ifndef COPT_H
#define COPT_H

#define MATRIX_INITIALIZE 0
#define ARRAY_INITIALIZE 1
#define FACTORIAL 2
#define MATRIX_MULTIPLY 3

const char *op_names[] = {
    "MATRIX_INIT",
    "ARRAY_INIT",
    "FACTORIAL",
    "MATRIX_MULTIPLY"
};

const char usage[] =
"Usage: copt OP N LOOP\n\n"

"copt measures the execution time impact of source level optimizations\n"
"in C. copt runs and times an unoptimized and optimized version of a given\n"
"operation and input size.\n\n"

"Argument description:\n\n"

"OP is the operation to run for this invocation of copt. There are four\n"
"possible operations, each of which takes exactly one argument N:\n"
"  0: initialize a pair of square integer matrices. N is the size of the\n"
"     matrices.\n"
"  1: initialize an integer array. N is the length of the array\n"
"  2: compute factorial with a recursive routine. N is the number for\n"
"     which the routine computes the factorial\n"
"  3: multiply two square integer matrices. N is the size of the matrix\n\n"

"LOOP is the number of times to run the given operation with the given\n"
"argument. Timing starts before the first operation begins and ends when\n"
"the last operation has completed.\n\n"

"OP, N, and LOOP all must be integers <= INT_MAX\n";

#endif /* COPT_H */
