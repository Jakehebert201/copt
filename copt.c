#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/times.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "copt.h"
#include "copt_fun.h"

/* XXX: students should not modify any code in this file. */ 

int ttime() {
   struct tms buffer;
   int utime;

   times(&buffer);
   utime = (buffer.tms_utime / 60.0) * 1000.0;
   return (utime);
}


void init_op(int op, int n, void (**unopt_fn)(struct fn_args *args),
    void (**opt_fn)(struct fn_args *args), struct fn_args *args) {

    if (op == MATRIX_INITIALIZE) {

        args->n = n;
        args->mem1 = malloc(sizeof(int) * (n * n));
        if (args->mem1 == NULL){
            exit(-ENOMEM);
        }

        args->mem2 = malloc(sizeof(int) * (n * n));
        if (args->mem2 == NULL){
            exit(-ENOMEM);
        }

        (*unopt_fn) = matrix_initialize_unopt;
        (*opt_fn) = matrix_initialize_opt;
    }
    
    else if (op == ARRAY_INITIALIZE) {

        args->n = n;
        args->mem1 = malloc(sizeof(int) * n);
        if (args->mem1 == NULL){
            exit(-ENOMEM);
        }

        (*unopt_fn) = array_initialize_unopt;
        (*opt_fn) = array_initialize_opt;
    }

    else if (op == FACTORIAL) {
        args->n = n;
        (*unopt_fn) = factorial_unopt;
        (*opt_fn) = factorial_opt;
    }

    else if (op == MATRIX_MULTIPLY) {

        args->n = n;
        args->mem1 = malloc(sizeof(int) * (n * n));
        if (args->mem1 == NULL){
            exit(-ENOMEM);
        }

        args->mem2 = malloc(sizeof(int) * (n * n));
        if (args->mem2 == NULL){
            exit(-ENOMEM);
        }

        args->mem3 = malloc(sizeof(int) * (n * n));
        if (args->mem3 == NULL){
            exit(-ENOMEM);
        }

        matrix_initialize_unopt(args);

        (*unopt_fn) = matrix_multiply_unopt;
        (*opt_fn) = matrix_multiply_opt;
    }
}

void store_op_result(int op, struct fn_args *args,
    int **arr_exp, unsigned long long *fac_exp) {

    int n;
    n = args->n;
    if (op == MATRIX_INITIALIZE) {

        (*arr_exp) = malloc((sizeof(int)*(n*n)*2));
        if ((*arr_exp) == NULL) {
            exit(-ENOMEM);
        }
        memcpy(&((*arr_exp)[0]), args->mem1, (sizeof(int)*(n*n)));
        memcpy(&((*arr_exp)[n*n]), args->mem2, (sizeof(int)*(n*n)));

    } else if (op == ARRAY_INITIALIZE) {

        (*arr_exp) = malloc(sizeof(int)*n);
        if ((*arr_exp) == NULL) {
            exit(-ENOMEM);
        }
        memcpy(&((*arr_exp)[0]), args->mem1, (sizeof(int)*n));

    } else if (op == FACTORIAL) {

        (*fac_exp) = args->fac;

    } else if (op == MATRIX_MULTIPLY) {

        (*arr_exp) = malloc((sizeof(int)*(n*n)));
        if ((*arr_exp) == NULL) {
            exit(-ENOMEM);
        }
        memcpy(&((*arr_exp)[0]), args->mem3, (sizeof(int)*(n*n)));
    }
}

void check_op_result(int op, struct fn_args *args, int *arr_exp,
    unsigned long long fac_exp) {

    int n, ret;

    ret = 0;
    n = args->n;
    if (op == MATRIX_INITIALIZE) {
        ret =  memcmp(&(arr_exp[0]), args->mem1, (sizeof(int)*(n*n)));
        ret += memcmp(&(arr_exp[n*n]), args->mem2, (sizeof(int)*(n*n)));
    } else if (op == ARRAY_INITIALIZE) {
        ret = memcmp(&(arr_exp[0]), args->mem1, (sizeof(int)*n));
    } else if (op == FACTORIAL) {
        ret = (fac_exp == args->fac) ? 0 : 1;
    } else if (op == MATRIX_MULTIPLY) {
        ret = memcmp(&(arr_exp[0]), args->mem3, (sizeof(int)*(n*n)));
    }

    if(ret != 0){
        fprintf(stderr, "result of optimized operation did not match result "
                        "of unoptimized operation\n");
        exit(1);
    }
}

float time_op(void (*fn) (struct fn_args *args),
    struct fn_args *args, int loop) {
    int i;
    float start, end;

    start = ttime();
    for(i = 0; i < loop; i++){
        (fn)(args);
    }
    end = ttime();

    return (end - start);
}

void parse_options(int argc, char **argv, int *op, int *n, int *loop) {
    if (argc != 4) {
        printf(usage);
        exit(-EINVAL);
    }

    *op = strtol(argv[1], NULL, 10);
    *n = strtol(argv[2], NULL, 10);
    *loop = strtol(argv[3], NULL, 10);

    if (errno) {
        printf(usage);
    }
}

int main(int argc, char **argv){
    int i, op, n, loop;
    int *arr_exp;
    float unopt_time, opt_time;
    struct fn_args fn_args;
    unsigned long long fac_exp;
    void (*unopt_fn) (struct fn_args *args);
    void (*opt_fn) (struct fn_args *args);

    parse_options(argc, argv, &op, &n, &loop);

    printf("Running %s with n = %d loop = %d\n\n", op_names[op], n, loop); fflush(stdout);

    memset(&fn_args,0,sizeof(fn_args));
    init_op(op, n, &unopt_fn, &opt_fn, &fn_args);
    unopt_time = time_op(unopt_fn, &fn_args, loop);

    printf("UNOPTIMIZED(ms):  %12.1f\n", unopt_time); fflush(stdout);

    store_op_result(op, &fn_args, &arr_exp, &fac_exp);

    memset(&fn_args,0,sizeof(fn_args));
    init_op(op, n, &unopt_fn, &opt_fn, &fn_args);
    opt_time = time_op(opt_fn, &fn_args, loop);

    check_op_result(op, &fn_args, arr_exp, fac_exp);

    printf("OPTIMIZED(ms):    %12.1f\n", opt_time);
    printf("SPEEDUP:          %12.1f\n", (unopt_time / opt_time));

    return 0;
}
