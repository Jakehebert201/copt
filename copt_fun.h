#ifndef COPT_FUN_H
#define COPT_FUN_H

#define min(a, b) (((a)<(b))?(a):(b))

#define X 500
#define Y 12
#define Z 8

struct fn_args {
    int n;
    int *mem1;
    int *mem2;
    int *mem3;
    unsigned long long fac;
};

void matrix_initialize_unopt(struct fn_args *);
void matrix_initialize_opt(struct fn_args *);
void array_initialize_unopt(struct fn_args *);
void array_initialize_opt(struct fn_args *);
void factorial_unopt(struct fn_args *);
void factorial_opt(struct fn_args *);
void matrix_multiply_unopt(struct fn_args *);
void matrix_multiply_opt(struct fn_args *);

#endif /* COPT_FUN_H */
