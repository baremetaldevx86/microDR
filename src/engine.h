#ifndef ENGINE_H
#define ENGINE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//
// Tensor object
//
typedef struct Tensor {
    float* data;      // data buffer (scalar or matrix)
    float* grad;      // gradient buffer

    int ndim;         // number of dimensions (0 = scalar, 2 = matrix)
    int* shape;       // shape array (e.g. [rows, cols])
    int size;         // total number of elements


    struct Tensor** parents;   // computation graph parents
    int n_parents;             // number of parents

    int ref_count;             // Need to manage memory manually

    void (*backward)(struct Tensor*);   // backward function
} Tensor;

//
// Tensor creation
//
Tensor* tensor_create(float x);                 // scalar tensor
Tensor* tensor_create_matrix(int rows, int cols);

//
// Memory management
//
void tensor_retain(Tensor* t);
void tensor_release(Tensor* t);

//
// Core ops (forward)
//
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_pow(Tensor* a, Tensor* b);
Tensor* tensor_expn(Tensor* a);
Tensor* tensor_relu(Tensor* a);
Tensor* tensor_Tanh(Tensor* a);
Tensor* tensor_matmul(Tensor* A, Tensor* B);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mean(Tensor* a);
Tensor* tensor_div(Tensor* a, Tensor* b);
Tensor* tensor_sqrt(Tensor* a);

//
// Backward engine
//
void tensor_backward(Tensor* t);

//
//Activation functions
//
Tensor* tensor_relu(Tensor* t);
Tensor* tensor_softmax(Tensor* t);

//
// Utility
//
void tensor_zero_grad(Tensor* t);
void tensor_print(Tensor* t, char* name); // Added for debugging

#endif

