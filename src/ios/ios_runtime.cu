#include <cstdio>
#include <sstream>
#include <fstream>
#include <map>
#include <iostream>
#include <string>
#include <cassert>
#include <cstdint>

#include <unistd.h>
#include <pthread.h>
#include <cudnn.h>

#include "ios/ops.h"
#include "ios/profile.h"
#include "utils/json.h"
#include "utils/utils.h"

#if defined(USE_FLOAT16)
typedef __half bias_data_type;
cudnnDataType_t cudnn_bias_data_type = CUDNN_DATA_HALF;
cudnnDataType_t cudnn_data_type = CUDNN_DATA_HALF;
cudnnDataType_t cudnn_conv_data_type = CUDNN_DATA_FLOAT;
cudnnTensorFormat_t cudnn_data_format = CUDNN_TENSOR_NHWC;
#elif defined(USE_INT8)
typedef float bias_data_type;
cudnnDataType_t cudnn_bias_data_type = CUDNN_DATA_FLOAT;
cudnnDataType_t cudnn_data_type = CUDNN_DATA_INT8;
cudnnDataType_t cudnn_conv_data_type = CUDNN_DATA_INT32;
cudnnTensorFormat_t cudnn_data_format = CUDNN_TENSOR_NHWC;
#else // USE_FLOAT32
typedef float bias_data_type;
cudnnDataType_t cudnn_bias_data_type = CUDNN_DATA_FLOAT;
cudnnDataType_t cudnn_data_type = CUDNN_DATA_FLOAT;
cudnnDataType_t cudnn_conv_data_type = CUDNN_DATA_FLOAT;
cudnnTensorFormat_t cudnn_data_format = CUDNN_TENSOR_NCHW;
#endif

#if defined(USE_TENSOR_CORE)
cudnnMathType_t cudnn_math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
#else
cudnnMathType_t cudnn_math_type = CUDNN_DEFAULT_MATH;
#endif

#define CONTEXT_WORKSPACE_SIZE 128 * 1024 * 1024 // 256 MB
#define MAX_NUM_GROUPS 10
#define MAX_NUM_VALUES 25
#define MAX_NUM_TERMS 25
#define MAX_NUM_NODES 1000
#define MAX_GROUP_SIZE 40

using std::vector;
using std::map;
using std::stringstream;
using std::string;
using std::size_t;


bool is_NCHW_layout(int* shape, int* stride)
{
    int cnt = 1;
    int expect_stride = 1;
    for (int i= 3; i >=0; i--){
        expect_stride = cnt;
        cnt *= shape[i];
        if (expect_stride != stride[i])
            return false;
    }
    return true;
}

bool is_NHWC_layout(int* shape, int* stride)
{
    int NHWC_stride_order[4] = {0, 2, 3, 1};
    int cnt = 1;
    int expect_stride = 1;
    for(int i= 3; i >=0; i--){
        expect_stride = cnt;
        cnt *= shape[NHWC_stride_order[i]];
        if(expect_stride != stride[NHWC_stride_order[i]])
            return false;
    }
    return true;
}

bool is_layout(int N, int C, int H, int W, int* stride, string layout)
{
    int shape[4] = {N, C, H, W};
    // we need to check stride in case of N111, where it could be both NCHW and NHWC
    if(layout == "NCHW" && is_NCHW_layout(shape, stride))
        return true;
    else if (layout == "NHWC" && is_NHWC_layout(shape, stride))
        return true;
    return false;
}

void get_stride(int N, int C, int H, int W, string layout, int* stride)
{
    int shape[4] = {N, C, H, W};
    if (layout == "NCHW") {
        int cnt = 1;
        for (int i= 3; i >=0; i--){
            stride[i] = cnt;
            cnt *= shape[i];
        }
    }
    else {
        assert(layout == "NHWC");
        int NHWC_stride_order[4] = {0, 2, 3, 1};
        int cnt = 1;
        for(int i= 3; i >=0; i--){
            stride[NHWC_stride_order[i]] = cnt;
            cnt *= shape[NHWC_stride_order[i]];
    }
        
    }
}



struct ConvKey {
    int attrs[14];
    ConvKey() {}
    ConvKey(int batch_size, int in_channels, int input_h, int input_w, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int groups,
        cudnnTensorFormat_t input_layout, cudnnTensorFormat_t output_layout) {
        attrs[0] = batch_size;
        attrs[1] = in_channels;
        attrs[2] = input_h;
        attrs[3] = input_w;
        attrs[4] = out_channels;
        attrs[5] = kernel_h;
        attrs[6] = kernel_w;
        attrs[7] = stride_h;
        attrs[8] = kernel_w;
        attrs[9] = padding_h;
        attrs[10] = padding_w;
        attrs[11] = groups;
        attrs[12] = input_layout;
        attrs[13] = output_layout;
    }
    void print() {
        fprintf(stderr, "input_shape: %d %d %d %d  out_channels: %d  kernel, stride, padding: %d %d, %d %d, %d %d  groups: %d input_layout: %d, output_layout: %d\n",
                attrs[0], attrs[1], attrs[2], attrs[3], attrs[4], attrs[5], attrs[6], attrs[7], attrs[8], attrs[9], attrs[10], attrs[11], attrs[12], attrs[13]);
    }
};

bool operator<(const ConvKey &lhs, const ConvKey &rhs) {
    const int n = sizeof(lhs.attrs) / sizeof(lhs.attrs[0]);
    for(int i = 0; i < n; i++) {
        if(lhs.attrs[i] != rhs.attrs[i])
            return lhs.attrs[i] < rhs.attrs[i];
    }
    return false;
}

cudnnTensorFormat_t getCudNNLayoutFromStr(string str) {
    if(str == "NCHW")
        return CUDNN_TENSOR_NCHW;
    assert(str == "NHWC");
    return CUDNN_TENSOR_NHWC;
}

struct ConvAlgMap {
    const char * config_filename = "conv2alg.txt";
    map<ConvKey, cudnnConvolutionFwdAlgo_t> conv2alg;

    ConvAlgMap() {
        std::ifstream fin(config_filename);
        if(fin.good()) {
            while(true) {
                ConvKey key;
                int alg;
                for(int & attr : key.attrs)
                    fin >> attr;
                fin >> alg;
                if(fin.eof())
                    break;
                conv2alg[key] = static_cast<cudnnConvolutionFwdAlgo_t>(alg);
            }
        }
    }
    size_t count(const ConvKey &key) {
        return conv2alg.count(key);
    }
    cudnnConvolutionFwdAlgo_t get(const ConvKey &key) {
        return conv2alg[key];
    }
    void put(const ConvKey &key, cudnnConvolutionFwdAlgo_t alg) {
        conv2alg[key] = alg;

        std::ofstream fout(config_filename, std::ios_base::out | std::ios_base::app);
        for(int i = 0; i < 14; i++)
            fout << key.attrs[i] << " ";
        fout << static_cast<int>(alg) << std::endl;
        fout.close();
    }
} conv_alg_map;

struct CudnnContext {
    cudnnHandle_t dnn;
    cudaStream_t stream;
    size_t max_size;
    data_type *space;
    CudnnContext(size_t max_size = CONTEXT_WORKSPACE_SIZE) {
        init_profile();
        this->max_size = max_size;
        checkCUDNN(cudnnCreate(&dnn));
        checkCUDA(cudaStreamCreate(&stream));
        checkCUDA(cudaMalloc(&space, max_size));
        checkCUDNN(cudnnSetStream(dnn, stream));
    }
    ~CudnnContext() {
        checkCUDNN(cudnnDestroy(dnn));
        checkCUDA(cudaFree(space));
        checkCUDA(cudaStreamDestroy(stream));
    }
};

CudnnContext contexts[MAX_NUM_GROUPS];


cudnnConvolutionFwdAlgo_t get_conv_alg(int batch_size, int in_channels, int input_h, int input_w, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int groups,
    cudnnTensorFormat_t input_layout, cudnnTensorFormat_t output_layout, bool disable_tc) {
    ConvKey key(batch_size, in_channels, input_h, input_w, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups, input_layout, output_layout);
    if(conv_alg_map.count(key))
        return conv_alg_map.get(key);
    data_type *input_data, *filter_data, *output_data;
    int output_h = 1 + (input_h - kernel_h + 2 * padding_h) / stride_h;
    int output_w = 1 + (input_w - kernel_w + 2 * padding_w) / stride_w;
    cudnnTensorDescriptor_t inputTensor, outputTensor;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, input_layout, cudnn_data_type, batch_size, in_channels, input_h, input_w));
    assert(in_channels % groups == 0);
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, cudnn_data_type, input_layout, out_channels, in_channels / groups, kernel_h, kernel_w));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, 1/*dilationH*/, 1/*dilationW*/, CUDNN_CROSS_CORRELATION, cudnn_conv_data_type));
    cudnnMathType_t conv_math_type = cudnn_math_type;
    if (disable_tc)
        conv_math_type = CUDNN_DEFAULT_MATH;
    checkCUDNN(cudnnSetConvolutionMathType(convDesc, conv_math_type));
    checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, groups));
    int n, c, h, w;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensor, filterDesc, &n, &c, &h, &w));
    assert(n == batch_size);
    assert(c == out_channels);
    assert(h == output_h);
    assert(w == output_w);
    checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, output_layout, cudnn_data_type, n, c, h, w));

    size_t input_size = sizeof(data_type) * batch_size * in_channels * input_h * input_w;
    size_t filter_size = sizeof(data_type) * out_channels * (in_channels / groups)* kernel_h * kernel_w;
    size_t output_size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
    checkCUDA(cudaMalloc(&input_data, input_size));
    checkCUDA(cudaMalloc(&filter_data, filter_size));
    checkCUDA(cudaMalloc(&output_data, output_size));


    cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    int returned;

    checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(
            contexts[0].dnn, inputTensor, input_data, filterDesc,
            filter_data, convDesc, outputTensor, output_data,
            CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned, perf,
            contexts[0].space, contexts[0].max_size));
    assert(returned == CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
    conv_alg_map.put(key, perf[0].algo);

    checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDA(cudaFree(input_data));
    checkCUDA(cudaFree(output_data));
    checkCUDA(cudaFree(filter_data));
    return perf[0].algo;
}

cudnnConvolutionFwdAlgo_t get_conv_alg_tc(int batch_size, int in_channels, int input_h, int input_w, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int groups,
    cudnnTensorFormat_t input_layout, cudnnTensorFormat_t output_layout) {
    ConvKey key(batch_size, in_channels, input_h, input_w, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups, input_layout, output_layout);
    if(conv_alg_map.count(key))
        return conv_alg_map.get(key);
    data_type *input_data, *filter_data, *output_data;
    int output_h = 1 + (input_h - kernel_h + 2 * padding_h) / stride_h;
    int output_w = 1 + (input_w - kernel_w + 2 * padding_w) / stride_w;
    cudnnTensorDescriptor_t inputTensor, outputTensor;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, input_layout, cudnn_data_type, batch_size, in_channels, input_h, input_w));
    assert(in_channels % groups == 0);
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, cudnn_data_type, input_layout, out_channels, in_channels / groups, kernel_h, kernel_w));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, 1/*dilationH*/, 1/*dilationW*/, CUDNN_CROSS_CORRELATION, cudnn_conv_data_type));
    cudnnMathType_t conv_math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    checkCUDNN(cudnnSetConvolutionMathType(convDesc, conv_math_type));
    checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, groups));
    int n, c, h, w;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensor, filterDesc, &n, &c, &h, &w));
    assert(n == batch_size);
    assert(c == out_channels);
    assert(h == output_h);
    assert(w == output_w);
    checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, output_layout, cudnn_data_type, n, c, h, w));

    size_t input_size = sizeof(data_type) * batch_size * in_channels * input_h * input_w;
    size_t filter_size = sizeof(data_type) * out_channels * (in_channels / groups)* kernel_h * kernel_w;
    size_t output_size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
    checkCUDA(cudaMalloc(&input_data, input_size));
    checkCUDA(cudaMalloc(&filter_data, filter_size));
    checkCUDA(cudaMalloc(&output_data, output_size));


    cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    int returned;

    checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(
            contexts[0].dnn, inputTensor, input_data, filterDesc,
            filter_data, convDesc, outputTensor, output_data,
            CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned, perf,
            contexts[0].space, contexts[0].max_size));
    assert(returned == CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
    int best_algo=0;
    for(;best_algo < CUDNN_CONVOLUTION_FWD_ALGO_COUNT; best_algo++)
    {
        // https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#tensor-ops-conv-functions-supported-algos
        // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM and CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
        // can be run as Tensor Core operations
        if ((best_algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) || (best_algo == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED))
        {
            conv_alg_map.put(key, perf[best_algo].algo);
            break;
        }
    }

    checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDA(cudaFree(input_data));
    checkCUDA(cudaFree(output_data));
    checkCUDA(cudaFree(filter_data));
    return perf[best_algo].algo;
}

struct ConvOP {
    int batch_size;
    int in_channels;
    int out_channels;
    int input_h;
    int input_w;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int groups;
    string act;
    bool has_act;

    int output_h;
    int output_w;

    cudnnTensorFormat_t input_layout;
    cudnnTensorFormat_t output_layout;

    CudnnContext *context;

    cudnnConvolutionFwdAlgo_t conv_alg;
    cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnActivationDescriptor_t actiDesc;

    data_type *input_data;
    data_type *output_data;
    data_type *filter_data;
    data_type *bias_data;

    bool disable_tc; // disable tensor core
    bool use_tc; // force to use tensor core

    void init(int batch_size, int in_channels, int input_h, int input_w, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int groups, string act,
        string input_layout, string output_layout, bool disable_tc, bool use_tc) {
        this->batch_size = batch_size;
        this->in_channels = in_channels;
        this->input_h = input_h;
        this->input_w = input_w;
        this->out_channels = out_channels;
        this->kernel_h = kernel_h;
        this->kernel_w = kernel_w;
        this->stride_h = stride_h;
        this->stride_w = stride_w;
        this->padding_h = padding_h;
        this->padding_w = padding_w;
        this->groups = groups;
        this->act = act;
        this->has_act = (act != "identity");
        this->output_h = 1 + (input_h - kernel_h + 2 * padding_h) / stride_h;
        this->output_w = 1 + (input_w - kernel_w + 2 * padding_w) / stride_w;
        this->input_layout = getCudNNLayoutFromStr(input_layout);
        this->output_layout = getCudNNLayoutFromStr(output_layout);
        this->disable_tc = disable_tc;
        this->use_tc = use_tc;
        assert(!(disable_tc && use_tc));
    }
    size_t get_filter_size() {
        return sizeof(data_type) * out_channels * (in_channels / groups) * kernel_h * kernel_w;
    }
    size_t get_bias_size() {
        return sizeof(bias_data_type) * out_channels;
    }
    void map(data_type *input_data, CudnnContext *context) {
        this->input_data = input_data;
        this->context = context;
        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, input_layout, cudnn_data_type, batch_size, in_channels, input_h, input_w));
        checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor, input_layout, cudnn_bias_data_type, 1, out_channels, 1, 1));
        assert(in_channels % groups == 0);
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, cudnn_data_type, input_layout, out_channels, in_channels / groups, kernel_h, kernel_w));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_h, stride_w, 1/*dilationH*/, 1/*dilationW*/, CUDNN_CROSS_CORRELATION, cudnn_conv_data_type));
        cudnnMathType_t conv_math_type = cudnn_math_type;
        if (disable_tc)
            conv_math_type = CUDNN_DEFAULT_MATH;
        checkCUDNN(cudnnSetConvolutionMathType(convDesc, conv_math_type));
        checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, groups));
        int n, c, h, w;
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensor, filterDesc, &n, &c, &h, &w));
        assert(n == batch_size);
        assert(c == out_channels);
        assert(h == output_h);
        assert(w == output_w);
        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, output_layout, cudnn_data_type, n, c, h, w));
        if (has_act) {
            cudnnActivationMode_t act_mode;
            if(act == "relu") {
                act_mode = CUDNN_ACTIVATION_RELU;
            } else if(act == "tanh") {
                act_mode = CUDNN_ACTIVATION_TANH;
            } else if(act == "sigmoid") {
                act_mode = CUDNN_ACTIVATION_SIGMOID;
            } else {
                FatalError("Wrong activation mode " + act);
            }
            checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
            checkCUDNN(cudnnSetActivationDescriptor(actiDesc, act_mode, CUDNN_NOT_PROPAGATE_NAN, 0.0));
        }

        size_t filter_size = get_filter_size();
        size_t bias_size = get_bias_size();
        size_t output_size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
        checkCUDA(cudaMalloc(&filter_data, filter_size));
        checkCUDA(cudaMalloc(&bias_data, bias_size));
        checkCUDA(cudaMalloc(&output_data, output_size));
        if(use_tc)
            this->conv_alg = get_conv_alg_tc(batch_size, in_channels, input_h, input_w, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups, input_layout, output_layout);
        else
            this->conv_alg = get_conv_alg(batch_size, in_channels, input_h, input_w, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups, input_layout, output_layout, disable_tc);
    }
    void forward() {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        if(has_act) {
#if defined(USE_TENSOR_CORE)
            if(! disable_tc)
            {
                checkCUDNN(cudnnConvolutionForward(
                        context->dnn, &alpha, inputTensor, input_data, filterDesc, filter_data,
                        convDesc, conv_alg, context->space, context->max_size,
                        &beta, outputTensor, output_data));
                checkCUDNN(cudnnAddTensor(context->dnn, &alpha, biasTensor, bias_data, &alpha, outputTensor, output_data));
                checkCUDNN(cudnnActivationForward(context->dnn, actiDesc, &alpha, outputTensor, output_data, &beta, outputTensor, output_data));
            }
            else{
                checkCUDNN(cudnnConvolutionBiasActivationForward(
                    context->dnn, &alpha, inputTensor, input_data, filterDesc, filter_data,
                    convDesc, conv_alg, context->space, context->max_size,
                    &beta, outputTensor, output_data, biasTensor, bias_data, actiDesc,
                    outputTensor, output_data));
            }
#else
            checkCUDNN(cudnnConvolutionBiasActivationForward(
                    context->dnn, &alpha, inputTensor, input_data, filterDesc, filter_data,
                    convDesc, conv_alg, context->space, context->max_size,
                    &beta, outputTensor, output_data, biasTensor, bias_data, actiDesc,
                    outputTensor, output_data));
#endif
        } else {
            // std::cout << " context->dnn " << context->dnn << " &alpha " << &alpha << " inputTensor " << inputTensor
            //         << " input_data " << input_data << " filterDesc " << filterDesc << " filter_data " << filter_data
            //         << " convDesc " << convDesc << " conv_alg " << conv_alg << " context->space " << context->space
            //         << " context->max_size " << context->max_size << " &beta " << &beta
            //         << " outputTensor " << outputTensor << " output_data " << output_data << std::endl;
            // std::cout << "check null or not" << (input_data == nullptr) << std::endl;
            checkCUDNN(cudnnConvolutionForward(
                    context->dnn, &alpha, inputTensor, input_data, filterDesc, filter_data,
                    convDesc, conv_alg, context->space, context->max_size,
                    &beta, outputTensor, output_data));
            checkCUDNN(cudnnAddTensor(context->dnn, &alpha, biasTensor, bias_data, &alpha, outputTensor, output_data));
        }
    }
    void unmap() {
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(biasTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        if (has_act) {
            checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
        }
        // free tensors
        checkCUDA(cudaFree(filter_data));
        checkCUDA(cudaFree(bias_data));
        checkCUDA(cudaFree(output_data));
    }
};

struct ActivationOP {
    static const int RELU = 1;
    static const int TANH = 2;
    static const int SIGMOID = 3;

    int batch_size;
    int in_channels;
    int input_h;
    int input_w;
    int out_channels;
    int act_type;
    bool inplace;

    int output_h;
    int output_w;

    cudnnTensorFormat_t input_layout;
    cudnnTensorDescriptor_t inputTensor;
    cudnnActivationDescriptor_t actiDesc;

    data_type *input_data;
    data_type *output_data;
    CudnnContext *context;

    void init(
        int batch_size, int in_channels, int input_h, int input_w, int act_type, bool inplace, string input_layout
    ) {
        this->batch_size = batch_size;
        this->in_channels = in_channels;
        this->input_h = input_h;
        this->input_w = input_w;
        this->act_type = act_type;
        this->inplace = inplace;
        this->out_channels = in_channels;
        this->output_h = input_h;
        this->output_w = input_w;
        this->context = nullptr;
        this->input_layout = getCudNNLayoutFromStr(input_layout);
    }
    void map(data_type *input_data, CudnnContext *context) {
        // create descriptors
        this->input_data = input_data;
        this->context = context;
        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, input_layout, cudnn_data_type, batch_size, in_channels, input_h, input_w));
        checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
        cudnnActivationMode_t mode;
        switch (act_type) {
            case RELU:
                mode = CUDNN_ACTIVATION_RELU;
                break;
            case SIGMOID:
                mode = CUDNN_ACTIVATION_SIGMOID;
                break;
            case TANH:
                mode = CUDNN_ACTIVATION_TANH;
                break;
            default:
                assert(false);
        }
        checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode, CUDNN_NOT_PROPAGATE_NAN, 0.0));
        if (!inplace) {
            size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
            checkCUDA(cudaMalloc(&output_data, size));
        } else {
            this->output_data = input_data;
        }
    }

    void unmap() {
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
        checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
        if (!inplace) {
            checkCUDA(cudaFree(output_data));
        }
    }

    void forward() {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCUDNN(cudnnActivationForward(context->dnn, actiDesc, &alpha, inputTensor, input_data, &beta, inputTensor, output_data));
    }
};

struct ElementOP {
    static const int MUL = 1;
    static const int ADD = 2;

    int batch_size;
    int channels;
    int h;
    int w;
    int op_type;
    cudnnTensorFormat_t input_layout_a;
    cudnnTensorFormat_t input_layout_b;

    cudnnTensorDescriptor_t inputTensor_a;
    cudnnTensorDescriptor_t inputTensor_b;
    cudnnOpTensorDescriptor_t opDesc;

    data_type *input_a;
    data_type *input_b;
    data_type *output_data;
    CudnnContext *context;

    void init(int batch_size, int channels, int h, int w, int op_type, string input_layout_a, string input_layout_b) {
        this->batch_size = batch_size;
        this->channels = channels;
        this->h = h;
        this->w = w;
        this->op_type = op_type;
        this->context = nullptr;
        this->input_layout_a = getCudNNLayoutFromStr(input_layout_a);
        this->input_layout_b = getCudNNLayoutFromStr(input_layout_b);
    }
    void map(data_type *input_a, data_type *input_b, CudnnContext *context) {
        this->input_a = input_a;
        this->input_b = input_b;
        size_t size = sizeof(data_type) * batch_size * channels * h * w;
        this->context = context;
        checkCUDA(cudaMalloc(&output_data, size));

        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor_a));
        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor_b));
        checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor_a, input_layout_a, cudnn_data_type, batch_size, channels, h, w));
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor_b, input_layout_b, cudnn_data_type, batch_size, channels, h, w));
        cudnnOpTensorOp_t opType;
        if(op_type == MUL)
            opType = CUDNN_OP_TENSOR_MUL;
        else if(op_type == ADD)
            opType = CUDNN_OP_TENSOR_ADD;
        else
            FatalError("not supported elementwise op");
        // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnOpTensor
        // opTensorCompType in opTensorDesc could not be HALF
        cudnnDataType_t op_data_type = CUDNN_DATA_FLOAT;
        checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, opType, op_data_type, CUDNN_NOT_PROPAGATE_NAN));
    }
    void forward() {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCUDNN(cudnnOpTensor(context->dnn, opDesc, &alpha, inputTensor_a, input_a, &alpha, inputTensor_b, input_b, &beta, inputTensor_a, output_data));
    }
    void unmap() {
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor_a));
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor_b));
        checkCUDNN(cudnnDestroyOpTensorDescriptor(opDesc));
        checkCUDA(cudaFree(output_data));
    }
};

struct PoolOP {
    static const int MAX_POOL = 1;
    static const int AVG_POOL = 2;
    std::string name;

    int batch_size;
    int in_channels;
    int input_h;
    int input_w;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int out_channels;
    int output_h;
    int output_w;
    int pool_type;

    cudnnTensorFormat_t input_layout;

    cudnnTensorDescriptor_t inputTensor, outputTensor;
    cudnnPoolingDescriptor_t poolDesc;

    CudnnContext *context;

    data_type *input_data;
    data_type *output_data;

    void init(int batch_size, int in_channels, int input_h, int input_w, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int pool_type, string input_layout) {
        this->batch_size = batch_size;
        this->in_channels = in_channels;
        this->input_h = input_h;
        this->input_w = input_w;
        this->kernel_h = kernel_h;
        this->kernel_w = kernel_w;
        this->stride_h = stride_h;
        this->stride_w = stride_w;
        this->padding_h = padding_h;
        this->padding_w = padding_w;
        this->pool_type = pool_type;
        this->input_layout = getCudNNLayoutFromStr(input_layout);

        this->out_channels = in_channels;
        this->output_h = 1 + (input_h - kernel_h + 2 * padding_h) / stride_h;
        this->output_w = 1 + (input_w - kernel_w + 2 * padding_w) / stride_w;
    }
    void map(data_type *input_data, CudnnContext *context) {
        this->input_data = input_data;
        this->context = context;
        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
        checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
        // set descriptors
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, input_layout,
                                              cudnn_data_type, batch_size, in_channels, input_h, input_w));
        cudnnPoolingMode_t mode;
        if(pool_type == MAX_POOL) {
            mode = CUDNN_POOLING_MAX;
        } else if(pool_type == AVG_POOL) {
            mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        } else {
            FatalError("unrecognized pooling type");
        }
        checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, mode, CUDNN_PROPAGATE_NAN, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w));
        int n, c, h, w;
        checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, inputTensor, &n, &c, &h, &w));
        assert(n == batch_size);
        assert(c == out_channels);
        assert(h == output_h);
        assert(w == output_w);
        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, input_layout, cudnn_data_type, n, c, h, w));

        size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
        checkCUDA(cudaMalloc(&output_data, size));
    }
    void forward() {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCUDNN(cudnnPoolingForward(context->dnn, poolDesc, &alpha, inputTensor, input_data, &beta, outputTensor, output_data));
    }
    void unmap() {
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
        checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
        checkCUDA(cudaFree(output_data));
    }

};


struct TransformOp {
    cudnnTensorDescriptor_t srcTensor, dstTensor;
    cudnnTensorFormat_t src_layout, dst_layout;

    data_type *input_data;
    data_type *output_data;

    int N, C, H, W;
    int stride[4];

    CudnnContext *context;
    bool needTransform;

    void init(int batch_size, int in_channels,
            int input_h, int input_w,
            int* input_stride,
            string _src_layout,
            string _dst_layout) {
        this->N = batch_size;
        this->C = in_channels;
        this->H = input_h;
        this->W = input_w;
        assert(is_layout(batch_size, in_channels, input_h, input_w, input_stride, _src_layout));
        this->src_layout = getCudNNLayoutFromStr(_src_layout);
        this->dst_layout = getCudNNLayoutFromStr(_dst_layout);
        if(_src_layout == _dst_layout || is_layout(batch_size, in_channels, input_h, input_w, input_stride, _dst_layout))
            needTransform = false;
        else
            needTransform = true;
    }

    void map(data_type *input_data, CudnnContext *context) {
        this->input_data = input_data;
        this->context = context;
        if (!this->needTransform)
        {
            output_data = input_data;
            return;
        }
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensor));

        // set descriptors
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensor, src_layout,
            cudnn_data_type, N, C, H, W));
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensor, dst_layout,
            cudnn_data_type, N, C, H, W));

        // allocate tensors
        size_t output_size = sizeof(data_type) * N * C * H * W;
        checkCUDA(cudaMalloc(&output_data, output_size));
    }

    void forward() {
        if (!this->needTransform) {
            return;
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCUDNN(cudnnTransformTensor(
            context->dnn, &alpha, srcTensor, input_data,
            &beta, dstTensor, output_data));

    }

    void unmap() {
        if (! this->needTransform)
            return;
        checkCUDNN(cudnnDestroyTensorDescriptor(srcTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(dstTensor));
        // free tensors
        checkCUDA(cudaFree(output_data));
    }
};


struct NodeBase {
    string name;
    CudnnContext *context = nullptr;
    int batch_size;
    int out_channels;
    int output_h;
    int output_w;
    int stride[4];
    string layout;
    data_type *output_data;
    void print_shape() {
        fprintf(stderr, "%s %d %d %d %d\n", name.c_str(), batch_size, out_channels, output_h, output_w);
    }
    void print_stride() {
        fprintf(stderr, "%s %d %d %d %d\n", name.c_str(), stride[0], stride[1], stride[2], stride[3]);
    }
    void set_context(CudnnContext *context_) {
        this->context = context_;
    }
    // init stride (assume default NCHW layout)
    void init_stride(string layout) {
        this->layout = layout;
        int shape[4] = {batch_size, out_channels, output_h, output_w};
        if (this->layout == "NCHW") {
            int cnt = 1;
            for (int i= 4-1; i >= 0; i--)
            {
                this->stride[i] = cnt;
                cnt *= shape[i];
            }
        }
        else {
            assert(this->layout == "NHWC");
            int NHWC_stride_order[4] = {0, 2, 3, 1};
            int cnt = 1;
            for (int i= 4-1; i >= 0; i--)
            {
                this->stride[NHWC_stride_order[i]] = cnt;
                cnt *= shape[NHWC_stride_order[i]];
            }
        }
    }
    virtual void map() = 0;
    virtual void forward() = 0;
    virtual void unmap() = 0;
};


struct Placeholder: NodeBase {
    void init(string name, int batch_size, int out_channels, int output_h, int output_w, string layout) {
        this->name = name;
        this->batch_size = batch_size;
        this->out_channels = out_channels;
        this->output_h = output_h;
        this->output_w = output_w;
        this->output_data = nullptr;
        this->init_stride(layout);
    }
    void init(int batch_size, const Json::Value &value_config) {
        // this->name = value_config["name"].asString();
        // this->batch_size = batch_size;
        // this->out_channels = value_config["output_shape"][0].asInt();
        // this->output_h = value_config["output_shape"][1].asInt();
        // this->output_w = value_config["output_shape"][2].asInt();
        // this->output_data = nullptr;
        init(
            (string)value_config["name"].asString(),
            batch_size,
            (int)value_config["output_shape"][0].asInt(),
            (int)value_config["output_shape"][1].asInt(),
            (int)value_config["output_shape"][2].asInt(),
            (string)value_config["layout"].asString()
        );
    }
    void map() override {
        size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
        checkCUDA(cudaMalloc(&output_data, size));
    }
    void forward() override {}
    void unmap() override {
        checkCUDA(cudaFree(output_data));
    }
};

struct Value: NodeBase {
    NodeBase *node;
    int begin;
    int end;

    void init(const Json::Value &value_config, const std::map<string,NodeBase*> &node_map) {
        this->node = node_map.at(value_config[0].asString());
        this->begin = value_config[1].asInt();
        this->end = value_config[2].asInt();
        this->batch_size = node->batch_size;
        this->out_channels = end - begin;
        this->output_h = node->output_h;
        this->output_w = node->output_w;
        this->context = nullptr;
        this->output_data = nullptr;
        this->init_stride(this->node->layout);
    }
    void map() override {
        if(batch_size == 1) {
            output_data = node->output_data + begin * node->output_h * node->output_w;
        } else {
            if (begin == 0 && end == node->out_channels) {
                output_data = node->output_data;
            } else {
                size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
                checkCUDA(cudaMalloc(&output_data, size));
            }
        }
        assert(output_data != nullptr);
    }
    void unmap() override {
        if(batch_size == 1) {
            return;
        } else {
            if (begin == 0 && end == node->out_channels)
                return;
            checkCUDA(cudaFree(output_data));
        }
    }
    void forward() override {
        if(batch_size == 1 || (begin == 0 && end == node->out_channels))
            return;
        int num_blocks = batch_size;
        int src_blk_size = node->out_channels * node->output_h * node->output_w;
        int dst_blk_size = out_channels * output_h * output_w;
        int offset = begin * output_h * output_w;
        int n = num_blocks * dst_blk_size;
        assign_with_stride_dst<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, context->stream>>>(output_data, node->output_data + offset, n,dst_blk_size, src_blk_size);
    }
};

struct Term: NodeBase {
    int num_values;
    Value values[MAX_NUM_VALUES];

    cudnnTensorDescriptor_t inputTensor;
    cudnnOpTensorDescriptor_t opDesc;


    void init(const Json::Value &term_config, const std::map<string,NodeBase*> &node_map) {
        this->num_values = (int)term_config.size();
        this->context = context;
        for(Json::ArrayIndex i = 0; i < term_config.size(); i++) {
            values[i].init(term_config[i], node_map);
        }
        this->batch_size = values[0].batch_size;
        this->out_channels = values[0].out_channels;
        this->output_h = values[0].output_h;
        this->output_w = values[0].output_w;
        for(int i = 1; i < num_values; i++) {
            assert(values[i].batch_size == batch_size);
            assert(values[i].out_channels == out_channels);
            assert(values[i].output_h == output_h);
            assert(values[i].output_w == output_w);
        }
        this->context = nullptr;
        this->output_data = nullptr;
        this->init_stride(values[0].layout);
    }
    void map() override {
        for(int i = 0; i < num_values; i++) {
            values[i].set_context(context);
            values[i].map();
        }
        if(num_values == 1) {
            this->output_data = values[0].output_data;
        } else {
            checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
            checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, cudnn_data_format, cudnn_data_type,
                    values[0].batch_size, values[0].out_channels, values[0].output_h, values[0].output_w));

            checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, CUDNN_OP_TENSOR_ADD, cudnn_data_type, CUDNN_NOT_PROPAGATE_NAN));
            size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
            checkCUDA(cudaMalloc(&output_data, size));
        }
    }
    void unmap() override {
        for(int i = 0; i < num_values; i++)
            values[i].unmap();
        if(num_values > 1) {
            checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
            checkCUDNN(cudnnDestroyOpTensorDescriptor(opDesc));
            checkCUDA(cudaFree(output_data));
        }
    }
    void forward() override {
        for(int i = 0; i < num_values; i++) {
            values[i].forward();
        }
        if(num_values > 1) {
            int n = batch_size * out_channels * output_h * output_w;
            switch(num_values) {
                case 2:
                    accumulate_sum_2<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, context->stream>>>(output_data, values[0].output_data, values[1].output_data, n);
                    break;
                case 3:
                    accumulate_sum_3<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, context->stream>>>(output_data, values[0].output_data, values[1].output_data, values[2].output_data, n);
                    break;
                case 4:
                    accumulate_sum_4<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, context->stream>>>(output_data, values[0].output_data, values[1].output_data, values[2].output_data, values[3].output_data, n);
                    break;
                case 5:
                    accumulate_sum_5<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, context->stream>>>(output_data, values[0].output_data, values[1].output_data, values[2].output_data, values[3].output_data, values[4].output_data, n);
                    break;
                default:
                    for(int i = 0; i < num_values; i++) {
                        const float alpha = 1.0;
                        const float beta = (i == 0 ? 0.0f : 1.0f);
                        checkCUDNN(cudnnAddTensor(context->dnn, &alpha, inputTensor, values[i].output_data, &beta, inputTensor, output_data));
                    }
            }
        }
    }
};

struct Input: NodeBase {
    int num_terms;
    Term terms[MAX_NUM_TERMS];

    void init(const Json::Value &input_config, const std::map<string,NodeBase*> &node_map) {
        this->num_terms = (int)input_config.size();
        this->context = context;
        int sum = 0;
        for(Json::ArrayIndex i = 0; i < input_config.size(); i++) {
            terms[i].init(input_config[i], node_map);
            sum += terms[i].out_channels;
        }
        this->batch_size = terms[0].batch_size;
        this->out_channels = sum;
        this->output_h = terms[0].output_h;
        this->output_w = terms[0].output_w;
        this->output_data = nullptr;
        this->context = nullptr;
        this->init_stride(terms[0].layout);
    }
    void map() override {
        for(int i = 0; i < num_terms; i++) {
            terms[i].set_context(context);
            terms[i].map();
        }
        if(num_terms == 1) {
            this->output_data = terms[0].output_data;
        } else {
            size_t size = sizeof(data_type) * batch_size * out_channels * output_h * output_w;
            checkCUDA(cudaMalloc(&output_data, size));
        }
    }
    void unmap() override {
        for(int i = 0; i < num_terms; i++)
            terms[i].unmap();
        if(num_terms > 1) {
            checkCUDA(cudaFree(output_data));
            output_data = nullptr;
        }
    }
    void forward() override {
        for(int i = 0; i < num_terms; i++) {
            terms[i].forward();
        }
        if(num_terms > 1) {
            int offset = 0;
            for(int i = 0; i < num_terms; i++) {
                int src_blk_size = terms[i].out_channels * output_h * output_w;
                int dst_blk_size = out_channels * output_h * output_w;
                int num_blocks = batch_size;
                int n = num_blocks * src_blk_size;
                assign_with_stride_src<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, context->stream>>>(output_data + offset, terms[i].output_data, n, dst_blk_size, src_blk_size);
                offset += src_blk_size;
            }
        }
    }
};

struct Conv: NodeBase {
    Input input;
    ConvOP conv_op;

    void init(const Json::Value &conv_config, const std::map<string,NodeBase*> &node_map) {
        this->name = conv_config["name"].asString();
        this->input.init(conv_config["inputs"], node_map);
        conv_op.init(input.batch_size, input.out_channels, input.output_h, input.output_w,
                     conv_config["out_channels"].asInt(),
                     conv_config["kernel"][0].asInt(),
                     conv_config["kernel"][1].asInt(),
                     conv_config["stride"][0].asInt(),
                     conv_config["stride"][1].asInt(),
                     conv_config["padding"][0].asInt(),
                     conv_config["padding"][1].asInt(),
                     conv_config["groups"].asInt(),
                     conv_config["act"].asString(),
                     input.layout,
                     conv_config["layout"].asString(),
                     conv_config["disable_tc"].asBool(),
                     conv_config["use_tc"].asBool()
                     );
        this->batch_size = input.batch_size;
        this->out_channels = conv_op.out_channels;
        this->output_h = conv_op.output_h;
        this->output_w = conv_op.output_w;
        this->context = nullptr;
        this->output_data = nullptr;
        this->init_stride(conv_config["layout"].asString());
    }
    void map() override {
        input.set_context(context);
        input.map();
        conv_op.map(input.output_data, context);
        this->output_data = conv_op.output_data;
    }
    void forward() override {
        input.forward();
        conv_op.forward();
    }
    void unmap() override {
        input.unmap();
        conv_op.unmap();
    }
};

// Transform + Conv + Transform; input layout == output layout
struct Transform_Conv: NodeBase {
    Input input;
    ConvOP conv_op;
    TransformOp transform_op0;
    TransformOp transform_op1;

    void virtual init(const Json::Value &conv_config, const std::map<string,NodeBase*> &node_map) {
        return;
    }

    void map() override {
        input.set_context(context);
        input.map();
        transform_op0.map(input.output_data, context);
        conv_op.map(transform_op0.output_data, context);
        transform_op1.map(conv_op.output_data, context);
        this->output_data = transform_op1.output_data;
    }
    void forward() override {
        input.forward();
        transform_op0.forward();
        conv_op.forward();
        transform_op1.forward();
    }
    void unmap() override {
        input.unmap();
        transform_op0.unmap();
        conv_op.unmap();
        transform_op1.unmap();
    }
};

// Transform(input.layout->NCHW) Conv(NCHW->NCHW) Transform(NCHW->input.layout)
struct Conv_NCHW_NCHW: Transform_Conv {

    void init(const Json::Value &conv_config, const std::map<string,NodeBase*> &node_map) {
        this->name = conv_config["name"].asString();
        this->input.init(conv_config["inputs"], node_map);
        transform_op0.init(
            input.batch_size, input.out_channels, input.output_h, input.output_w,
            input.stride, input.layout, "NCHW"
        );
        conv_op.init(input.batch_size, input.out_channels, input.output_h, input.output_w,
                     conv_config["out_channels"].asInt(),
                     conv_config["kernel"][0].asInt(),
                     conv_config["kernel"][1].asInt(),
                     conv_config["stride"][0].asInt(),
                     conv_config["stride"][1].asInt(),
                     conv_config["padding"][0].asInt(),
                     conv_config["padding"][1].asInt(),
                     conv_config["groups"].asInt(),
                     conv_config["act"].asString(),
                     "NCHW",
                     "NCHW",
                     conv_config["disable_tc"].asBool(),
                     conv_config["use_tc"].asBool());
        int transform_op1_input_stride[4];
        get_stride(
            conv_op.batch_size, conv_op.out_channels, conv_op.output_h, conv_op.output_w,
            "NCHW", transform_op1_input_stride
        );
        transform_op1.init(
            conv_op.batch_size, conv_op.out_channels, conv_op.output_h, conv_op.output_w,
            transform_op1_input_stride, "NCHW", input.layout
        );
        this->batch_size = input.batch_size;
        this->out_channels = conv_op.out_channels;
        this->output_h = conv_op.output_h;
        this->output_w = conv_op.output_w;
        this->context = nullptr;
        this->output_data = nullptr;
        this->init_stride(input.layout);
    }
};

// Transform(input.layout->NCHW) Conv(NCHW->NHWC) Transform(NHWC->input.layout)
struct Conv_NCHW_NHWC: Transform_Conv {

    void init(const Json::Value &conv_config, const std::map<string,NodeBase*> &node_map) {
        this->name = conv_config["name"].asString();
        this->input.init(conv_config["inputs"], node_map);
        transform_op0.init(
            input.batch_size, input.out_channels, input.output_h, input.output_w,
            input.stride, input.layout, "NCHW"
        );
        conv_op.init(input.batch_size, input.out_channels, input.output_h, input.output_w,
                     conv_config["out_channels"].asInt(),
                     conv_config["kernel"][0].asInt(),
                     conv_config["kernel"][1].asInt(),
                     conv_config["stride"][0].asInt(),
                     conv_config["stride"][1].asInt(),
                     conv_config["padding"][0].asInt(),
                     conv_config["padding"][1].asInt(),
                     conv_config["groups"].asInt(),
                     conv_config["act"].asString(),
                     "NCHW",
                     "NHWC",
                     conv_config["disable_tc"].asBool(),
                     conv_config["use_tc"].asBool());
        int transform_op1_input_stride[4];
        get_stride(
            conv_op.batch_size, conv_op.out_channels, conv_op.output_h, conv_op.output_w,
            "NHWC", transform_op1_input_stride
        );
        transform_op1.init(
            conv_op.batch_size, conv_op.out_channels, conv_op.output_h, conv_op.output_w,
            transform_op1_input_stride, "NHWC", input.layout
        );
        this->batch_size = input.batch_size;
        this->out_channels = conv_op.out_channels;
        this->output_h = conv_op.output_h;
        this->output_w = conv_op.output_w;
        this->context = nullptr;
        this->output_data = nullptr;
        this->init_stride(input.layout);
    }
};

// Transform(input.layout->NHWC) Conv(NHWC->NCHW) Transform(NCHW->input.layout)
struct Conv_NHWC_NCHW: Transform_Conv {

    void init(const Json::Value &conv_config, const std::map<string,NodeBase*> &node_map) {
        this->name = conv_config["name"].asString();
        this->input.init(conv_config["inputs"], node_map);
        transform_op0.init(
            input.batch_size, input.out_channels, input.output_h, input.output_w,
            input.stride, input.layout, "NHWC"
        );
        conv_op.init(input.batch_size, input.out_channels, input.output_h, input.output_w,
                     conv_config["out_channels"].asInt(),
                     conv_config["kernel"][0].asInt(),
                     conv_config["kernel"][1].asInt(),
                     conv_config["stride"][0].asInt(),
                     conv_config["stride"][1].asInt(),
                     conv_config["padding"][0].asInt(),
                     conv_config["padding"][1].asInt(),
                     conv_config["groups"].asInt(),
                     conv_config["act"].asString(),
                     "NHWC",
                     "NCHW",
                     conv_config["disable_tc"].asBool(),
                     conv_config["use_tc"].asBool());
        int transform_op1_input_stride[4];
        get_stride(
            conv_op.batch_size, conv_op.out_channels, conv_op.output_h, conv_op.output_w,
            "NCHW", transform_op1_input_stride
        );
        transform_op1.init(
            conv_op.batch_size, conv_op.out_channels, conv_op.output_h, conv_op.output_w,
            transform_op1_input_stride, "NCHW", input.layout
        );
        this->batch_size = input.batch_size;
        this->out_channels = conv_op.out_channels;
        this->output_h = conv_op.output_h;
        this->output_w = conv_op.output_w;
        this->context = nullptr;
        this->output_data = nullptr;
        this->init_stride(input.layout);
    }
};

// Transform(input.layout->NHWC) Conv(NHWC->NHWC) Transform(NHWC->input.layout)
struct Conv_NHWC_NHWC: Transform_Conv {

    void init(const Json::Value &conv_config, const std::map<string,NodeBase*> &node_map) {
        this->name = conv_config["name"].asString();
        this->input.init(conv_config["inputs"], node_map);
        transform_op0.init(
            input.batch_size, input.out_channels, input.output_h, input.output_w,
            input.stride, input.layout, "NHWC"
        );
        conv_op.init(input.batch_size, input.out_channels, input.output_h, input.output_w,
                     conv_config["out_channels"].asInt(),
                     conv_config["kernel"][0].asInt(),
                     conv_config["kernel"][1].asInt(),
                     conv_config["stride"][0].asInt(),
                     conv_config["stride"][1].asInt(),
                     conv_config["padding"][0].asInt(),
                     conv_config["padding"][1].asInt(),
                     conv_config["groups"].asInt(),
                     conv_config["act"].asString(),
                     "NHWC",
                     "NHWC",
                     conv_config["disable_tc"].asBool(),
                     conv_config["use_tc"].asBool());
        int transform_op1_input_stride[4];
        get_stride(
            conv_op.batch_size, conv_op.out_channels, conv_op.output_h, conv_op.output_w,
            "NHWC", transform_op1_input_stride
        );
        transform_op1.init(
            conv_op.batch_size, conv_op.out_channels, conv_op.output_h, conv_op.output_w,
            transform_op1_input_stride, "NHWC", input.layout
        );
        this->batch_size = input.batch_size;
        this->out_channels = conv_op.out_channels;
        this->output_h = conv_op.output_h;
        this->output_w = conv_op.output_w;
        this->context = nullptr;
        this->output_data = nullptr;
        this->init_stride(input.layout);
    }
};

struct Element: NodeBase {
    Value inputs[2];
    ElementOP elem_op;

    NodeBase * get_input(Json::Value inputs_config, const std::map<string,NodeBase*> &node_map, int idx) {
        Json::Value cinput = inputs_config[0];
        assert(cinput.size() == 2);
        assert(0 <= idx && idx < 2);
        cinput = cinput[idx];
        NodeBase *node = node_map.at(cinput[0].asString());
        int begin = cinput[1].asInt();
        int end = cinput[2].asInt();
        assert(begin == 0 && end == node->out_channels);
        return node;
    }

    void init(const Json::Value &elem_config, const std::map<string,NodeBase*> &node_map) {
        name = elem_config["name"].asString();
        Json::Value inputs_config = elem_config["inputs"];
        assert(inputs_config.size() == 1);
        int op_type;
        if(elem_config["op_type"].asString() == "mul") {
            op_type = ElementOP::MUL;
        } else if(elem_config["op_type"].asString() == "add") {
            op_type = ElementOP::ADD;
        } else {
            FatalError("");
        }
        this->inputs[0].init(inputs_config[0][0], node_map);
        this->inputs[1].init(inputs_config[0][1], node_map);
        elem_op.init(
            inputs[0].batch_size, inputs[0].out_channels, inputs[0].output_h, inputs[0].output_w, op_type, inputs[0].layout, inputs[1].layout
        );
        this->context = nullptr;
        this->output_data = nullptr;
        this->batch_size = inputs[0].batch_size;
        this->out_channels = inputs[0].out_channels;
        this->output_h = inputs[0].output_h;
        this->output_w = inputs[0].output_w;
        this->init_stride(elem_config["layout"].asString());
    }
    void map() override {
        inputs[0].map();
        inputs[1].map();
        elem_op.map(inputs[0].output_data, inputs[1].output_data, this->context);
        this->output_data = elem_op.output_data;
    }
    void forward() override {
        inputs[0].forward();
        inputs[1].forward();
        elem_op.forward();
    }
    void unmap() override {
        inputs[0].unmap();
        inputs[1].unmap();
        elem_op.unmap();
    }
};


struct Graph;
struct Sequential: NodeBase {
    std::vector<NodeBase*> nodes;

    void init(const Json::Value &config, std::map<string,NodeBase*> &node_map, Graph *graph);

    void map() {
        for(auto node : nodes) {
            node->context = context;
            node->map();
        }
        this->output_data = nodes.back()->output_data;
    }

    void forward() {
        for(auto node : nodes) {
            node->forward();
        }
    }

    void unmap() {
        for(auto node : nodes) {
            node->unmap();
        }
    }

};

struct Activation: NodeBase {
    Input input;
    ActivationOP act;

    void init(const Json::Value &config, const std::map<string,NodeBase*> &node_map) {
        name = config["name"].asString();
        input.init(config["inputs"], node_map);
        int act_type;
        string act_str = config["act_type"].asString();
        if(act_str == "relu")
            act_type = ActivationOP::RELU;
        else if(act_str == "tanh")
            act_type = ActivationOP::TANH;
        else if(act_str == "sigmoid")
            act_type = ActivationOP::SIGMOID;
        else
            FatalError("unsupported activation mode " + act_str);
        assert(config.isMember("inplace"));
        act.init(
            input.batch_size, input.out_channels, input.output_h, input.output_w, act_type,
            config["inplace"].asBool(), input.layout
        );
        this->batch_size = act.batch_size;
        this->out_channels = act.out_channels;
        this->output_h = act.output_h;
        this->output_w = act.output_w;
        this->output_data = nullptr;
        this->context = nullptr;
        this->init_stride(config["layout"].asString());
    }
    void map() {
        input.set_context(context);
        input.map();
        act.map(input.output_data, context);
        this->output_data = act.output_data;
    }
    void forward() {
        input.forward();
        act.forward();
    }
    void unmap() {
        input.unmap();
        act.unmap();
    }
};

struct Relu: NodeBase {
    Input input;
    ActivationOP act_relu;

    void init(const Json::Value &config, const std::map<string,NodeBase*> &node_map) {
        name = config["name"].asString();
        input.init(config["inputs"], node_map);
        act_relu.init(
            input.batch_size, input.out_channels, input.output_h, input.output_w, ActivationOP::RELU,
            false, input.layout
        );
        this->batch_size = act_relu.batch_size;
        this->out_channels = act_relu.out_channels;
        this->output_h = act_relu.output_h;
        this->output_w = act_relu.output_w;
        this->output_data = nullptr;
        this->context = nullptr;
        this->init_stride(config["layout"].asString());
    }
    void map() {
        input.set_context(context);
        input.map();
        act_relu.map(input.output_data, context);
        this->output_data = act_relu.output_data;
    }
    void forward() {
        input.forward();
        act_relu.forward();
    }
    void unmap() {
        input.unmap();
        act_relu.unmap();
    }
};

struct Identity: NodeBase {
    Input input;

    void init(const Json::Value &config, const std::map<string,NodeBase*> &node_map) {
        this->name = config["name"].asString();
        this->context = context;
        this->input.init(config["inputs"], node_map);
        this->batch_size = input.batch_size;
        this->out_channels = input.out_channels;
        this->output_h = input.output_h;
        this->output_w = input.output_w;
        this->init_stride(config["layout"].asString());
    }
    void map() override {
        input.set_context(context);
        input.map();
        this->output_data = input.output_data;
    }
    void forward() override {
        input.forward();
    }
    void unmap() override {
        input.unmap();
    }
};

struct Pool: NodeBase {
    Input input;
    PoolOP pool_op;

    void init(const Json::Value &config, const std::map<string,NodeBase*> &node_map) {
        this->name = config["name"].asString();
        this->input.init(config["inputs"], node_map);
        int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;

        int pool_type;
        if(config["pool_type"].asString().find("max") != string::npos)
            pool_type = PoolOP::MAX_POOL;
        else
            pool_type = PoolOP::AVG_POOL;
        if(config["pool_type"].asString().find("global") != string::npos) {
            kernel_h = input.output_h;
            kernel_w = input.output_w;
            stride_h = 1;
            stride_w = 1;
            padding_h = 0;
            padding_w = 0;
        } else {
            kernel_h = config["kernel"][0].asInt();
            kernel_w = config["kernel"][1].asInt();
            stride_h = config["stride"][0].asInt();
            stride_w = config["stride"][1].asInt();
            padding_h = config["padding"][0].asInt();
            padding_w = config["padding"][1].asInt();
        }
        pool_op.init(
            input.batch_size, input.out_channels, input.output_h, input.output_w, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, pool_type, input.layout
        );
        pool_op.name = name;

        this->batch_size = pool_op.batch_size;
        this->out_channels = pool_op.out_channels;
        this->output_h = pool_op.output_h;
        this->output_w = pool_op.output_w;
        this->init_stride(config["layout"].asString());
    }
    void map() override {
        input.set_context(context);
        input.map();

        pool_op.map(input.output_data, context);
        this->output_data = pool_op.output_data;
    }
    void forward() override {
        input.forward();
        pool_op.forward();
    }
    void unmap() override {
        input.unmap();
        pool_op.unmap();
    }
};

struct Transform: NodeBase {
    Input input;
    TransformOp transform_op;

    void init(const Json::Value &config, const std::map<string,NodeBase*> &node_map) {
        this->name = config["name"].asString();
        this->input.init(config["inputs"], node_map);
        transform_op.init(input.batch_size, input.out_channels, input.output_h, input.output_w,
                          input.stride, input.layout, config["dst_layout"].asString());
        this->batch_size = input.batch_size;
        this->out_channels =  input.out_channels;
        this->output_h =  input.output_h;
        this->output_w = input.output_w;
        this->context = context;
        this->output_data = nullptr;
        this->init_stride(config["dst_layout"].asString());
    }

    void map() override {
        input.set_context(context);
        input.map();
        transform_op.map(input.output_data, context);
        this->output_data = input.output_data;
    }
    void forward() override {
        input.forward();
        transform_op.forward();
    }
    void unmap() override {
        input.unmap();
        transform_op.unmap();
    }
};

struct SplitBatch: NodeBase {
    Input input;
    int batch_begin;
    int batch_end;
    
    void init(const Json::Value &batch_config, const std::map<string,NodeBase*> &node_map) {
        this->name = batch_config["name"].asString();
        this->input.init(batch_config["inputs"], node_map);
        this->batch_begin = batch_config["batch_begin"].asInt();
        this->batch_end = batch_config["batch_end"].asInt();
        this->context = nullptr;
        this->output_data = nullptr;
        this->batch_size = batch_end - batch_begin;
        this->out_channels = input.out_channels;
        this->output_h = input.output_h;
        this->output_w = input.output_w;
        this->init_stride(batch_config["layout"].asString());
    }

    void map() override {
        input.set_context(context);
        input.map();
        if (batch_begin > batch_end)
            FatalError("batch_begin > batch_end");
        if (batch_end > input.batch_size)
            FatalError("batch_end should not > input's batch_size");
        output_data = input.output_data + batch_begin * out_channels * output_h * output_w;
        assert(output_data != nullptr);
    }

    void unmap() override {
        return;
    }

    void forward() override {
        return;
    }
};

struct Graph {
    int num_inputs;
    Placeholder inputs[MAX_NUM_NODES];
    int num_convs;
    Conv convs[MAX_NUM_NODES];
    int num_pools;
    Pool pools[MAX_NUM_NODES];
    int num_idents;
    Identity idents[MAX_NUM_NODES];
    int num_elem;
    Element elems[MAX_NUM_NODES];
    int num_relus;
    Relu relus[MAX_NUM_NODES];
    int num_acts;
    Activation acts[MAX_NUM_NODES];
    int num_sequential;
    Sequential sequential[MAX_NUM_NODES];
    int num_transform;
    Transform transform[MAX_NUM_NODES];
    int num_transform_conv_NCHW_NCHW;
    Conv_NCHW_NCHW transform_conv_NCHW_NCHW[MAX_NUM_NODES];
    int num_transform_conv_NCHW_NHWC;
    Conv_NCHW_NHWC transform_conv_NCHW_NHWC[MAX_NUM_NODES];
    int num_transform_conv_NHWC_NCHW;
    Conv_NHWC_NCHW transform_conv_NHWC_NCHW[MAX_NUM_NODES];
    int num_transform_conv_NHWC_NHWC;
    Conv_NHWC_NHWC transform_conv_NHWC_NHWC[MAX_NUM_NODES];
    int num_split_batch = 0;
    SplitBatch split_batch[MAX_NUM_NODES];

    int num_stages;
    int stage_num_seq[MAX_NUM_NODES];
    int stage_seq_num_op[MAX_NUM_NODES][MAX_NUM_GROUPS];
    NodeBase* stages[MAX_NUM_NODES][MAX_NUM_GROUPS][MAX_GROUP_SIZE];

    void reset() {
        num_inputs = num_convs = num_pools = num_idents = num_relus = num_elem =
        num_acts = num_sequential = num_stages = num_transform = 
        num_transform_conv_NCHW_NCHW = num_transform_conv_NCHW_NHWC =
        num_transform_conv_NHWC_NCHW = num_transform_conv_NHWC_NHWC = 
        num_split_batch = 0;
    }

    NodeBase *add_node(Json::Value node_config, std::map<string, NodeBase*> &node_map) {
        NodeBase *nb = nullptr;
        if(node_config["type"].asString() == "conv") {
            assert(num_convs < MAX_NUM_NODES);
            convs[num_convs].init(node_config, node_map);
            nb = convs + num_convs++;
        } else if(node_config["type"].asString() == "pool") {
            assert(num_pools < MAX_NUM_NODES);
            pools[num_pools].init(node_config, node_map);
            nb = pools + num_pools++;
        } else if(node_config["type"].asString() == "identity") {
            assert(num_idents < MAX_NUM_NODES);
            idents[num_idents].init(node_config, node_map);
            nb = idents + num_idents++;
        } else if(node_config["type"].asString() == "activation") {
            assert(num_idents < MAX_NUM_NODES);
            acts[num_acts].init(node_config, node_map);
            nb = acts + num_acts++;
        } else if(node_config["type"].asString() == "element") {
            assert(num_elem < MAX_NUM_NODES);
            elems[num_elem].init(node_config, node_map);
            nb = elems + num_elem++;
        } else if(node_config["type"].asString() == "relu") {
            assert(num_relus < MAX_NUM_NODES);
            relus[num_relus].init(node_config, node_map);
            nb = relus + num_relus++;
        } else if(node_config["type"].asString() == "sequential") {
            assert(num_sequential < MAX_NUM_NODES);
            sequential[num_sequential].init(node_config, node_map, this);
            nb = sequential + num_sequential++;
        } else if(node_config["type"].asString() == "transform") {
            assert(num_transform < MAX_NUM_NODES);
            transform[num_transform].init(node_config, node_map);
            nb = transform + num_transform++;
        } else if(node_config["type"].asString() == "transform_conv_NCHW_NCHW") {
            assert(num_transform_conv_NCHW_NCHW < MAX_NUM_NODES);
            transform_conv_NCHW_NCHW[num_transform_conv_NCHW_NCHW].init(node_config, node_map);
            nb = transform_conv_NCHW_NCHW + num_transform_conv_NCHW_NCHW++;
        } else if(node_config["type"].asString() == "transform_conv_NCHW_NHWC") {
            assert(num_transform_conv_NCHW_NHWC < MAX_NUM_NODES);
            transform_conv_NCHW_NHWC[num_transform_conv_NCHW_NHWC].init(node_config, node_map);
            nb = transform_conv_NCHW_NHWC + num_transform_conv_NCHW_NHWC++;
        } else if(node_config["type"].asString() == "transform_conv_NHWC_NCHW") {
            assert(num_transform_conv_NHWC_NCHW < MAX_NUM_NODES);
            transform_conv_NHWC_NCHW[num_transform_conv_NHWC_NCHW].init(node_config, node_map);
            nb = transform_conv_NHWC_NCHW + num_transform_conv_NHWC_NCHW++;
        } else if(node_config["type"].asString() == "transform_conv_NHWC_NHWC") {
            assert(num_transform_conv_NHWC_NHWC < MAX_NUM_NODES);
            transform_conv_NHWC_NHWC[num_transform_conv_NHWC_NHWC].init(node_config, node_map);
            nb = transform_conv_NHWC_NHWC + num_transform_conv_NHWC_NHWC++;
        } else if(node_config["type"].asString() == "split_batch") {
            assert(num_split_batch < MAX_NUM_NODES);
            split_batch[num_split_batch].init(node_config, node_map);
            nb = split_batch + num_split_batch++;
        }
        else {
            FatalError("unsupported type " + node_config["type"].asString());
        }
        return nb;
    }
    void init_graph(int batch_size, Json::Value graph_config) {
        std::map<string, NodeBase*> node_map;
        reset();
        num_inputs = 1;
        inputs[0].init(batch_size, graph_config["input"]);
        node_map[inputs[0].name] = &inputs[0];
        for(Json::Value block : graph_config["blocks"]) {
            NodeBase *nb;
            for(const Json::Value& node : block["inner_nodes"]) {
                nb = add_node(node, node_map);
                node_map[nb->name] = nb;
            }
            nb = add_node(block["exit_node"], node_map);
            node_map[nb->name] = nb;

            for(const Json::Value& stage_config : block["stages"]) {
                stage_num_seq[num_stages] = stage_config.size();
                for(Json::ArrayIndex i = 0; i < stage_config.size(); i++) {
                    const Json::Value& seq_config = stage_config[i];
                    stage_seq_num_op[num_stages][i] = seq_config.size();
                    for(Json::ArrayIndex j = 0; j < seq_config.size(); j++) {
                        NodeBase *pnode = node_map.at(stage_config[i][j].asString());
                        stages[num_stages][i][j] = pnode;
                        pnode->set_context(contexts + i);
                    }
                }
                num_stages++;
            }
        }
    }
    void init_block(int batch_size, Json::Value block_config) {
        std::map<string, NodeBase*> node_map;
        reset();
        Json::Value enter_node = block_config["enter_node"];
        Json::Value input_shape = enter_node["output_shape"];
        num_inputs = 1;
        inputs[0].init(
            enter_node["name"].asString(),
            batch_size,
            input_shape[0].asInt(),
            input_shape[1].asInt(),
            input_shape[2].asInt(),
            enter_node["layout"].asString()
        );
        node_map[enter_node["name"].asString()] = &inputs[0];

        NodeBase *nb;
        for(const Json::Value& node : block_config["inner_nodes"]) {
            nb = add_node(node, node_map);
            node_map[nb->name] = nb;
        }
        nb = add_node(block_config["exit_node"], node_map);
        node_map[nb->name] = nb;

        for(const Json::Value& stage_config : block_config["stages"]) {
            stage_num_seq[num_stages] = stage_config.size();
            for(Json::ArrayIndex i = 0; i < stage_config.size(); i++) {
                const Json::Value& seq_config = stage_config[i];
                stage_seq_num_op[num_stages][i] = seq_config.size();
                for(Json::ArrayIndex j = 0; j < seq_config.size(); j++) {
                    NodeBase *pnode = node_map.at(stage_config[i][j].asString());
                    stages[num_stages][i][j] = pnode;
                    pnode->set_context(contexts + i);
                }
            }
            num_stages++;
        }
    }
    void init_stage(int batch_size, Json::Value stage_config, Json::Value input_config) {
        std::map<string, NodeBase*> node_map;
        reset();
        for(Json::Value::const_iterator it = input_config.begin(); it != input_config.end(); it++) {
            string name = it.key().asString();
            int out_channels = (*it)[0].asInt();
            int output_h = (*it)[1].asInt();
            int outupt_w = (*it)[2].asInt();
            // TODO: where is layout??
            string layout = (*it)[3].asString();
            inputs[num_inputs].init(name, batch_size, out_channels, output_h, outupt_w, layout);
            inputs[num_inputs].set_context(contexts + 0);
            node_map[name] = inputs + num_inputs;
            num_inputs++;
        }
        stage_num_seq[num_stages] = stage_config.size();
        for(Json::ArrayIndex i = 0; i < stage_config.size(); i++) {
            const Json::Value& seq_config = stage_config[i];
            stage_seq_num_op[num_stages][i] = seq_config.size();
            for(Json::ArrayIndex j = 0; j < seq_config.size(); j++) {
                Json::Value node_config = stage_config[i][j];
                NodeBase *pnode = add_node(node_config, node_map);
                node_map[pnode->name] = pnode;
                stages[num_stages][i][j] = pnode;
                pnode->set_context(contexts + i);
            }
        }
        num_stages++;
    }
    void set_input(data_type *input_data /*in host memory space*/) {
        size_t size = sizeof(data_type) * inputs[0].batch_size * inputs[0].out_channels * inputs[0].output_h * inputs[0].output_w;
        checkCUDA(cudaMemcpy(inputs[0].output_data, input_data, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    }
    void get_output(data_type *output_data) {
        assert(stage_num_seq[num_stages-1] == 1);
        NodeBase *nb = stages[num_stages-1][0][stage_seq_num_op[num_stages-1][0]-1];
        size_t size = sizeof(data_type) * nb->batch_size * nb->out_channels * nb->output_h * nb->output_w;
        checkCUDA(cudaMemcpy(output_data, nb->output_data, size, cudaMemcpyDeviceToHost));
    }

    void set_conv_weights(char *name, data_type *weight, data_type *bias) {
        Conv *conv = nullptr;
        for(int i = 0; i < num_convs; i++) {
            if(convs[i].name == string(name)) {
                conv = convs + i;
                break;
            }
        }
        if(conv == nullptr) {
            printf("%s not found\n", name);
            return;
        }
        if(weight != nullptr)
            checkCUDA(cudaMemcpy(conv->conv_op.filter_data, weight, conv->conv_op.get_filter_size(), cudaMemcpyHostToDevice));
        else
            assert(false);
        if(bias != nullptr)
            checkCUDA(cudaMemcpy(conv->conv_op.bias_data, bias, conv->conv_op.get_bias_size(), cudaMemcpyHostToDevice));
        else
            assert(false);
    }
    void map() {
        for(int i = 0; i < num_inputs; i++)
            inputs[i].map();
        for(int i = 0; i < num_stages; i++) {
            for(int j = 0; j < stage_num_seq[i]; j++)
                for(int k = 0; k < stage_seq_num_op[i][j]; k++)
                    stages[i][j][k]->map();
            if(stage_num_seq[i] > MAX_NUM_GROUPS) {
                fprintf(stderr, "The number of nodes in stage %d exceed the number of available streams %d\n", (int)stage_num_seq[i], MAX_NUM_GROUPS);
                assert(stage_num_seq[i] <= MAX_NUM_GROUPS);
            }
        }
    }
    static void print(const char *str, data_type *device_data, int cnt) {
        size_t size = sizeof(data_type) * cnt;
        auto * host_data = (data_type*)malloc(size);
        checkCUDA(cudaMemcpy(host_data, device_data, size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        printf("%s ", str);
        for(int i = 0; i < cnt; i++)
            printf("%+2.3f ", (double)host_data[i]);
        printf("\n");
        free(host_data);
    }

    void forward(bool record_events, cudaEvent_t *events) {
        for(int i = 0; i < num_stages; i++) {
            int stage_size = stage_num_seq[i];
            for (int j = 0; j < stage_size; j++) {
                int seq_length = stage_seq_num_op[i][j];
                for (int k = 0; k < seq_length; k++)
                    stages[i][j][k]->forward();
            }
            if(record_events) {
                if(i + 1 < num_stages)
                    checkCUDA(cudaEventRecord(events[i], 0)); // profile stage latency
            } else {
                if(i + 1 == num_stages || (stage_size == 1 && stage_num_seq[i+1] == 1))
                    continue;
                else
                    checkCUDA(cudaDeviceSynchronize());
            }
        }
    }


    void unmap() {
        for(int i = 0; i < num_inputs; i++)
            inputs[i].unmap();
        for(int i = 0; i < num_stages; i++) {
            for(int j = 0; j < stage_num_seq[i]; j++)
                for(int k = 0; k < stage_seq_num_op[i][j]; k++) {
                    stages[i][j][k]->unmap();
                }
        }
    }
    void measure_stage_latency(int warmup, int number, int repeat, float* results, bool profile_stage_latency, float *stage_results) {
        assert(!profile_stage_latency);
        cudaEvent_t start, end;

        checkCUDA(cudaEventCreate(&start));
        checkCUDA(cudaEventCreate(&end));
        map();

        // warmup
        for(int i = 0; i < warmup; i++)
            forward(false, 0);

        // measure latency
        for(int i = 0; i < repeat; i++) {
            checkCUDA(cudaDeviceSynchronize());
            checkCUDA(cudaEventRecord(start, 0));
            for (int t = 0; t < number; t++) {
                forward(false, 0);
                if(stage_num_seq[0] != 1)
                    checkCUDA(cudaDeviceSynchronize());
            }
            checkCUDA(cudaEventRecord(end, 0));
            checkCUDA(cudaDeviceSynchronize());
            float latency = 0.0f;
            checkCUDA(cudaEventElapsedTime(&latency, start, end));
            results[i] = latency / float(number);
        }

        if(profile_stage_latency) {
            for(int i = 0; i < repeat; i++)
                for(int j = 0; j < num_stages; j++)
                    stage_results[i * num_stages + j] /= (float) number;
        }

        // release resources
        unmap();
    }
    void measure_latency(int warmup, int number, int repeat, float* results, bool profile_stage_latency, float *stage_results) {
        cudaEvent_t start, end, events[MAX_NUM_NODES];

        // allocate resources
        if(profile_stage_latency) {
            for (int i = 0; i < num_stages; i++)
                checkCUDA(cudaEventCreate(events + i));
        }
        checkCUDA(cudaEventCreate(&start));
        checkCUDA(cudaEventCreate(&end));
        map();


        bool profile = (warmup >= 10000); // magic number to enable profiling
        if(profile) {
            volatile int test_complete;
            volatile int test_start;
            test_complete = 0;
            test_start = 0;
            testComplete = &test_complete;
            testStart = &test_start;
            int status;
            pthread_t pThread;
            const char *eventNames[] = {
                "active_warps_pm",
                "l2_subp0_read_sector_misses",
                "l2_subp0_total_read_sector_queries",
                "l2_subp1_read_sector_misses",
                "l2_subp1_total_read_sector_queries"
            };
            assert(warmup-10000 < sizeof(eventNames)/sizeof(const char*));
            eventName = eventNames[warmup-10000];

            status = pthread_create(&pThread, NULL, sampling_func, NULL);
            if (status != 0) {
                perror("pthread_create");
                exit(-1);
            }
            int T = number * repeat;
            while(*testStart == 0)
                usleep(1);
            while(T--)
                forward(false, events);
            test_complete = 1;
            pthread_join(pThread, NULL);
            warmup = 2;
        }


        // warmup
        for(int i = 0; i < warmup; i++)
            forward(profile_stage_latency, events);

        // measure latency
        for(int i = 0; i < repeat; i++) {
            results[i] = 0.0;
            for(int j = 0; j < num_stages; j++)
                stage_results[i * num_stages + j] = 0.0f;
            for (int t = 0; t < number; t++) {
                checkCUDA(cudaDeviceSynchronize());
                checkCUDA(cudaEventRecord(start, 0));
                forward(profile_stage_latency, events);
                // here do not need to synchronize, because and the end of last stage, there is a synchronization
                checkCUDA(cudaEventRecord(end, 0));
                checkCUDA(cudaDeviceSynchronize());
                float latency;
                if(profile_stage_latency) {
                    for (int j = 0; j < num_stages; j++) {
                        checkCUDA(cudaEventElapsedTime(&latency, j == 0 ? start : events[j - 1], j == num_stages - 1 ? end : events[j]));
                        stage_results[i * num_stages + j] += latency;
                    }
                }
                checkCUDA(cudaEventElapsedTime(&latency, start, end));
                results[i] += latency;
            }
            results[i] /= float(number);
        }

        if(profile_stage_latency) {
            for(int i = 0; i < repeat; i++)
                for(int j = 0; j < num_stages; j++)
                    stage_results[i * num_stages + j] /= (float) number;
        }

        // release resources
        unmap();
        if(profile_stage_latency) {
            for (int i = 0; i < num_stages; i++)
                checkCUDA(cudaEventDestroy(events[i]));
        }
    }
};
void Sequential::init(const Json::Value &config, std::map<string,NodeBase*> &node_map, Graph *graph) {
    name = config["name"].asString();
    nodes.clear();
    auto nodes_config = config["nodes"];
    int n = nodes_config.size();
    for(int i = 0; i < n; i++) {
        NodeBase *node = graph->add_node(nodes_config[i], node_map);
        nodes.push_back(node);
        node_map[node->name] = node;
    }
    NodeBase *tail = nodes.back();
    batch_size = tail->batch_size;
    out_channels = tail->out_channels;
    output_h = tail->output_h;
    output_w = tail->output_w;
    context = nullptr;
    output_data = nullptr;
}

Graph graph;

extern "C" {

DLL void graph_latency(const char *graph_json, int batch_size, int warmup, int number, int repeat, int profile_stage_latency, float *results, float *stage_results) {
    Json::Value graph_config = json_from_cstr(graph_json);
    graph.init_graph(batch_size, graph_config);
    graph.measure_latency(warmup, number, repeat, results, profile_stage_latency, stage_results);
}

DLL void block_latency(const char *block_json, int batch_size, int warmup, int number, int repeat, int profile_stage_latency, float *results, float *stage_results) {
    Json::Value block_config = json_from_cstr(block_json);
    graph.init_block(batch_size, block_config);
    graph.measure_latency(warmup, number, repeat, results, profile_stage_latency, stage_results);
}

DLL void stage_latency(const char *stage_json, const char *input_json, int batch_size, int warmup, int number, int repeat, int profile_stage_latency, float *results, float *stage_results) {
    Json::Value stage_config = json_from_cstr(stage_json);
    Json::Value input_config = json_from_cstr(input_json);
    graph.init_stage(batch_size, stage_config, input_config);
//    graph.measure_latency(warmup, number, repeat, results, profile_stage_latency, stage_results);
    graph.measure_stage_latency(warmup, number, repeat, results, profile_stage_latency, stage_results);
}

DLL void graph_inference(const char *graph_json, int batch_size, data_type *input,
                         int num_convs, char **conv_names, data_type **filter_data, data_type **bias_data,
                         data_type *output) {
    Json::Value graph_config;
    stringstream in(graph_json);
    in >> graph_config;
    graph.init_graph(batch_size, graph_config);

    graph.map();
    graph.set_input(input);
    for(int i = 0; i < num_convs; i++)
        graph.set_conv_weights(conv_names[i], filter_data[i], bias_data[i]);

    graph.forward(false, 0);

    graph.get_output(output);
    graph.unmap();
}

}

