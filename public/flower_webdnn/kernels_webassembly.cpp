
#include <stdlib.h>
#include <math.h>

float static_buffer[26527378];
float* dynamic_buffer = nullptr;

int meta_buf_0[] = {26464917,25444370,1,3,75,75,75,75,3,3,1,1,1,1,1,1};
int meta_buf_1[] = {25444370,0,24196818,5625,64,27};
int meta_buf_2[] = {1728,24196818,24196818,64,64,64,5625};
int meta_buf_3[] = {24196818,15258450,1,64,75,75,75,75,3,3,1,1,1,1,1,1};
int meta_buf_4[] = {15258450,1792,23836818,5625,64,576};
int meta_buf_5[] = {38656,23836818,23476818,64,64,64,5625};
int meta_buf_6[] = {23476818,25596245,1,75,75,64,37,37,2,2,2,2,0,0};
int meta_buf_7[] = {25596245,20075538,1,64,37,37,37,37,3,3,1,1,1,1,1,1};
int meta_buf_8[] = {20075538,38720,25269138,1369,128,576};
int meta_buf_9[] = {112448,25269138,24743442,128,128,128,1369};
int meta_buf_10[] = {24743442,18498450,1,128,37,37,37,37,3,3,1,1,1,1,1,1};
int meta_buf_11[] = {18498450,112576,24918674,1369,128,1152};
int meta_buf_12[] = {260032,24918674,25093906,128,128,128,1369};
int meta_buf_13[] = {25093906,26319765,1,37,37,128,18,18,2,2,2,2,0,0};
int meta_buf_14[] = {26319765,23103570,1,128,18,18,18,18,3,3,1,1,1,1,1,1};
int meta_buf_15[] = {23103570,260160,25683861,324,256,1152};
int meta_buf_16[] = {555072,25683861,25683861,256,256,256,324};
int meta_buf_17[] = {25683861,20864082,1,256,18,18,18,18,3,3,1,1,1,1,1,1};
int meta_buf_18[] = {20864082,555328,25766805,324,256,2304};
int meta_buf_19[] = {1145152,25766805,25766805,256,256,256,324};
int meta_buf_20[] = {25766805,21610578,1,256,18,18,18,18,3,3,1,1,1,1,1,1};
int meta_buf_21[] = {21610578,1145408,25849749,324,256,2304};
int meta_buf_22[] = {1735232,25849749,25932693,256,256,256,324};
int meta_buf_23[] = {25932693,26444181,1,18,18,256,9,9,2,2,2,2,0,0};
int meta_buf_24[] = {26444181,24556818,1,256,9,9,9,9,3,3,1,1,1,1,1,1};
int meta_buf_25[] = {24556818,1735488,26402709,81,512,2304};
int meta_buf_26[] = {2915136,26402709,26402709,512,512,512,81};
int meta_buf_27[] = {26402709,22357074,1,512,9,9,9,9,3,3,1,1,1,1,1,1};
int meta_buf_28[] = {22357074,2915648,26278293,81,512,4608};
int meta_buf_29[] = {5274944,26278293,26361237,512,512,512,81};
int meta_buf_30[] = {26361237,22730322,1,512,9,9,9,9,3,3,1,1,1,1,1,1};
int meta_buf_31[] = {22730322,5275456,26236821,81,512,4608};
int meta_buf_32[] = {7634752,26236821,26236821,512,512,512,81};
int meta_buf_33[] = {26236821,26489984,1,9,9,512,4,4,2,2,2,2,0,0};
int meta_buf_34[] = {26489984,26015637,1,512,4,4,4,4,3,3,1,1,1,1,1,1};
int meta_buf_35[] = {26015637,7635264,26498176,16,512,4608};
int meta_buf_36[] = {9994560,26498176,26506368,512,512,512,16};
int meta_buf_37[] = {26506368,26089365,1,512,4,4,4,4,3,3,1,1,1,1,1,1};
int meta_buf_38[] = {26089365,9995072,26481792,16,512,4608};
int meta_buf_39[] = {12354368,26481792,26481792,512,512,512,16};
int meta_buf_40[] = {26481792,26163093,1,512,4,4,4,4,3,3,1,1,1,1,1,1};
int meta_buf_41[] = {26163093,12354880,26514560,16,512,4608};
int meta_buf_42[] = {14714176,26514560,26514560,512,512,512,16};
int meta_buf_43[] = {26514560,26522752,1,4,4,512,2,2,2,2,2,2,0,0};
int meta_buf_44[] = {26522752,26526848,1,2,2,512,1,1,2,2,1,1,0,0};
int meta_buf_45[] = {26526848,14714688,26524800,1,1024,512};
int meta_buf_46[] = {15238976,26524800,26525824,1024};
int meta_buf_47[] = {26525824,15240000,26527360,1,18,1024};
int meta_buf_48[] = {15258432,26527360,26527360,18};
int meta_buf_49[] = {26527360,26527360,1,18};
int* meta_buffers[] = {meta_buf_0,meta_buf_1,meta_buf_2,meta_buf_3,meta_buf_4,meta_buf_5,meta_buf_6,meta_buf_7,meta_buf_8,meta_buf_9,meta_buf_10,meta_buf_11,meta_buf_12,meta_buf_13,meta_buf_14,meta_buf_15,meta_buf_16,meta_buf_17,meta_buf_18,meta_buf_19,meta_buf_20,meta_buf_21,meta_buf_22,meta_buf_23,meta_buf_24,meta_buf_25,meta_buf_26,meta_buf_27,meta_buf_28,meta_buf_29,meta_buf_30,meta_buf_31,meta_buf_32,meta_buf_33,meta_buf_34,meta_buf_35,meta_buf_36,meta_buf_37,meta_buf_38,meta_buf_39,meta_buf_40,meta_buf_41,meta_buf_42,meta_buf_43,meta_buf_44,meta_buf_45,meta_buf_46,meta_buf_47,meta_buf_48,meta_buf_49};

extern "C" void init() {
    //static_buffer = (float*)malloc(26527378 * sizeof(float));
}

extern "C" float* get_static_buffer(void) {
    return static_buffer;
}

extern "C" float* allocate_dynamic_buffer(int count) {
    if (dynamic_buffer) {
        free(dynamic_buffer);
        dynamic_buffer = nullptr;
    }
    dynamic_buffer = (float*)malloc(count * sizeof(float));
    return dynamic_buffer;
}

extern "C" float* get_dynamic_buffer(void) {
    return dynamic_buffer;
}
extern "C" void set_placeholder_value(int kernel_order, int offset, int value) {
    meta_buffers[kernel_order][offset] = value;
}

void im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(const int * meta_buffer)
{
    const float *im = (static_buffer + meta_buffer[0]);
    float *col = (static_buffer + meta_buffer[1]);

    const int N = meta_buffer[2];
    const int C1 = meta_buffer[3];
    const int H1 = meta_buffer[4];
    const int W1 = meta_buffer[5];
    const int H2 = meta_buffer[6];
    const int W2 = meta_buffer[7];
    const int KH = meta_buffer[8];
    const int KW = meta_buffer[9];
    const int DH = meta_buffer[10];
    const int DW = meta_buffer[11];
    const int SH = meta_buffer[12];
    const int SW = meta_buffer[13];
    const int PH = meta_buffer[14];
    const int PW = meta_buffer[15];

    for (int gid = 0; gid < N*H2*W2*KH*KW*C1; gid += 1) {
        const int c1 = gid % C1;
        const int kw = gid / C1 % KW;
        const int kh = gid / C1 / KW % KH;
        const int w2 = gid / C1 / KW / KH % W2;
        const int h2 = gid / C1 / KW / KH / W2 % H2;
        const int  n = gid / C1 / KW / KH / W2 / H2;

        const int h1 = h2 * SH - PH + kh * DH;
        const int w1 = w2 * SW - PW + kw * DW;

        col[gid] = (h1 < 0 || h1 >= H1 || w1 < 0 || w1 >= W1) ? 0 : im[((n*H1+h1)*W1+w1)*C1+c1];
    }
}


#ifndef INCLUDE_EIGEN
#define INCLUDE_EIGEN
#include <Eigen/Dense>
#endif

void tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(const int * meta_buffer)
{
    float *A = (static_buffer + meta_buffer[0]);
    float *B = (static_buffer + meta_buffer[1]);
    float *C = (static_buffer + meta_buffer[2]);

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > a_mat(A, meta_buffer[3], meta_buffer[5]);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > b_mat(B, meta_buffer[5], meta_buffer[4]);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > c_mat(C, meta_buffer[3], meta_buffer[4]);

    c_mat.noalias() = a_mat * b_mat;
}


void fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    const float * v2 = (static_buffer + meta_buffer[1]);
    float * v3 = (static_buffer + meta_buffer[2]);
    const int v4 = meta_buffer[3];
    const int v5 = meta_buffer[4];
    const int D0 = meta_buffer[5];
    const int D1 = meta_buffer[6];
    int d0;
    for (d0 = ((1 > 8) ? (0 % (1 / 8)) : 0); d0 < D0; d0 += ((1 > 8) ? (1 / 8) : 1)) {
        const float v6 = v1[d0];
        int d1;
        for (d1 = ((1 > 8) ? (0 / (1 / 8)) : 0); d1 < D1; d1 += ((1 > 8) ? 8 : 1)) {
            const float v7 = v2[d0 + d1*v4];
            float v8;
            {
                v8 = v7 + v6;
            }
            float v9;
            {
                v9 = v8 > 0 ? v8 : 0;
            }
            v3[d0 + d1*v5] = v9;
        }
    }
}


void maxpooling2d_386ebef75ea973afd833e12e5ea8f6c97456944cd3812afb94b9bb4e(const int * meta_buffer)
{
    const float *X = (static_buffer + meta_buffer[0]);
    float *Y = (static_buffer + meta_buffer[1]);
    const int N = meta_buffer[2];
    const int H1 = meta_buffer[3];
    const int W1 = meta_buffer[4];
    const int C = meta_buffer[5];
    const int H2 = meta_buffer[6];
    const int W2 = meta_buffer[7];
    const int KH = meta_buffer[8];
    const int KW = meta_buffer[9];
    const int SH = meta_buffer[10];
    const int SW = meta_buffer[11];
    const int PH = meta_buffer[12];
    const int PW = meta_buffer[13];

    for (int gid = 0; gid < N * H2 * W2 * C; gid += 1) {
        const int c = gid % C;
        const int w2 = gid / C % W2;
        const int h2 = gid / C / W2 % H2;
        const int n = gid / C / W2 / H2;

        float v = -1e7;
        for (int kh = 0; kh < KH; kh++) {
            const int h1 = h2 * SH - PH + kh;
            if (h1 < 0 || h1 >= H1) continue;

            for (int kw = 0; kw < KW; kw++) {
                const int w1 = w2 * SW - PW + kw;
                if (w1 < 0 || w1 >= W1) continue;

                v = v > X[((n * H1 + h1) * W1 + w1) * C + c] ? v : X[((n * H1 + h1) * W1 + w1) * C + c];
            }
        }

        Y[gid] = v;
    }
}


void averagepooling2d_315133a9f01ff68a853da10590f9591258950d2d5ab8b7bac49ab0ea(const int * meta_buffer)
{
    const float *X = (static_buffer + meta_buffer[0]);
    float *Y = (static_buffer + meta_buffer[1]);
    const int N = meta_buffer[2];
    const int H1 = meta_buffer[3];
    const int W1 = meta_buffer[4];
    const int C = meta_buffer[5];
    const int H2 = meta_buffer[6];
    const int W2 = meta_buffer[7];

    const int KH = meta_buffer[8];
    const int KW = meta_buffer[9];
    const int SH = meta_buffer[10];
    const int SW = meta_buffer[11];
    const int PH = meta_buffer[12];
    const int PW = meta_buffer[13];

    for (int gid = 0; gid < N * H2 * W2 * C; gid += 1) {
        const int c = gid % C;
        const int w2 = gid / C % W2;
        const int h2 = gid / C / W2 % H2;
        const int n = gid / C / W2 / H2;

        float v = 0;
        
        for (int kh = 0; kh < KH; kh++) {
            const int h1 = h2 * SH - PH + kh;
            if (h1 < 0 || h1 >= H1) continue;

            for (int kw = 0; kw < KW; kw++) {
                const int w1 = w2 * SW - PW + kw;
                if (w1 < 0 || w1 >= W1) continue;

                v += X[((n * H1 + h1) * W1 + w1) * C + c];
                
            }
        }
        v /= KH * KW;

        Y[gid] = v;
    }
}


void fusedelementwise_9d5b58d4f5a00a595e84c1773b477ca77b891512fffc1c7d24d35e3d(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    const float * v2 = (static_buffer + meta_buffer[1]);
    float * v3 = (static_buffer + meta_buffer[2]);
    const int D0 = meta_buffer[3];
    int d0;
    for (d0 = 0; d0 < D0; d0 += 1) {
        const float v4 = v1[d0];
        const float v5 = v2[d0];
        float v6;
        {
            v6 = v5 + v4;
        }
        float v7;
        {
            v7 = v6 > 0 ? v6 : 0;
        }
        v3[d0] = v7;
    }
}


void elementwiseadd_98450d361d3b8b06ad718e1e961bda61506f2baab3157c18aecbf897(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    const float * v2 = (static_buffer + meta_buffer[1]);
    float * v3 = (static_buffer + meta_buffer[2]);
    const int D0 = meta_buffer[3];
    int d0;
    for (d0 = 0; d0 < D0; d0 += 1) {
        const float v4 = v1[d0];
        const float v5 = v2[d0];
        float v6;
        {
            v6 = v5 + v4;
        }
        v3[d0] = v6;
    }
}


void softmax_be4c2fae32b9326b0ca891226ac9234de828aecba508a30394d8a57e(const int * meta_buffer)
{
    const float *X = (static_buffer + meta_buffer[0]);
    float *Y = (static_buffer + meta_buffer[1]);
    const int N = meta_buffer[2];
    const int C = meta_buffer[3];

    for (int n = 0; n < N; n++) {
        float set_max = X[n * C];
        for (int c = 0; c < C; c++) {
            float val = X[n * C + c];
            if (val > set_max) {
                set_max = val;
            }
        }

        float sum_exp = 0.0F;
        for (int c = 0; c < C; c++) {
            float val = X[n * C + c];
            float exp_x = expf(val - set_max);
            sum_exp += exp_x;
            Y[n * C + c] = exp_x;
        }

        for (int c = 0; c < C; c++) {
            Y[n * C + c] /= sum_exp;
        }
    }
}

extern "C" void run() {
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_0);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_1);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_2);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_3);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_4);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_5);
maxpooling2d_386ebef75ea973afd833e12e5ea8f6c97456944cd3812afb94b9bb4e(meta_buf_6);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_7);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_8);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_9);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_10);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_11);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_12);
maxpooling2d_386ebef75ea973afd833e12e5ea8f6c97456944cd3812afb94b9bb4e(meta_buf_13);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_14);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_15);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_16);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_17);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_18);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_19);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_20);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_21);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_22);
maxpooling2d_386ebef75ea973afd833e12e5ea8f6c97456944cd3812afb94b9bb4e(meta_buf_23);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_24);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_25);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_26);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_27);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_28);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_29);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_30);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_31);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_32);
maxpooling2d_386ebef75ea973afd833e12e5ea8f6c97456944cd3812afb94b9bb4e(meta_buf_33);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_34);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_35);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_36);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_37);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_38);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_39);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_40);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_41);
fusedelementwise_39f491a753592bd6d9a83c6f76917012b5eb3c0ac597e095aa174319(meta_buf_42);
maxpooling2d_386ebef75ea973afd833e12e5ea8f6c97456944cd3812afb94b9bb4e(meta_buf_43);
averagepooling2d_315133a9f01ff68a853da10590f9591258950d2d5ab8b7bac49ab0ea(meta_buf_44);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_45);
fusedelementwise_9d5b58d4f5a00a595e84c1773b477ca77b891512fffc1c7d24d35e3d(meta_buf_46);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_47);
elementwiseadd_98450d361d3b8b06ad718e1e961bda61506f2baab3157c18aecbf897(meta_buf_48);
softmax_be4c2fae32b9326b0ca891226ac9234de828aecba508a30394d8a57e(meta_buf_49);

}

