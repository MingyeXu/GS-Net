#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"
#include "cuda_utils.h"


__global__ void ball_query_kernel_fast(int b, int n, int m, float radius, int nsample, 
    const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx, float *__restrict__ his) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    //      his: (B, M, nsample/3)
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;
    his += bs_idx * m * 8 + pt_idx * 8;

    float radius2 = radius * radius;
//    float radius1 = pow(radius*pow((2.0/3.0),1.0/3.0),2);
//    float radius0 = pow(radius*pow((1.0/3.0),1.0/3.0),2);   
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int d2_arr[24];
    int cnt_arr[8];
    for (int i=0;i<8;++i){cnt_arr[i]=0;}
    int sum_cnt = 0;

    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        float dx = new_x - x;
        float dy = new_y - y;
        float dz = new_z - z;
        if (k == 0){
            for (int l = 0; l < nsample; ++l) {
                idx[l] = pt_idx;
                his[l] = 0;
            }
        }        
        if (d2 < radius2 && d2!=0){
                if (dx>0 && dy>0 && dz>0){
                    if(cnt_arr[0]>2){
                        if (d2_arr[0]>d2){d2_arr[0]=d2;idx[0] = k;}
                        else if (d2_arr[1]>d2){d2_arr[1]=d2;idx[1] = k;}
                        else if (d2_arr[2]>d2){d2_arr[2]=d2;idx[2] = k;}
                    }
                    else{
                        idx[cnt_arr[0]] = k;
                        d2_arr[cnt_arr[0]] = d2;
                    }
                    cnt_arr[0]=cnt_arr[0]+1; 
                    sum_cnt=sum_cnt+1;
                    }
                else if (dx>0 && dy>0 && dz<0){
                    int index_offset = 3;
                    if(cnt_arr[1]>2){
                        if (d2_arr[index_offset + 0]>d2){d2_arr[index_offset + 0]=d2; idx[index_offset + 0] = k;}
                        else if (d2_arr[index_offset + 1]>d2){d2_arr[index_offset + 1]=d2; idx[index_offset + 1] = k;}
                        else if (d2_arr[index_offset + 2]>d2){d2_arr[index_offset + 2]=d2; idx[index_offset + 2] = k;}
                    }
                    else{
                        idx[index_offset+cnt_arr[1]] = k;
                        d2_arr[index_offset+cnt_arr[1]] = d2;
                    }                    
                    cnt_arr[1]=cnt_arr[1]+1; 
                    sum_cnt=sum_cnt+1;
                    }
                else if (dx>0 && dy<0 && dz>0){
                    int index_offset = 6;
                    if(cnt_arr[2]>2){
                        if (d2_arr[index_offset + 0]>d2){d2_arr[index_offset + 0]=d2; idx[index_offset + 0] = k;}
                        else if (d2_arr[index_offset + 1]>d2){d2_arr[index_offset + 1]=d2; idx[index_offset + 1] = k;}
                        else if (d2_arr[index_offset + 2]>d2){d2_arr[index_offset + 2]=d2; idx[index_offset + 2] = k;}
                    }
                    else{
                        idx[index_offset+cnt_arr[2]] = k;
                        d2_arr[index_offset+cnt_arr[2]] = d2;
                    }   
                    cnt_arr[2]=cnt_arr[2]+1; 
                    sum_cnt=sum_cnt+1;
                    }
                else if (dx<0 && dy>0 && dz>0){
                    int index_offset = 9;
                    if(cnt_arr[3]>2){
                        if (d2_arr[index_offset + 0]>d2){d2_arr[index_offset + 0]=d2; idx[index_offset + 0] = k;}
                        else if (d2_arr[index_offset + 1]>d2){d2_arr[index_offset + 1]=d2; idx[index_offset + 1] = k;}
                        else if (d2_arr[index_offset + 2]>d2){d2_arr[index_offset + 2]=d2; idx[index_offset + 2] = k;}
                    }
                    else{
                        idx[index_offset+cnt_arr[3]] = k;
                        d2_arr[index_offset+cnt_arr[3]] = d2;
                    }   
                    cnt_arr[3]=cnt_arr[3]+1; 
                    sum_cnt=sum_cnt+1;
                    }
                else if (dx>0 && dy<0 && dz<0){
                    int index_offset = 12;
                    if(cnt_arr[4]>2){
                        if (d2_arr[index_offset + 0]>d2){d2_arr[index_offset + 0]=d2; idx[index_offset + 0] = k;}
                        else if (d2_arr[index_offset + 1]>d2){d2_arr[index_offset + 1]=d2; idx[index_offset + 1] = k;}
                        else if (d2_arr[index_offset + 2]>d2){d2_arr[index_offset + 2]=d2; idx[index_offset + 2] = k;}
                    }
                    else{
                        idx[index_offset+cnt_arr[4]] = k;
                        d2_arr[index_offset+cnt_arr[4]] = d2;
                    }   
                    cnt_arr[4]=cnt_arr[4]+1; 
                    sum_cnt=sum_cnt+1;
                    }
                else if (dx<0 && dy>0 && dz<0){
                    int index_offset = 15;
                    if(cnt_arr[5]>2){
                        if (d2_arr[index_offset + 0]>d2){d2_arr[index_offset + 0]=d2; idx[index_offset + 0] = k;}
                        else if (d2_arr[index_offset + 1]>d2){d2_arr[index_offset + 1]=d2; idx[index_offset + 1] = k;}
                        else if (d2_arr[index_offset + 2]>d2){d2_arr[index_offset + 2]=d2; idx[index_offset + 2] = k;}
                    }
                    else{
                        idx[index_offset+cnt_arr[5]] = k;
                        d2_arr[index_offset+cnt_arr[5]] = d2;
                    }   
                    cnt_arr[5]=cnt_arr[5]+1; 
                    sum_cnt=sum_cnt+1;
                    }
                else if (dx<0 && dy<0 && dz>0){
                    int index_offset = 18;
                    if(cnt_arr[6]>2){
                        if (d2_arr[index_offset + 0]>d2){d2_arr[index_offset + 0]=d2; idx[index_offset + 0] = k;}
                        else if (d2_arr[index_offset + 1]>d2){d2_arr[index_offset + 1]=d2; idx[index_offset + 1] = k;}
                        else if (d2_arr[index_offset + 2]>d2){d2_arr[index_offset + 2]=d2; idx[index_offset + 2] = k;}
                    }
                    else{
                        idx[index_offset+cnt_arr[6]] = k;
                        d2_arr[index_offset+cnt_arr[6]] = d2;
                    }   
                    cnt_arr[6]=cnt_arr[6]+1; 
                    sum_cnt=sum_cnt+1;
                    }
                else if (dx<0 && dy<0 && dz<0){
                    int index_offset = 21;
                    if(cnt_arr[7]>2){
                        if (d2_arr[index_offset + 0]>d2){d2_arr[index_offset + 0]=d2; idx[index_offset + 0] = k;}
                        else if (d2_arr[index_offset + 1]>d2){d2_arr[index_offset + 1]=d2; idx[index_offset + 1] = k;}
                        else if (d2_arr[index_offset + 2]>d2){d2_arr[index_offset + 2]=d2; idx[index_offset + 2] = k;}
                    }
                    else{
                        idx[index_offset+cnt_arr[7]] = k;
                        d2_arr[index_offset+cnt_arr[7]] = d2;
                    }    
                    cnt_arr[7]=cnt_arr[7]+1; 
                    sum_cnt=sum_cnt+1;
                    }
        }
    }

    for (int i=0;i<8;++i){
        his[i]=(float)cnt_arr[i]/(float)sum_cnt;
    }




}


void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, \
    const float *new_xyz, const float *xyz, int *idx, float *his, cudaStream_t stream) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample, new_xyz, xyz, idx, his);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}