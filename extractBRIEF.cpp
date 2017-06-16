/*
 Mattias P. Heinrich
 Universitaet Luebeck, 2016
 
 published and documented in:
 
 MP Heinrich, M Blendowski.
 "Multi-Organ Segmentation using Vantage Point Forests and Binary Context Features"
 Medical Image Computing and Computer Assisted Intervention (MICCAI) 2016. LNCS Springer (2016)
 (please cite if used in other works)
 
 Please see license.txt and readme.txt for more information.
 */

#include "mex.h"
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include <vector>

using namespace std;
#define printf mexPrintf

//INPUT: 3D image (float), offset sampling layout (int32) 6x64xL, list of image coordinates (int32) 1 x length
//OUTPUT: feature array, L x length (uint64)

//random-pixel offset features, coordinates are pre-defined in xy
void randomFeatures(uint64_t* features,float* image,int m,int n,int o,int* xy,int L,int S,int* indices,int length){
    
    //uint64 are used to store 64 bits each
    //total number of features is 64*L, image size m x n x o
    
    for(int f=0;f<L;f++){
        for(int k=0;k<length;k++){
            
            //indices to image coordinates where BRIEF is to be extracted
            int ind=indices[k];
            int z=ind/(m*n);
            int j=(ind-z*m*n)/m;
            int i=ind-z*m*n-j*m;
            
            uint64_t bin=0;
            uint64_t one=1;
            
            for(uint64_t s=0;s<S;s++){
                //each offset row contains two 3D coordinates
                int x1=i+xy[(s+f*S)*6]; int x2=i+xy[(s+f*S)*6+3];
                int y1=j+xy[(s+f*S)*6+1]; int y2=j+xy[(s+f*S)*6+4];
                int z1=z+xy[(s+f*S)*6+2]; int z2=z+xy[(s+f*S)*6+5];
                float value1=0; float value2=0; //zero-padding
                if(x1>=0&x1<m&y1>=0&y1<n&z1>=0&z1<o){
                    value1=image[x1+y1*m+z1*m*n];
                }
                if(x2>=0&x2<m&y2>=0&y2<n&z2>=0&z2<o){
                    value2=image[x2+y2*m+z2*m*n];
                }
                if(value1>value2){
                    bin|=one<<s; //set single bit in uint64 string
                }
            }
            features[f+L*k]=bin;
            
            
        }
        
    }
}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	timeval time1,time2;
	float* image=(float*)mxGetData(prhs[0]);
    int* xy=(int*)mxGetData(prhs[1]);
    int* indices=(int*)mxGetData(prhs[2]);
    const mwSize* indDim=mxGetDimensions(prhs[2]);
    int length=max(indDim[0],indDim[1]);
    
    const mwSize* dimsA=mxGetDimensions(prhs[1]);
    int S=dimsA[1]; //should be 64
    int L=dimsA[2];
	const mwSize* dims1=mxGetDimensions(prhs[0]);
    int m=dims1[0]; int n=dims1[1]; int o=dims1[2]; int sz=m*n*o;

    int dims2[]={L,length};
    plhs[0]=mxCreateNumericArray(2,dims2,mxUINT64_CLASS,mxREAL);
	uint64_t* output=(uint64_t*)mxGetData(plhs[0]);
    
    gettimeofday(&time1,NULL);

    randomFeatures(output,image,m,n,o,xy,L,S,indices,length);
    
    gettimeofday(&time2,NULL);
    
    float timeP1=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    //printf("Time extracting features: %4.2f secs.\n",timeP1);

	
    return;
}