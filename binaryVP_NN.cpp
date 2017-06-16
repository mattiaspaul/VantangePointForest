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
#include <iostream>
#include <cmath>
#include <fstream>
#include <sys/time.h>
#include <algorithm>
#include <vector>
#include <stack>
#include <list>
#include <queue>
#include <x86intrin.h>
#include <functional>
#include <numeric>
#include <cstring>

#define printf mexPrintf

using namespace std;


//INPUT: train feature array featDim x szTrain, test array featDim x szTest, ..
//(scalars) number of k-Nearest Neigbours to find, bucket/leaf_size, number of VP trees

//OUTPUT: indices of nearest neighbours (knn x szTest) and optionally their Hamming distance and count of samples in buckets

//struct for vp-tree
struct Node{
    int ind; float median;
    vector<pair<float,int>> data;
    Node* left=NULL;
    Node* right=NULL;
};

//method to count and delete data from nodes in tree
int countNodes(Node *root){
    stack<Node*> mystack;
    mystack.push(root);
    int count=0;
    while (!mystack.empty())
    {
        Node* current=mystack.top();
        mystack.pop();
        
        while(current!=NULL){
            current->data.clear();

            if(current->right!=NULL){
                mystack.push(current->right);
            }
            current=current->left;
            count++;
        }
    }
    return count;
}

// Hamming search tree with clusteres/buckets of min_size
Node* buildTree(uint64_t* dataTrain,int szTrain,int featDim,int min_size){
    
    timeval time1,time2;

    Node* root=new Node;
    root->ind=0; root->median=10000;
    
    //elements is list of nodes (with data), which still has to be processed
    for(int i=0;i<szTrain;i++){
        root->data.push_back({0.0f,i});
    }
    list<Node*> elements;
    
    elements.push_back(root);
    
    while(!elements.empty()){
        //retrieve current set of elements
        Node* current_node=elements.front();
        elements.pop_front();
        vector<pair<float,int>> current=current_node->data;
        
        if(current.size()>min_size){ //partition further
            current_node->data.clear();
            int vp=rand()%current.size();
            int i1=current[vp].second;
            current_node->ind=i1;
            
            //compare vantage-point with all elements in partition (lower+1 to upper)
            for(auto& el:current){
                int i2=el.second;
                float distT=0;
                for(int k=0;k<featDim;k++){
                    distT+=__builtin_popcountll(dataTrain[i1*featDim+k]^dataTrain[i2*featDim+k]);
                }
                el.first=distT;
            }
            //split data around median distance
            int median=current.size()/2;
            nth_element(begin(current),begin(current)+median,end(current));
            current_node->median=current[median].first;
            
            //insert elements to left and right subsets
            vector<pair<float,int>> left(begin(current),begin(current)+median);
            vector<pair<float,int>> right(begin(current)+median,end(current));
            Node* left_node=new Node;
            Node* right_node=new Node;
            current_node->left=left_node;
            current_node->right=right_node;
            
            left_node->data=left;
            right_node->data=right;

            elements.push_back(left_node);
            elements.push_back(right_node);
        }
        else{
            
        }
    }
    return root;
    
}

//approximate nearest neighbour search (BBF) within single tree
Node* searchTree(Node* root,uint64_t* dataTrain,uint64_t* dataTest,int featDim){
    
    Node* current_node=root;

    while(true){
        if(current_node->left==NULL&&current_node->right==NULL){
            return current_node;
        }
            
        int i1=current_node->ind;
        float distT=0;
        for(int k=0;k<featDim;k++){
            distT+=__builtin_popcountll(dataTrain[i1*featDim+k]^dataTest[k]);
        }
        //traverse tree based on whether current Hamming distance is smaller than median distance
        if(distT<current_node->median){
            if(current_node->left==NULL){
                return current_node;}
            else{
                current_node=current_node->left;}
        }
        else{
            if(current_node->right==NULL){
                return current_node;}
            else{
                current_node=current_node->right;}
        }
    }
    return current_node;
}

//VP Forest kNN-search (using multiple trees) and linear search of candidates
void knnTree(int* idxNN,int* countNN,float* distNN,uint64_t* dataTrain,uint64_t* dataTest,int featDim,int szTrain,int szTest,int leaf_limit,int knn,int num_tree){
    timeval time1,time2;
   
    gettimeofday(&time1, NULL);

    //build several search trees
    vector<Node*> trees;
    for(int i=0;i<num_tree;i++){
        trees.push_back(buildTree(dataTrain,szTrain,featDim,leaf_limit));
    }
    
    gettimeofday(&time2, NULL);

    float timeT=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    int count=countNodes(trees[0]);
    printf("%d trees with %d nodes each built in %3.4f msecs.\n",num_tree,count,timeT*1000.0f);
    
    

    gettimeofday(&time1, NULL);
    
    //for each query (test) element
    for(int qi=0;qi<szTest;qi++){
        
        //concatenate all bucket/leaf elements from forest
        vector<pair<float,int>> found_set;
        for(int i=0;i<trees.size();i++){
            Node* best_node=searchTree(trees[i],dataTrain,dataTest+qi*featDim,featDim);
            found_set.insert(begin(found_set),begin(best_node->data),end(best_node->data));
        }
        //remove duplicate data-elements
        auto less=[](pair<float,int> a,pair<float,int> b){return a.second<b.second;};
        sort(found_set.begin(),found_set.end(),less);
        auto same=[](pair<float,int> a,pair<float,int> b){return a.second==b.second;};
        found_set.erase(unique(found_set.begin(),found_set.end(),same),found_set.end());

        //perform linear search (calculate Hamming distance) and sort kNN set
        for(auto& el:found_set){
            int i1=el.second;
            float distT=0;
            for(int k=0;k<featDim;k++){
                distT+=__builtin_popcountll(dataTrain[i1*featDim+k]^dataTest[qi*featDim+k]);
            }
            el.first=distT;

        }
        int mink=min((int)found_set.size(),knn);
        nth_element(begin(found_set),begin(found_set)+mink,end(found_set));
        sort(begin(found_set),begin(found_set)+mink);
        
        countNN[qi*2]=mink;
        countNN[qi*2+1]=found_set.size();
        for(int k1=0;k1<mink;k1++){
            idxNN[k1+qi*knn]=found_set[k1].second;
            distNN[k1+qi*knn]=found_set[k1].first;
        }
        
        
        
    }
    gettimeofday(&time2, NULL);
    
    float timeP=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    printf("Search (and sort) time: %3.4f msecs\n",timeP*1000);
    
    //clean up trees
    for(int i=0;i<trees.size();i++){
        int count=countNodes(trees[i]);
    }


    
}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    timeval time1,time2,time1a,time2a;
    
    uint64_t* dataTrain=(uint64_t*)mxGetData(prhs[0]);
    uint64_t* dataTest=(uint64_t*)mxGetData(prhs[1]);
    
    int knn=(int)mxGetScalar(prhs[2]);
    
    int leaf_limit=200;
    if(nrhs>3){
        leaf_limit=(int)mxGetScalar(prhs[3]);
    }
    
    int num_tree=10;
    if(nrhs>4){
        num_tree=(int)mxGetScalar(prhs[4]);
    }
    
    const mwSize* dimTrain=mxGetDimensions(prhs[0]);
    int featDim=dimTrain[0];
    int szTrain=dimTrain[1];

    const mwSize* dimTest=mxGetDimensions(prhs[1]);
    int featuresTest=dimTest[0];
    int szTest=dimTest[1];
    
    
    int dims1[]={knn,szTest};
    int dims2[]={2,szTest};
    
    plhs[0]=mxCreateNumericArray(2,dims1,mxINT32_CLASS,mxREAL);
    int* indexKNN=(int*)mxGetData(plhs[0]);
    plhs[1]=mxCreateNumericArray(2,dims1,mxSINGLE_CLASS,mxREAL);
    float* distKNN=(float*)mxGetData(plhs[1]);
    plhs[2]=mxCreateNumericArray(2,dims2,mxINT32_CLASS,mxREAL);
    int* countKNN=(int*)mxGetData(plhs[2]);
    
    knnTree(indexKNN,countKNN,distKNN,dataTrain,dataTest,featDim,szTrain,szTest,leaf_limit,knn,num_tree);

    
}

