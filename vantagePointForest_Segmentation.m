%% Multi-Organ Segmentation using Vantage Point Forests and Binary Context Features
% Medical Image Computing and Computer Assisted Intervention (MICCAI) 2016
% by M.P. Heinrich and M. Blendowski
% (please cite paper if used, and see licence.txt)

%% before usage the following mex-Files have to be compiled
% change flags if your CPU does not support AVX2/SSE4
% important: compilation requires -std=c++11 to be possible and set
mex CXXOPTIMFLAGS='$CXXOPTIMFLAGS -O3 -std=c++11' extractBRIEF.cpp
mex CXXOPTIMFLAGS='$CXXOPTIMFLAGS -O3 -std=c++11 -msse4.2 -mavx2' binaryVP_NN.cpp
% the post-processing step uses the Eigen-library 
unix('tar -zxf eigen.tar.gz');
mex CXXOPTIMFLAGS='$CXXOPTIMFLAGS -O3 -std=c++11 -msse4.2 -mavx2' postProcessRegularise.cpp

%% requires the following input (all scans should have similar voxel-spacings):
% scansTrain    (cell array of 3D volumes of grayvalue training scans)
% segmentsTrain (cell array of 3D volumes of same size with GT segmentations of scans)
% masksTrain    (cell array of 3D volumes of masks - region of interest)

% scanTest      (a 3D volume of test scan)
% maskTest      (region of interest for the above)

% output is a 3D volume segmentTest - the segmentation of the given scan

%% parameters
hs=fspecial('gaussian',[9,1],3.00); % Gaussian kernel for patch/image smoothing
num_labels=7;   % largest label-index of foreground segmentation
knn=21;         % number of k-nearest Neighbours
num_feat=640;   % number of BRIEF feature-bits
radii=[20,40];  % sampling radii for BRIEF context
strideTrain=6;  % stride for training sampling locations
strideTest=4;   % stride for test scan
leaf_limit=15;  % bucket/leaf-node size
num_trees=15;   % number of VP trees in forest
reg_lambda=20;  % regularisation weighting in random walk post-processing
reg_sigma=10;   % sigma_w for intensity gradients (adapted for CT scans)

%% define BRIEF sampling layout and preprocess (smooth) scans
% see Section 2.1 Contextual Binary Similarity in paper
for i=1:length(scansTrain);
    scansTrain{i}=imfilter(imfilter(imfilter(scansTrain{i},hs),hs'),reshape(hs,1,1,[]));
end
scanTestFilt=imfilter(imfilter(imfilter(scanTest,hs),hs'),reshape(hs,1,1,[]));
L=num_feat/64;
% BRIEF sampling layout
samplingCoords=round(min(max(randn(6,64,L)*radii(1),-radii(1)*2.5),radii(1)*2.5));
samplingCoords=cat(3,samplingCoords,min(max(randn(6,64,L)*radii(2),-radii(2)*2.5),radii(2)*2.5));
% include LBP like features (with 0,0,0 centre)
samplingCoords(4:6,rand(64*L*2,1)>0.667)=0;

%% extract BRIEF features (and GT labels at sampling locations)
% for each training scan
featTrain=[]; labelTrain=[];

for i=1:length(scansTrain)
    % sampling locations in images (with strideTrain)
    [m,n,o]=size(scansTrain{i});
    [x_train,y_train,z_train]=meshgrid(2:strideTrain:n,2:strideTrain:m,2:strideTrain:o);
    ind_train=sub2ind([m,n,o],y_train(:)',x_train(:)',z_train(:)');
    % within region of interest mask
    ind_mask=ind_train(masksTrain{i}(ind_train(:))>0);
    featTrain=cat(2,featTrain,extractBRIEF(single(scansTrain{i}),int32(samplingCoords),int32(ind_mask-1)));
    labelTrain=cat(2,labelTrain,segmentsTrain{i}(ind_mask));
end
% and for the test scan
[m,n,o]=size(scanTest);
[x_test,y_test,z_test]=meshgrid(2:strideTest:n,2:strideTest:m,2:strideTest:o);
ind_test=sub2ind([m,n,o],y_test(:)',x_test(:)',z_test(:)');
ind_mask=ind_test(maskTest(ind_test(:))>0);
featTest=extractBRIEF(single(scanTestFilt),int32(samplingCoords),int32(ind_mask-1));

%% find approximate k-nearest neighbours
% see Section 2.2 Vantage Point Forests in paper
idxKNN=binaryVP_NN(uint64(featTrain),uint64(featTest),knn,leaf_limit,num_trees);
%probVP=labelTrain(idxKNN+1); labelMode=mode(probVP,1);
% sparse probabilistic segmentation output (for sampling locations)
sparseLabelProb=hist(labelTrain(idxKNN+1),0:num_labels);
sparseLabelProb=sparseLabelProb./repmat(sum(sparseLabelProb,1),num_labels+1,1);

%% post processing with random-walk regularisation
% see Section 2.3 Spatial Regularisation using Multi-Label Random Walk
% upsample sparse probabilities to dense grid
denseLabelProb=zeros([size(scanTest),num_labels+1]);
[x,y,z]=meshgrid(1:size(scanTest,2),1:size(scanTest,1),1:size(scanTest,3));
for i=1:num_labels+1;
    % resize to coarse grid
    prob1=zeros(size(x_test));
    prob1(maskTest(ind_test(:))>0)=sparseLabelProb(i,:);
    prob1=reshape(prob1,size(x_test));
    % linear upsampling
    denseLabelProb(:,:,:,i)=interp3(x_test,y_test,z_test,prob1,x,y,z,'*linear');
end
maskedLabelProb=single(reshape(denseLabelProb,[],num_labels+1)'); maskedLabelProb=maskedLabelProb(:,maskTest(:)>0);
maskedLabelProb(isnan(maskedLabelProb(:)))=0;
% call the mex-function for regularisation and largest-connected component
[labelReg,labelLCC,maskedRegularProb]=postProcessRegularise(maskedLabelProb,single(scanTest),uint8(maskTest(:)>0),reg_lambda,reg_sigma);
segmentTest=labelLCC; % final output
