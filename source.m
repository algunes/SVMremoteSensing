

%This part of code load the data and calls the functions that require for classifying,

clear all
clc
  
load('C:\aliyar_EE490.mat');
 
    % trainSeperate function seperates zeroes and ones from label matrix 
    %(in this case 1-damaged, 0-undamaged) then
    % takes random samples from zeroes and ones matrices equal number and randomly and
    % returns train, test and their label's index numbers
[trainIdx, testIdx, trainLabel, testLabel] = trainSeperate(Comp_dist_r1, Damage_vect, 10);
    % trainTest function only generates train and test matrices respect to
    % trainIdx and testIdx
[cR1Train, cR1Test] = trainTest(Comp_dist_r1, trainIdx, testIdx);
[cR2Train, cR2Test] = trainTest(Comp_dist_r2, trainIdx, testIdx);
[cR3Train, cR3Test] = trainTest(Comp_dist_r3, trainIdx, testIdx);
[bR1Train, bR1Test] = trainTest(Bhat_dist_r1, trainIdx, testIdx);
[bR2Train, bR2Test] = trainTest(Bhat_dist_r2, trainIdx, testIdx);
[bR3Train, bR3Test] = trainTest(Bhat_dist_r3, trainIdx, testIdx);
 
 
% ALL TRAIN DATA
massDataTrain = [cR1Train cR2Train cR3Train bR1Train bR2Train bR3Train];
 
 
% Only first 3 columns Train
massDataTrainC = [cR1Train cR2Train cR3Train];
 
 
% First 3 column + B1 Train
massDataTrainCb1 = [cR1Train cR2Train cR3Train bR1Train];
 
 
% First 3 columns + B2 Train
massDataTrainCb2 = [cR1Train cR2Train cR3Train bR2Train];
 
 
% First 3 columns + B3 Train
massDataTrainCb3 = [cR1Train cR2Train cR3Train bR3Train];
 
 
% first 3 column only trained
massDataTrainCx3 = [cR1Train cR2Train cR3Train];
 
 
 
% ALL TEST DATA
massDataTest = [cR1Test cR2Test cR3Test bR1Test bR2Test bR3Test];
 
 
% Only first 3 columns Test
massDataTestC = [cR1Test cR2Test cR3Test];
 
 
% First 3 column + B1 Test
massDataTestCb1 = [cR1Test cR2Test cR3Test bR1Test];
 
 
% First 3 column + B2 Test
massDataTestCb2 = [cR1Test cR2Test cR3Test bR2Test];
 % First 3 column + B3 Test
massDataTestCb3 = [cR1Test cR2Test cR3Test bR3Test];
 
 
% first 3 column only test
massDataTestCx3 = [cR1Test cR2Test cR3Test];
  
% SVM Part
 
% Case1 - First 3 column + Bhat_dist_r1
[detect_damage1, genEror1, precision_damaged1, precision_undamaged1, recall_damaged1,...
    recall_undamaged1, TPrate1, FPrate1] = svmVary(massDataTrainCb1,...
    trainLabel,massDataTestCb1, testLabel);
 
% Case2 - First 3 column + Bhat_dist_r2
[detect_damage2, genEror2, precision_damaged2, precision_undamaged2, recall_damaged2,...
    recall_undamaged2, TPrate2, FPrate2] = svmVary(massDataTrainCb2,...
    trainLabel,massDataTestCb2, testLabel);
 
% Case3 - First 3 column + Bhat_dist_r3
[detect_damage3, genEror3, precision_damaged3, precision_undamaged3, recall_damaged3,...
    recall_undamaged3, TPrate3, FPrate3] = svmVary(massDataTrainCb3,...
    trainLabel,massDataTestCb3, testLabel);
 
% Case4 - All Bhat_dist + Comp_dist
[detect_damage4, genEror4, precision_damaged4, precision_undamaged4, recall_damaged4,...
    recall_undamaged4, TPrate4, FPrate4] = svmVary(massDataTrain,...
    trainLabel,massDataTest, testLabel);
 


%case 5 All C vaues
[detect_damage5, genEror5, precision_damaged5, precision_undamaged5, recall_damaged5,...
    recall_undamaged5, TPrate5, FPrate5] = svmVary(massDataTrainCx3,...
    trainLabel,massDataTestCx3, testLabel);
 
 
% Result Table
cases = {'Ct + B1';'Ct + B2';'Ct + B3';'Ct + Bt';'C'};
general_error = [genEror1;genEror2;genEror3;genEror4;genEror5]; 
precision_damaged = [precision_damaged1;precision_damaged2;precision_damaged3;precision_damaged4;precision_damaged5];
recall_damaged = [recall_damaged1;recall_damaged2;recall_damaged3;recall_damaged4;recall_damaged5];
precision_undamaged = [precision_undamaged1;precision_undamaged2;precision_undamaged3;precision_undamaged4;precision_undamaged5];
recall_undamaged = [recall_undamaged1;recall_undamaged2;recall_undamaged3;recall_undamaged4;recall_undamaged5];
tPrate = [TPrate1;TPrate2;TPrate3;TPrate4;TPrate5];
fPrate = [FPrate1;FPrate2;FPrate3;FPrate4;FPrate5];
 
T = table(general_error,precision_damaged,recall_damaged,precision_undamaged,recall_undamaged,tPrate,fPrate,...
    'RowNames',cases)




% *******************

% This function for train the support vector machine properly and generating damaged and undamaged buildings' stats,

% x-Train Data, y-Train Labels, z-Test Data, q-Test Data Labels
 
function [detect_damage, genError, precision_damaged, precision_undamaged, recall_damaged, recall_undamaged,...
    TPrate, FPrate] = svmVary(x, y, z, q) 
 
% load('C:\Users\aliyar\Desktop\learner.mat');
 
svmModel = fitcsvm(x,y,'KernelFunction','linear','KFold',8,...
     'CrossVal','on'); 
 
 cSsvmModel = svmModel.Trained{1}; % Extract trained, compact classifier
 
[detect_damage, score] = predict(cSsvmModel, z);

 
genError = kfoldLoss(svmModel); % On average, the generalization error is
 
 
sumNumberOfTruePositives  = sum(  detect_damage & q==1);
sumNumberOfFalsePositives = sum(  detect_damage & q==0);
 
sumNumberOfTrueNegatives  = sum((~detect_damage) & q==0);
sumNumberOfFalseNegatives = sum((~detect_damage) & q==1);
 
precision_damaged = sumNumberOfTruePositives / (sumNumberOfTruePositives + sumNumberOfFalsePositives);
numberOfDamagedBuildings = sumNumberOfTruePositives + sumNumberOfFalseNegatives;
recall_damaged = sumNumberOfTruePositives / numberOfDamagedBuildings;
precision_undamaged = sumNumberOfTrueNegatives / (sumNumberOfTrueNegatives + sumNumberOfFalseNegatives);
numberOfUndamagedBuildings = sumNumberOfTrueNegatives + sumNumberOfFalsePositives;
recall_undamaged = sumNumberOfTrueNegatives / numberOfUndamagedBuildings;
 
TPrate = sumNumberOfTruePositives/(sumNumberOfTruePositives+sumNumberOfFalseNegatives);
FPrate = sumNumberOfFalsePositives/(sumNumberOfFalsePositives+sumNumberOfTrueNegatives);
 
end

% **************

% This function seperate train and test data from each other, and set the random sampling rate, and generate random sampling index numbers,
% x-data, y-data labels, z-train percent
 function [trainIdx, testIdx, trainLabel, testLabel] = trainSeperate(x, y, z) 
 
[rows, ~] = size(x);
rowsTemp = [1:rows]';
percent = int16(rows*(z/100));
 
trainLabel11 = [];
trainLabel00 = [];
map0 = [];
map1 = [];
for i = 1 : rows
    if y(i,:)==0
        trainLabel00 = vertcat(trainLabel00, y(i,:));
        map0 = [map0 ; i];
    else
        trainLabel11=vertcat(trainLabel11, y(i,:));
        map1 = [map1 ; i];
    end
end
  
temp1 = randperm(numel(map1),int16(percent/2)).';
temp0 = randperm(numel(map0),int16(percent/2)).';
  
trainIdx1 = [map1(temp1,:)];
trainIdx0 = [map0(temp0,:)];
trainIdx = [trainIdx1; trainIdx0];
trainLabel = y(trainIdx,:);
 
testIdxFlags = ~ismember(rowsTemp,trainIdx,'rows');
testIdx = find(testIdxFlags);
testLabel = y(testIdx,:);
end



& ***********
% x-data, y-trainIdx, z-testIdx
 
% This function create test and train data matrices,

function [train, test] = trainTest(x, y, z) 
 
train = [x(y,:)];
test = [x(z,:)];
 
end
