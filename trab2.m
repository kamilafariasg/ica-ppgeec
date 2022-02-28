%% 
clc;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
clear;

%% Importing data
ht_sensor = importdata('HT_Sensor_dataset.dat');
T = array2table(ht_sensor.data, 'VariableNames', ht_sensor.colheaders);
ht_sensor_metadata = readtable('HT_Sensor_metadata.csv');

%% Join tables by id
T = join(T, ht_sensor_metadata);

%% Preparing data
% Removing columns
T.id = [];
T.('Temp.') = [];
T.Humidity = [];
T.('time	') = [];
T.date = [];
T.t0 = [];
T.dt = [];

%1. Normalization Z-Score
%for column = 1:8
%    vector_column = T{:, column};
%    T{:, column} = (vector_column - mean(vector_column))/std(vector_column);
%end

% 2. Normalization Minmax
%for column = 1:8
%    vector_column = T{:, column};
%    T{:, column} = (vector_column - min(vector_column))/(max(vector_column) - min(vector_column));
%end

% Sub-sampling
samples = 15000;
T = datasample(T, samples);

% Get samples by class
banana = T(string(T{:, 9})=='banana', :);
wine = T(string(T{:, 9})=='wine', :);
bg = T(string(T{:, 9})=='background', :);

% Discretizing classes
banana_bin = array2table(ones(size(banana, 1), 3).*[1 0 0], 'VariableNames', {'y1', 'y2', 'y3'});
wine_bin = array2table(ones(size(wine, 1), 3).*[0 1 0], 'VariableNames', {'y1', 'y2', 'y3'});
bg_bin = array2table(ones(size(bg, 1), 3).*[0 0 1], 'VariableNames', {'y1', 'y2', 'y3'});
banana = [banana(:, 1:8) banana_bin];
wine = [wine(:, 1:8) wine_bin];
bg = [bg(:, 1:8) bg_bin];

% Data same quantity classes (balanced)
size_classes = [size(banana, 1) size(wine, 1) size(banana_bin, 1)];
n_min = min(size_classes);
T_b = [datasample(banana, n_min); datasample(wine, n_min); datasample(bg, n_min)];

T_b = table2array(T_b);

test_p = 0.15;
cv = cvpartition(size(T_b, 1), 'holdout', test_p);

%% Separate to training and test data

test_data = T_b(cv.test, :);
train_data = T_b(cv.training, :);

train_y = train_data(:, 9:11);
train_x = train_data(:, 1:8);
test_y = test_data(:, 9:11);
test_x = test_data(:, 1:8);


%% RBF

goal = 0; % Error Limit
DF = 25; % Intervals between neurons
MN = 1200; % Max neurons
spread = 50;
new_train = true;

if new_train == true
    rbf_net = newrb(train_x', train_y', goal, spread, MN, DF);
else
    load rbf_net_restore;
    rbf_net = rbf_net_restore;
end

predict_y = rbf_net(test_x');
vec_ind_test = vec2ind(test_y');
vec_ind_pred = vec2ind(predict_y);
hit1 = sum(vec_ind_test == 1 & vec_ind_pred == 1);
hit2 = sum(vec_ind_test == 2 & vec_ind_pred == 2);
hit3 = sum(vec_ind_test == 3 & vec_ind_pred == 3);
hits = hit1 + hit2 + hit3;
acc = hits/size(vec_ind_pred, 2);

C = confusionmat(vec_ind_test, vec_ind_pred);
confusionchart(C);

rbf_net_restore = rbf_net;
save rbf_net_restore;

% Norm, Samples, %Test, Neurons, Spread, Hit1, Hit2, Hit3, Hits, Acc
dlmwrite('rbf.csv', [0, samples, test_p, MN, spread, hit1, hit2, hit3, hits, acc], 'delimiter', ';', '-append');
   
finish


%% MLP
layer1 = 50;
layer2 = 50;
net = feedforwardnet([layer1, layer2]);

% 1 'trainlm'	Levenberg-Marquardt
% 2 'trainbr'	Bayesian Regularization
% 3 'trainbfg' BFGS Quasi-Newton
% 4 'trainrp'	Resilient Backpropagation
% 5 'trainscg' Scaled Conjugate Gradient
% 6 'traincgb' Conjugate Gradient with Powell/Beale Restarts
% 7 'traincgf' Fletcher-Powell Conjugate Gradient
% 8 'traincgp' Polak-Ribiére Conjugate Gradient
% 9 'trainoss' One Step Secant
% 10 'traingdx' Variable Learning Rate Gradient Descent
% 11 'traingdm' Gradient Descent with Momentum
% 12 'traingd' Gradient Descent 

net.trainFcn = 'trainlm';
net.trainParam.epochs = 500;
new_train = true;

if new_train == true
    mlp_net = train(net, train_x', train_y');
else
    load mlp_net_restore_9589;
    mlp_net = mlp_net_restore;
    
end

predict_y = mlp_net(test_x');
vec_ind_test = vec2ind(test_y');
vec_ind_pred = vec2ind(predict_y);
hit1 = sum(vec_ind_test == 1 & vec_ind_pred == 1);
hit2 = sum(vec_ind_test == 2 & vec_ind_pred == 2);
hit3 = sum(vec_ind_test == 3 & vec_ind_pred == 3);
hits = hit1 + hit2 + hit3;
acc = hits/size(vec_ind_pred, 2);

C = confusionmat(vec_ind_test, vec_ind_pred);
confusionchart(C);

mlp_net_restore = mlp_net;
save mlp_net_restore;

% Norm, Samples, %Test, Layer1, Layer2, trainFcn, epochs, Hit1, Hit2,
% Hit3, Hits, Acc
dlmwrite('mlp.csv', [1, samples, test_p, layer1, layer2, 1, 1000, hit1, hit2, hit3, hits, acc], 'delimiter', ';', '-append');

finish


