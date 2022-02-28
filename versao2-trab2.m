%% INTELIGENCIA COMPUTACIONAL APLICADA - TRABALHO PRÁTICO
clc;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
clear;

%% Importing data
addpath('dataset/');
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

% Normalization data
% 0: No, 1: Z-score, 2: Minmax (integer)
norm = 1;
switch norm
    case 1
        for column = 1:8
            vector_column = T{:, column};
            T{:, column} = (vector_column - mean(vector_column))/std(vector_column);
        end
    case 2
        for column = 1:8
            vector_column = T{:, column};
            T{:, column} = (vector_column - min(vector_column))/(max(vector_column) - min(vector_column));
        end
end

% Get samples by class
banana = T(string(T{:, 9})=='banana', :);
wine = T(string(T{:, 9})=='wine', :);
bg = T(string(T{:, 9})=='background', :);

% Discretizing classes
banana = [banana(:, 1:8) array2table(ones(size(banana, 1), 3).*[1 0 0], 'VariableNames', {'y1', 'y2', 'y3'})];
wine = [wine(:, 1:8) array2table(ones(size(wine, 1), 3).*[0 1 0], 'VariableNames', {'y1', 'y2', 'y3'});];
bg = [bg(:, 1:8) array2table(ones(size(bg, 1), 3).*[0 0 1], 'VariableNames', {'y1', 'y2', 'y3'});];
T = [banana; wine; bg];
T = table2array(T);


%% Get samples, randomize and separe train-test by class
a1 = 1;
a2 = 50000;
b1 = 50001;
b2 = 100000;

banana = T(T(:, 9)==1, :);
banana = datasample(banana, size(banana, 1));
train_banana = banana(a1:a2, :);
test_banana = banana(b1:b2, :);

wine = T(T(:, 10)==1, :);
wine = datasample(wine, size(wine, 1));
train_wine = wine(a1:a2, :);
test_wine = wine(b1:b2, :);

bg = T(T(:, 11)==1, :);
bg = datasample(bg, size(bg, 1));
train_bg = bg(a1:a2, :);
test_bg = bg(b1:b2, :);

samples_class_train = 5000;
samples_class_test = 500;

%% Main loop 
for i = 1:10
    %% Training net
 
    train_lim_inf_class = (samples_class_train * (i-1)) + 1;
    train_lim_sup_class = samples_class_train * i;
    
    train_banana_i = train_banana(train_lim_inf_class:train_lim_sup_class, :);
    train_wine_i = train_wine(train_lim_inf_class:train_lim_sup_class, :);
    train_bg_i = train_bg(train_lim_inf_class:train_lim_sup_class, :);
    
    train_data = [train_banana_i; train_wine_i; train_bg_i;];
    train_data = datasample(train_data, size(train_data, 1));
    
    [trained_net, path_model, t_end] = mlp_net(train_data, i);
    % [trained_net, path_model] = rbf_net(train_data);
    
    [acc, C, hit1, hit2, hit3, hits] = get_results(train_data, trained_net);
    dlmwrite('./logs/results.csv', [i, 0, train_lim_inf_class, train_lim_sup_class, t_end, hit1, hit2, hit3, hits, acc], 'delimiter', ';', '-append');
  
    %% Test trained net
    for j = 1:10
        test_lim_inf_class = (samples_class_test * (j-1)) + 1;
        test_lim_inf_class = ((samples_class_test*10) * (i-1)) + test_lim_inf_class;
        test_lim_sup_class = samples_class_test * j;
        test_lim_sup_class = ((samples_class_test*10) * (i-1)) + test_lim_sup_class;
        
        test_banana_j = test_banana(test_lim_inf_class:test_lim_sup_class, :);
        test_wine_j = test_wine(test_lim_inf_class:test_lim_sup_class, :);
        test_bg_j = test_bg(test_lim_inf_class:test_lim_sup_class, :);
        
        test_data = [test_banana_j; test_wine_j; test_bg_j];
        test_data = datasample(test_data, size(test_data, 1));
       
        load(path_model, 'trained_net');
        
        [acc, C, hit1, hit2, hit3, hits] = get_results(test_data, trained_net);
        dlmwrite('./logs/results.csv', [i, j, test_lim_inf_class, test_lim_sup_class, 0, hit1, hit2, hit3, hits, acc], 'delimiter', ';', '-append');
    end   
    dlmwrite('./logs/results.csv', ' ', '-append');
end

dlmwrite('./logs/results.csv', ' ', '-append');
dlmwrite('./logs/results.csv', ' ', '-append');
dlmwrite('./logs/results.csv', ' ', '-append');

%% Aux function to calcule and plot results
function [acc, C, hit1, hit2, hit3, hits] = get_results(test_data, trained_net)
    test_y = test_data(:, 9:11)';
    test_x = test_data(:, 1:8)';
    predict_y = trained_net(test_x);
    vec_ind_test = vec2ind(test_y);
    vec_ind_pred = vec2ind(predict_y);
    vec_size = size(vec_ind_pred, 2);
    hit1 = sum(vec_ind_test == 1 & vec_ind_pred == 1);
    hit2 = sum(vec_ind_test == 2 & vec_ind_pred == 2);
    hit3 = sum(vec_ind_test == 3 & vec_ind_pred == 3);
    hits = hit1 + hit2 + hit3;
    acc = hits/vec_size * 100;
    C = confusionmat(vec_ind_test, vec_ind_pred);
end

%% MLP Net
function [trained_net, path_model, t_end] = mlp_net(train_data, i)
    path_model = sprintf('./models/mlp_%d_restore.mat', i);
    layer1 = 50;
    layer2 = 50;
    f_training = 'trainlm';
    epochs = 100;
    feedforward = feedforwardnet([layer1, layer2]);
    feedforward.trainFcn = f_training;
    feedforward.trainParam.epochs = epochs;
    train_y = train_data(:, 9:11);
    train_x = train_data(:, 1:8);
    t_start = tic;
    trained_net = train(feedforward, train_x', train_y');
    t_end = toc(t_start);
    save(path_model, 'trained_net');  
end

%% RBF Net
function [trained_net, path_model, t_end] = rbf_net(train_data, i)
    path_model = sprintf('./models/rbf_%d_restore.mat', i);
    goal = 0;
    DF = 25; 
    MN = 100;
    spread = 50;
    train_y = train_data(:, 9:11)';
    train_x = train_data(:, 1:8)';
    t_start = tic;
    trained_net = newrb(train_x, train_y, goal, spread, MN, DF);
    t_end = toc(t_start);
    save(path_model, 'trained_net');
end