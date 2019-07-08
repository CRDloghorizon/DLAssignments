% DD2424 deeplearning 2019 assignment2
% two layer NN
% LogHorizon

%clear all; 
clc;

disp("1: run test part;");
disp("2: run training part with cyclical learning rates;");
disp("3: Coarse-to-fine random search to set lambda.");
exercise = input("Input 1, 2 or 3: ");

if exercise == 1
    k = 10;
    % loadbatch
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    [validX, validY, validy] = LoadBatch('data_batch_2.mat');
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    trainX = trainX(:, 1:100);
    trainY = trainY(:, 1:100);
    trainy = trainy(:, 1:100);
    validX = validX(:, 1:100);
    validY = validY(:, 1:100);
    testX = testX(:, 1:100);
    testy = testy(:, 1:100);
    [d, N] = size(trainX);
    % pre-processing
    mean_X = mean(trainX, 2);
	trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
    validX = validX - repmat(mean_X, [1, size(validX, 2)]);
    testX = testX - repmat(mean_X, [1, size(testX, 2)]);
    
    m = 2;
    [W, b] = IniParams(m, d, k);
    
    [P, h] = EvaluateClassifier(trainX, W, b);
    lambda = 0;
    [grad_W, grad_b] = ComputeGradients(trainX, trainY, P, h, W, lambda);
    [ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX, trainY, W, b, lambda, 1e-5);
    % absolutes difference are small (<=1e-9)
    abs_errw = {grad_W{1}-ngrad_W{1}, grad_W{2}-ngrad_W{2}};
    abs_errb = {grad_b{1}-ngrad_b{1}, grad_b{2}-ngrad_b{2}};
    %disp(max(max(abs_errw{i}))); 1.05e-10; 7.52e-11; 
    %disp(max(abs_errb{i})); 4.8e-12; 3.5e-11;
    % eps test
    err_w1 = abs(grad_W{1} - ngrad_W{1})./(max(eps,abs(grad_W{1}) + abs(ngrad_W{1})));
    err_w1 = max(max(err_w1));
    err_w2 = abs(grad_W{2} - ngrad_W{2})./(max(eps,abs(grad_W{2}) + abs(ngrad_W{2})));
    err_w2 = max(max(err_w2));
    err_b1 = abs(grad_b{1} - ngrad_b{1})./(max(eps,abs(grad_b{1}) + abs(ngrad_b{1})));
    err_b1 = max(max(err_b1));
    err_b2 = abs(grad_b{2} - ngrad_b{2})./(max(eps,abs(grad_b{2}) + abs(ngrad_b{2})));
    err_b2 = max(max(err_b2));
    disp("first layer maximum relative errors are " + err_w1 + " and " + err_b1);
    disp("second layer maximum relative errors are " + err_w2 + " and " + err_b2);
%     first layer maximum relative errors are 0.00019129 and 1.4126e-08
%     second layer maximum relative errors are 7.0543e-08 and 4.2615e-09
      
    % test the mini batch GD function
    n_batch = 50;
    eta = 0.02; % 0.02
    n_epochs = 200;
    GDparams = ObjParam(n_batch, eta, n_epochs);
    m = 50;
    [W, b] = IniParams(m, d, k);
    [Wstar, bstar, trainloss, validationloss] = MiniBatchGD(trainX, trainY, validX, validY, GDparams, W, b, lambda);
    acc1 = ComputeAccuracy(testX, testy, Wstar, bstar);
    acc2 = ComputeAccuracy(trainX, trainy, Wstar, bstar);
    % plot and result
    disp("Accuracy on test set is " + acc1);
    disp("Accuracy on train set is " + acc2);
    % Accuracy on test set is 0.17 for lr=0.01; 0.15 for lr=0.02
    % Accuracy on train set is 0.91 for lr=0.01; 0.99 for lr=0.02
    epoch = 1:n_epochs;
    plot(epoch, trainloss, 'g')
    legend('training loss')
    xlabel('epoch')
    ylabel('loss')
    % print("Result_Pics/t1.png", '-dpng'); % t2.png
    % close gcf

end

if exercise == 2
    % test part with cyclical learning rates
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    [validX, validY, validy] = LoadBatch('data_batch_2.mat');
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    [d, N] = size(trainX);
    k = 10;
    % pre-processing and initialize parameters
    mean_X = mean(trainX, 2);
	trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
    validX = validX - repmat(mean_X, [1, size(validX, 2)]);
    testX = testX - repmat(mean_X, [1, size(testX, 2)]);
    m = 50;
    [W, b] = IniParams(m, d, k);
    
    lambda = 0.01;
    n_epochs = 48;
    n_batch = 100;
    eta = 0.01;
    GDparams = ObjParam(n_batch, eta, n_epochs);
    [Wstar, bstar, trainloss, validationloss, acctr, accva] = GDcyclr(trainX, trainY, trainy, validX, validY, validy, GDparams, W, b, lambda);
    acc = ComputeAccuracy(testX, testy, Wstar, bstar);
    % plot and result
    disp("Accuracy on test set is " + acc);
    % loss plot
    step = (1:(n_epochs/2)) .* (2*N / n_batch);
    plot(step, trainloss, 'g', step, validationloss, 'r');
    legend('training loss', 'validation loss')
    xlabel('update step')
    ylabel('loss')
    print("Result_Pics/f4_3.png", '-dpng');
    %pause;
    close gcf
    % acc plot
    plot(step, acctr, 'g', step, accva, 'r');
    legend({'training acc', 'validation acc'}, 'Location','southeast')
    legend('boxoff')
    xlabel('update step')
    ylabel('accuracy')
    print("Result_Pics/f4_4.png", '-dpng');
    %pause;
    close gcf
end

% Accuracy on test set ns=500 is 0.4285
% Accuracy on test set ns=800 is 0.4631

if exercise == 3
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    for i=2:5
        [tmpX, tmpY, tmpy] =  LoadBatch("data_batch_"+i+".mat");
        trainX = cat(2, trainX, tmpX);
        trainY = cat(2, trainY, tmpY);
        trainy = cat(2, trainy, tmpy);
    end
    validX = trainX(:, 1:1000);
    validY = trainY(:, 1:1000);
    validy = trainy(:, 1:1000);
    trainX = trainX(:, 1001:size(trainX, 2));
    trainY = trainY(:, 1001:size(trainY, 2));
    trainy = trainy(:, 1001:size(trainy, 2));
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    [d, N] = size(trainX);
    k = 10;
    % pre-processing and initialize parameters
    mean_X = mean(trainX, 2);
	trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
    validX = validX - repmat(mean_X, [1, size(validX, 2)]);
    testX = testX - repmat(mean_X, [1, size(testX, 2)]);
    m = 50;
    [W, b] = IniParams(m, d, k);
    % ns=980, 3cycle=5880, nepoch=12
    n_epochs = 16;
    n_batch = 100;
    eta = 0.01;
    GDparams = ObjParam(n_batch, eta, n_epochs);
    % go through l values
%     l_min = -4.01;
%     l_max = -3.79;
%     l = l_min + (l_max - l_min)*rand(1, 10);
%     lambda = 10.^l;
%     acc = zeros(1, 10);
    lambda = 0.00012643;

    [Wstar, bstar, trainloss, validationloss, acctr, accva] = GDcyclr(trainX, trainY, trainy, validX, validY, validy, GDparams, W, b, lambda);
    acc = ComputeAccuracy(testX, testy, Wstar, bstar);
    % plot and result
    disp("Accuracy on test set is " + acc);
    % loss plot
    step = 1:n_epochs;
    plot(step, trainloss, 'g', step, validationloss, 'r');
    legend('training loss', 'validation loss')
    xlabel('epochs')
    ylabel('loss')
    print("Result_Pics/f5_1.png", '-dpng');
    close gcf
    % acc plot
    plot(step, acctr, 'g', step, accva, 'r');
    legend({'training acc', 'validation acc'}, 'Location','southeast')
    xlabel('epochs')
    ylabel('accuracy')
    print("Result_Pics/f5_2.png", '-dpng');
    close gcf  
end
% Accuracy on test set is 0.514
% Accuracy on test set is 0.5123

function [X, Y, y] = LoadBatch(filename)
	addpath Datasets/cifar-10-batches-mat/;
	A = load(filename);
	X = double(A.data')./255.0;
    N = size(X, 2);
    Y = zeros(10, N);
	y = A.labels'+1;
    % y(i) = (1~10), i in 10000
    for i = 1:N
        Y(y(i), i) = 1;
    end
end

function [W, b] = IniParams(m, d, k)
    % W1 m * d (50, 3072) b1 m * 1
    % W2 k * m (10, 50)   b2 k * 1
    % rng(400);
    W1 = randn(m, d, 'double') * (1/sqrt(d));
    W2 = randn(k, m, 'double') * (1/sqrt(m));
    b1 = zeros(m, 1);
    b2 = zeros(k, 1);
    % array cell W and b
    W = {W1, W2};
    b = {b1, b2};
end

function GDparams = ObjParam(n_batch, eta, n_epochs)
    % object GDparams
    GDparams.n_batch = n_batch;
    GDparams.eta = eta;
    GDparams.n_epochs = n_epochs;
end

function [P, h] = EvaluateClassifier(X, W, b)
    % layer 1: s1 = w1x+b1; h = max(0, s1)
    % layer 2: s2 = w2h+b2; p = softmax(s2)
    % h: m * n 50*10000; P: k * n 10*10000
    s1 = W{1} * X;
    h = s1 + b{1};
    h = max(0, h);
    s2 = W{2} * h + b{2};    
    P = exp(s2)./(double(ones(size(W{2}, 1)))*exp(s2));
end

function J = ComputeCost(X, Y, W, b, lambda)
    % loss = -log(yT P)  
    % j = sum(diag -log(Y'*P))/n + lambda*(sum(W1^2) + sum(W2^2))
    [P, ~] = EvaluateClassifier(X, W, b);
    N = size(X, 2);
    % out of memory for 49000^2!!!!!
    % lcross = sum(diag(-log(Y' * P)))/N; 
    result = 0;
    for i = 1:N
        result = result - log(Y(:,i)'*P(:,i));
    end
    J = result / N + lambda * (sum(sum(W{1}.^2)) + sum(sum(W{2}.^2)));
end

function acc = ComputeAccuracy(X, y, W, b)
    % accracy of labels
    N = size(X, 2);
    [P, ~] = EvaluateClassifier(X, W, b);
    % Ignore the output using a tilde (~)
    [~, predict] = max(P);
    correct = find(y == predict);
    acc = length(correct)/N;
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, W, lambda)
    % efficient matlab computation
    N = size(X, 2);
%     grad_W1 = zeros(size(W{1}));
%     grad_W2 = zeros(size(W{2}));
%     grad_b1 = zeros(size(b{1}));
%     grad_b2 = zeros(size(b{2}));
    % first w2 and b2 h: m * n 50*10000; P: k * n 10*10000; 
    % W2 10*50; b2 10*1; W1(50 * 3072) b1 (50 * 1)
    G = P - Y;
    grad_b2 = (G * ones(N, 1))/N;
    grad_W2 = (G * h')/N + 2*lambda*W{2};
    g = (W{2})' * G;
    h(h > 0) = 1;
    g = g .* h;
    grad_W1 = (g * X')/N;
    grad_b1 = (g * ones(N, 1))/N;
    grad_W = {grad_W1, grad_W2};
    grad_b = {grad_b1, grad_b2};
end


function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)
    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);
    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));
        for i=1:length(b{j})
            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b_try, lambda);
            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);
            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end
    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));
        for i=1:numel(W{j})
            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W_try, b, lambda);
            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W_try, b, lambda);
            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end

% learning rate
function lr = TriLr(eta_min, eta_max, ns, t)
    kp = floor(t/ns);
    kpm = mod(kp, 2);
    if kpm == 0
        lr = eta_min + (t - kp * ns) * (eta_max - eta_min) / ns;
    else
        lr = eta_max - (t - kp * ns) * (eta_max - eta_min) / ns;
    end
end

% vanilla mini-batch gradient descent with cyclical learning rates
function [Wstar, bstar, trainloss, validationloss, acctr, accva] = GDcyclr(X, Y, y, VX, VY, vy, GDparams, W, b, lambda)
    eta_min = 1e-5;
    eta_max = 1e-1;
    n_batch = GDparams.n_batch;
    % eta = GDparams.eta;
    n_epochs = GDparams.n_epochs;
    Wstar = W;
    bstar = b;
    N = size(X, 2);
    % ns = 800; %500 900 980
    ns = 2*floor(N/n_batch);
    trainloss = zeros(n_epochs, 1);
    validationloss = zeros(n_epochs, 1);
    acctr = zeros(n_epochs, 1);
    accva = zeros(n_epochs, 1);
    % do n_epochs step, update w, b, log train/validation loss for mini batch
    for i=1:n_epochs
        for j=1:(N/n_batch)
            j_start = (j - 1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            [P, h]= EvaluateClassifier(Xbatch, Wstar, bstar);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, h, Wstar, lambda);
            % after grad compute, do update
            t = j + (i-1) * (N/n_batch);
            eta = TriLr(eta_min, eta_max, ns, t);
            Wstar{1} = Wstar{1} - eta*grad_W{1};
            Wstar{2} = Wstar{2} - eta*grad_W{2};
            bstar{1} = bstar{1} - eta*grad_b{1};
            bstar{2} = bstar{2} - eta*grad_b{2};
%             if (mod(t, 200)==0)
%                 pl = t/200;
%                 trainloss(pl) = ComputeCost(X, Y, Wstar, bstar, lambda);
%                 validationloss(pl) = ComputeCost(VX, VY, Wstar, bstar, lambda);
%                 acctr(pl) = ComputeAccuracy(X, y, Wstar, bstar);
%                 accva(pl) = ComputeAccuracy(VX, vy, Wstar, bstar);
%             end
        end
        trainloss(i) = ComputeCost(X, Y, Wstar, bstar, lambda);
        validationloss(i) = ComputeCost(VX, VY, Wstar, bstar, lambda);
        acctr(i) = ComputeAccuracy(X, y, Wstar, bstar);
        accva(i) = ComputeAccuracy(VX, vy, Wstar, bstar);
    end
end

function [Wstar, bstar, trainloss, validationloss] = MiniBatchGD(X, Y, VX, VY, GDparams, W, b, lambda)
    n_batch = GDparams.n_batch;
    eta = GDparams.eta;
    n_epochs = GDparams.n_epochs;
    Wstar = W;
    bstar = b;
    N = size(X, 2);
    trainloss = zeros(n_epochs, 1);
    validationloss = zeros(n_epochs, 1);
    % do n_epochs step, update w, b, log train/validation loss for mini batch
    for i=1:n_epochs
        for j=1:(N/n_batch)
            j_start = (j - 1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            [P, h]= EvaluateClassifier(Xbatch, Wstar, bstar);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, h, Wstar, lambda);
            % after grad compute, do update
            Wstar{1} = Wstar{1} - eta*grad_W{1};
            Wstar{2} = Wstar{2} - eta*grad_W{2};
            bstar{1} = bstar{1} - eta*grad_b{1};
            bstar{2} = bstar{2} - eta*grad_b{2};
        end
        trainloss(i) = ComputeCost(X, Y, Wstar, bstar, lambda);
        validationloss(i) = ComputeCost(VX, VY, Wstar, bstar, lambda);           
    end   
end