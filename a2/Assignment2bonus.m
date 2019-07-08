% DD2424 deeplearning 2019 assignment2 bonus part
% two layer NN optimization
% LogHorizon

%clear all; 
clc;

disp("1: Optimize the performance of the network;");
disp("2: Set eta_min and eta_max tothe guidelines in the paper.");
exercise = input("Input 1 or 2: ");

if exercise == 1
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    for i=2:5
        [tmpX, tmpY, tmpy] =  LoadBatch("data_batch_"+i+".mat");
        trainX = cat(2, trainX, tmpX);
        trainY = cat(2, trainY, tmpY);
        trainy = cat(2, trainy, tmpy);
    end
    validX = trainX(:, 1:5000);
    validY = trainY(:, 1:5000);
    validy = trainy(:, 1:5000);
    trainX = trainX(:, 5001:size(trainX, 2));
    trainY = trainY(:, 5001:size(trainY, 2));
    trainy = trainy(:, 5001:size(trainy, 2));
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    [d, N] = size(trainX);
    k = 10;
    % pre-processing and initialize parameters
    mean_X = mean(trainX, 2);
	trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
    validX = validX - repmat(mean_X, [1, size(validX, 2)]);
    testX = testX - repmat(mean_X, [1, size(testX, 2)]);
    % --------------------------
    % opt2: add more hidden layer 50 -> 80 -> 100
    m = 100;
    % --------------------------
    [W, b] = IniParams(m, d, k);
    n_batch = 100;
    eta = 0.01;
%     acc = zeros(1, 7);
    lambda = 0.00012643;
    % -----------------------------
    % opt1: length of cycle ns -> 2 to 8 * 450, run 3 cycles
    % -----------------------------
%     for ns = 2:8
%         n_epochs = 6 * ns;
%         GDparams = ObjParam(n_batch, eta, n_epochs);
%         [Wstar, bstar, trainloss, validationloss, acctr, accva] = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, GDparams, W, b, lambda, ns);
%         acc(ns-1) = accva;
%         disp(ns + " "+ accva);
%     end 
    n_epochs = 30;
    GDparams = ObjParam(n_batch, eta, n_epochs);
    [Wstar, bstar, trainloss, validationloss, acctr, accva] = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, GDparams, W, b, lambda);
    acc = ComputeAccuracy(testX, testy, Wstar, bstar);
    % plot and result
    disp("Accuracy on test set is " + acc);
    % loss and acc plot
    step = 1:n_epochs;
    plot(step, trainloss, 'g', step, validationloss, 'r');
    legend('training loss', 'validation loss')
    xlabel('epochs')
    ylabel('loss')
    print("Result_Pics/bf1.png", '-dpng');
    close gcf
    plot(step, acctr, 'g', step, accva, 'r');
    legend({'training acc', 'validation acc'}, 'Location','southeast')
    xlabel('epochs')
    ylabel('accuracy')
    print("Result_Pics/bf2.png", '-dpng');
    close gcf     
end
% Accuracy on test set is 0.5253

if exercise == 2
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    for i=2:5
        [tmpX, tmpY, tmpy] =  LoadBatch("data_batch_"+i+".mat");
        trainX = cat(2, trainX, tmpX);
        trainY = cat(2, trainY, tmpY);
        trainy = cat(2, trainy, tmpy);
    end
    validX = trainX(:, 1:5000);
    validY = trainY(:, 1:5000);
    validy = trainy(:, 1:5000);
    trainX = trainX(:, 5001:size(trainX, 2));
    trainY = trainY(:, 5001:size(trainY, 2));
    trainy = trainy(:, 5001:size(trainy, 2));
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    [d, N] = size(trainX);
    k = 10;
    % pre-processing and initialize parameters
    mean_X = mean(trainX, 2);
	trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
    validX = validX - repmat(mean_X, [1, size(validX, 2)]);
    testX = testX - repmat(mean_X, [1, size(testX, 2)]);
    m = 100; % 100
    [W, b] = IniParams(m, d, k);
    n_batch = 100;
    eta = 0.01;
    lambda = 0.00012643; % 0.0001
    n_epochs = 30; 
%     GDparams = ObjParam(n_batch, eta, n_epochs);
%     [Wstar, bstar, trainloss, validationloss, lr, acctr, accva] = GDcyclr(trainX, trainY, trainy, validX, validY, validy, GDparams, W, b, lambda);
%      % acc vs lr plot [smith 2015] step size = 8*450 = 3600
%     plot(lr, acctr, 'b');
%     xlabel('Learning rate')
%     ylabel('Accuracy')
%     print("Result_Pics/b2t2.png", '-dpng');
%     close gcf
%     plot(lr, accva, 'b');
%     xlabel('Learning rate')
%     ylabel('Accuracy')
%     print("Result_Pics/b2v2.png", '-dpng');
%     close gcf
    GDparams = ObjParam(n_batch, eta, n_epochs);
    [Wstar, bstar, trainloss, validationloss, acctr, accva] = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, GDparams, W, b, lambda);
    acc = ComputeAccuracy(testX, testy, Wstar, bstar);
    disp("Accuracy on test set is " + acc);
    step = 1:n_epochs;
    plot(step, trainloss, 'g', step, validationloss, 'r');
    legend('training loss', 'validation loss')
    xlabel('epochs')
    ylabel('loss')
    print("Result_Pics/b2f3.png", '-dpng');
    close gcf
    plot(step, acctr, 'g', step, accva, 'r');
    legend({'training acc', 'validation acc'}, 'Location','southeast')
    xlabel('epochs')
    ylabel('accuracy')
    print("Result_Pics/b2f4.png", '-dpng');
    close gcf 
end
% Accuracy on test set is 0.5241



% opt3 add jitter, jitter should be small (like 1e-4)
function JX = AddJitter(X, jitter)
    [d, n] = size(X);
    JX = jitter .* randn(d, n);
    JX = X + JX;
end

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
function [Wstar, bstar, trainloss, validationloss, lr, acctr, accva] = GDcyclr(X, Y, y, VX, VY, vy, GDparams, W, b, lambda)
    eta_min = 1e-5;
    eta_max = 0.5;
    n_batch = GDparams.n_batch;
    % eta = GDparams.eta;
    n_epochs = GDparams.n_epochs;
    Wstar = W;
    bstar = b;
    N = size(X, 2);
    % ns = 800; %500 900 980
    ns = 5*floor(N/n_batch);
%     trainloss = zeros(n_epochs, 1);
%     validationloss = zeros(n_epochs, 1);
    pt = 50;
    acctr = zeros(ns/pt, 1);
    accva = zeros(ns/pt, 1);
    lr = zeros(ns/pt, 1);
    % do n_epochs step, update w, b, log train/validation loss for mini batch
    for i=1:n_epochs
        for j=1:(N/n_batch)
            j_start = (j - 1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            % -------------------------------
            % add jitter
            Xbatch = AddJitter(Xbatch, 1e-4);
            % -------------------------------
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
            if mod(t, pt)==0
                accva(t/pt) = ComputeAccuracy(VX, vy, Wstar, bstar);
                acctr(t/pt) = ComputeAccuracy(X, y, Wstar, bstar);
                lr(t/pt) = eta;
            end
        end
%         trainloss(i) = ComputeCost(X, Y, Wstar, bstar, lambda);
%         validationloss(i) = ComputeCost(VX, VY, Wstar, bstar, lambda);
    end
    trainloss = 1;
    validationloss = 1;
end


function [Wstar, bstar, trainloss, validationloss, acctr, accva] = MiniBatchGD(X, Y, y, VX, VY, vy, GDparams, W, b, lambda)
    eta_min = 0.0001;
    eta_max = 0.15;
    n_batch = GDparams.n_batch;
    % eta = GDparams.eta;
    n_epochs = GDparams.n_epochs;
    Wstar = W;
    bstar = b;
    N = size(X, 2);
    % ns = 800; %500 900 980
    ns = 5*floor(N/n_batch);
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
            % -------------------------------
            % add jitter
            Xbatch = AddJitter(Xbatch, 1e-4);
            % -------------------------------
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
        end
        trainloss(i) = ComputeCost(X, Y, Wstar, bstar, lambda);
        validationloss(i) = ComputeCost(VX, VY, Wstar, bstar, lambda);
        acctr(i) = ComputeAccuracy(X, y, Wstar, bstar);
        accva(i) = ComputeAccuracy(VX, vy, Wstar, bstar);
    end
%         trainloss = 1;
%         validationloss = 1;
%         acctr = 1;
%         accva = ComputeAccuracy(VX, vy, Wstar, bstar);
end
