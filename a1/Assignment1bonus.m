% dd2424 deeplearning 2019 assignment1 bonus part
% loghorizon

%clear all; 
clc;

disp("1: run optimize part;");
disp("2: run training part using SVM.");
exercise = input("press 1 or 2: ");

if exercise == 1
    % (a) Use all the available training data for training and size 1000
    % for validation set (first 1000)
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
     % (b) run for a longer time: around 100
    [d, N] = size(trainX);
    K = 10;
    W = randn(K, d, 'double') * 0.01;
    b = randn(K, 1, 'double') * 0.01;
    lambda = 0;
    n_epochs = 80;
    n_batch = 100;
    eta = 0.01;
    fname1 = ["Result_Pics/opt1.png","Result_Pics/opt2.png", "Result_Pics/opt3.png"];
    decay = [0, 1, 2];
    for i=3
        GDparams = [n_batch, eta, n_epochs];
        [Wstar, bstar, trainloss, validationloss] = MiniBatchGD_Opt(trainX, trainY, validX, validY, GDparams, W, b, lambda, decay(i));
        acc = ComputeAccuracy(testX, testy, Wstar, bstar);
        disp("Accuracy using decay " + i + " is " + acc);
        
        epoch = 1:n_epochs;
        plot(epoch, trainloss, 'g', epoch, validationloss, 'r');
        legend('training loss', 'validation loss')
        xlabel('epoch')
        ylabel('loss')
        print(fname1(i), '-dpng');
        pause;
        close gcf
    end
end
% Accuracy using decay 0 is 0.4068
% Accuracy using decay 1 is 0.4025
% Accuracy using decay 3 is 0.4027

if exercise == 2
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    %[validX, validY, validy] = LoadBatch('data_batch_2.mat');
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
    K = 10;
    W = randn(K, d, 'double') * 0.01;
    b = randn(K, 1, 'double') * 0.01;
    lambda = 0;
    n_epochs = 40;
    n_batch = 100;
    eta = 0.01;
    GDparams = [n_batch, eta, n_epochs];
    [Wstar, bstar, trainloss, validationloss] = MiniBatchGD_SVM(trainX, trainy, validX, validy, GDparams, W, b, lambda);
    acc = ComputeAccuracy_SVM(testX, testy, Wstar, bstar);
    disp("Accuracy on test set is " + acc);
    
    epoch = 1:n_epochs;
    plot(epoch, trainloss, 'g', epoch, validationloss, 'r');
    legend('training loss', 'validation loss');
    xlabel('epoch');
    ylabel('loss');
    print("Result_Pics/svm2.png", '-dpng');
    close gcf
end
% Accuracy on test set is 0.356
% Accuracy on test set is 0.3026
% Accuracy on test set is 0.2731


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



function P = EvaluateClassifier(X, W, b)
    % s=wx+b, p=softmax(s)
    % x: d*n, p: k*n, W: k*d, b: k*1
    s = W*X + b;
    P = exp(s)./(double(ones(10))*exp(s));
end

function P = EvaluateClassifier_SVM(X, W, b)
    P = W*X + b;
end

function J = ComputeCost(X, Y, W, b, lambda)
    % loss = -log(yT P)  
    % j = sum(-log(YT P))/n + lambda sum(W^2)
    P = EvaluateClassifier(X, W, b);
    N = size(X, 2);
    result = 0;
    for i = 1:N
        result = result - log(Y(:,i)'*P(:,i));
    end
    J = result/N + lambda * sum(sum(W.*W));
end

function J = ComputeCost_SVM(X, y, W, b, lambda)
    % loss = sum(l)/n + lambda sum(W^2)
    % loss = sum(max(0,pj-py+1))(j in k and j!=y)
    P = EvaluateClassifier_SVM(X, W, b);
    N = size(X, 2);
    result = 0;
    for i = 1:N
       P(:, i) = P(:, i) - P(y(i), i) + 1;
       P(:, i) = max(0, P(:, i));
       % each column with an extra 1, remove it
       result = result + sum(P(:, i)) - 1;
    end
    J = result/N + lambda * sum(sum(W.*W));
end

function acc = ComputeAccuracy(X, y, W, b)
    % accracy of labels
    N = size(X, 2);
    correct = 0;
    P = EvaluateClassifier(X, W, b);
    % Ignore the first output using a tilde (~)
    [~, predict] = max(P);
    for i = 1:N
        if y(i)==predict(i)
            correct = correct + 1;
        end
    end
    acc = correct/N;
end

function acc = ComputeAccuracy_SVM(X, y, W, b)
    N = size(X, 2);
    correct = 0;
    % not softmax, just s
    P = EvaluateClassifier_SVM(X, W, b);
    [~, predict] = max(P);
    for i = 1:N
        if y(i)==predict(i)
            correct = correct + 1;
        end
    end
    acc = correct/N;
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    % grad_w = dL/dW +2lambda*W 10*3072
    % grad_b = dL/db 10*1
    % gbatch 10*1, xbatch 3072*1
    N = size(X, 2);
    [K, D] = size(W);
    gwsum = zeros(K, D);
    gbsum = zeros(K, 1);
    for i=1:N
        xbatch = X(:, i);
        gbsum = gbsum + (P(:, i) - Y(:, i));
        gwsum = gwsum + (P(:, i) - Y(:, i))*xbatch';
    end
    grad_b = gbsum/N;
    grad_W = gwsum/N + 2*lambda*W;
end

function [grad_W, grad_b] = ComputeGradients_SVM(X, y, P, W, lambda)
    % dLi/dWj = 1(xiwj-xiWy_i+1>0)xi
    N = size(X, 2);
    [K, D] = size(W);
    grad_b = zeros(K, 1);
    grad_W = zeros(K, D);
    for i=1:N
        for j=1:K
            if j~=y(j)
                margin = P(j,i) - P(y(i), i) + 1;
                if margin>0
                    grad_W(y(i), :) = grad_W(y(i), :) - X(:, i)';
                    grad_W(j, :) = grad_W(j, :) + X(:, i)';
                end
            end
        end
    end
    grad_W = grad_W/N + 2*lambda*W;
end


function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)
    no = size(W, 1);
    d = size(X, 1);
    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);
    c = ComputeCost(X, Y, W, b, lambda);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c) / h;
    end

    for i=1:numel(W)   

        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);

        grad_W(i) = (c2-c) / h;
    end
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)
    no = size(W, 1);
    d = size(X, 1);
    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end

    for i=1:numel(W)

        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);

        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);

        grad_W(i) = (c2-c1) / (2*h);
    end
end

function [Wstar, bstar, trainloss, validationloss] = MiniBatchGD_Opt(X, Y, VX, VY, GDparams, W, b, lambda, decay)
    n_batch = GDparams(1);
    eta = GDparams(2);
    n_epochs = GDparams(3);
    Wstar = W;
    bstar = b;
    N = size(X, 2);
    trainloss = zeros(n_epochs, 1);
    validationloss = zeros(n_epochs, 1);
    % do n_epochs step, update w,b, log train/validation loss for mini batch
    for i=1:n_epochs
        % (g) shuffle the training data at the begining of each epoch 
        % using randperm shuffle column index to get X_shuffle and Y_shuffle
        shuffle_index = randperm(N);
        X_shuffle = X(:,shuffle_index);
        Y_shuffle = Y(:,shuffle_index);
        for j=1:(N/n_batch)
            j_start = (j - 1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X_shuffle(:, j_start:j_end);
            Ybatch = Y_shuffle(:, j_start:j_end);
            P = EvaluateClassifier(Xbatch, Wstar, bstar);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
            Wstar = Wstar - eta*grad_W;
            bstar = bstar - eta*grad_b;
        end
        trainloss(i) = ComputeCost(X_shuffle, Y_shuffle, Wstar, bstar, lambda);
        validationloss(i) = ComputeCost(VX, VY, Wstar, bstar, lambda); 
        % (d) decaying the learning rate by a factor ?0.9 after each epoch
        % time-based decay (1) lr = lr * decay rate
        % or step decay (2), half evey 10 epoches
        if decay == 1
            eta = eta * 0.9;
        elseif decay == 2
            eta = eta * power(0.5, floor(i/10));
        end
    end   
end

function [Wstar, bstar, trainloss, validationloss] = MiniBatchGD_SVM(X, y, VX, Vy, GDparams, W, b, lambda)
    n_batch = GDparams(1);
    eta = GDparams(2);
    n_epochs = GDparams(3);
    Wstar = W;
    bstar = b;
    N = size(X, 2);
    trainloss = zeros(n_epochs, 1);
    validationloss = zeros(n_epochs, 1);
    % do n_epochs step, update w,b, log train/validation loss for mini batch
    for i=1:n_epochs
        for j=1:(N/n_batch)
            j_start = (j - 1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            ybatch = y(:, j_start:j_end);
            P = EvaluateClassifier_SVM(Xbatch, Wstar, bstar);
            [grad_W, grad_b] = ComputeGradients_SVM(Xbatch, ybatch, P, Wstar, lambda);
            Wstar = Wstar - eta*grad_W;
            bstar = bstar - eta*grad_b;
        end
        trainloss(i) = ComputeCost_SVM(X, y, Wstar, bstar, lambda);
        validationloss(i) = ComputeCost_SVM(VX, Vy, Wstar, bstar, lambda);           
    end   
end