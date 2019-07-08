% dd2424 deeplearning 2019 assignment1
% loghorizon

%clear all; 
clc;

disp("1: run test part;");
disp("2: run training part with different parameters.");
exercise = input("press 1 or 2: ");

if exercise == 1
    % test the gradient function and the sample figure
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    [validX, validY, validy] = LoadBatch('data_batch_2.mat');
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    [d, N] = size(trainX);
    K = 10;
    rng(400);
    W = randn(K, d, 'double') * 0.01;
    b = randn(K, 1, 'double') * 0.01;
    P = EvaluateClassifier(trainX, W, b);
    lambda = 0.1;
    [grad_W, grad_b] = ComputeGradients(trainX(:,1:50), trainY(:,1:50), P(:,1:50), W, lambda);
    [ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(:,1:50), trainY(:,1:50), W, b, lambda, 1e-6);
    % absolutes difference are small (<=1e-9)
    abs_errw = grad_W-ngrad_W;
    abs_errb = grad_b-ngrad_b;
    %disp(abs_errw);
    %disp(abs_errb);
    % eps test
    err_w = abs(grad_W - ngrad_W)./(max(eps,abs(grad_W) + abs(ngrad_W)));
    err_w = max(max(err_w));
    err_b = abs(grad_b - ngrad_b)./(max(eps,abs(grad_b) + abs(ngrad_b)));
    err_b = max(max(err_b));
    disp("maximum relative errors are " + err_w + " and " + err_b);
    
    n_batch = 100;
    eta = 0.01;
    n_epochs = 40;
    lambda = 0;
    GDparams = [n_batch, eta, n_epochs];
    [Wstar, bstar, trainloss, validationloss] = MiniBatchGD(trainX, trainY, validX, validY, GDparams, W, b, lambda);
    acc = ComputeAccuracy(testX, testy, Wstar, bstar);
    % plot and result
    disp("Accuracy on test set is " + acc);
    
    epoch = 1:n_epochs;
    fig = figure;
    plot(epoch, trainloss, 'g', epoch, validationloss, 'r')
    legend('training loss', 'validation loss')
    xlabel('epoch')
    ylabel('loss')
    print(fig, 'Result_Pics/t1.png', '-dpng');
    close gcf
    clear s_im;
    for i =1:10
        im = reshape(Wstar(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:)))/(max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
    end
    montage(s_im, 'Size', [1, 10]);
    print('Result_Pics/mt1.png', '-dpng');
    close gcf
end

if exercise == 2
    % train part with different parameters
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    [validX, validY, validy] = LoadBatch('data_batch_2.mat');
    [testX, testY, testy] = LoadBatch('test_batch.mat');
    [d, N] = size(trainX);
    K = 10;
    W = randn(K, d, 'double') * 0.01;
    b = randn(K, 1, 'double') * 0.01;
    lambda = [0, 0, 0.1, 1];
    n_epochs = [40, 40, 40, 40];
    n_batch = [100, 100, 100, 100];
    eta = [0.1, 0.01, 0.01, 0.01];
    
    fname1 = ["Result_Pics/tr1.png","Result_Pics/tr2.png","Result_Pics/tr3.png","Result_Pics/tr4.png"];
    fname2 = ["Result_Pics/mtr1.png","Result_Pics/mtr2.png","Result_Pics/mtr3.png","Result_Pics/mtr4.png"];
    for i=1:size(n_batch, 2)
        GDparams = [n_batch(i), eta(i), n_epochs(i)];
        [Wstar, bstar, trainloss, validationloss] = MiniBatchGD(trainX, trainY, validX, validY, GDparams, W, b, lambda(i));
        acc = ComputeAccuracy(testX, testy, Wstar, bstar);
        % plot and result
        disp("Accuracy on test set (" + i + ") is " + acc);
        epoch = 1:n_epochs(i);
        plot(epoch, trainloss, 'g', epoch, validationloss, 'r');
        legend('training loss', 'validation loss')
        xlabel('epoch')
        ylabel('loss')
        print(fname1(i), '-dpng');
        pause;
        close gcf

        clear s_im;
        for c =1:10
            im = reshape(Wstar(c, :), 32, 32, 3);
            s_im{c} = (im - min(im(:)))/(max(im(:)) - min(im(:)));
            s_im{c} = permute(s_im{c}, [2, 1, 3]);
        end
        montage(s_im, 'Size', [1, 10]);
        print(fname2(i), '-dpng');
        pause;
        close gcf
    end
end

% Accuracy on test set (1) is 0.2852
% Accuracy on test set (2) is 0.3677
% Accuracy on test set (3) is 0.3338
% Accuracy on test set (4) is 0.2194

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

function [Wstar, bstar, trainloss, validationloss] = MiniBatchGD(X, Y, VX, VY, GDparams, W, b, lambda)
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
            Ybatch = Y(:, j_start:j_end);
            P = EvaluateClassifier(Xbatch, Wstar, bstar);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
            Wstar = Wstar - eta*grad_W;
            bstar = bstar - eta*grad_b;
        end
        trainloss(i) = ComputeCost(X, Y, Wstar, bstar, lambda);
        validationloss(i) = ComputeCost(VX, VY, Wstar, bstar, lambda);           
    end   
end