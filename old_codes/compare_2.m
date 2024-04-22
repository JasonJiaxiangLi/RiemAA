clear; clc;
close all;

%% Parameters
N = 5000; % Max iteration

fid = 1;
fprintf(fid,'==============================================================================================\n');

seed = round(rand() * 10000000);
rng(seed);

%% Brockett cost function on the Stiefel manifold St(n,p)
% min trace(X'AXM) s.t. X\in St(n,p) 
% where M = diag([1,2,...,p]);

n = 10; p = 5;
A = randn(n,n);
A = .5*(A + A');
M = 1:p;
M = diag(M);


%%
f = @(x) fval(x,A,M);
grad = @(x) Rgrad(x,A,M);
R = @(x,eta) Retr(x,eta);
Tran = @(x1,x2,d) Transp(x1,x2,d);

[phi_init, ~] = svd(randn(n,p),0); % random initialization
x0 = phi_init;
err = 1e-5;
%% Gradient descent

L = norm(A);
stepSize = .02;
[x_gd,gd_time, gd_f, gd_grad] = Rie_gd(x0, f, grad,R, stepSize, N, err);


%% Rie_AA
m = 10;
c = 0.9;
stepSize = .02;
err = 1e-8;
[x_RAA, RAA_f, RAA_gf, sindex, array_lambda] = Rie_AA_stiefel(x0, f, grad, R, Tran, stepSize, err, N, m ,c);

[V,D] = eig(A);
disp([f(x_gd), f(x_RAA)])
% V
% x_gd
% V
% x_RAA


figure(1);
plot(1:length(gd_f), gd_f,'r-','LineWidth',1.5);
hold on
plot(1:length(RAA_f), RAA_f,'bo--','LineWidth',1.5)
hold on
legend({'RGD','RAA'});
xlabel('Iteration');
ylabel('fval');

figure(2)
semilogy(1:length(gd_grad), gd_grad, 'r-','LineWidth',1.5);
hold on
semilogy(1:length(RAA_gf),RAA_gf,'bo--','LineWidth',1.5);
hold on
legend({'RGD','RAA'});
xlabel('Iteration');
ylabel('|grad|');

function [output] = fval(X, A, M)
   output = trace(X'*A*X*M);
end

function [output] = Rgrad(X, A, M)
   
   U = 2*A*X*M;
   output = U - X*X'*U + .5*X*(X'*U - U'*X);
   %output = 2*U*M - X*X'*U*M -X*M*X'*U;
end

function [output] = Retr(X, eta)
   tmp = X + eta;
   [~, SIGMA, V] = svd(tmp,0);
   SIGMA = diag(SIGMA);
   output = tmp * (V * diag(1./SIGMA) * V');
end

function [output] = Transp(x1,x2,eta)
    output = eta - x2*x2'*eta + .5*x2*(x2'*eta - eta'*x2);
end 
   