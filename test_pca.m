clear;
clc;
rng('default');
rng(22);

% Generate some random data to test the function
d = 1000;
CN = 1000;
D = 1000*diag(logspace(-log10(CN), 0, d)); fprintf('Exponential decay of singular values with CN %d.\n \n\n', CN);
[Q, R] = qr(randn(d)); 
A = Q*D*Q';
A = (A + A')/2;
p = 1;

max_eig = max(diag(D));
min_eig = min(diag(D));
costopt = -max_eig/2; 

M = spherefactory_exp(d);
problem.M = M;
problem.cost = @(X)    -.5*trace(X'*A*X);
problem.grad = @(X)    -M.egrad2rgrad(X, A*X);


x0 = M.rand();


maxiter = 500;
stepsize = 1/(max_eig - min_eig);
lambda = 1e-8;
tolgradnorm = 1e-6;

%% rgd with fixed stepsize
clear options;
options.stepsize = stepsize;
options.maxiter = maxiter;
options.tolgradnorm = tolgradnorm;
[x, cost, info_gd] = steepestdescent_mod(problem, x0, options);
%}


%% RNAG-C
clear options;
options.stepsize = stepsize;
options.maxiter = maxiter;
options.tolgradnorm = tolgradnorm;
options.xi = 1;
[x, cost, info_rnag] = RNAG_C(problem, x0, options);
%}

%% RNG-SC
clear options;
mu = 10;
options.stepsize = stepsize;
options.maxiter = maxiter;
options.tolgradnorm = tolgradnorm;
options.xi = 1;
options.q = mu * stepsize;
[x, cost, info_rnagsc] = RNAG_SC(problem, x0, options);
%keyboard;

%% RAGD
clear options;
options.stepsize = stepsize;
options.maxiter = maxiter;
options.tolgradnorm = tolgradnorm;
options.mu = mu;
[x, cost, info_ragd] = RAGD(problem, x0, options);


%% RiemNA
clear options;
k = 10;
options.stepsize = stepsize;
options.maxiter = maxiter;
options.tolgradnorm = tolgradnorm;
options.memory = k;
options.reg_lambda = lambda;
options.average = @mfd_average;

[x, cost, info_riemna] = RiemNA(problem, x0, options);




%% plots
lw = 2.0;
ms = 2.4;
fs = 21;
colors = {[55, 126, 184]/255, [228, 26, 28]/255, [247, 129, 191]/255, ...
      [166, 86, 40]/255, [255, 255, 51]/255, [255, 127, 0]/255, ...
      [152, 78, 163]/255, [77, 175, 74]/255}; 


optgap_gd = abs([info_gd.cost] - costopt); 
optgap_riemna = abs([info_riemna.cost] - costopt);
optgap_rnag = abs([info_rnag.cost] - costopt);
optgap_rnag_sc = abs([info_rnagsc.cost] - costopt);
optgap_ragd = abs([info_ragd.cost] - costopt);


h1 = figure;
semilogy([info_gd.iter], [info_gd.gradnorm], '-o', 'color', colors{1}, 'LineWidth', lw, 'MarkerSize',ms);  hold on;
semilogy([info_ragd.iter], [info_ragd.gradnorm], '-^', 'color', colors{3},  'LineWidth', lw, 'MarkerSize',ms); hold on;
semilogy([info_rnag.iter], [info_rnag.gradnorm], '-x', 'color', colors{7},  'LineWidth', lw, 'MarkerSize',ms); hold on;
semilogy([info_rnagsc.iter], [info_rnagsc.gradnorm], '-d', 'color', colors{6},  'LineWidth', lw, 'MarkerSize',ms); hold on;
semilogy([info_riemna.iter], [info_riemna.gradnorm], '-*', 'color', colors{2}, 'LineWidth', lw, 'MarkerSize',ms);  hold on;
ax1 = gca;
set(ax1,'FontSize', fs);
xlabel('Iterations', 'fontsize', fs);
ylabel('Gradnorm', 'fontsize', fs);
legend({'RGD', 'RAGD', 'RNAG-C', 'RNAG-SC', 'RGD+RiemNA'}, 'fontsize', fs-5);



h2 = figure;
semilogy([info_gd.iter], optgap_gd, '-o', 'color', colors{1}, 'LineWidth', lw, 'MarkerSize',ms);  hold on;
semilogy([info_ragd.iter], optgap_ragd, '-^', 'color', colors{3},  'LineWidth', lw, 'MarkerSize',ms); hold on;
semilogy([info_rnag.iter], optgap_rnag, '-x', 'color', colors{7},  'LineWidth', lw, 'MarkerSize',ms); hold on;
semilogy([info_rnagsc.iter], optgap_rnag_sc, '-d', 'color', colors{6},  'LineWidth', lw, 'MarkerSize',ms); hold on;
semilogy([info_riemna.iter], optgap_riemna, '-*', 'color', colors{2}, 'LineWidth', lw, 'MarkerSize',ms);  hold on;
ax1 = gca;
set(ax1,'FontSize', fs);
xlabel('Iterations', 'fontsize', fs);
ylabel('Optimality gap', 'fontsize', fs);
legend({'RGD', 'RAGD', 'RNAG-C', 'RNAG-SC', 'RGD+RiemNA'}, 'fontsize', fs-5);



h3 = figure;
semilogy([info_gd.time], optgap_gd, '-o', 'color', colors{1}, 'LineWidth', lw, 'MarkerSize',ms);  hold on;
semilogy([info_ragd.time], optgap_ragd, '-^', 'color', colors{3},  'LineWidth', lw, 'MarkerSize',ms); hold on;
semilogy([info_rnag.time], optgap_rnag, '-x', 'color', colors{7},  'LineWidth', lw, 'MarkerSize',ms); hold on;
semilogy([info_rnagsc.time], optgap_rnag_sc, '-d', 'color', colors{6},  'LineWidth', lw, 'MarkerSize',ms); hold on;
semilogy([info_riemna.time], optgap_riemna, '-*', 'color', colors{2}, 'LineWidth', lw, 'MarkerSize',ms);  hold on;
ax1 = gca;
set(ax1,'FontSize', fs);
xlabel('Time (s)', 'fontsize', fs);
ylabel('Optimality gap', 'fontsize', fs);
legend({'RGD', 'RAGD', 'RNAG-C', 'RNAG-SC', 'RGD+RiemNA'}, 'fontsize', fs-5);

