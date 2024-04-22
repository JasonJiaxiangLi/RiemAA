function test_grass_nleig()
% This example is motivated in the paper and is included in ManOpt
% "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems",
% Zhi Zhao, Zheng-Jian Bai, and Xiao-Qing Jin,
% SIAM Journal on Matrix Analysis and Applications, 36(2), 752-774, 2015.
    
    clear 
    clc
    rng('default');
    rng(22);

    if ~exist('L', 'var') || isempty(L)
        n = 100;
        L = gallery('tridiag', n, -1, 2, -1);
    end
    
    n = size(L, 1);
    assert(size(L, 2) == n, 'L must be square.');
    
    if ~exist('k', 'var') || isempty(k) || k > n
        k = 5;
    end
    
    if ~exist('alpha', 'var') || isempty(alpha)
        alpha = 1;
    end
    
    M = grassmannfactory(n, k);
    problem.M = M;
    
    % Cost function evaluation
    problem.cost =  @cost;
    function val = cost(X)
        rhoX = sum(X.^2, 2); % diag(X*X'); 
        val = 0.5*trace(X'*(L*X)) + (alpha/4)*(rhoX'*(L\rhoX));
    end
    
    % Euclidean gradient evaluation
    problem.egrad = @egrad;
    function g = egrad(X)
        rhoX = sum(X.^2, 2); % diag(X*X');
        g = L*X + alpha*diag(L\rhoX)*X;
    end
    
    x0 = M.rand();


    maxiter = 500;
    stepsize = 0.1;
    lambda = 1e-8;
    tolgradnorm = 1e-6;
    
    
    % compute optimal solution using rlbfgs
    options.tolgradnorm = 1e-10;
    options.memory = 5;
    [Xopt, costopt] = rlbfgs(problem, [], options);
    

    %% rgd with fixed stepsize
    clear options;
    options.stepsize = stepsize;
    options.maxiter = maxiter;
    options.tolgradnorm = tolgradnorm;
    [x, costgd, info_gd] = steepestdescent_mod(problem, x0, options);
    
    %% RNAG-C
    clear options;
    stepsize_c = 0.1;
    options.stepsize = stepsize_c;
    options.maxiter = maxiter;
    options.tolgradnorm = tolgradnorm;
    options.xi = 1;
    [x, costrnag, info_rnag] = RNAG_C(problem, x0, options);
    %}
    
    %% RNAG-SC
    clear options;
    mu = 5;
    stepsize_rng_sc = 0.1;
    options.stepsize = stepsize_rng_sc;
    options.maxiter = maxiter;
    options.tolgradnorm = tolgradnorm;
    options.xi = 1;
    options.q = mu * stepsize_rng_sc;
    [xnagsc, costnagsc, info_rnagsc] = RNAG_SC(problem, x0, options);
    
    %% RAGD
    stepsize_ragd = 0.1;
    options.stepsize = stepsize_ragd;
    options.maxiter = maxiter;
    options.tolgradnorm = tolgradnorm;
    options.mu = mu;
    [x, costragd, info_ragd] = RAGD(problem, x0, options);
    
    
    %% RiemNA
    clear options
    k = 5;
    options.stepsize = stepsize;
    options.maxiter = maxiter;
    options.tolgradnorm = tolgradnorm;
    options.memory = k;
    options.reg_lambda = lambda;
    options.average = @mfd_average;
    
    [xrna, costrna, info_riemna] = RiemNA(problem, x0, options);
    
    
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
    
end

