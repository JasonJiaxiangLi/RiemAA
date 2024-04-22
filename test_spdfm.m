function test_spdfm()
% test on frechet mean computation on SPD manifold

    clear;
    clc;
    rng('default');
    rng(22);
    
    d = 10;
    N = 100;

    A = zeros(d,d,N);
    M = sympositivedefinitefactory_exp(d);
    
    % generate random SPD matrices
    ref = diag(max(.1, 1+.1*randn(d, 1)));
    for i = 1 : N
        noise = 1*randn(d);
        noise = (noise + noise')/2;
        [V, D] = eig(ref + noise);
        A(:, :, i) = V*diag(max(.01, diag(D)))*V';
    end
    
    problem.M = M;
    problem.cost = @cost;    
    function f = cost(X)
        f = 0;
        for jj = 1 : N
            f = f + 0.5*(problem.M.norm(X, problem.M.log(X,  A(:,:,jj)))^2);
        end
        f = f/N;
        
    end

    % Riemannian gradient of the cost function
    problem.grad = @rgrad;      
    function g = rgrad(X)
        g = 0;
        for jj = 1 : N
            g = g - problem.M.log(X,   A(:,:,jj));
        end
        g = g/N;
    end

    % compute optimal solution using rlbfgs
    options.tolgradnorm = 1e-10;
    [Xopt, costopt] = rlbfgs(problem, [], options);
    

    %% parameters 
    stepsize = 0.5;
    maxiter = 100;
    tolgradnorm = 1e-6;
    x0 = M.rand();

    %% rgd with fixed stepsize
    clear options
    options.stepsize = stepsize;
    options.maxiter = maxiter;
    options.tolgradnorm = tolgradnorm;
    [xgd, costgd, info_gd] = steepestdescent_mod(problem, x0, options);
    


    %% RNAG-C
    clear options
    stepsize_rng_c = 0.5;
    options.stepsize = stepsize_rng_c;
    options.maxiter = maxiter;
    options.tolgradnorm = tolgradnorm;
    options.xi = 1;
    [xnag, costnag, info_rnag] = RNAG_C(problem, x0, options);
    
    
    %% RNAG-SC
    clear options;
    mu = 1;
    stepsize_rng_sc = 0.5;
    options.stepsize = stepsize_rng_sc;
    options.maxiter = maxiter;
    options.tolgradnorm = tolgradnorm;
    options.xi = 1;
    options.q = mu * stepsize_rng_sc;
    [xnagsc, costnagsc, info_rnagsc] = RNAG_SC(problem, x0, options);
    
    %% RAGD
    clear options;
    mu = 1;
    stepsize_ragd = 0.5;
    options.stepsize = stepsize_ragd;
    options.maxiter = maxiter;
    options.tolgradnorm = tolgradnorm;
    options.mu = mu;
    [x, costragd, info_ragd] = RAGD(problem, x0, options);

    %% RiemNA
    clear options
    k = 5;
    lambda = 1e-8;
    options.stepsize = stepsize;
    options.maxiter = maxiter;
    options.tolgradnorm = tolgradnorm;
    options.memory = k;
    options.reg_lambda = lambda;
    options.average = @mfd_average;
    
    [xrna, costrna, info_riemna] = RiemNA(problem, x0, options);
    %}

    
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