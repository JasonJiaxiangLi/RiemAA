function [x, cost, info, options] = RNAG_C(problem, x, options)
% Riemannian Nesterov accelerated gradient. We follow the recent paper to
% write the code: "Accelerated Gradient Methods for Geodesically Convex Optimization:
% Tractable Algorithms and Convergence Analysis" in ICML 2022.
% The source code is written in python and can be found on https://github.com/jungbinkim1/RNAG
%
% This is the version for convex function.
%
% The algorithm requires retr and invretr, transp to be properly defined. To use
% exponential map and parallel transport. Please redefine a factory file
% with the field retr/invretr/transp defined as exp/log/parallel transport

    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
                'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the
        % user seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
               ['No gradient provided. Using an FD approximation instead (slow).\n' ...
                'It may be necessary to increase options.tolgradnorm.\n' ...
                'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end

    % Set local defaults here.
    localdefaults.minstepsize = 1e-10;
    localdefaults.maxiter = 1000;
    localdefaults.tolgradnorm = 1e-6; 

    localdefaults.stepsize = 0.1;
    localdefaults.xi = 1;


    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    timetic = tic();
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end

    % Create a store database and get a key for the current x.
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x.
    [cost, grad] = getCostGrad(problem, x, storedb, key);
    gradnorm = problem.M.norm(x, grad);


    % Iteration counter.
    % At any point, iter is the number of fully executed iterations so far.
    iter = 0;


    % Save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];

    if options.verbosity >= 2
        fprintf(' iter\t               cost val\t    grad. norm\n');
    end

    M = problem.M;

    stepsize = options.stepsize;
    xi = options.xi;
    vbar = M.invretr(x,x);

    % Start iterating until stopping criterion triggers.
    while true

        % Display iteration information.
        if options.verbosity >= 2
            fprintf('%5d\t%+.16e\t%.8e\n', iter, cost, gradnorm);
        end

        % Start timing this iteration.
        timetic = tic();

        % Run standard stopping criterion checks.
        [stop, reason] = stoppingcriterion(problem, x, options, ...
                                                             info, iter+1);

        % If none triggered, run specific stopping criterion check.
        if ~stop && stats.stepsize < options.minstepsize
            stop = true;
            reason = sprintf(['Last stepsize smaller than minimum '  ...
                              'allowed; options.minstepsize = %g.'], ...
                              options.minstepsize);
        end
        if stop
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end

        lambda = (iter + 2*xi)/2;
        
        y = M.retr(x, M.lincomb(x, (xi/ (lambda + xi -1)), vbar) );

        [~, grady] = getCostGrad(problem, y);
        dir = M.lincomb(y, -stepsize, grady);
        xnew = M.retr(y, dir);

        v = M.transp(x, y, M.lincomb(x, 1, vbar, -1, M.invretr(x, y)) );
        vbar = M.lincomb(y, 1, v, (-stepsize*lambda/xi), grady);
        vbar = M.transp(y, xnew, M.lincomb(y, 1, vbar, -1, dir)  );

        
        
        newkey = storedb.getNewKey();
        % Compute the new cost-related quantities for x
        [newcost, newgrad] = getCostGrad(problem, xnew, storedb, newkey);
        newgradnorm = problem.M.norm(xnew, newgrad);

        x = xnew;
        key = newkey;
        cost = newcost;
        grad = newgrad;
        gradnorm = newgradnorm;


        % Make sure we don't use too much memory for the store database.
        storedb.purge();
        
        % iter is the number of iterations we have accomplished.
        iter = iter + 1;
        
        % Log statistics for freshly executed iteration.
        stats = savestats();
        info(iter+1) = stats;


    end
    info = info(1:iter+1);


    % Routine in charge of collecting the current iteration stats.
    function stats = savestats()
        stats.iter = iter;
        stats.cost = cost;
        stats.gradnorm = gradnorm;
        if iter == 0
            stats.stepsize = NaN;
            stats.time = toc(timetic);
            %stats.linesearch = [];
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic);
            %stats.linesearch = lsstats;
        end
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end

end

