function [x, time_t,error_history,error_gd_g] = gradient_descent(x0,f,gradient, stepsize,N)
    % Initialization
    % Set x = x0
    x = x0;
    % Optimal step size for convex functions
    stepSize =  stepsize;
    error_history = zeros(N,1);
    error_gd_g= zeros(N,1);
    time_t=zeros(N,1);
    total_time=0;
    for i = 1:N
        tic;
        grad=gradient(x);
        x = x - stepSize * grad;
        t2=toc;
        total_time=total_time+t2;
        time_t(i)=total_time;
        error_history(i) = f(x);
        error_gd_g(i)=norm(stepSize*grad);
    end
end