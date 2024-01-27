function [x, time_t, fval, gfval] = Rie_gd(x0, f, grad, R, stepSize, N, err)
   % Initialization
   % Set x = x0
   x = x0;
   % Optimal step size for convex functions

   fval = [];
   gfval = [];
   time_t = [];

   total_time = 0;
   for i = 1:N
       tic;
       gval = grad(x);
       x = R(x, -stepSize*gval);
       t2 = toc;
       total_time = total_time + t2;
       time_t(i) = total_time;
       fval(i) = f(x);
       gfval(i) = norm(gval,'fro');
       fprintf('iter:%d, fval:%d, grad:%d\n', i,fval(i), gfval(i));
       if norm(gval,'fro')<= err
           break;
       end
   end


end

