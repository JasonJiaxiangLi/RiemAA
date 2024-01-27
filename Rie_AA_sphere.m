function [x_result, fval, norm_gf, sindex, array_lambda] = Rie_AA_sphere(x0, f, grad, Retr, Tran, step_length, err, N, m ,c)
   % Input:
   % x0: initial x
   % f : function value
   % grad: gradient of objective function
   % step_length: stepsize
   % N :max iteration
   % m: Anserson depth
   % c: Lipschitz constang for g

   % Output:
   % x_result: final result
   % error_f: 
   % error_gf:
   
   n = length(x0);

   % parameters
   mu_min = 0; mu = 1e0;
   p1 = 0.01; p2 = 0.25;
   eta1 = 2; eta2 = 0.25;
   
   % Initialization
   sindex = []; array_lambda = [];
   Delta_X = zeros(n, m);

   X = zeros(n,m+1);
   hat_R = zeros(n,m+1); % original search direction
   R = zeros(n,m+1);
   R_norm = zeros(1, m);


   x = x0; 
   for k=1:m
      X(:,k) = x;
      hat_R(:,k) = -step_length * grad(x);
      R(:,k) = hat_R(:,k); R_norm(:,k) = norm(R(:,k),'fro');
      Delta_X(:,k) = hat_R(:,k);
      if(k>=2)
         for i =1:k-1
            Delta_X(:,i) = Tran(X(:,k-1), Delta_X(:,k-1), Delta_X(:,i));
            R(:,i) = Tran(X(:,k-1), Delta_X(:,k-1), R(:,i));
            R_norm(:,i) = norm(R(:,i),'fro');
         end
      end

      fval(k) = f(x);
      norm_gf(k) = norm(grad(x),'fro');
      fprintf("iter:%d, fval:%d, gf:%d \n",k, fval(k), norm_gf(k));

      x = Retr(X(:,k), hat_R(:,k));

   end
 

X(:,m+1) = x;
hat_R(:,m+1) = -step_length * grad(x);
R(:,m+1) = hat_R(:,m+1); R_norm(:,m+1) = norm(R(:,m+1),'fro');

   for k = m+1: N


       if(R_norm(k) <= err)
           break;
       end

       for i = k-m : k-1
           Delta_X(:,i) = Tran(X(:,k-1), Delta_X(:,k-1), Delta_X(:,i));
           R(:,i) = Tran(X(:,k-1), Delta_X(:,k-1), R(:,i));
           R_norm(:,i) = norm(R(:,i),'fro');
       end
       
       %% update max R_norm
       max_R = max(R_norm(k-m:k));

       r_k = R(:,k);
       x_k = X(:,k);

       %% generate Delta R
       temp_R = R(:,k-m:k-1) - r_k;
       
       %% generate Delta X
       bar_DeltaX = Delta_X;
       bar_DeltaX(:,k) = 0;
       for j = k-1:-1:k-m
          bar_DeltaX(:,j) = bar_DeltaX(:,j+1)-Delta_X(:,j);
       end
       temp_X = bar_DeltaX(:,k-m:k-1);

       % set lambda
       lambda = mu; % lambda = mu*min_F^2;

       % set alpha
       J = temp_R;
       M = J'*J + lambda * eye(m);
       b = -J'*r_k;
       alpha = lsqminnorm(M,b);

       w = max_R;
     
       % generate trial point
       r_hat = r_k + temp_R * alpha;
       tilde_DeltaX = temp_X * alpha + r_hat;
       g_hat = Retr(x_k, tilde_DeltaX);
       trial_r = -step_length * grad(g_hat); 

       pred = w - c*norm(r_hat,'fro');
       ared = w - norm(trial_r, 'fro');
       rho = ared/pred;
       array_lambda(k) = lambda;
       fprintf('iteration:%d, lambda:%d, ared:%d, pred:%d, rho:%d\n', k,lambda, ared, pred, rho);
       
       % update mu
       if(rho <= p1)
           mu = eta1 * mu;
       elseif(rho >= p2)
           mu = max(eta2*mu, mu_min);
       end

       % update new iteration
       if(rho >= p1)
           x = g_hat;
           X(:,k+1) = x;
           hat_R(:,k+1) = trial_r;
           R(:,k+1) = hat_R(:,k+1);
           R_norm(:,k+1) = norm(R(:,k+1),'fro');

           Delta_X(:,k) = tilde_DeltaX;
           sindex(k) = true;
       else
           x = Retr(x_k,r_k);
           X(:,k+1) = x;
           hat_R(:,k+1) = -step_length*grad(x);
           R(:,k+1) = hat_R(:,k+1);
           R_norm(:,k+1) = norm(R(:,k+1),'fro');

           Delta_X(:,k) = r_k;
           sindex(k) = false;
       end

       fval(k) = f(x);
       norm_gf(k) = norm(grad(x),'fro');
       fprintf("iter:%d, fval:%d, gf:%d \n",k,fval(k), norm_gf(k));
   end
   x_result = x;
end



















