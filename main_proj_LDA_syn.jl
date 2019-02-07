@everywhere using Distances
@everywhere using BenchmarkTools
@everywhere using MAT
@everywhere using Distributions
#@everywhere using MTH229
@everywhere using SpecialFunctions
#@everywhere include("/Users/macbookpro/Dropbox/Large_deviatinos/module_projections.jl")
@everywhere include("module_projections.jl")
# CDF of a standard Gaussian
@everywhere Phi(x) = 1/2*(1 + erf(x/sqrt(2)))

""" Data stats for artificial data """
p = 400; # data dimension
#Sigma = eye(p); # identity model
rho = 0.4; # correlation factor
v = 0:p-1;
expon = abs.(v*ones(p)' - ones(p)*v');
Sigma = (rho*ones(p, p)).^expon; # Sigma_{i,j} = rho^(abs(i-j))
mu0 = zeros(p);
mu1 = mu0 + 3*ones(p) / sqrt(p);
mu = mu0 - mu1;
eigens, eigvect = eig(Sigma);
inv_Sigma = eigvect * diagm(ones(p)./eigens) * eigvect';
Sigma_r = eigvect * diagm(sqrt.(eigens)) * eigvect';

# Estimate stats.
n_train = 2000;
n0_train = Int64(ceil(n_train/2));
n1_train = Int64(n_train-n0_train);
train0 = mu0 * ones(1, n0_train) + Sigma_r*randn(p, n0_train);
train1 = mu1 * ones(1, n1_train) + Sigma_r*randn(p, n1_train);
stats = get_stats(train0, train1);
Sigma_ = stats[1];
mu0_ = stats[2];
mu1_ = stats[3];
eigens_ = stats[4];
eigvect_ = stats[5];
pi0_ = stats[6]; pi1_ = stats[7];
inv_Sigma_ = eigvect_ * diagm(ones(p)./eigens_) * eigvect_';

""" Experiments on Gaussian data """
n = 10000; # number of testing points
n0 = Int64(ceil(n/2));
n1 = Int64(n-n0);
pi0 = n0/n; pi1 = n1/n;
X0 = mu0 * ones(1, n0) + Sigma_r*randn(p, n0);
X1 = mu1 * ones(1, n1) + Sigma_r*randn(p, n1);
test = [X0 X1]; # data points
y = [zeros(n0);ones(n1)]; # labels

ratio = linspace(1e-2, 0.7, 30); # the ratio d/p
risk_array = pmap(1:length(ratio)) do i
    d = Int64(ceil(ratio[i]*p));
    # Bayes risk
    eps_bayes = Phi(-0.5sqrt(mu'*inv_Sigma*mu));
    #y_bayes = Fisher_LDA(test, mu0, mu1, inv_Sigma, pi0, pi1);
    #eps_bayes = sum(xor(convert(Array{Int64}, y), convert(Array{Int64}, y_bayes))) / length(y);

    # P-LDA theoretical performance
    eps_p_lda_th = th_risk(p, d, eigens_, eigvect_, mu0_-mu1_, pi0_, pi1_);
    #eps_p_lda_th = th_risk(p, d, eigens, eigvect, mu0-mu1, pi0_, pi1_);
    # P-LDA empirical performance
    Iterations = 1;
    avg = map(1:Iterations) do iter
        println("d= ", d, " iter= ", iter)
        # Gaussian projection
        W = randn(d, p)/sqrt(p);
        # Bernoulli projection
        #W = (1-2*rand(Bernoulli(0.5), d, p)) / sqrt(p);
        #y_proj = Fisher_LDA(W*X, W*mu0, W*mu1, inv(W*Sigma*W'), 0.5, 0.5);
        y_proj = Fisher_LDA(W*test, W*mu0_, W*mu1_, inv(W*Sigma_*W'), pi0_, pi1_);
        #return sum(xor(convert(Array{Int64}, y), convert(Array{Int64}, y_proj))) / n;
        return sum(xor(convert(Array{Int64}, y), convert(Array{Int64}, y_proj))) / length(y);
    end
    eps_p_lda_emp = mean(avg);

    # Experiments on MNIST data

    return eps_bayes, eps_p_lda_th, eps_p_lda_emp
end

risk_array = tuple_to_array(risk_array, 3)
risk_bayes = risk_array[:, 1];
risk_plda_th = risk_array[:, 2];
risk_plda_emp = risk_array[:, 3];

# save to a mat file
#rm("results.mat");
matwrite("results.mat", Dict(
                "risk_bayes" => risk_bayes, "risk_plda_th" => risk_plda_th,
                "risk_plda_emp" => risk_plda_emp;
         ));
#=
matwrite("/Users/macbookpro/Dropbox/Large_deviatinos/results.mat", Dict(
                "risk_bayes" => risk_bayes, "risk_plda_th" => risk_plda_th,
                "risk_plda_emp" => risk_plda_emp;
         ));
=#
