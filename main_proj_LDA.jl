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

 """ Experiments on MNIST data """
digit0 = 2;
digit1 = 3;
train0, test0, train1, test1 = MNIST_data(digit0, digit1);
stats = get_stat(train0, train1);

Sigma_ = stats[1];
p, _ = size(Sigma_);
mu0_ = stats[2];
mu1_ = stats[3];
eigens_ = stats[4];
eigvect_ = stats[5];
pi0_ = stats[6]; pi1_ = stats[7];
test = [test0 test1];
y_ = [zeros(length(test0)); ones(length(test1))];
inv_Sigma_ = eigvect_ * diagm(ones(p)./eigens_) * eigvect_';
ratio = linspace(1e-2, 0.7, 30); # the ratio d/p

risk_array = pmap(1:length(ratio)) do i
    d = Int64(ceil(ratio[i]*p));
    # Bayes risk
    #eps_bayes = Phi(-0.5sqrt(mu'*inv_Sigma*mu));
    y_bayes = Fisher_LDA(test, mu0_, mu1_, inv_Sigma_, pi0_, pi1_);
    eps_bayes = sum(xor(convert(Array{Int64}, y_), convert(Array{Int64}, y_bayes))) / length(y_)
    # P-LDA theoretical performance
    eps_p_lda_th = th_risk(p, d, eigens_, eigvect_, mu0_-mu1_, pi0_, pi1_);

    # P-LDA empirical performance
    Iterations = 1;
    avg = map(1:Iterations) do iter
        println("d= ", d, " iter= ", iter)
        # Gaussian projection
        #W = randn(d, p)/sqrt(p);
        # Bernoulli projection
        W = (1-2*rand(Bernoulli(0.5), d, p)) / sqrt(p);
        #y_proj = Fisher_LDA(W*X, W*mu0, W*mu1, inv(W*Sigma*W'), 0.5, 0.5);
        y_proj_ = Fisher_LDA(W*test, W*mu0_, W*mu1_, inv(W*Sigma_*W'), pi0_, pi1_);
        #return sum(xor(convert(Array{Int64}, y), convert(Array{Int64}, y_proj))) / n;
        return sum(xor(convert(Array{Int64}, y_), convert(Array{Int64}, y_proj_))) / length(y_);
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
