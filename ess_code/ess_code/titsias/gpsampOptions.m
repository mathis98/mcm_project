function options = gpsampOptions(gplik) 
%
%

options.kern = 'rbf';
options.constraints.kernHyper = 'free'; % 'free' or 'fixed'

switch gplik 
    case 'regression'
        % Gaussian likelihood 
        options.Likelihood = 'Gaussian';
    case 'classification' % binary classification 
        % Probit likelihood
        options.Likelihood = 'Sigmoid';
    case 'poissonRegr' % Poisson regression
        % Poisson likelihood
        options.Likelihood = 'Poisson';
    case 'offsetPoissonRegr' % Poisson Regression with an offset on the log rate
        % Poisson likelihood after offsetting the log rate
        options.Likelihood = 'OffsetPoisson';
    case 'ODE' 
        % not included 
end

options.constraints.likHyper = 'free'; % 'free' or 'fixed'

        
