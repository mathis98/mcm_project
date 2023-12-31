function mcmcoptions = mcmcOptions(algorithm) 
%
%

% MCMC OPTIONS (you can change these options if you like)   
mcmcoptions.train.StoreEvery = 100; % keep samples every StoreEvery MCMC iterations
mcmcoptions.train.Burnin = 5000;  % burn in time
mcmcoptions.train.T = 50000; % sampling time
mcmcoptions.train.Store = 0;  % store the results regularly in a file         
mcmcoptions.train.disp = 0;  % display information during sampling (if applicable)

% define also options required when perform the adaptive MCMC phase 
switch algorithm 
    case 'controlPnts'
        %        
        mcmcoptions.adapt.T = 200;          
        mcmcoptions.adapt.Burnin = 100;
        mcmcoptions.adapt.StoreEvery = 10; 
        mcmcoptions.adapt.disp = 1;
        mcmcoptions.adapt.initialNumContrPnts = 3; 
        mcmcoptions.adapt.incrNumContrBy = 1;
        %
    case 'localRegion'
        %
        mcmcoptions.adapt.T = 200;          
        mcmcoptions.adapt.Burnin = 100;
        mcmcoptions.adapt.StoreEvery = 10; 
        mcmcoptions.adapt.disp = 1;
        mcmcoptions.adapt.initialNumRegions = 3; 
        mcmcoptions.adapt.incrNumRegionBy = 1;
        %
    case 'gibbs' 
        % for Gibbs sampling or Gibbs-like algorthm no adaptive MCMC is
        % needed
        mcmcoptions.adapt = [];
        %
end