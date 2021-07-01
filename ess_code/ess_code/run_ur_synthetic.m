function run_ur_synthetic()
% Run the underrelaxed on the synthetic data.

experiment_setup()

% Structure containing precomputed data for all synthetic experiments:
setup = setup_synthetic();

for num_dims = 1:setup.max_dim
    name = sprintf('ur_synthetic_%dd', num_dims);
    fn = @(run) synthetic_run(setup, num_dims, run);
    success = experiment_run(name, setup.runs, fn, true);
end


function ur_results = synthetic_run(setup, num_dims, run)

step_size = 0.05;

UNPACK_STRUCT(setup);

counting_log_like_fn = add_call_counter(@(x) -0.5*numel(x)*log(2*pi*noise_variance) ...
  - (0.5/noise_variance)*sum((data{num_dims}.obs_f(:)-x(:)).^2));

ff           = data{num_dims}.init_f;
cur_log_like = counting_log_like_fn(ff);
idx          = 0;
samples      = zeros([num_data iterations]);
loglikes     = zeros([1 iterations]);
num_calls    = zeros([1 iterations + burn]);
tic;
for ii = (1-burn):iterations
  if mod(ii,100) == 0
    fprintf('Run %03d/%3d %02d-d Iter %05d / %05d\r', run, runs, num_dims, ii, iterations);
  end

  [ff, cur_log_like] = gppu_underrelax(step_size, ff, data{num_dims}.chol_cov, counting_log_like_fn, cur_log_like);

  num_calls(ii + burn) = counting_log_like_fn();

  if ii > 0
    samples(:, ii) = ff;
    loglikes(ii)   = cur_log_like;
  end
end
elapsed = toc;

ur_results.num_calls         = num_calls;
ur_results.loglikes          = loglikes;
ur_results.elapsed           = elapsed;
ur_results.effective_samples = effective_size_rcoda(loglikes(:));
fprintf('Run %03d/%3d %02d-d Effective Samples: %0.2f  %0.2f secs\n\n', ...
  run, runs, num_dims, ur_results.effective_samples, elapsed);

