# Understanding metrics

## Preliminaries

For every request, we define the following key metrics:

1. Request arrival time ($a_r$): The time at which a request enters the system.
2. Request schedule time ($s_r$): The time at which a given request is scheduled for the first time (irrespective of subsequent restarts).
3. Request completion time ($c_r$): The time at which a request completes.
4. Request prefill completion time ($f_r$): The time at which prefill completes and the first output token is produced.
5. Request execution time ($e_r$): The total amount of time a request spends executing on GPUs (across all attempts) - excluding the time request is allocated on a replica but not executing due to pipeline bubbles, stalled decodes in vLLM scheduler etc.
6. Request preemption time ($p_r$): The total amount of time a request spends request is allocated on a replica but not executing due to pipeline bubbles, scheduling preemptions, the time between restarts, etc (aggregate across all attempts).
7. Request scheduling delay ($d_r$): the total amount for which the request is waiting before getting scheduled ($s_r - a_r$).

Note that arrival, schedule and completion time refer to a specific point in time, whereas, execution time, preemption time, and scheduling delay refer to a period.

## Logged Metrics

### From the perspective of requests

1. `request_inter_arrival_delay_histogram`: Histogram of difference between arrival times of adjacent requests ($a_{r+1} - a_r$).
1. `request_num_tokens_histogram`: Histogram of number of tokens (prefill + decode) across all requests.
1. `request_num_restarts_histogram`: Histogram of number of restarts for a given request. Note that this is expected to be a non-zero entity only when using schedulers that use dynamic KV cache allocation - which restart requests in case a replica runs out of memory.
1. `request_e2e_time_cdf`: CDF of end-to-end request latency ($c_r - a_r$).
1. `request_e2e_time_normalised_cdf`: CDF of end-to-end request latency normalized by the number of output tokens.
1. `request_execution_plus_preemption_times_cdf`: CDF of the total time each request spends in the system excluding its initial scheduling delay ($c_r - s_r$).
1. `request_scheduling_delay_cdf`: CDF of request scheduling delay ($s_r - a_r$).
1. `request_execution_time_cdf`: CDF of request execution time ($e_r$).
1. `request_preempted_time_cdf`: CDF of request preemption time ($p_r$).
1. `decode_token_execution_plus_preemption_times`: CDF of per decode token execution time and preemption time - i.e. inter-token delay observed by the user.
1. `request_arrivals_time_series`: Time series of request arrival timestamps.
1. `request_completions_time_series`: Time series of request completion times - this provides an indicator for makespan and helps in identifying the request processing rate (requests per second) by analyzing the slope of the curve.

### From the perspective of prefills and decodes

1. `prefill_time_e2e_cdf`: CDF of end-to-end latency to the first output token (time-to-first-byte), i.e, the time elapsed since the request arrival to the point where first output is generated ($f_r - a_r$).
1. `prefill_time_execution_plus_preemption_cdf`: CDF of total prefill process time excluding the initial scheduling delay ($f_r - s_r$). This metric is useful for tracking the prefill efficiency.
1. `prefill_time_execution_plus_preemption_normalized_cdf`: Similar to `prefill_time_execution_plus_preemption_cdf`, but normalized by the number of prefill tokens. This provides distribution independent of request prefill length, and thus, is easier to analyze.
1. `decode_time_execution_plus_preemption_normalized_cdf`: CDF of total time spent processing decodes ($c_r - f_r$) normalized by the number of decode tokens. This provides an indicator similar to `decode_token_execution_plus_preemption_times`, however, this metric presents an average over all decode tokens in the request.
1. `prefill_completions_time_series`: Time series of prefill token completion times - helps in identifying the prefill processing rate (prefill tokens per second) by analyzing the slope of the curve.
1. `decode_completions_time_series`: Time series of decode completion times - helps in identifying the decode processing rate (decode tokens per second) by analyzing the slope of the curve.

### From the perspective of batches

1. `batch_num_tokens_cdf`: CDF of the total number of tokens to be processed in a batch (sum of prefill tokens + one per decode request). This distribution is useful for understanding how the compute load is distributed across batches. Note that with iteration level scheduling a batch is formed at every iteration.
1. `batch_size_cdf`: CDF of batch sizes - usually larger batch sizes imply higher throughput.
