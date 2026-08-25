Context: We proposed and calibrated 3 models, i.e. 1-factor, 2-factor OU process and 1-factor OU process with correlation to underlying spot for chinese stock index csi1000(000852) carry curve $C_t$ calibration. See `session_log.md`. 

Goal: We goto a practical problem explained in `put_on_carry.md`. The goal is to come out with pricing code for such an American put.

Todo: Read the 2 mentioned markdowns above. Consider what scheme we should use for the put pricing. I proposed a binomial tree, but I am not 100% true is feasibility. You may propose new model. The parameters for 2-factor OU and GBM should be the input of the pricing function. As for the example call, for 2-factor OU parameters, used the one already calibrated inside folder `im_2factor_ou_carry/outputs` and for GBM of the stock price, use risk-free rate $r = 0.014$ and volatility $\sigma = 0.25$. If we need any more parameters, check with me first. 

Do not jump right into coding, read the above files and codes and report back to me. Let me check that we are on the common ground then goto coding.

You can use the python environment `GUOYUAN`. Try to stick to existing packages but in case you do need to install any new package, use `conda install`.