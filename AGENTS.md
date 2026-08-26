Context: We proposed and calibrated 3 models, i.e. 1-factor, 2-factor OU process and 1-factor OU process with correlation to underlying spot for chinese stock index csi1000(000852) carry curve $C_t$ calibration and also proposed a pricing scheme for a carry-put option. See `session_log.md`. 

Goal: We have the pieces ready. Next we generate a demo using the option pricing engine and 2-factor OU process.

Todo: Read `session_log.md` first for the background. Assume the evaluation date is 8/21/2026.Calibrate the 2-factor OU process using the past 244 trading days' data. Calculate the carry-put option price. For 2-factor OU process, use parameters from calibration, for underlying stock process, use historical volatility and risk-free rate $r = 0.014$. Use `IM2609` as the underlying futures contract, hence we can identify the carry-put strike $q_{0,T}$

Generate everything into a new folder `Demo`. Generate the demostration code into a `.ipynb` for readability and demostration. The calibration/pricing functions and implementations can be stored in seprate `.py` files so that the demostration notebook won't look too messy. 

Do not jump right into coding, read the above file and report back to me. Let me check that we are on the common ground then goto coding.

You can use the python environment `spyder-env`. Try to stick to existing packages but in case you do need to install any new package, use `conda install`. **DO NOT** use anaconda official channel, you can use `conda-forge` channel.