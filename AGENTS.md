Context: 1-factor and 2-factor OU process for chinese stock index csi1000(000852) carry curve $C_t$ calibration. See `session_log.md`. 

Goal: I have proposed a new model adding correlation between the carry curve $C_t$ and the stock price $S_t$, see `csi1000_im_correlated_ou_codex_prompt.md`

Todo: Read the 2 mentioned markdowns above and existing code in both `im_ou_carry` for the 1-factor OU process calibration scheme. Apply the newly proposed correlated OU scheme and generate results into new folder `im_corr_ou_1factor`

Do not jump right into coding, read the above files and codes and report back to me. Let me check that we are on the common ground then goto coding.

You can use the python environment `GUOYUAN`. Try to stick to existing packages but in case you do need to install any new package, use `conda install`.