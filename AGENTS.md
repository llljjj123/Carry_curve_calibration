context: read `summary_for_AI.md` for the summary of the project. This time we focus on solving `delta`. Read `Delta_hedging_explained.md` for detailed knowledge on the delta hedging of exposure to both fast and slow factors.

goal: Using 2 futures contracts as the hedging instruments, we can use `IM2609` and `IM2703` as example. Calculate the corresponding hedging ratio for the 2 futures. Update this hedge ratio to `carry_put_pricing` and `Demo`. Make the choice of the 2 futures as a input, check for singularity of the hedging matrix. If singular output an warning and don't calculate the corresponding delta, if not, calculate delta.

todo: 1. Read files mentioned above, report back to me, no coding at this stage.
      2. I have some question prepared in `question.md`, answer these question.
      3. update codes in `carry_put_pricing` and `Demo`.

