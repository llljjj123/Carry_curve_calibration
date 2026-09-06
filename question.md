# 1st question

When using $1$ futures instead of $2$ or more in hedging, we have the following:

$$
\Delta_s^F
=
\frac{\partial V/\partial x_s}{\partial F/\partial x_s},
\qquad
\Delta_f^F
=
\frac{\partial V/\partial x_f}{\partial F/\partial x_f}.
$$

Consider we totally hedge $\Delta_s^F$ using underlying $F$. Does that mean we have locally eliminated slow factor impact while only partially offset(or maybe overly offest) the fast factor?

# 2nd question

The carry put option is based on $1$ single futures option, let's call it $F_0$. But if we hedge the option using $2$ or more options, $F_0,F_1,...,F_n$, we would be introducing an extra mechanism on the relation between these futures contracts. If we believe the 2-factor ou model governs the entire mechanism of the carry $q$, then everything is fine. But in reality the model can't explain all the mechanisms between the index and all the futures contracts. So is it wise to introduce more futures, or should we just stick to $1$ futures. Or maybe we should do back-testings on this matter?