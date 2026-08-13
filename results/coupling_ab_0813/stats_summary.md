# coupling_ab 順序検定 (t50 = 半立ち上がり時刻, seed bootstrap)

pair X<Y: 仕様の因果順で X が Y に先行するか。p_order = P(lag>0), lag = t50_Y - t50_X (step)。n は両指標が発現した seed 数。

exp  width   c                pair  n  p_order       lag_mean    lag_lo    lag_hi
  A      5 NaN srank_drop<post_err  5  1.00000  127380.000000   27600.0  233020.0
  A      5 NaN        post_err<trC  3  1.00000   61700.000000   10300.0  152000.0
  A      5 NaN            trC<dead  3  0.74200   21866.666667 -171300.0  119500.0
  A    100 NaN srank_drop<post_err  5  0.99275  197920.000000   41680.0  310740.0
  A    100 NaN        post_err<trC  5  0.00075 -273700.000000 -391700.0 -119700.0
  A    100 NaN            trC<dead  5  0.89900   26140.000000  -14100.0   64400.0
  B      5 0.0 srank_drop<post_err  5  1.00000  118400.000000   50620.0  196020.0
  B      5 0.0        post_err<trC  5  0.00000 -119740.000000 -181740.0  -61700.0
  B      5 0.0            trC<dead  0      NaN            NaN       NaN       NaN
  B      5 2.0 srank_drop<post_err  5  1.00000  133740.000000  105320.0  154340.0
  B      5 2.0        post_err<trC  5  0.00375 -117400.000000 -173140.0  -33640.0
  B      5 2.0            trC<dead  0      NaN            NaN       NaN       NaN
  B    100 0.0 srank_drop<post_err  5  0.38925  -19640.000000 -191020.0  136680.0
  B    100 0.0        post_err<trC  1      NaN            NaN       NaN       NaN
  B    100 0.0            trC<dead  0      NaN            NaN       NaN       NaN
  B    100 2.0 srank_drop<post_err  5  1.00000  235720.000000  133000.0  340920.0
  B    100 2.0        post_err<trC  0      NaN            NaN       NaN       NaN
  B    100 2.0            trC<dead  0      NaN            NaN       NaN       NaN
