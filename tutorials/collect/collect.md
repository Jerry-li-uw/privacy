
#### DPSGD on Census (KDD)

1 layer, 64 neurals, $C=4$, $\sigma=1$

1 test run (each)

![1](auc_test_kdd.png)

![1](auc_train_kdd.png)

average over 5 runs (each)

![1](auc_avg.png)

#### averaging gradient norms on Census (no privacy)

$C=\text{average of norms of previous epoch}$

1 test run

![1](avr_20.png)

#### AdaClip on normal MNIST 

$\epsilon = 1, \sigma=10^{-5}, 100 neurals$

DPSGD

![1](adaclip_def.png)

AdaClip

![1](adaclip_new.png)

(no much difference)

#### Next

- AdaClip on Census
