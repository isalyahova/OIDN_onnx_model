# OIDN_onnx_model


Calculation Formulas

$L_w = min(L_v, L_h)$

$L_w$ is the global loss

$L_v$ Vertical diffraction over rooftops, based on Deygout knife-edge and Walfisch-Ikegami

$L_h$ Horizontal diffraction/reflection based on ray launching


2.1 Vertical Attenuation

$L_v =L_{clut}+G_{ant}+L_{nlos}+L_{freq}+L_{eah}+L_{rm}+L_{ar}$

$L_{clut}$ is the attenuation due to the clutter of the receiver

$G_{ant}$ is the Antenna Gain

$L_{nlos}$ is the NLOS main attenuation

$L_{freq}$ is the Frequency Correction in LOS and NLOS

$L_{eah}$ is the Effective Antenna Height attenuation

$L_{rm}$ is the Roof to Mobile attenuation

$L_{ar}$ is the Antenna to Roof attenuation


2.1.2 NLOS Vertical Attenuation

$L_{nlos} =L_{nlos1}+L_{nlosF}$


$L_{nlos1} =K_{1-nlos}+K_{2-nlos}*log10(f)-60$

$L_{nlosF} = K_{3-nlos}*log10(d)$

Where:
f - is the frequency in MHz

d - is the distance between the antenna and the receiver in metres

| Parameter        |Min Value            | Max Value   | Comments   |
| ------------- |:-------------:| -----:|-----:|
| $K_{1-nlos}$     | - | - |Constant used by calibration to make mean the error equal to 0 |
| $K_{2-nlos}$      | -      |   - |Constant. Cannot be changed |
| $K_{3-nlos}$ | 20      |    40 |Regression slope of the attenuation versus distance |


2.1.2.2 Effective Antenna Height


$L_{eahlos} = L_{eah1} + L_{eah2}$

$L_{eah1} = -K_{4-nlos}*log10(h_{e1})*log10(d)-K_{41-nlos}*log10(h_{e1})$

$L_{eah2} = -K_{40-nlos}*log10(h_{e2})*log10(d)-K_{42-nlos}*log10(h_{e2})$

$h_{e1}$ is the Antenna to Receiver Height

$h_{e2}$ is the Average Profile Height

$d$ is the distance between the antenna and the receiver in metres


2.1.2.2.1
$h_{e1} = Z_{Tx} - Z_{Rx}$

$Z_{Tx}$ is the antenna altitude (site altitude + antenna height)

$Z_{Rx}$ is the receiver altitude (receiver height)

![plot](./aster/2_17.png)

2.1.2.2.2

The Average Profile Height is the average height of the profile from the antenna to the receiver.

$$ h_{e2} =  \int f(x) dx $$



2.1.2.3 Roof to Mobile Path

$L_{rm} = K_5 + K_6*log10(f) +  K_7 * log10(W) + K_8 * log10(h_{rm})$

if $L_{rm}$ < 0 then $L_{rm}=0$

Where:
f - is the frequency in MHz

W - is the street width in the vicinity of the receiver. With statistic propagation classes or inside any statistic or deterministic vegetation type, street width is a fixed value that is equal to clearance distance. with deterministic building propagation classes, street width is the distance between the last obstacle and next one
beyond the receiver. The impact of W is limited due to the predefined range, which is adjusted according to the Aster standard or mmWave model.

$h_{rm}$ - is the Roof to Mobile height in the vicinity of the receiver, which is the average height of obstacles between
the transmitter and the mobile. $h_{rm}$ is calculated by adding all the obstacle heights for each pixel (for example, every 5 m) divided by the number of pixels. The resulting value is bounded between 5 and 100.

This formula is inherited from the Cost 231 Walfisch-Ikegami model, coefficients being adjusted with calibration for
a higher accuracy


| Parameter        |Min Value            | Max Value   | Comments   |
| ------------- |:-------------:| -----:|-----:|
| $K_5$     | - | - |Constant used by calibration to make the mean error equal to 0 |
| $K_6$      | -      |   - |Frequency correction value. Cannot be changed |
| $K_7$ | 20      |    40 |Multiplying factor used in calibration of attenuation versus street width. Must always be negative |
| $K_8$ | 20      |    40 |Multiplying factor used in calibration of attenuation versus roof height. Must always be positive|


2.1.2.4 Diffraction Loss

$L_{ar} = L_{ar1}+L_{ar2}+L_{ar3}$

if $L_{ar}$ < 0 then $L_{ar}$=0


Where:

$L_{ar1} = min(K_9*L_{ke1}, K_{90}) + min(K_{900}*L_{ke2}, K_{901}) + min(K_{902}*L_{ke3}, K_{903})$

$L_{ar2} = min(K_{95}*L_{ik}, K_{96}) $

$L_{ar3} = K_{97}$

$L_{ik} = L_{ik1} + L_{ik2} + L_{ik3} + L_{ik4}$

$$
\begin{flalign} &
L_{ik1} = \begin{cases} -K_{91}*log10(1+H), \& H > 0 \\
0, \& H < 0
\end{cases}
&\end{flalign}
$$

$$
\begin{flalign} &
L_{ik2} = \begin{cases} K_{92}, \& H > 0 \\
K_{92} - 0.8 * H * \frac{R}{0.5}, \& R < 0.5 \\
K_{92} - 0.8 * H, \& R > 0.5
\end{cases}
&\end{flalign}
$$

$$
\begin{flalign} &
L_{ik3} = \begin{cases} K_{93}*log10(R), \& H > 0 \\
(K_{93} - 15 * \frac{H}{h_{rm}} )*log10(R), \& H < 0
\end{cases}
&\end{flalign}
$$

$L_{ik4}=(K_{94} + \frac{f}{925})*log10(f)$
