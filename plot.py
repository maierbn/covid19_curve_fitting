#!/usr/bin/python3
from datetime import *
import scipy.optimize
import numpy as np

# data from RKI Lagebericht for Germany and Baden-Württemberg
data = [(date(2021,3,22),107,103),
        (date(2021,3,21),104,100),
        (date(2021,3,20),100,92),
        (date(2021,3,19),96,89),
        (date(2021,3,18),90,85),
        (date(2021,3,17),86,80),
        (date(2021,3,16),84,75),
        (date(2021,3,15),83,76),
        (date(2021,3,14),79,74),
        (date(2021,3,13),76,69),
        (date(2021,3,12),72,67),
        (date(2021,3,11),69,63),
        (date(2021,3,10),65,61),
        (date(2021,3,9),68,60),
        (date(2021,3,8),68,60),
        (date(2021,3,6),66,57),
        (date(2021,3,5),65,56),
        (date(2021,3,4),65,54),
        (date(2021,3,3),64,52),
        (date(2021,3,2),65,52),
        (date(2021,3,1),66,52),
        (date(2021,2,28),64,50),
        (date(2021,2,21),60,44),
        (date(2021,2,14),57,49),
        (date(2021,2,7),76,61),
        (date(2021,1,31),90,74),
        (date(2021,1,24),111,90),
        (date(2021,1,17),136,111),
        (date(2021,1,10),162,139)]

# fix a particular date as starting point for the model
date_start = date(2021,1,10)
t_start = date_start.toordinal()

# extract values
date_list = [d for d,_,_ in data]
t_list    = np.array([d.toordinal() - t_start for d in date_list])
yd_list   = np.array([yd for _,yd,_ in data])      # Inzidenz Deutschland
ybw_list  = np.array([ybw for _,_,ybw in data])    # Inzidenz Baden-Württemberg

# model
# incidence of B.1.1.7
def curve1(t, a,b,c,d):
  return a*np.exp(b*t)

# incidence of other type
def curve2(t, a,b,c,d):
  return c*np.exp(d*t)
  
# total incidence is the sum of both
def model(t, a,b,c,d):
  return curve1(t, a,b,c,d) + curve2(t, a,b,c,d)
  
# differentiation w.r.t parameters
def model_diff(t, a,b,c,d):
  return np.array([np.exp(b*t), a*np.exp(b*t)*t, np.exp(d*t), c*np.exp(d*t)*t]).T
  
# manually tuned initial guess
initial_guess_d = [yd_list[-1],-4e-2,3,5e-2]

# optimize parameters
parameters_d, covariance_d = scipy.optimize.curve_fit(model, t_list, yd_list, initial_guess_d, jac=model_diff, maxfev=1000)
parameters_bw, covariance_bw = scipy.optimize.curve_fit(model, t_list, ybw_list, initial_guess_d, jac=model_diff, maxfev=1000)

error_d = np.sqrt(np.diag(covariance_d))
error_bw = np.sqrt(np.diag(covariance_bw))

print("D  parameters: {}, error: {}".format(parameters_d, error_d))
print("BW parameters: {}, error: {}".format(parameters_bw, error_bw))

# create plots
import matplotlib.pyplot as plt

# plot everything for Germany
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6), gridspec_kw={'height_ratios': [3,1]})
fig.suptitle("Deutschland")

# plot data points
ax1.plot(date_list, yd_list, "bo-", label="data")

# prepare interpolated dates and times
t_list_extrapolate = np.array(list(range(min(t_list),max(t_list)+13)))
t_list_extrapolate.sort()
date_list_extrapolate = [date.fromordinal(t+t_start) for t in t_list_extrapolate]

# plot predicted curves
ax1.plot(date_list_extrapolate, model(t_list_extrapolate, *parameters_d), "+--", label="predicted")
ax1.plot(date_list_extrapolate, curve2(t_list_extrapolate, *parameters_d), "r:", label="B.1.1.7 type")
ax1.plot(date_list_extrapolate, curve1(t_list_extrapolate, *parameters_d), "g:", label="other type")

# plot confidence interval
ax1.fill_between(date_list_extrapolate, 
  model(t_list_extrapolate, *(parameters_d+error_d)),
  model(t_list_extrapolate, *(parameters_d-error_d)),alpha=0.2)

ax1.legend(loc="lower left")
ax1.xaxis_date()
ax1.autoscale_view()
ax1.grid(which="both")
ax1.set_ylabel("7-Tage Inzidenz\n pro 100.000 EW")

# plot portion of curve1 and curve2
ax2.plot(t_list_extrapolate, 
         curve2(t_list_extrapolate, *parameters_d) / model(t_list_extrapolate, *parameters_d), 
         "r-", label="portion of B.1.1.7")
ax2.set_ylim([0,1])
ax2.legend()
ax2.grid()
plt.savefig("corona_d.png")


# -----------------------------
# plot everything for Baden-Württemberg
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios': [3,1]})
fig.suptitle("Baden-Württemberg")

# plot data points
ax1.plot(date_list, ybw_list, "bo-", label="data")

# plot predicted curves
ax1.plot(date_list_extrapolate, model(t_list_extrapolate, *parameters_bw), "+--", label="predicted")
ax1.plot(date_list_extrapolate, curve2(t_list_extrapolate, *parameters_bw), "r:", label="B.1.1.7 type")
ax1.plot(date_list_extrapolate, curve1(t_list_extrapolate, *parameters_bw), "g:", label="other type")

# plot confidence interval
ax1.fill_between(date_list_extrapolate, 
  model(t_list_extrapolate, *(parameters_bw+error_bw)),
  model(t_list_extrapolate, *(parameters_bw-error_bw)),alpha=0.2)

#ax.plot(date_list_extrapolate, model(t_list_extrapolate, *parameters_bw), "m+--", label="BW predicted")

ax1.legend(loc="lower left")
ax1.xaxis_date()
ax1.autoscale_view()
ax1.grid(which="both")
ax1.set_ylabel("7-Tage Inzidenz\n pro 100.000 EW")

# plot portion of curve1 and curve2
ax2.plot(t_list_extrapolate, 
         curve2(t_list_extrapolate, *parameters_bw) / model(t_list_extrapolate, *parameters_bw), 
         "r-", label="portion of B.1.1.7")
ax2.set_ylim([0,1])
ax2.legend()
ax2.grid()
plt.savefig("corona_bw.png")

print("prediction for D  at 4.4: {}".format(model(date(2021,4,4).toordinal() - t_start, *parameters_d)))
print("prediction for BW at 4.4: {}".format(model(date(2021,4,4).toordinal() - t_start, *parameters_bw)))

print("\nThe model as one-liner:\n")
print("""python -c 'import datetime, sys, numpy as np;tstart=datetime.date.today().toordinal()-737800;[sys.stdout.write("{}: {:.0f}\\n".format(datetime.date.fromordinal(737800+t), 162.638295*np.exp(-3.07381933e-02*t)+1.40679245*np.exp(5.83740594e-02*t))) for t in range(tstart-1,tstart+20)]'""")

plt.show()
