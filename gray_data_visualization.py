import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import glob 
import scipy.io as scio
import scipy.signal
import os

nmonkey = 0
nses    = 0
nevt    = 0
ntype   = 0

#--------------------------------------------------------------------------
# Directories
#--------------------------------------------------------------------------
dirs = {'rawdata':'GrayLab/',
		'results':'Results/',
		'monkey' :['lucy', 'ethyl'],
		'session':'session01',
		'date'   :[['150128'], [],]}

#--------------------------------------------------------------------------
# Create session dicitionary
#--------------------------------------------------------------------------
session = {'dir'        :dirs['rawdata']+dirs['monkey'][nmonkey]+'/'+dirs['date'][nmonkey][nses]+'/'+str(dirs['session']+'/'),
		   'dir_out'    :dirs['results']+dirs['monkey'][nmonkey]+'/'+dirs['date'][nmonkey][nses]+'/'+str(dirs['session']+'/'),
		   'fname_base' :dirs['date'][nmonkey][nses],
		   'evt_names'  :['samplecor','sampleinc','samplecorinc'],
		   'evt_trinfo' :['sample_on','match_on'],
		   'evt_dt'     :[[ -0.65,1.75 ],[ -0.8,2.5 ]],
		   'fc'         :np.arange(6, 62, 2),
		   'df'         :4,
		   'dt'         :250,
		   'step'       :25,}
# Create out folder
os.makedirs(session['dir_out'])

#--------------------------------------------------------------------------
# Recording and trials info dicitionaries
#--------------------------------------------------------------------------
info = ['/recording_info.mat', '/trial_info.mat']
ri = scio.loadmat(session['dir']+info[0])['recording_info']
ti = h5py.File(session['dir']+info[1])['trial_info']
recording_info = {'channel_count': ri['channel_count'].astype(int)[0][0],
				  'channel_numbers':ri['channel_numbers'][0,0][0],
				  'fsample': ri['lfp_sampling_rate'].astype(int)[0][0],
				  'ms_mod': ri['ms_mod'][0,0][0],                        #
				  'slvr': ri['slvr'][0,0][0],}                           #
trial_info     = {'num_trials': int(ti['num_trials'][0,0]),
				  'trial_type': ti['trial_type'][:].T[0],
				  'behavioral_response': ti['behavioral_response'][:].T[0],
				  'sample_image': ti['sample_image'][:].T[0],
				  'nonmatch_image': ti['nonmatch_image'][:].T[0],
				  'match_image': ti['match_image'][:].T[0],
				  'reaction_time': ti['reaction_time'][:].T[0],
				  'sample_on': ti['sample_on'][:].T[0], #1th image is shown
				  'match_on': ti['match_on'][:].T[0],   #2nd image is shown
				  'sample_off': ti['sample_off'][:].T[0],}

# Find zero time wrt to selected event
t0 = trial_info[session['evt_trinfo'][nevt]]
#--------------------------------------------------------------------------
# Reading data
#--------------------------------------------------------------------------
files   = sorted(glob.glob(session['dir']+'/'+dirs['date'][nmonkey][nses]+'*'))
'''
# Reading data from
data = {}
for i in range(1, trial_info['num_trials']+1): 
	try:
		print i
		data[str(i)] = h5py.File(files[i])
	except:
		continue
'''

# Get only LFP channels does not contain slvr and ms_mod
indch  = (recording_info['slvr'] == 0) & (recording_info['ms_mod'] == 0)
indch  = np.arange(1, recording_info['channel_count']+1, 1)[indch]
# Trial type
if  session['evt_names'][ntype] == 'samplecor':
	# Use only completed trials and correct
	indt = (trial_info['trial_type'] == 1) & (trial_info['behavioral_response'] == 1)
	indt = np.arange(1, trial_info['num_trials']+1, 1)[indt]
elif session['evt_names'][ntype] == 'sampleinc':
	# Use only completed trials and incorrect
	indt = (trial_info['trial_type'] == 1) & (trial_info['behavioral_response'] == 0)
	indt = np.arange(1, trial_info['num_trials']+1, 1)[indt]
elif session['evt_names'][ntype] == 'samplecorinc':
	# Use all completed correct and incorrect trials 
	indt = (trial_info['trial_type'] == 1) 
	indt = np.arange(1, trial_info['num_trials']+1, 1)[indt]

# Recode choices, i.e. find which oculomotor choice was performed
choice = np.nan*np.ones(trial_info['sample_image'].shape[0])
# Incorrect response means the monkey chose the nonmatch image
ind = trial_info['behavioral_response'] == 0
choice[ind] = trial_info['nonmatch_image'][ind];
# Correct response means the monkey chose the match image
ind = trial_info['behavioral_response'] == 1
choice[ind] = trial_info['match_image'][ind];
trial_info['choice'] = choice;

# Loop over trials
i = 1
data_prep = {'trial':[], 'time':[], 'fsample':recording_info['fsample'], 'trial_info': np.zeros(5)}
delta_t   = 1.0 / data_prep['fsample']
for nt in indt:
	lfp_data = np.transpose( h5py.File(files[nt-1])['lfp_data'] )
	nch, ns  = lfp_data.shape
	indb = int(t0[nt-1] + 1000*session['evt_dt'][nevt][0])
	inde = int(t0[nt-1] + 1000*session['evt_dt'][nevt][1])
	# Find time index
	ind = np.arange(indb, inde+1).astype(int)
	# cell-array containing a time axis for each trial (1 X Ntrial), each time axis is a 1 X Nsamples vector
	data_prep['trial'].append( lfp_data[indch-1, indb:inde+1] )
	del lfp_data
	# cell-array containing a time axis for each trial (1 X Ntrial), each time axis is a 1 X Nsamples vector
	data_prep['time'].append( np.arange(session['evt_dt'][nevt][0], session['evt_dt'][nevt][1]+delta_t, delta_t) ) 
	# Keep track of [ real trial number, sample image, choice, outcome (correct or incorrect), reaction time ]
	info_aux = np.array([nt, trial_info['sample_image'][nt-1], trial_info['choice'][nt-1], trial_info['behavioral_response'][nt-1], trial_info['reaction_time'][nt-1]/1000.0])
	data_prep['trial_info'] = np.vstack((data_prep['trial_info'], info_aux))
	np.savetxt(session['dir_out']+'trial'+str(i)+'.dat', data_prep['trial'][i-1])
	i = i + 1

labels = np.zeros(len(indch))
for nc in range(len(indch)):
	labels = np.append(labels,  recording_info['channel_numbers'][indch[nc]-1] )

# Parameters
N = len(labels)
j, i = np.where( np.tril(np.ones(N), -1) ); pairs = np.array([i, j]).T
nP = pairs.shape[0]
nT = data_prep['trial_info'].shape[0]
data_prep['label'] = np.array( labels )


'''
ch = [0, 10, 37, 45]
trial = ['1', '250', '470', '540']
i = 1
for t in trial:
	for c in ch:
		plt.subplot(2,2,i)
		plt.title('Trial = ' + t)
		plt.plot(data_prep['time'][t], data_prep['trial'][t][c,:])
	i+=1
plt.legend(['ch = 0', 'ch = 10', 'ch = 37', 'ch = 45'])
plt.tight_layout()
'''


'''
plt.plot(data['0']['lfp_data'][:, 0])
plt.plot(data['0']['lfp_data'][:, 1])
plt.plot(data['0']['lfp_data'][:, 2])
plt.plot(data['0']['lfp_data'][:, 3])
'''
