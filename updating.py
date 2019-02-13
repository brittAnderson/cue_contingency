import numpy as np
import matplotlib.pyplot as pl
np.random.seed  = 112358


n_trials = 60       # How many trials to run
true_cue = 1        # 1 == Visual Cue is predictive; 2 == Audio Cue is predictive
tau = 0.9           # Probability that Visual Cue is telling the truth
free_tau = 0.77     # Model's value for tau
delta = 0.68        # Reversal probability
my_model = 1        # 1 == I think Visual Cue is predictive; 2 == I think Audio Cue is predictive
rev = np.array([[delta, 1 - delta], [1 - delta, delta]])    # Reversal probability matrix for fitting contingency switch beliefs

# A Cue is a list of 3 items:
#   a tuple with the string name of the cue pairing
#   the probability that you should get rewarded given the Visual Cue (V1 == good; V2 == bad)
#   the probability that you should get rewarded given the Audio Cue (A1 == good; A2 == bad)
cues = [[('V1A1'), tau, tau], [('V2A2'), 1-tau, 1-tau], [('V1A2'), tau, 1-tau], [('V2A1'), 1-tau, tau]] # List containing the different cue pairings

# Info metric stuff
def surprise(likelihood):
    ''' Return the surprisal given a likelihood. '''
    return -1 * np.log(likelihood)


def KLD(posterior, prior):
    ''' Return their definition of the Bayesian surprise; per Wikipedia this is the expected number of bits
        of useful information we gained from our observation during the update step. '''
    KLD = 0
    for i in range(len(posterior)):
        KLD += posterior[i] * np.log2(posterior[i] / prior[i])

    return KLD

# Bayes Updating
# This likelihood function is where there is need for tweaking
def L(cue, model, outcome, tau=tau):
    ''' Return the probability of an outcome given a cue pairing and your belief in which of the cues is predictive.
        model == 0 => Auditory
        model == 1 => Visual '''
    if (cue == 'V1A1'):
        if model:
            if outcome:
                return tau
            else:
                return 1 - tau
        else:
            if outcome:
                return 1 - tau
            else:
                return tau

    elif (cue == 'V1A2'):
        if model:
            if outcome:
                return tau
            else:
                return 1 - tau
        else:
            if outcome:
                return tau
            else:
                return 1 - tau

    elif (cue == 'V2A1'):
        if model:
            if outcome:
                return 1 - tau
            else:
                return tau
        else:
            if outcome:
                return 1 - tau
            else:
                return tau

    elif (cue == 'V2A2'):
        if model:
            if outcome:
                return 1 - tau
            else:
                return tau
        else:
            if outcome:
                return tau
            else:
                return 1 - tau


def E(posterior, reversal_matrix=rev):
    ''' Modify the posterior distribution following normal updating with the reversal probability matrix. '''
    return np.dot(posterior, reversal_matrix)


def Z(cue, outcome, prior, tau=tau):
    ''' Return the denominator for Bayes' theorem corresponding to a given cue pairing, model belief, and outcome.
        prior is a 2-item list such that sum(prior) == 1 and prior[0] is the prior that V is the true_cue. '''
    v_term = L(cue=cue, model=1, outcome=outcome, tau=tau)
    v_term *= prior[0]

    a_term = L(cue=cue, model=0, outcome=outcome, tau=tau)
    a_term *= prior[1]

    return v_term + a_term


# Simulating data sets
# record is a list of tuples containing (trial number, cue name, which cue is truly predictive on this trial, whether the predictive cue switched after last trial, outcome on that trial)
# this information is useful for inspecting things afterward
record = []


def det_outcome(cue, predictive_cue_index):
    ''' Return the outcome of a trial that respects the cue contingencies. '''
    reward_threshold = cue[predictive_cue_index]
    test_value = np.random.random() # min: 0 max: 0.9999... => if we want something 50% of the time then we want 0:0.4999...

    # Case: test value is more extreme than our threshold, implying that we don't give a reward.
    if (test_value > reward_threshold):
        return 0
    # Case: test value is equal to or less than our threshold, implying that we give a reward.
    else:
        return 1


def pick_a_cue(cues):
    ''' Return one of the cue types from a list of cues as follows:
        V1A1 and V2A2 occur half the time (uniform among themselves)
        V1A2 and V2A1 occur half the time (uniform among themselves) '''
    if (np.random.random() < 0.5):
        cues = cues[:2]
    else:
        cues = cues[2:]

    cue_index = np.random.randint(0, len(cues))
    return cues[cue_index]


def switch_true_cue(curr_cue, p_switch=0.1):
    ''' Switch the true cue stochastically; since we represent the different
        models as 1 and 2, subtracting the current cue from 3 will flip them. '''
    if (np.random.random() < p_switch):
        return [True, 3 - curr_cue]
    else:
        return [False, curr_cue]


def sim(n_trials=n_trials, cues=cues, true_cue=true_cue, switch=0.0):
    ''' Simulate n_trials of the protocol. '''
    record = []
    for i in range(n_trials):
        this_cue = pick_a_cue(cues)
        # Switch_cue_info is a list with the first element a boolean
        # indicating if we switched the predictive cue from the preceding
        # trial, and the second element the number representing the
        # the cue that is now the predictive cue
        # By predictive cue I mean the cue which was used to
        # determine if this trial gave a reward or not
        switch_cue_info = switch_true_cue(true_cue, switch)
        true_cue = switch_cue_info[1]
        this_outcome = det_outcome(this_cue, true_cue)
        record.append((i, this_cue, true_cue, switch_cue_info[0], this_outcome))
    return record

# Run two simulations assuming true_cue == 2 with switch rate of 0.1
record = sim(true_cue=2, switch=0.1)
record2 = sim(true_cue=2, switch=0.1)


# Final Piece for doing bayesian updating on the simulated data sets
def update_one_step(prior, likelihood, z, rev=False):
    ''' Return the result of a single Bayesian update step;
        rev = True will modify the posterior with the reversal
        probability matrix. '''
    result = (prior[0] * likelihood) / z
    result = np.array([result, 1 - result])
    if rev:
        result = E(result)
    return result


def bayes(record, free_tau=free_tau, init_prior=np.array([0.5, 0.5]), rev=False):
    ''' Optimal Bayesian updating with true parameters known.
        record is a list of simulated trial data from the sim(.) function;
        free_tau is the model's estimate of the visual cue validity;
        init_prior is the assumed starting prior over which of the cues is predictive;
        rev is a flag controlling whether we weight posteriors with the reversal matrix following updating.

        The result is a dictionary with keys === posteriors, surprisals, KLD,
        and values === lists of these quantities for each trial in the record. '''

    n_iter = len(record)
    posteriors = []
    surprisals = []
    KLDs = []
    my_model = 1
    for i in range(n_iter):
        if (i == 0):
            prior = init_prior
        else:
            prior = posteriors[i - 1]

        this_trial = record[i]          # Grab the next trial from the simulated data record
        this_cue = this_trial[1][0]     # Which cue pairing did participants see on this trial?
        this_outcome = this_trial[-1]   # What was our outcome on this trial?

        likelihood = L(cue=this_cue, model=my_model, outcome=this_outcome, tau=free_tau)
        z = Z(cue=this_cue, outcome=this_outcome, prior=prior, tau=free_tau)
        posterior = update_one_step(prior, likelihood, z, rev=rev)
        posteriors.append(posterior)
        # Record the info metrics as well
        surprisal = surprise(likelihood)
        kld = KLD(posterior, prior)
        surprisals.append(surprisal)
        KLDs.append(kld)

    return {'posteriors': posteriors, 'surprisal': surprisals, 'KLD': KLDs}


# Stuff for plotting/visualization
# Just run a few bayes simulations with a record of simulated trial data
# and vary init_prior and whether we use the reversal matrix to
# get a feel for their impact on the process
rec1 = bayes(record, rev=False)
rec2 = bayes(record, init_prior=np.array([0.25, 0.75]), rev=False)
rec3 = bayes(record, rev=True)
rec4 = bayes(record, init_prior=np.array([0.25, 0.75]), rev=True)

# Produce values for the x-axis in our plots (1 for each trial)
x = np.arange(0, len(rec1['posteriors']), 1)

# Just a kludge to bridge the gap from bayes output dictionary
# to plot function
def prep_y(bayes_output, key, label=''):
    ''' Extract a part of a bayes simulation output for plotting. '''
    y = [label]
    if (key=='posteriors'):
        y = y + [a[0] for a in bayes_output['posteriors']]
    elif (key in ('surprisal', 'KLD')):
        y = y + bayes_output[key]

    return y


# Prep the y-axis additions with labels and the keys to get the info
# from the dictionary bayes output
y0 = prep_y(rec1, 'posteriors', 'P(Vision is Cue)')
y1 = prep_y(rec1, 'surprisal', 'Surprisal')
y2 = prep_y(rec1, 'KLD', 'KLD')

y3 = prep_y(rec3, 'posteriors', 'P(Vision is Cue')
y4 = prep_y(rec3, 'surprisal', 'Surprisal')
y5 = prep_y(rec3, 'KLD', 'KLD')


# Grab the trial numbers where we switched from the record
# data so we can overlay vertical lines on the plot on these
# trials
def rec_switches(record):
    ''' Return a list of trial numbers where the true cue switched over. '''
    switch_trials = []
    for rec in record:
        if (rec[-2] == True):
            switch_trials.append(rec[0])
    return switch_trials

# Grab the vertical line indices (trials when the predictive cue switched)
VL = rec_switches(record)


# Example use of show_data:
# show_data([y0, y1, y2], vlines=VL) would plot
# the posteriors, surprisals, and KLDS for each trial
# in the simulated data set record which we used to
# run bayes(.) to generate rec1, with VL showing
# the trials we switched as vertical lines on the plot
# (see: y0 definitions above).
def show_data(y_list, vlines=[], x=x):
    ''' Quick plotting tool. '''
    for y in y_list:
        # Kludge so we can pass labels for data points as
        # parts of the same list of values we want to plot
        if type(y[0])==str:
            pl.plot(x, y[1:], label=y[0])
        else:
            pl.plot(x, y)
    # Put the vertical lines on if we have any
    for vline in vlines:
        pl.axvline(vline, color='black')
    # Add a legend
    pl.legend()

    # detect the appropriate limits for the y-axis on the
    # basis of values in the list we actually wish to plot
    # i.e., ignoring any labels we passed in
    ymax = 1.0
    ymin = 0.0
    for y in y_list:
        no_str_y = list(filter(lambda yi: type(yi)!=str, y))
        ymax = max(ymax, max(no_str_y))
        ymin = min(ymin, min(no_str_y))
    pl.ylim(ymin, ymax)
    # Show the plot
    pl.show()







