#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import neurogym as ngym
from neurogym import spaces


class FindSensePairs(ngym.TrialEnv):
    r"""An altered version of --- Delayed paired-association task.

    The agent is shown a pair of two stimuli separated by a delay period. For
    half of the stimuli-pairs shown, the agent should choose the Go response.
    The agent is rewarded if it chose the Go response correctly.
    """

    metadata = {
        'paper_link': 'https://elifesciences.org/articles/43191',
        'paper_name': 'Active information maintenance in working memory' +
        ' by a sensory cortex',
        'tags': ['perceptual', 'working memory', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, N=4, interpolate_delay=2):
        super().__init__(dt=dt)
        self.choices = [0, 1]
        # trial conditions
        self.pairs = []
        # Give an error if N isn't even
        assert N % 2 == 0, 'N must be even'

        self.N = N
        self.N_div2 = int(N/2)
        for i in range(self.N_div2):
            for j in range(self.N_div2):
                self.pairs.append((i+1, j+self.N_div2+1))


        self.association = 0  # GO if np.diff(self.pair)[0]%2==self.association
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        # Durations (stimulus duration will be drawn from an exponential)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': -1., 'miss': 0.}
        if rewards:
            self.rewards.update(rewards)

        #stim 1 is first stimuli, stim 2 is first and second, stim 3 is none, stim 4 is second stimuli
        if interpolate_delay > 0:
            stim2 = interpolate_delay*100
            stim3 = 0
        elif interpolate_delay < 0:
            stim2 = 0
            stim3 = -interpolate_delay*100
        elif interpolate_delay == 0:
            stim2 = 0
            stim3 = 0

        self.timing = {
            'fixation': 0,
            'stim1': 200,
            'stim2': stim2,
            'stim3': stim3,
            'stim4': 200, #np.random.randint(500, 1000),
            'decision': 3000}
        if timing:
            self.timing.update(timing)

        self.abort = False
        # action and observation spaces
        name = {'fixation': 0, 'stimulus': range(1, self.N+1)}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.N+1,),
                                            dtype=np.float32, name=name)

        self.action_space = spaces.Discrete(2, name={'fixation': 0, 'go': 1})

    def _new_trial(self, **kwargs):
        pair = self.pairs[self.rng.choice(len(self.pairs))]
        trial = {
            'pair': pair,
            'ground_truth': pair[0] == pair[1]- (self.N_div2),
        }
        trial.update(kwargs)
        pair = trial['pair']

        periods = ['fixation', 'stim1', 'stim2', 'stim3', 'stim4', 'decision']
        self.add_period(periods)

        # set observations
        self.add_ob(1, where='fixation')
        self.add_ob(1, 'stim1', where=pair[0])
        self.add_ob(1, 'stim2', where=pair[0])
        self.add_ob(1, 'stim2', where=pair[1])
        self.add_ob(1, 'stim4', where=pair[1])
        self.set_ob(0, 'decision')
        # set ground truth
        self.set_groundtruth(trial['ground_truth'], 'decision')

        # if trial is GO the reward is set to R_MISS and  to 0 otherwise
        self.r_tmax = self.rewards['miss']*trial['ground_truth']
        self.performance = 1-trial['ground_truth']

        return trial

    def _step(self, action, **kwargs):
        new_trial = False
        # rewards
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']
                    self.performance = 0
                new_trial = True

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}