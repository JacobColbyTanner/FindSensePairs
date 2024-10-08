#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import neurogym as ngym
from neurogym import spaces


class ContextObjectStimuli(ngym.TrialEnv):
    r"""Context-Object-Action task.

    The agent is presented with a context stimulus and object stimuli.
    Based on the context and object, the agent must choose the correct action.
    Action 0 is reserved as a 'no action' or 'wait' action.
    """

    metadata = {
        'paper_link': None,
        'paper_name': None,
        'tags': ['perceptual', 'context-dependent', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None,
                 num_contexts=2, num_objects=4, num_actions=1,
                 context_functions=None, trial_type="object_transition"):
        super().__init__(dt=dt)

        # Task parameters
        self.num_contexts = num_contexts
        self.num_objects = num_objects
        self.num_actions = num_actions + 1  # +1 for the 'no action'
        self.context_functions = context_functions or self._default_context_functions()
        self.trial_type = trial_type

        assert trial_type in ["context_transition",
                              "object_transition", "single_object"]

        # Rewards
        self.rewards = {'correct': +1., 'fail': 0., 'no_action': 0.}
        if rewards:
            self.rewards.update(rewards)

        # Timing
        if self.trial_type == "context_transition":
            self.timing = {
                'stim1': 1000,
                'stim2': 1000,
                'stim3': 1000,
                'stim4': 1000,
            }
        elif self.trial_type == "object_transition":
            self.timing = {
                'stim1': 1000,
                'stim2': 1000,
            }
        else:  # single_object
            self.timing = {
                'stim': 1000,
            }
        if timing:
            self.timing.update(timing)

        # Observation and action spaces
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(num_contexts + num_objects,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_actions)

    def _default_context_functions(self):
        # Default context functions: randomly map each object to an action for each context
        return [
            {obj: self.rng.choice(range(self.num_actions))
             for obj in range(self.num_objects)}
            for _ in range(self.num_contexts)
        ]

    def _new_trial(self, **kwargs):
        if self.trial_type == "context_transition":
            return self._new_context_transition_trial(**kwargs)
        elif self.trial_type == "object_transition":
            return self._new_object_transition_trial(**kwargs)
        else:  # single_object
            return self._new_single_object_trial(**kwargs)

    def _new_single_object_trial(self, **kwargs):
        # Choose context and object
        context = self.rng.choice(self.num_contexts)
        object = self.rng.choice(self.num_objects)

        # Determine correct action
        action = self.context_functions[context](object)

        trial = {
            'context': context,
            'object': object,
            'action': action,
        }
        trial.update(kwargs)

        # Set up period
        self.add_period(['stim'])

        # Set observation
        self.add_ob(1, 'stim', where=[context, self.num_contexts + object])

        # Set ground truth
        self.set_groundtruth(action, 'stim')

        return trial

    def _new_object_transition_trial(self, **kwargs):
        # This is the same as the previous _new_single_trial method
        context = self.rng.choice(self.num_contexts)
        object1 = self.rng.choice(self.num_objects)
        object2 = self.rng.choice(self.num_objects)

        action1 = self.context_functions[context][object1]
        action2 = self.context_functions[context][object2]

        trial = {
            'context': context,
            'object1': object1,
            'object2': object2,
            'action1': action1,
            'action2': action2,
        }
        trial.update(kwargs)

        self.add_period(['stim1', 'stim2'])

        self.add_ob(1, 'stim1', where=[context, self.num_contexts + object1])
        self.add_ob(1, 'stim2', where=[context, self.num_contexts + object2])

        self.set_groundtruth(action1, 'stim1')
        self.set_groundtruth(action2, 'stim2')

        return trial

    def _new_context_transition_trial(self, **kwargs):
        # This is the same as the previous _new_transition_trial method
        context1 = self.rng.choice(self.num_contexts)
        context2 = self.rng.choice(self.num_contexts)
        while context2 == context1:
            context2 = self.rng.choice(self.num_contexts)

        object1 = self.rng.choice(self.num_objects)
        object2 = self.rng.choice(self.num_objects)
        object3 = self.rng.choice(self.num_objects)
        object4 = self.rng.choice(self.num_objects)

        action1 = self.context_functions[context1][object1]
        action2 = self.context_functions[context1][object2]
        action3 = self.context_functions[context2][object3]
        action4 = self.context_functions[context2][object4]

        trial = {
            'context1': context1,
            'context2': context2,
            'object1': object1,
            'object2': object2,
            'object3': object3,
            'object4': object4,
            'action1': action1,
            'action2': action2,
            'action3': action3,
            'action4': action4,
        }
        trial.update(kwargs)

        self.add_period(['stim1', 'stim2', 'stim3', 'stim4'])

        self.add_ob(1, 'stim1', where=[context1, self.num_contexts + object1])
        self.add_ob(1, 'stim2', where=[context1, self.num_contexts + object2])
        self.add_ob(1, 'stim3', where=[context2, self.num_contexts + object3])
        self.add_ob(1, 'stim4', where=[context2, self.num_contexts + object4])

        self.set_groundtruth(action1, 'stim1')
        self.set_groundtruth(action2, 'stim2')
        self.set_groundtruth(action3, 'stim3')
        self.set_groundtruth(action4, 'stim4')

        return trial

    def _step(self, action):
        new_trial = False
        reward = 0
        ob = self.ob_now
        gt = self.gt_now

        if action != 0:  # Any non-zero action is considered a decision
            new_trial = True
            if action == gt:
                reward = self.rewards['correct']
            else:
                reward = self.rewards['fail']
        else:
            reward = self.rewards['no_action']

        done = new_trial or self.t >= self.tmax
        return ob, reward, done, {'new_trial': new_trial, 'gt': gt}
