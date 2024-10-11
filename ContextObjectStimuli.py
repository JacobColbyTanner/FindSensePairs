#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import neurogym as ngym
from neurogym import spaces
import random


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
                 num_contexts=2, num_objects=4, num_actions=1, object_sequence_length=4,
                 context_functions=None, trial_type="object_transition"):
        super().__init__(dt=dt)

        # Task parameters
        self.num_contexts = num_contexts
        self.num_objects = num_objects
        self.num_actions = num_actions + 1  # +1 for the 'no action'
        self.context_functions = context_functions or self._default_context_functions()
        self.trial_type = trial_type
        self.object_sequence_length = object_sequence_length

        assert trial_type in ["context_memory",
                              "object_transition", "single_object"]

        # Rewards
        self.rewards = {'correct': +1., 'fail': 0., 'no_action': 0.}
        if rewards:
            self.rewards.update(rewards)

        # Timing
        if self.trial_type == "context_memory":
            self.timing = {
                f'stim{i+1}': 1000 for i in range(self.object_sequence_length)}
        elif self.trial_type == "object_transition":
            self.timing = {
                'stim1': random.randint(300, 700),
                'stim2': random.randint(300, 700),
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
        if self.trial_type == "context_memory":
            return self._new_context_memory_trial(**kwargs)
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

    def _new_context_memory_trial(self, **kwargs):
        # Choose a single context
        context = self.rng.choice(self.num_contexts)

        # Choose objects based on object_sequence_length
        objects = self.rng.choice(
            self.num_objects, size=self.object_sequence_length)

        # Determine correct actions for each object
        actions = [self.context_functions[context][obj] for obj in objects]

        trial = {
            'context': context,
            'objects': objects,
            'actions': actions,
        }
        trial.update(kwargs)

        # Add periods for each stimulus
        self.add_period(
            [f'stim{i+1}' for i in range(self.object_sequence_length)])

        # Add context only for the first object
        self.add_ob(1, 'stim1', where=[
                    context, self.num_contexts + objects[0]])

        # For the remaining objects, only show the object, not the context
        for i in range(1, self.object_sequence_length):
            self.add_ob(1, f'stim{i+1}',
                        where=[self.num_contexts + objects[i]])

        # Set ground truth for all stimuli
        for i, action in enumerate(actions):
            self.set_groundtruth(action, f'stim{i+1}')

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
