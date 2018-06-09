import numpy as np
"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""

class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class AdaptiveSchedule(object):
    def __init__(self, linear_timesteps, linear_final_p, episodes_mem=5, min_p=0.02, max_p=0.5):

        """Adaptive to the differences between rewards
        Starting as a Linear Schedule for linear_timesteps
        Afterward, the returned p value changes according to the changes in rewards
        Parameters
        ----------
        linear_timesteps, linear_final_p : LinearSchedule parameters
        episodes_mem: int
            How many episodes should be played between updates to p
        min_p: float
            minimal p value while in adaptive mode
        min_p: float
            maximal p value while in adaptive mode
        """
        self.linear_schedule = LinearSchedule(linear_timesteps, linear_final_p)
        self.delta_p = (1 / 1000000) * episodes_mem * 2000
        self.last_reward = - 20
        self.adaptive_p = 1
        self.min_p = min_p
        self.max_p = max_p
        self.episodes_mem = episodes_mem
        self.episodes_count = 1

    def value(self, t):
        """See Schedule.value"""
        if t < self.linear_schedule.schedule_timesteps:
            self.adaptive_p = self.linear_schedule.value(t)
        return self.adaptive_p

    def add_reward(self, episode_rewards):
        cur_count = len(episode_rewards)
        if cur_count != self.episodes_count and cur_count % self.episodes_mem == 1:
            """change p according to last_reward vs new_reward """
            new_reward = np.mean(episode_rewards[-self.episodes_mem - 1: -1])
            if new_reward >= self.last_reward:
                self.adaptive_p -= self.delta_p
                self.adaptive_p = max(self.min_p, self.adaptive_p)
            else:
                self.adaptive_p += self.delta_p
                self.adaptive_p = min(self.max_p, self.adaptive_p)
            self.last_reward = new_reward
            self.episodes_count = cur_count
