"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def train(self):
        """Training procedure for HAPPO."""
        actor_train_infos = []

        # factor is used for considering updates made by previous agents
        # 初始化 factor [episode_length, 进程数量, 1] 初始值都是1
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            # gae / advantage -- 对比 OnPolicyCriticBufferEP.compute_returns
            # self.critic_buffer.returns： V网络的标签值 = GAE(step) + V网络(step)
            # self.critic_buffer.value_preds： V网络(step)

            # advantages [episode_length, 进程数量, 1] = Q - V
            advantages = self.critic_buffer.returns[:-1] - \
                         self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            # gae / advantage -- 对比 OnPolicyCriticBufferEP.compute_returns
            advantages = (self.critic_buffer.returns[:-1]
                          - self.critic_buffer.value_preds[:-1])

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # 是否打乱agent顺序
        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())

        # 对于每个agent来说
        for agent_id in agent_order:
            # 在buffer里更新当前的actor的factor
            self.actor_buffer[agent_id].update_factor(factor)  # current actor save factor

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # compute action log probs for the actor before update.
            # 计算策略更新前actor的动作概率
            # 伪代码12行 - 分母
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id].obs[:-1].reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id].rnn_states[0:1].reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(-1, *self.actor_buffer[agent_id].actions.shape[2:]),
                self.actor_buffer[agent_id].masks[:-1].reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id].active_masks[:-1].reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update actor
            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id],
                    advantages.copy(),
                    "EP"
                )
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id],
                    advantages[:, :, agent_id].copy(),
                    "FP"
                )

            # compute action log probs for updated agent
            # 计算策略更新后actor的动作概率
            # 伪代码12行 - 分子
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id].obs[:-1].reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id].rnn_states[0:1].reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(-1, *self.actor_buffer[agent_id].actions.shape[2:]),
                self.actor_buffer[agent_id].masks[:-1].reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id].active_masks[:-1].reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update factor for next agent
            # 更新下一个agent的factor -- 伪代码12行
            # 把刚计算好的agent_id的前后策略的比值，乘到factor上
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # critic更新
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
